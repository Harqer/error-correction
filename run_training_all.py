#!/usr/bin/env python3
"""
Train **one model per experiment** by iterating over all ``.npz`` files under
``pretrain_data`` (recursively) and launching ``ai_models/model_mla.py`` once per
file. With ``--npu`` and multiple NPUs/GPUs available, jobs are scheduled in
parallel.

Usage:
    python run_training_all.py [--npu] [--data-root PRETRAIN_DIR]...
"""

import argparse
import os
import socket
import subprocess
import sys
import time
from itertools import cycle
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple

from ai_models.model_mla import get_basis_from_filename, model_stem_from_npz
from ai_models.pauli_plus_dataset import find_label_key

try:
    import torch  # type: ignore
except ImportError:
    # torch may not be available at import time; handle gracefully
    torch = None  # type: ignore

# Training parameters.  Strings are used here as these values are passed
# directly on the command line to the child process.
DEFAULT_EPOCHS: int = 20
DEFAULT_BATCH_SIZE: int = 16
MODEL_DIR: Path = Path("ai_models") / "models"


def main() -> None:
    """Entry point of the training launcher."""
    parser = argparse.ArgumentParser(
        description=(
            "Train all NPZ files serially or in parallel across NPUs/GPUs. "
            "When --npu is supplied and multiple NPUs are available the "
            "training tasks will be dispatched concurrently across devices."
        )
    )
    parser.add_argument(
        "--npu",
        action="store_true",
        help=(
            "Use NPUs for training (requires Ascend PyTorch with torch.npu support). "
            "If multiple devices are available the launcher will schedule jobs "
            "across them in parallel."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs to pass through each dataset (default: 20)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Mini-batch size for each training run (default: 16)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit the number of samples loaded per dataset when invoking the trainer",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        action="append",
        dest="data_roots",
        help=(
            "Directory containing .npz training datasets. May be supplied "
            "multiple times. Without this flag the script searches both "
            "pretrain_data/ and simulated_data/. Providing --data-root "
            "overrides the defaults, so use it repeatedly to enumerate all "
            "desired dataset roots."
        ),
    )
    args = parser.parse_args()

    # Collect all npz files from the requested data directories.  When the user
    # does not provide ``--data-root`` we prefer ``pretrain_data/`` (populated by
    # ``make_all_pretraining_noise.py``) and only fall back to ``simulated_data``
    # if no datasets were discovered.  This avoids training each experiment
    # twice when the helper script has already copied the generated ``.npz``
    # files into ``pretrain_data/``.
    fallback_roots: List[Path]
    if args.data_roots:
        data_roots = [Path(root) for root in args.data_roots]
        fallback_roots = []
    else:
        data_roots = [Path("pretrain_data")]
        fallback_roots = [Path("simulated_data")]

    searched_roots: List[Path] = []
    all_npz: List[Path] = []
    seen: Set[Path] = set()

    def collect(root: Path) -> int:
        searched_roots.append(root)
        if not root.is_dir():
            print(f"No dataset directory found at {root}; skipping")
            return 0
        found = 0
        for npz in sorted(root.rglob("*.npz")):
            resolved = npz.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            all_npz.append(npz)
            found += 1
        return found

    discovered = sum(collect(root) for root in data_roots)
    if discovered == 0 and not args.data_roots:
        for root in fallback_roots:
            collect(root)

    invalid_logged: Set[Path] = set()
    label_cache: Dict[Path, Optional[str]] = {}

    def filter_valid_npz(candidates: Sequence[Path]) -> List[Path]:
        valid: List[Path] = []
        for npz in candidates:
            basis = get_basis_from_filename(npz.name)
            if basis == -1:
                basis = 0
            if npz in label_cache:
                label_key = label_cache[npz]
            else:
                label_key = find_label_key(npz, basis)
                label_cache[npz] = label_key
            if label_key is None:
                if npz not in invalid_logged:
                    print(f"Skipping {npz} – no valid label array found")
                    invalid_logged.add(npz)
                continue
            valid.append(npz)
        return valid

    if not all_npz:
        joined = ", ".join(str(root) for root in searched_roots)
        print(f"No .npz files found in any of: {joined}")
        if not args.data_roots:
            print(
                "Hint: run 'python make_all_pretraining_noise.py' to populate "
                "pretrain_data/ before launching training."
            )
        return

    npz_files = filter_valid_npz(all_npz)

    if not npz_files and fallback_roots and not args.data_roots:
        # No usable datasets were found under the primary roots (for example
        # ``pretrain_data`` may contain partially generated files without
        # labels).  Try the fallback directories before giving up so that the
        # launcher can still train on datasets under ``simulated_data``.
        new_found = 0
        for root in fallback_roots:
            new_found += collect(root)
        if new_found:
            npz_files = filter_valid_npz(all_npz)

    if not npz_files:
        joined = ", ".join(str(root) for root in searched_roots)
        print(f"No datasets with valid labels found under {joined}; aborting")
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    expected_models: Dict[Path, Path] = {npz: _get_model_path(npz) for npz in npz_files}
    print("Planned model outputs:")
    for npz, model_path in expected_models.items():
        print(f"  {npz} -> {model_path}")
    run_start_time = time.time()

    # Determine the accelerator indices that are available.  For NPUs we rely on
    # ``torch.npu`` when possible, but also honour ``ASCEND_VISIBLE_DEVICES`` and
    # related environment variables – these are commonly used to expose multiple
    # devices to the process while ``torch.npu.device_count()`` may still report
    # ``1``.  When no accelerators are present the list falls back to ``[0]`` so
    # the remainder of the launcher logic can treat it uniformly.
    device_indices: Sequence[int]
    if args.npu:
        device_indices = _discover_npu_devices()
    else:
        device_indices = _discover_gpu_devices()

    device_count = len(device_indices)

    # Helper to build the command list for a single training invocation
    def build_cmd(
        npz: Path,
        model_path: Path,
        device_index: Optional[int] = None,
        tqdm_position: Optional[int] = None,
    ) -> List[str]:
        cmd: List[str] = [
            sys.executable,
            "-m",
            "ai_models.model_mla",
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--npz_file",
            str(npz),
            "--model-save-path",
            str(model_path),
        ]
        if args.npu:
            cmd.append("--npu")
        if device_index is not None:
            cmd.extend(["--device_index", str(device_index)])
        if tqdm_position is not None:
            cmd.extend(["--tqdm_position", str(max(tqdm_position, 0))])
        if args.max_samples is not None:
            cmd.extend(["--max_samples", str(args.max_samples)])
        return cmd

    # If more than one device is available we run training tasks in parallel.
    # Each spawned process is pinned to a single device using the appropriate
    # visibility environment variable.  The concurrency level is capped at
    # the number of devices to avoid oversubscription.
    if device_count > 1:
        device_desc = ", ".join(str(idx) for idx in device_indices)
        print(
            f"Detected {device_count} {'NPUs' if args.npu else 'GPUs'} ({device_desc}). "
            "Launching tasks in parallel."
        )
        processes: List[Tuple[subprocess.Popen, List[str], int]] = []
        allocated_ports: Set[int] = set()
        device_cycle = cycle(device_indices)
        for npz in npz_files:
            device_idx = next(device_cycle)
            env = os.environ.copy()

            if args.npu:
                # Ascend PyTorch honours these environment variables when selecting
                # a default device.  Setting them ensures libraries that bypass
                # ``torch.npu.set_device`` still remain on the assigned device.
                _apply_npu_env(env, device_idx)

            model_path = expected_models[npz]
            cmd = build_cmd(npz, model_path, device_idx, device_idx)
            master_port = _allocate_master_port(allocated_ports)
            env.setdefault("MASTER_ADDR", "127.0.0.1")
            env["MASTER_PORT"] = str(master_port)
            env.setdefault("PYTHONUNBUFFERED", "1")
            print(
                "[async] starting on device "
                f"{device_idx}: {' '.join(cmd)} (MASTER_PORT={master_port})"
            )
            processes.append((subprocess.Popen(cmd, env=env), cmd, master_port))
            # Limit the number of concurrent processes to the number of devices
            if len(processes) >= device_count:
                proc, proc_cmd, port = processes.pop(0)
                _wait_for_process(proc, proc_cmd)
                allocated_ports.discard(port)
        # Wait for any remaining processes to finish
        for proc, proc_cmd, port in processes:
            _wait_for_process(proc, proc_cmd)
            allocated_ports.discard(port)
        _verify_models(expected_models, run_start_time)
        return

    # Serial fallback: one training process at a time
    position_cycle: Optional[Iterator[int]] = None
    if device_count:
        position_cycle = cycle(device_indices)

    for npz in npz_files:
        position = next(position_cycle) if position_cycle is not None else None
        model_path = expected_models[npz]
        cmd = build_cmd(npz, model_path, position, position)
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        if args.npu and position is not None:
            _apply_npu_env(env, position)
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, env=env)

    _verify_models(expected_models, run_start_time)


def _get_model_path(npz_file: Path) -> Path:
    """Return the expected checkpoint path for a given dataset."""

    stem = model_stem_from_npz(npz_file)
    return MODEL_DIR / f"{stem}.pth"


def _wait_for_process(process: subprocess.Popen, cmd: List[str]) -> None:
    """Wait for ``process`` to finish and raise if it exits with an error."""

    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


def _allocate_master_port(allocated_ports: Set[int]) -> int:
    """Return a TCP port that is currently free on the host.

    Torch's distributed initialisation uses ``MASTER_PORT`` and defaults to a
    static value, which causes contention when multiple independent training
    processes start concurrently.  This helper finds a free port and reserves it
    for the lifetime of the spawned subprocess to avoid collisions.
    """

    for _ in range(100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            port = sock.getsockname()[1]
        if port not in allocated_ports:
            allocated_ports.add(port)
            return port
    raise RuntimeError("Unable to allocate a free MASTER_PORT for distributed training")


def _verify_models(models: Dict[Path, Path], start_time: float) -> None:
    """Ensure that every dataset produced a fresh model checkpoint."""

    missing: List[str] = []
    stale: List[str] = []
    tolerance = 1.0  # seconds; accounts for filesystem timestamp precision

    for npz, model_path in models.items():
        if not model_path.exists():
            missing.append(f"{npz} -> {model_path}")
            continue
        if model_path.stat().st_mtime < start_time - tolerance:
            stale.append(f"{npz} -> {model_path}")

    if missing or stale:
        error_lines = ["Model verification failed after training."]
        if missing:
            error_lines.append("Missing checkpoints:\n  " + "\n  ".join(missing))
        if stale:
            error_lines.append("Stale checkpoints (not updated in this run):\n  " + "\n  ".join(stale))
        raise RuntimeError("\n".join(error_lines))

    print(f"All models saved successfully in {MODEL_DIR.resolve()}.")


def _discover_npu_devices() -> Sequence[int]:
    """Return the list of NPU device indices visible to the process."""

    env_devices = _parse_visible_devices(
        os.environ,
        (
            "ASCEND_VISIBLE_DEVICES",
            "ASCEND_RT_VISIBLE_DEVICES",
            "NPU_VISIBLE_DEVICES",
            "DEVICE_ID_LIST",
        ),
    )

    # Ensure torch is initialised with NPU support if possible.  Some Ascend
    # installations require importing ``torch_npu`` before ``torch.npu`` becomes
    # available.  Any import error is ignored so we can still fall back to CPU.
    if torch is not None and not hasattr(torch, "npu"):
        try:
            import importlib

            importlib.import_module("torch_npu")
        except Exception:
            pass

    count = 0
    if torch is not None and hasattr(torch, "npu"):
        try:
            count = int(max(getattr(torch.npu, "device_count", lambda: 0)(), 0))
        except Exception:
            count = 0

    if env_devices:
        devices = env_devices
        # When ``torch.npu.device_count`` under-reports the available hardware we
        # still honour the explicit environment list so that the launcher can
        # distribute work across the provided indices.
        if count and max(devices, default=-1) >= count:
            print(
                "[warn] torch.npu.device_count() returned fewer devices than "
                "the environment exposes; proceeding with the environment list."
            )
    elif count > 0:
        devices = list(range(count))
    else:
        devices = [0]

    return tuple(devices)


def _discover_gpu_devices() -> Sequence[int]:
    """Return the list of GPU device indices visible to the process."""

    env_devices = _parse_visible_devices(os.environ, ("CUDA_VISIBLE_DEVICES",))
    if torch is not None and torch.cuda.is_available():
        count = torch.cuda.device_count()
    else:
        count = 0

    if env_devices:
        # CUDA exposes devices in the order provided by CUDA_VISIBLE_DEVICES.
        devices = env_devices
    elif count > 0:
        devices = list(range(count))
    else:
        devices = [0]

    return tuple(devices)


def _parse_visible_devices(env: Dict[str, str], keys: Sequence[str]) -> Optional[List[int]]:
    """Parse accelerator visibility environment variables into integer lists."""

    for key in keys:
        raw = env.get(key)
        if not raw:
            continue
        tokens = raw.replace(";", ",").split(",")
        devices: List[int] = []
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            try:
                devices.append(int(token))
            except ValueError:
                # Ignore non-integer tokens; leave the loop so the next key can
                # be considered instead of returning a partial result.
                devices = []
                break
        if devices:
            return devices
    return None


def _apply_npu_env(env: Dict[str, str], device_idx: int) -> None:
    """Restrict a child process to ``device_idx`` for Ascend NPUs."""

    value = str(device_idx)
    env["ASCEND_DEVICE_ID"] = value
    env["DEVICE_ID"] = value
    env["ASCEND_VISIBLE_DEVICES"] = value
    env["ASCEND_RT_VISIBLE_DEVICES"] = value
    env["NPU_VISIBLE_DEVICES"] = value


if __name__ == "__main__":
    main()
