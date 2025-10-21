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
from contextlib import suppress
from itertools import cycle
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

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
THIS_DIR = Path(__file__).resolve().parent
RUN_CREATE_ALL = THIS_DIR / "run_create_all_samples.py"
EXPERIMENT_ROOT = THIS_DIR / "experiment_data"
ENV_EXPERIMENT_ROOTS = "ALPHAQUBIT_EXPERIMENT_ROOTS"


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
    parser.add_argument(
        "--experiment-root",
        type=Path,
        action="append",
        dest="experiment_roots",
        help=(
            "Stim experiment directory used when automatically generating datasets. "
            "May be supplied multiple times. Defaults to experiment_data/ or the "
            "ALPHAQUBIT_EXPERIMENT_ROOTS environment variable if set."
        ),
    )
    args = parser.parse_args()

    experiment_roots = _resolve_experiment_roots(args.experiment_roots)

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
        auto_generated = _auto_generate_datasets(
            fallback_roots, use_npu=args.npu, experiment_roots=experiment_roots
        )
        if auto_generated:
            all_npz.clear()
            seen.clear()
            searched_roots.clear()
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
                local_device_idx = _apply_npu_env(env, device_idx)
            else:
                local_device_idx = device_idx

            model_path = expected_models[npz]
            cmd = build_cmd(npz, model_path, local_device_idx, device_idx)
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
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        if args.npu and position is not None:
            device_index = _apply_npu_env(env, position)
        else:
            device_index = position
        cmd = build_cmd(npz, model_path, device_index, position)
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


def _resolve_experiment_roots(cli_roots: Optional[Sequence[Path]]) -> List[Path]:
    """Return the list of experiment directories to probe for Stim circuits."""

    if cli_roots:
        return [root.resolve() for root in cli_roots]

    env_value = os.environ.get(ENV_EXPERIMENT_ROOTS, "")
    roots: List[Path] = []
    if env_value:
        for chunk in env_value.split(os.pathsep):
            chunk = chunk.strip()
            if not chunk:
                continue
            roots.append(Path(chunk).expanduser().resolve())

    if not roots:
        roots = [EXPERIMENT_ROOT]

    return roots


def _auto_generate_datasets(
    fallback_roots: Iterable[Path], *, use_npu: bool, experiment_roots: Sequence[Path]
) -> bool:
    """Generate missing experiment datasets using ``run_create_all_samples``."""

    if not fallback_roots:
        return False
    if not RUN_CREATE_ALL.exists() or not RUN_CREATE_ALL.is_file():
        return False
    if not experiment_roots:
        return False

    existing_roots = [root for root in experiment_roots if root.exists()]
    if not existing_roots:
        return False

    try:
        first_root = next(iter(fallback_roots))
    except StopIteration:
        return False

    first_root.mkdir(parents=True, exist_ok=True)

    device = _default_simulator_device(use_npu=use_npu)
    cmd = [
        sys.executable,
        str(RUN_CREATE_ALL),
        "--output-dir",
        str(first_root),
        "--layout",
        "by_experiment",
        "--skip-existing",
        "--device",
        device,
    ]

    for root in existing_roots:
        cmd.append(str(root))

    print(
        "No datasets detected – generating simulated samples for all experiments\n"
        f"Running: {' '.join(cmd)}"
    )

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(
            "Automatic dataset generation failed; proceeding without generated"
            f" samples (return code {exc.returncode})."
        )
        return False

    return True


def _default_simulator_device(*, use_npu: bool) -> str:
    """Select an appropriate device flag for ``run_create_all_samples``."""

    if use_npu:
        return "npu"
    if torch is not None:
        with suppress(Exception):
            if torch.cuda.is_available():
                return "cuda"
    return "cpu"


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
        devices, failures = _validate_npu_devices(env_devices)
        if failures:
            failure_msgs = ", ".join(
                f"{idx}: {str(exc).splitlines()[0]}" for idx, exc in failures
            )
            print(
                "[warn] Ignoring invalid NPU indices from environment: "
                f"{failure_msgs}"
            )
        if devices:
            # When ``torch.npu.device_count`` under-reports the available
            # hardware we still honour the explicit environment list so that the
            # launcher can distribute work across the provided indices.
            if count and max(devices, default=-1) >= count:
                print(
                    "[warn] torch.npu.device_count() returned fewer devices "
                    "than the environment exposes; proceeding with the "
                    "environment list."
                )
        elif count > 0:
            print(
                "[warn] No usable NPU indices remained after filtering the "
                "environment list; falling back to torch.npu.device_count()."
            )
            devices = list(range(count))
        else:
            print(
                "[warn] No usable NPU indices remained after filtering the "
                "environment list; defaulting to device 0."
            )
            devices = [0]
    elif count > 0:
        devices = list(range(count))
    else:
        devices = [0]

    return tuple(devices)


def _validate_npu_devices(devices: Sequence[int]) -> Tuple[List[int], List[Tuple[int, Exception]]]:
    """Return devices that torch.npu can select and the failures encountered."""

    if torch is None or not hasattr(torch, "npu"):
        return list(dict.fromkeys(devices)), []

    unique_devices: List[int] = []
    seen: Set[int] = set()
    for idx in devices:
        if idx in seen:
            continue
        seen.add(idx)
        unique_devices.append(idx)

    valid: List[int] = []
    failures: List[Tuple[int, Exception]] = []
    current_device: Optional[int] = None
    with suppress(Exception):
        current_device = torch.npu.current_device()

    for idx in unique_devices:
        try:
            torch.npu.set_device(idx)
        except Exception as exc:  # pragma: no cover - depends on hardware
            failures.append((idx, exc))
        else:  # pragma: no cover - depends on hardware
            valid.append(idx)

    if current_device is not None:
        with suppress(Exception):
            torch.npu.set_device(current_device)

    return valid, failures


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


def _apply_npu_env(env: Dict[str, str], device_idx: int) -> int:
    """Restrict a child process to ``device_idx`` for Ascend NPUs.

    Returns the device index that should be forwarded to the training command.
    When the visibility environment variables expose a single accelerator the
    runtime renumbers it to ``0``.  Passing the physical index directly would
    therefore fail inside the child process.  Instead this helper keeps the
    physical ID in the environment while returning the logical index that the
    spawned trainer should request via ``--device_index``.
    """

    value = str(device_idx)
    env["ASCEND_DEVICE_ID"] = value
    env["DEVICE_ID"] = value
    env["ASCEND_VISIBLE_DEVICES"] = value
    env["ASCEND_RT_VISIBLE_DEVICES"] = value
    env["NPU_VISIBLE_DEVICES"] = value

    # Child processes see only the devices listed in the visibility variables.
    # With a single entry this means ``torch.npu`` exposes it as ``0``.
    return 0


if __name__ == "__main__":
    main()
