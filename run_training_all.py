#!/usr/bin/env python3
"""
run_training_all.py

Enhanced training launcher for ALPHAQUBIT.  It iterates over all `.npz`
files under ``simulated_data`` and trains a model on each using
``ai_models/model_mla.py``.  When running with the ``--npu`` flag and
multiple Ascend NPUs are available the launcher dispatches multiple
training processes concurrently, pinning each process to a single device
via the ``NPU_VISIBLE_DEVICES`` environment variable.  This allows
utilisation of all available devices.  If NPUs are not present or
the ``--npu`` flag is omitted the script falls back to the original
serial behaviour and uses GPUs or the CPU as appropriate.

Usage:
    python run_training_all.py [--npu]

"""

import argparse
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from ai_models.model_mla import model_stem_from_npz

try:
    import torch  # type: ignore
except ImportError:
    # torch may not be available at import time; handle gracefully
    torch = None  # type: ignore

# Training parameters.  Strings are used here as these values are passed
# directly on the command line to the child process.
EPOCHS: str = "20"
BATCH_SIZE: str = "16"
SCRIPT: Path = Path("ai_models/model_mla.py")
DATA_DIR: Path = Path("simulated_data")
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
    args = parser.parse_args()

    # Collect all npz files in the data directory
    npz_files: List[Path] = sorted(DATA_DIR.glob("*.npz"))
    if not npz_files:
        print(f"No .npz files found in {DATA_DIR}")
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    expected_models: Dict[Path, Path] = {npz: _get_model_path(npz) for npz in npz_files}
    print("Planned model outputs:")
    for npz, model_path in expected_models.items():
        print(f"  {npz} -> {model_path}")
    run_start_time = time.time()

    # Determine the number of devices available.  Prefer NPUs when --npu is
    # specified, otherwise fall back to CUDA.  When torch is unavailable we
    # assume a single CPU device.
    device_count = 1
    if args.npu and torch is not None and hasattr(torch, "npu"):
        try:
            device_count = getattr(torch.npu, "device_count", lambda: 1)()
        except Exception:
            device_count = 1
    elif not args.npu and torch is not None and torch.cuda.is_available():
        device_count = torch.cuda.device_count()

    # Helper to build the command list for a single training invocation
    def build_cmd(
        npz: Path, device_index: Optional[int] = None, tqdm_position: Optional[int] = None
    ) -> List[str]:
        cmd: List[str] = [
            "python",
            str(SCRIPT),
            "--epochs",
            EPOCHS,
            "--batch_size",
            BATCH_SIZE,
            "--npz_file",
            str(npz),
        ]
        if args.npu:
            cmd.append("--npu")
        if device_index is not None:
            cmd.extend(["--device_index", str(device_index)])
        if tqdm_position is not None:
            cmd.extend(["--tqdm_position", str(max(tqdm_position, 0))])
        return cmd

    # If more than one device is available we run training tasks in parallel.
    # Each spawned process is pinned to a single device using the appropriate
    # visibility environment variable.  The concurrency level is capped at
    # the number of devices to avoid oversubscription.
    if device_count > 1:
        print(
            f"Detected {device_count} {'NPUs' if args.npu else 'GPUs'}. "
            "Launching tasks in parallel."
        )
        processes: List[Tuple[subprocess.Popen, List[str], int]] = []
        allocated_ports: Set[int] = set()
        for idx, npz in enumerate(npz_files):
            device_idx = idx % device_count
            env = os.environ.copy()
            cmd = build_cmd(npz, device_idx, device_idx)
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
    for npz in npz_files:
        position = 0 if (args.npu or (torch is not None and torch.cuda.is_available())) else None
        cmd = build_cmd(npz, position, position)
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
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


if __name__ == "__main__":
    main()
