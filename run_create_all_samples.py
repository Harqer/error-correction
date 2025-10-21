"""Batch helper for generating all experiment .npz bundles.

The google_qec_simulator CLI only accepts a *single* experiment directory at a
time. This script discovers every directory under ``experiment_data/`` that
contains ``*.stim`` circuits. It then sequentially invokes the simulator so
that we obtain one ``samples_<experiment>.npz`` per experiment under
``simulated_data/``.

By default we **preserve the experiment folder structure** under
``simulated_data/`` so later stages (pretraining/training) can keep a
one-model-per-experiment workflow.

Example
-------

.. code-block:: bash

   python run_create_all_samples.py --shots 2000 --device cuda

"""

from __future__ import annotations

import argparse
import subprocess
import threading
import time
import sys
from pathlib import Path


def discover_experiments(root: Path) -> list[Path]:
    """Return unique directories underneath ``root`` that host ``*.stim`` files."""

    stim_parents = {path.parent for path in root.rglob("*.stim") if path.is_file()}
    return sorted(stim_parents, key=lambda p: p.relative_to(root).as_posix())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate .npz bundles for every experiment")
    parser.add_argument(
        "--experiment-root",
        type=Path,
        action="append",
        dest="experiment_roots",
        help=(
            "Top-level directory that holds experiment subfolders. May be supplied "
            "multiple times; when omitted the repository's experiment_data/ is used."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "simulated_data",
        help="Destination directory for generated samples_*.npz files",
    )
    parser.add_argument(
        "--layout",
        choices=("by_experiment", "flat"),
        default="by_experiment",
        help="Output layout: create subfolders per experiment (default) or a flat directory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device passed to google_qec_simulator (cpu|cuda|npu)",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=1000,
        help="Monte-Carlo shots per circuit",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip experiments whose samples_*.npz already exist in output-dir",
    )
    args = parser.parse_args()

    exp_roots = args.experiment_roots or [Path(__file__).resolve().parent / "experiment_data"]
    exp_roots = [root.resolve() for root in exp_roots]
    out_root = args.output_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    multi_root = len(exp_roots) > 1

    experiments: list[tuple[Path, Path]] = []
    for root in exp_roots:
        if not root.exists():
            print(f"Warning: experiment root {root} does not exist; skipping")
            continue
        found = discover_experiments(root)
        if not found:
            print(f"Warning: no experiments with *.stim found under {root}")
            continue
        print(f"Found {len(found)} experiment folder(s) under {root}")
        experiments.extend((root, exp_dir) for exp_dir in found)

    if not experiments:
        raise SystemExit("No experiments with *.stim found in the provided roots")

    total = len(experiments)

    for idx, (exp_root, exp_dir) in enumerate(experiments, start=1):
        rel = exp_dir.relative_to(exp_root)
        rel_name = rel.as_posix()
        if args.layout == "flat":
            safe_name = rel_name.replace("/", "_")
            if multi_root:
                safe_name = f"{exp_root.name}_{safe_name}"
            out_file = out_root / f"samples_{safe_name}.npz"
        else:
            # Preserve experiment folder structure and optionally prefix by root name
            rel_dir = out_root
            if multi_root:
                rel_dir /= exp_root.name
            rel_dir /= rel
            rel_dir.mkdir(parents=True, exist_ok=True)
            out_file = rel_dir / f"samples_{exp_dir.name}.npz"

        if args.skip_existing and out_file.exists():
            print(f"[{idx}/{total}] Skip existing {out_file.name}")
            continue

        cmd = [
            sys.executable,
            "-m",
            "google_qec_simulator.main",
            str(exp_dir),
            "--shots",
            str(args.shots),
            "--device",
            args.device,
            "--out",
            str(out_file),
        ]

        print(f"[{idx}/{total}] Running {' '.join(cmd)}", flush=True)

        # Stream the child process output so long-running simulations visibly progress.
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as proc:
            assert proc.stdout is not None  # for type-checkers

            # Emit a heartbeat if the simulator is silent for long stretches.
            last_line_time = time.monotonic()

            def heartbeat() -> None:
                while proc.poll() is None:
                    time.sleep(30)
                    if time.monotonic() - last_line_time >= 30 and proc.poll() is None:
                        print(f"[{idx}/{total}] … still running", flush=True)

            hb_thread = threading.Thread(target=heartbeat, daemon=True)
            hb_thread.start()

            try:
                for line in proc.stdout:
                    last_line_time = time.monotonic()
                    message = line.rstrip()
                    print(
                        f"[{idx}/{total}] | {message}" if message else f"[{idx}/{total}] |",
                        flush=True,
                    )
            except KeyboardInterrupt:
                proc.terminate()
                proc.wait()
                raise

            retcode = proc.wait()
            if retcode:
                raise subprocess.CalledProcessError(retcode, cmd)

        print(f"[{idx}/{total}] DONE → {out_file}", flush=True)


if __name__ == "__main__":
    main()
