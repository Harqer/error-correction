"""Batch helper for generating all experiment .npz bundles.

The google_qec_simulator CLI only accepts a *single* experiment directory at a
time.  This script discovers every directory under ``experiment_data/`` that
contains ``*.stim`` circuits and sequentially invokes the simulator so that we
obtain one ``samples_<experiment>.npz`` per experiment in ``simulated_data/``.

Example
-------

.. code-block:: bash

   python run_create_all_samples.py --shots 2000 --device cuda

"""

from __future__ import annotations

import argparse
import subprocess
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
        default=Path(__file__).resolve().parent / "experiment_data",
        help="Top-level directory that holds experiment subfolders",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "simulated_data",
        help="Destination directory for generated samples_*.npz files",
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

    exp_root = args.experiment_root.resolve()
    out_root = args.output_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    experiments = discover_experiments(exp_root)
    if not experiments:
        raise SystemExit(f"No experiments with *.stim found under {exp_root}")

    print(f"Found {len(experiments)} experiment folder(s) under {exp_root}")

    for idx, exp_dir in enumerate(experiments, start=1):
        rel_name = exp_dir.relative_to(exp_root).as_posix()
        safe_name = rel_name.replace("/", "_")
        out_file = out_root / f"samples_{safe_name}.npz"

        if args.skip_existing and out_file.exists():
            print(f"[{idx}/{len(experiments)}] Skip existing {out_file.name}")
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

        print(f"[{idx}/{len(experiments)}] Running {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print(f"[{idx}/{len(experiments)}] DONE → {out_file}")


if __name__ == "__main__":
    main()
