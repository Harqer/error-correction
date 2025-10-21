"""Batch helper for generating all experiment .npz bundles.

The google_qec_simulator CLI only accepts a *single* experiment directory at a
time. This script discovers every directory under the default
``~/work/google_qec3v5_experiment_data`` folder (or any user supplied paths)
that contains ``*.stim`` circuits. It then sequentially invokes the simulator so
that we obtain one ``samples_<experiment>.npz`` per experiment under
``pretrain_data/``.

By default we **preserve the experiment folder structure** under
``pretrain_data/`` so later stages (pretraining/training) can keep a
one-model-per-experiment workflow.

Example
-------

.. code-block:: bash

   python run_create_all_samples.py ~/work/google_qec3v5_experiment_data --shots 2000 --device cuda

"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from google_qec_simulator.main import simulate_folder

DEFAULT_EXPERIMENT_ROOT = Path.home() / "work/google_qec3v5_experiment_data"

def discover_experiments(root: Path) -> list[Path]:
    """Return unique directories underneath ``root`` that host ``*.stim`` files."""

    stim_parents = {path.parent for path in root.rglob("*.stim") if path.is_file()}
    return sorted(stim_parents, key=lambda p: p.relative_to(root).as_posix())


def format_duration(seconds: float) -> str:
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{sec:02d}s"
    if minutes:
        return f"{minutes}m{sec:02d}s"
    return f"{sec}s"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate .npz bundles for every experiment")
    parser.add_argument(
        "experiment_roots",
        nargs="*",
        type=Path,
        help=(
            "Optional directories containing experiment subfolders. Multiple "
            "directories may be supplied; when omitted the default "
            f"{DEFAULT_EXPERIMENT_ROOT} is scanned."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "pretrain_data",
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
    parser.add_argument(
        "--experiment-root",
        dest="experiment_root",
        action="append",
        type=Path,
        help=(
            "Additional experiment directory (alias for positional experiment_roots). "
            "May be passed multiple times."
        ),
    )
    args = parser.parse_args()

    exp_roots = list(args.experiment_roots)
    if args.experiment_root:
        exp_roots.extend(args.experiment_root)
    if not exp_roots:
        exp_roots = [DEFAULT_EXPERIMENT_ROOT]
    else:
        # Preserve user-specified ordering while removing duplicates.
        exp_roots = list(dict.fromkeys(exp_roots))
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

    print(
        f"Preparing to run {total} experiment(s) with device={args.device}, shots={args.shots}",
        flush=True,
    )

    for idx, (exp_root, exp_dir) in enumerate(experiments, start=1):
        rel = exp_dir.relative_to(exp_root)
        rel_name = rel.as_posix()
        prefix = f"[{idx}/{total}]"
        print(f"{prefix} Experiment {rel_name}", flush=True)

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

        print(f"{prefix} Running experiment with simulate_folder → {out_file}", flush=True)

        start_time = time.monotonic()
        try:
            simulate_folder(
                exp_dir=exp_dir,
                out_file=out_file,
                shots=args.shots,
                device=args.device,
            )
        except Exception as exc:  # pragma: no cover - surfaced to CLI for debugging
            raise RuntimeError(f"{prefix} Simulation failed for {rel_name}") from exc

        elapsed = time.monotonic() - start_time
        print(f"{prefix} DONE in {format_duration(elapsed)} → {out_file}", flush=True)


if __name__ == "__main__":
    main()
