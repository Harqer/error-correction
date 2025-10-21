#!/usr/bin/env python3
"""
Generate ALL pretraining noise datasets for ALPHAQUBIT in one go.

Produces (example):
  pretrain_data/
    dem/                            # DEM syndromes/logicals (.npy)
    si1000/                         # SI1000 syndromes/logicals (.npy), optionally for a grid of p
    <experiment_A>/samples_*.npz    # Soft ("I/Q") readout .npz produced by google_qec_simulator
    <experiment_B>/samples_*.npz

Requirements: run from the repo root (where generate_data.py is).
Refs:
- `python generate_data.py --model dem|si1000|pauli_plus --samples N` (repo README)  # noqa
- `python run_create_all_samples.py` or `python google_qec_simulator/main.py ...`     # noqa

Example:
  # From within the repo root
  python make_all_pretraining_noise.py \
    --dem-samples 500000 \
    --si1000-samples 500000 \
    --si1000-p-grid 0.006,0.010,0.014 \
    --soft-shots 200000 \
    --soft-device auto \
    --out-dir pretrain_data
"""
import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

try:
    import yaml  # pyyaml is listed in requirements
except Exception as e:
    print("This script needs PyYAML (pip install pyyaml).", file=sys.stderr)
    raise

REPO_ROOT = Path(__file__).resolve().parent
GEN_SCRIPT = REPO_ROOT / "generate_data.py"
RUN_CREATE_ALL = REPO_ROOT / "run_create_all_samples.py"
GQEC_MAIN = REPO_ROOT / "google_qec_simulator" / "main.py"
OUTPUT_DIR = REPO_ROOT / "output"
SIMDATA_DIR = REPO_ROOT / "simulated_data"

def _run(cmd, cwd=None):
    print(f"\n$ {' '.join(map(str, cmd))}")
    res = subprocess.run(
        cmd, cwd=cwd or REPO_ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    print(res.stdout)
    if res.returncode != 0:
        raise RuntimeError(
            f"Command failed ({res.returncode}): {' '.join(map(str, cmd))}\n{res.stdout}"
        )
    return res.stdout

def _snapshot(dirpath: Path):
    """Return a mapping of file path -> (mtime_ns, size)."""

    dirpath.mkdir(parents=True, exist_ok=True)
    snap = {}
    for p in dirpath.rglob("*"):
        if not p.is_file():
            continue
        try:
            stat = p.stat()
        except FileNotFoundError:
            # File vanished between rglob() and stat(); skip it.
            continue
        snap[p.resolve()] = (stat.st_mtime_ns, stat.st_size)
    return snap


def _new_files(dirpath: Path, before):
    """Return files that are new or modified compared to ``before`` snapshot."""

    after = _snapshot(dirpath)
    changed = []
    for path, meta in after.items():
        if path not in before or before[path] != meta:
            changed.append(path)
    return sorted(changed)

def _parse_saved_paths(stdout_text: str):
    # generate_data.py prints the save paths; parse them so we can move & rename deterministically.
    syn = None
    log = None
    m1 = re.findall(r"Syndromes saved to:\s*(.+)", stdout_text)
    m2 = re.findall(r"Logical errors saved to:\s*(.+)", stdout_text)
    if m1:
        syn = Path(m1[-1].strip()).resolve()
    if m2:
        log = Path(m2[-1].strip()).resolve()
    return syn, log

def _safe_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"→ wrote {dst}")

def _detect_device(auto_choice: str):
    if auto_choice != "auto":
        return auto_choice
    try:
        import torch
        if hasattr(torch, "npu") and getattr(torch.npu, "is_available", lambda: False)():
            return "npu"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"

def _backup_text(path: Path):
    path = Path(path)
    if not path.exists():
        return None
    bak = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, bak)
    return bak

def _restore_text(path: Path, backup: Path):
    if backup and Path(backup).exists():
        shutil.move(str(backup), str(path))

def _load_yaml(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _dump_yaml(obj, path: Path):
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def _discover_experiments(exp_root: Path):
    """Return experiment directories detected by ``run_create_all_samples``."""

    if not RUN_CREATE_ALL.exists():
        return []

    spec = importlib.util.spec_from_file_location(
        "_run_create_all_samples", RUN_CREATE_ALL
    )
    if spec is None or spec.loader is None:
        return []

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        return []

    discover = getattr(module, "discover_experiments", None)
    if discover is None:
        return []

    try:
        return discover(exp_root)
    except Exception:
        return []


def generate_dem(dem_samples: int, dest_root: Path, manifest: list):
    print("\n=== [DEM] Generating DEM data ===")
    before = _snapshot(OUTPUT_DIR)
    out = _run([sys.executable, str(GEN_SCRIPT), "--model", "dem", "--samples", str(dem_samples)])
    syn, log = _parse_saved_paths(out)
    # Fallback if parsing fails: infer by diffing output/ dir
    if not syn or not log:
        created = _new_files(OUTPUT_DIR, before)
        # Heuristics: pick last two .npy files
        npys = [p for p in created if p.suffix == ".npy"]
        npys.sort()
        if len(npys) >= 2:
            syn, log = npys[-2], npys[-1]
    # Archive to pretrain_data tree
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if syn:
        _safe_copy(syn, dest_root / f"dem_syndromes_{ts}.npy")
    if log:
        _safe_copy(log, dest_root / f"dem_logicals_{ts}.npy")
    manifest.append({"kind": "dem", "samples": dem_samples,
                     "files": [str(dest_root / f"dem_syndromes_{ts}.npy"),
                               str(dest_root / f"dem_logicals_{ts}.npy")]})

def generate_si1000(si1000_samples: int, p_grid: list, dest_root: Path, manifest: list):
    print("\n=== [SI1000] Generating SI1000 data ===")
    cfg_path = REPO_ROOT / "configs" / "si1000.yaml"
    base_cfg = _load_yaml(cfg_path) if cfg_path.exists() else {}
    bak = _backup_text(cfg_path)
    try:
        for p in p_grid:
            cfg = dict(base_cfg) if base_cfg else {"p": p}
            cfg["p"] = float(p)
            _dump_yaml(cfg, cfg_path)
            before = _snapshot(OUTPUT_DIR)
            out = _run([sys.executable, str(GEN_SCRIPT), "--model", "si1000", "--samples", str(si1000_samples)])
            syn, log = _parse_saved_paths(out)
            if not syn or not log:
                created = _new_files(OUTPUT_DIR, before)
                npys = [x for x in created if x.suffix == ".npy"]
                npys.sort()
                if len(npys) >= 2:
                    syn, log = npys[-2], npys[-1]
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            tag = f"p{str(p).replace('.', 'p')}"
            if syn:
                _safe_copy(syn, dest_root / f"si1000_syndromes_{tag}_{ts}.npy")
            if log:
                _safe_copy(log, dest_root / f"si1000_logicals_{tag}_{ts}.npy")
            manifest.append({"kind": "si1000", "p": float(p), "samples": si1000_samples,
                             "files": [str(dest_root / f"si1000_syndromes_{tag}_{ts}.npy"),
                                       str(dest_root / f"si1000_logicals_{tag}_{ts}.npy")]} )
    finally:
        _restore_text(cfg_path, bak)

def generate_soft(
    soft_shots: int,
    device: str,
    dest_root: Path,
    experiment_roots: list[Path],
    manifest: list,
):
    print("\n=== [SOFT] Generating soft (I/Q) readout data ===")
    device = _detect_device(device)
    print(f"[SOFT] Selected device: {device}")
    before = _snapshot(SIMDATA_DIR)
    multi_root = len(experiment_roots) > 1
    experiments_by_root: dict[Path, list[Path]] = {}
    total_experiments = 0
    for root in experiment_roots:
        if not root.exists():
            print(f"[SOFT] Warning: experiment root {root} does not exist; skipping")
            continue
        discovered = _discover_experiments(root)
        experiments_by_root[root] = discovered
        total_experiments += len(discovered)
        if discovered:
            header = f"[SOFT] Experiments under {root}:"
            rel_paths = [p.relative_to(root).as_posix() for p in discovered]
            print(header + "\n  - " + "\n  - ".join(rel_paths))
        else:
            print(f"[SOFT] Warning: No experiments discovered under {root}")

    if total_experiments == 0:
        print(
            "[SOFT] Warning: No experiments discovered in the provided roots. "
            "Only datasets produced during this run will be collected."
        )

    if RUN_CREATE_ALL.exists():
        # Use the batch helper so *all* experiments under experiment_data/ are generated.
        # Forward shots & device so the caller's CLI flags actually take effect.
        cmd = [
            sys.executable,
            str(RUN_CREATE_ALL),
            "--layout",
            "by_experiment",
            "--shots",
            str(soft_shots),
            "--device",
            device,
        ]
        if experiment_roots:
            for root in experiment_roots:
                cmd += ["--experiment-root", str(root)]
        _run(cmd)
    else:
        # Fallback: call google_qec_simulator/main.py directly on a plausible experiment dir.
        # README shows: python google_qec_simulator/main.py path/to/exp --shots N --device <cpu|cuda|npu>
        exp_dir = None
        for root in experiment_roots:
            candidate = root / "surface_code"
            if candidate.exists():
                exp_dir = candidate
                break
        if exp_dir is None:
            exp_dir = REPO_ROOT / "experiment_data" / "surface_code"
        if not exp_dir.exists():
            # Try tests as a fallback
            exp_dir = REPO_ROOT / "test_experiment_simulator"
        cmd = [sys.executable, str(GQEC_MAIN), str(exp_dir), "--shots", str(soft_shots)]
        if device != "cpu":
            cmd += ["--device", device]
        _run(cmd)

    created = {p.resolve() for p in _new_files(SIMDATA_DIR, before) if p.suffix == ".npz"}

    copied_any = False
    missing_experiments: list[str] = []

    if total_experiments:
        for root, exp_list in experiments_by_root.items():
            if not exp_list:
                continue
            for exp_dir in exp_list:
                rel = exp_dir.relative_to(root)
                sim_dir = SIMDATA_DIR
                dest_dir = dest_root
                if multi_root:
                    sim_dir = sim_dir / root.name
                    dest_dir = dest_dir / root.name
                sim_dir = sim_dir / rel
                dest_dir = dest_dir / rel
                files = sorted(sim_dir.glob("*.npz")) if sim_dir.exists() else []
                if not files:
                    missing_experiments.append(f"{root}:{rel.as_posix()}")
                    continue

                copied_files: list[str] = []
                for src in files:
                    dst = dest_dir / src.name
                    _safe_copy(src, dst)
                    copied_files.append(str(dst))
                    copied_any = True

                manifest.append({
                    "kind": "soft",
                    "shots": soft_shots,
                    "device": device,
                    "files": copied_files,
                    "experiment": (Path(root.name) / rel).as_posix() if multi_root else rel.as_posix(),
                    "new_sources": [str(src) for src in files if src.resolve() in created],
                })
    else:
        # Fallback: no experiment discovery available – copy every .npz in simulated_data.
        all_npz = sorted(SIMDATA_DIR.rglob("*.npz"))
        if not all_npz:
            print(
                "[SOFT] No .npz files detected under simulated_data/. "
                "Ensure run_create_all_samples.py or google_qec_simulator wrote outputs.",
                file=sys.stderr,
            )
        for src in all_npz:
            try:
                rel = src.resolve().relative_to(SIMDATA_DIR.resolve())
            except Exception:
                rel = Path(src.name)
            dst = dest_root / rel
            _safe_copy(src, dst)
            manifest.append({
                "kind": "soft",
                "shots": soft_shots,
                "device": device,
                "files": [str(dst)],
                "experiment": str(dst.parent.relative_to(dest_root)),
                "new_sources": [str(src)] if src.resolve() in created else [],
            })
            copied_any = True

    if missing_experiments:
        missing_list = "\n  - ".join(missing_experiments)
        raise RuntimeError(
            "[SOFT] Failed to produce samples for all experiments. Missing:\n  - "
            + missing_list
        )

    if not copied_any:
        raise RuntimeError(
            "[SOFT] No experiment datasets were copied into the pretraining tree. "
            "Check that run_create_all_samples.py succeeded."
        )

def main():
    parser = argparse.ArgumentParser(description="Generate ALL pretraining noise datasets for ALPHAQUBIT.")
    parser.add_argument("--dem-samples", type=int, default=200_000,
                        help="Number of DEM samples to generate in one call (generate_data.py).")
    parser.add_argument("--si1000-samples", type=int, default=200_000,
                        help="Number of SI1000 samples per p in one call (generate_data.py).")
    parser.add_argument("--si1000-p-grid", type=str, default="0.006,0.010,0.014",
                        help="Comma-separated p grid for SI1000 (e.g., 0.002,0.004,...).")
    parser.add_argument("--soft-shots", type=int, default=100_000,
                        help="Shots *per experiment* for soft/IQ sampling (forwarded to run_create_all_samples.py or google_qec_simulator).")
    parser.add_argument("--soft-device", type=str, default="auto", choices=["auto", "cpu", "cuda", "npu"],
                        help="Device for soft/IQ sampling (auto tries NPU, then CUDA).")
    parser.add_argument("--out-dir", type=str, default="pretrain_data",
                        help="Where to collect consolidated datasets.")
    parser.add_argument(
        "--experiment-root",
        type=Path,
        action="append",
        dest="experiment_roots",
        help=(
            "Additional directories containing Stim experiments. May be supplied "
            "multiple times; defaults to the repository's experiment_data/."
        ),
    )
    args = parser.parse_args()

    # Sanity checks
    if not GEN_SCRIPT.exists():
        raise SystemExit("Run this script from the repository root (generate_data.py not found).")
    if not (RUN_CREATE_ALL.exists() or GQEC_MAIN.exists()):
        print("Warning: neither run_create_all_samples.py nor google_qec_simulator/main.py is visible.")

    out_root = REPO_ROOT / args.out_dir
    dem_dir = out_root / "dem"
    si1k_dir = out_root / "si1000"
    soft_dir = out_root
    experiment_roots = args.experiment_roots or [REPO_ROOT / "experiment_data"]
    experiment_roots = [root.resolve() for root in experiment_roots]
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = []

    # 1) DEM
    generate_dem(args.dem_samples, dem_dir, manifest)

    # 2) SI1000 (optionally across p grid)
    p_grid = [float(x.strip()) for x in args.si1000_p_grid.split(",") if x.strip()]
    generate_si1000(args.si1000_samples, p_grid, si1k_dir, manifest)

    # 3) Soft/IQ (google_qec_simulator)
    generate_soft(args.soft_shots, args.soft_device, soft_dir, experiment_roots, manifest)

    # Write manifest
    mf_path = out_root / "MANIFEST.json"
    with open(mf_path, "w") as f:
        json.dump({"created_at": datetime.now().isoformat(), "items": manifest}, f, indent=2)
    print(f"\nAll done. Manifest: {mf_path}")

    # Helpful note re: paper_aligned & leakysim stub (not needed for pretraining, but FYI)
    if (REPO_ROOT / "leakysim.py").exists():
        print("Note: A local 'leakysim.py' exists. This is fine for DEM/SI1000 pretraining.\n"
              "If you later generate 'paper_aligned' physical-noise data for fine-tuning,\n"
              "ensure the real 'leakysim' package is used instead of the stub.", file=sys.stderr)

if __name__ == "__main__":
    main()
