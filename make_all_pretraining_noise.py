#!/usr/bin/env python3
"""
Generate ALL pretraining noise datasets for ALPHAQUBIT in one go.

Produces:
  pretrain_data/
    dem/         # DEM syndromes/logicals (.npy)
    si1000/      # SI1000 syndromes/logicals (.npy), optionally for a grid of p
    soft/        # Soft ("I/Q") readout .npz produced by google_qec_simulator

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

def generate_soft(soft_shots: int, device: str, dest_root: Path, manifest: list):
    print("\n=== [SOFT] Generating soft (I/Q) readout data ===")
    device = _detect_device(device)
    print(f"[SOFT] Selected device: {device}")
    before = _snapshot(SIMDATA_DIR)

    if RUN_CREATE_ALL.exists():
        # The README shows this wrapper for generating .npz across circuits under simulated_data/.
        # It doesn't document flags; just call it and let it drive google_qec_simulator.
        _run([sys.executable, str(RUN_CREATE_ALL)])
    else:
        # Fallback: call google_qec_simulator/main.py directly on a plausible experiment dir.
        # README shows: python google_qec_simulator/main.py path/to/exp --shots N --device <cpu|cuda|npu>
        exp_dir = REPO_ROOT / "experiment_data" / "surface_code"
        if not exp_dir.exists():
            # Try tests as a fallback
            exp_dir = REPO_ROOT / "test_experiment_simulator"
        cmd = [sys.executable, str(GQEC_MAIN), str(exp_dir), "--shots", str(soft_shots)]
        if device != "cpu":
            cmd += ["--device", device]
        _run(cmd)

    created = _new_files(SIMDATA_DIR, before)
    npzs = [p for p in created if p.suffix == ".npz"]
    if not npzs:
        print("[SOFT] No new .npz detected under simulated_data/. "
              "Ensure run_create_all_samples.py or google_qec_simulator wrote outputs.", file=sys.stderr)
    for src in npzs:
        dst = dest_root / src.name
        _safe_copy(src, dst)
        manifest.append({"kind": "soft", "shots": soft_shots, "device": device, "files": [str(dst)]})

def main():
    parser = argparse.ArgumentParser(description="Generate ALL pretraining noise datasets for ALPHAQUBIT.")
    parser.add_argument("--dem-samples", type=int, default=200_000,
                        help="Number of DEM samples to generate in one call (generate_data.py).")
    parser.add_argument("--si1000-samples", type=int, default=200_000,
                        help="Number of SI1000 samples per p in one call (generate_data.py).")
    parser.add_argument("--si1000-p-grid", type=str, default="0.006,0.010,0.014",
                        help="Comma-separated p grid for SI1000 (e.g., 0.002,0.004,...).")
    parser.add_argument("--soft-shots", type=int, default=100_000,
                        help="Shots for soft/IQ sampling when calling google_qec_simulator directly.")
    parser.add_argument("--soft-device", type=str, default="auto", choices=["auto", "cpu", "cuda", "npu"],
                        help="Device for soft/IQ sampling (auto tries NPU, then CUDA).")
    parser.add_argument("--out-dir", type=str, default="pretrain_data",
                        help="Where to collect consolidated datasets.")
    args = parser.parse_args()

    # Sanity checks
    if not GEN_SCRIPT.exists():
        raise SystemExit("Run this script from the repository root (generate_data.py not found).")
    if not (RUN_CREATE_ALL.exists() or GQEC_MAIN.exists()):
        print("Warning: neither run_create_all_samples.py nor google_qec_simulator/main.py is visible.")

    out_root = REPO_ROOT / args.out_dir
    dem_dir = out_root / "dem"
    si1k_dir = out_root / "si1000"
    soft_dir = out_root / "soft"
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = []

    # 1) DEM
    generate_dem(args.dem_samples, dem_dir, manifest)

    # 2) SI1000 (optionally across p grid)
    p_grid = [float(x.strip()) for x in args.si1000_p_grid.split(",") if x.strip()]
    generate_si1000(args.si1000_samples, p_grid, si1k_dir, manifest)

    # 3) Soft/IQ (google_qec_simulator)
    generate_soft(args.soft_shots, args.soft_device, soft_dir, manifest)

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
