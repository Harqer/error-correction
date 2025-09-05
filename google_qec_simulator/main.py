# google_qec_simulator/main.py
# --------------------------------------------------------------------
"""
Generate one soft-channel .npz bundle for a SINGLE experiment folder.

Usage
-----
    python -m google_qec_simulator.main  <exp_subdir>  [--shots 2000] [--out ./custom.npz]

Arguments
---------
<exp_subdir> : path of ONE folder that holds *.stim (and optional *.dem)
--shots      : Monte-Carlo repetitions per circuit (default 1000)
--out        : output filename; if omitted we write
                 samples_<exp_subdir_name>.npz  into the *same* folder
--device     : torch device for soft-channel sampling (cpu, cuda, npu)
"""

from pathlib import Path
import argparse
import numpy as np

from stim_helpers  import extract_rounds_and_dets
from data_helpers  import reshape_detectors, soft_channels
from data_manager  import DataManager
from circuit_utils import sample_detectors_obs


# --------------------------------------------------------------------
def list_stim_files(folder: Path) -> list[Path]:
    """Return *.stim files that live *directly* in <folder> (no recursion)."""
    return sorted(p for p in folder.iterdir() if p.suffix == ".stim" and p.is_file())


def simulate_folder(
    exp_dir: Path,
    out_file: Path,
    shots: int = 1000,
    snr: float = 10.0,
    t: float   = 0.01,
    device: str = "cpu",
) -> None:
    dm = DataManager()
    stim_files = list_stim_files(exp_dir)

    if not stim_files:
        raise RuntimeError(f"No *.stim in {exp_dir}")

    print(f"[scan] {len(stim_files)} stim files in {exp_dir.name}")

    for stim_path in stim_files:
        print(f"[sample] {stim_path.name}  shots={shots}")
        det_flat, obs = sample_detectors_obs(stim_path, shots)
        det          = reshape_detectors(det_flat, stim_path, shots)  # (N,R,S,1)

        R, S         = extract_rounds_and_dets(stim_path)
        post1, post2 = soft_channels(shots * R * S, snr, t, device=device)
        post1        = post1.reshape(shots, R, S, 1)
        post2        = post2.reshape(shots, R, S, 1)

        dm.store(np.concatenate([det, post1, post2], axis=-1), obs)

    dm.save(out_file)


# =====================================================================
# CLI entry-point
# =====================================================================
if __name__ == "__main__":
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser()
    ap.add_argument("exp_dir", type=Path,
                    help="ONE experiment sub-folder that contains *.stim files")
    ap.add_argument("--shots", type=int, default=1000,
                    help="Monte-Carlo shots per circuit (default 1000)")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output .npz (default = <ROOT>/output/samples_<folder>.npz)")
    ap.add_argument("--device", type=str, default="cpu",
                    help="Torch device for soft-channel sampling (cpu, cuda, npu)")
    args = ap.parse_args()

    if not args.exp_dir.is_dir():
        raise SystemExit(f"❌  {args.exp_dir} is not a directory")

    # -----------------------------------------------------------------
    # DEFAULT OUTPUT LOCATION
    #   <root>/output/samples_<subfolder>.npz
    # -----------------------------------------------------------------
    # -----------------------------------------------------------------
    # DEFAULT OUTPUT:
    #   <PROJECT_ROOT>/output/samples_<subfolder>.npz
    #   (PROJECT_ROOT = the folder that contains this repo, i.e. two
    #    levels above this main.py file)
    # -----------------------------------------------------------------
    if args.out is None:
        # main.py → google_qec_simulator → <PROJECT_ROOT>
        project_root = Path(__file__).resolve().parents[1]
        out_dir      = project_root / "simulated_data"
        out_dir.mkdir(exist_ok=True)
        args.out     = out_dir / f"samples_{args.exp_dir.name}.npz"


    simulate_folder(
        exp_dir  = args.exp_dir,
        out_file = args.out,
        shots    = args.shots,
        device   = args.device,
    )
    print("DONE →", args.out.resolve())

