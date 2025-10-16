# google_qec_simulator/stim_helpers.py
import json
import re
from pathlib import Path
import stim


# ---------------------------------------------------------------------
def find_matching_dem(stim_path: Path) -> Path | None:
    stem = stim_path.stem
    for p in stim_path.parent.iterdir():
        if p.is_file() and p.stem.startswith(stem) and ".dem" in p.suffixes:
            return p
    return None


def _dem_from_stim(stim_path: Path) -> stim.DetectorErrorModel:
    """Always returns a DEM object (from .dem file or from circuit)."""
    dem_path = find_matching_dem(stim_path)
    if dem_path:
        return stim.DetectorErrorModel.from_file(dem_path)
    return stim.Circuit.from_file(stim_path).detector_error_model()


# ---------------------------------------------------------------------
_RX_R = re.compile(r"_r(\d+)")
_RX_D = re.compile(r"_d(\d+)")

def _parse_rounds_distance(path: Path) -> tuple[int | None, int | None]:
    """Try to read `_rXX` and `_dX` tags from any component of the path."""
    txt = str(path)
    r_m = _RX_R.search(txt)
    d_m = _RX_D.search(txt)
    rounds = int(r_m.group(1)) if r_m else None
    dist   = int(d_m.group(1)) if d_m else None
    return rounds, dist


# ---------------------------------------------------------------------
def extract_rounds_and_dets(stim_path: Path) -> tuple[int, int]:
    """
    Return (rounds, detectors_per_round) for *one* circuit.

    Order of preference
    -------------------
    1. Use dem.num_ticks / dem.num_detectors   (Stim ≥1.14)
    2. Count "TICK"/"DETECTOR" tokens in DEM text (older Stim)
    3. Parse folder/file name "_rXX", "_dX"  and set
           rounds = XX
           dets   = d² - 1
    """
    dem = _dem_from_stim(stim_path)

    # 1. modern Stim exposes structured counts
    if hasattr(dem, "num_ticks") and dem.num_ticks:
        rounds = dem.num_ticks
        dets_total = dem.num_detectors
        return rounds, dets_total // rounds

    # 2. older Stim: count tokens (case-insensitive)
    words = str(dem).lower().split()
    rounds = words.count("tick")
    dets_total = words.count("detector")
    if rounds:
        return rounds, dets_total // rounds

    # Stim 1.13 (and some minimal circuits) report ``num_ticks=None`` but
    # still expose ``num_detectors``.  In that case we assume a single
    # round containing all detectors.  This mirrors the behaviour of the
    # token-counting fallback while avoiding a division by zero.
    if hasattr(dem, "num_detectors") and dem.num_detectors:
        return 1, dem.num_detectors

    # 3. fallback: infer from path tags
    rounds, dist = _parse_rounds_distance(stim_path)
    if rounds is None or dist is None:
        raise ValueError(f"Cannot determine rounds / dets for {stim_path}")
    dets_per_round = dist * dist - 1     # surface-code stabilisers
    return rounds, dets_per_round


# Test each function in this file
if __name__ == "__main__":
    test_stim_file = Path("path_to_a_stim_file/stim_example.stim")

    # Test if we can find the matching DEM
    matching_dem = find_matching_dem(test_stim_file)
    print(f"Matching DEM: {matching_dem}")

    # Test DEM extraction and rounds/dets_per_round calculation
    rounds, dets_per_round = extract_rounds_and_dets(test_stim_file)
    print(f"Rounds: {rounds}, Detectors per round: {dets_per_round}")
