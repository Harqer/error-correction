import subprocess
import sys

import numpy as np


def test_skips_npz_without_labels(tmp_path):
    samples = np.zeros((1, 1, 1, 3), dtype=np.float32)
    observables = np.empty((1, 0), dtype=np.float32)
    np.savez(tmp_path / "samples_surface_code.npz", data=samples, observables=observables)

    result = subprocess.run(
        [sys.executable, "run_training_all.py", "--data-root", str(tmp_path), "--npu"],
        check=True,
        capture_output=True,
        text=True,
    )

    stdout = result.stdout
    assert "Skipping" in stdout
    assert "No datasets with valid labels found" in stdout
