import pathlib
import sys

import numpy as np
import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from ai_models.pauli_plus_dataset import PauliPlusDataset


def test_prefers_basis_specific_observable(tmp_path):
    samples = np.zeros((2, 1, 3, 3), dtype=np.float32)
    obs_empty = np.empty((2, 0), dtype=np.float32)
    obs_x = np.array([[0.0], [1.0]], dtype=np.float32)
    obs_z = np.array([[1.0], [0.0]], dtype=np.float32)

    np.savez(
        tmp_path / "sample.npz",
        data=samples,
        obs=obs_empty,
        obs_x=obs_x,
        obs_z=obs_z,
    )

    dataset = PauliPlusDataset(str(tmp_path / "sample.npz"), basis_id=0)

    assert torch.equal(dataset.y, torch.tensor([0.0, 1.0], dtype=torch.float32))


def test_raises_when_no_valid_label(tmp_path):
    samples = np.zeros((1, 1, 3, 3), dtype=np.float32)
    invalid = np.empty((1, 0), dtype=np.float32)

    np.savez(tmp_path / "no_labels.npz", data=samples, obs=invalid)

    try:
        PauliPlusDataset(str(tmp_path / "no_labels.npz"), basis_id=0)
    except ValueError as exc:
        assert "Could not find a valid label array" in str(exc)
    else:
        raise AssertionError("Expected ValueError when labels are missing")
