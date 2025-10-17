import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_models import model_mla


def test_model_stem_from_npz_includes_origin(monkeypatch, tmp_path):
    pretrain_root = tmp_path / "pretrain_data"
    simulated_root = tmp_path / "simulated_data"
    pretrain_root.mkdir()
    simulated_root.mkdir()

    monkeypatch.setattr(model_mla, "PRETRAIN_DATA_DIR", pretrain_root)
    monkeypatch.setattr(model_mla, "SIMULATED_DATA_DIR", simulated_root)

    file_name = "samples_surface_code_bX_d5_r01_center_5_5.npz"
    pretrain_file = pretrain_root / file_name
    simulated_file = simulated_root / file_name

    pretrain_file.touch()
    simulated_file.touch()

    stem_pretrain = model_mla.model_stem_from_npz(pretrain_file)
    stem_simulated = model_mla.model_stem_from_npz(simulated_file)

    assert stem_pretrain != stem_simulated
    assert stem_pretrain.startswith("pretrain_")
    assert stem_simulated.startswith("simulated_")
