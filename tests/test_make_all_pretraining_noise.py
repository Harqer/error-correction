import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "_make_all_noise", ROOT / "make_all_pretraining_noise.py"
)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)  # type: ignore[arg-type]


def test_generate_soft_forwards_shots_and_device(monkeypatch, tmp_path):
    run_create = tmp_path / "run_create_all_samples.py"
    run_create.write_text("# stub")
    module.RUN_CREATE_ALL = run_create
    simdata_dir = tmp_path / "simulated_data"
    simdata_dir.mkdir()
    module.SIMDATA_DIR = simdata_dir

    calls = []

    def fake_run(cmd, cwd=None):
        calls.append(cmd)
        return ""

    monkeypatch.setattr(module, "_run", fake_run)
    monkeypatch.setattr(module, "_snapshot", lambda path: {})
    monkeypatch.setattr(module, "_new_files", lambda path, before: [])
    monkeypatch.setattr(module, "_safe_copy", lambda src, dst: None)
    monkeypatch.setattr(module, "_detect_device", lambda choice: "npu")

    manifest = []
    module.generate_soft(soft_shots=123456, device="auto", dest_root=tmp_path, manifest=manifest)

    assert calls, "generate_soft should invoke run_create_all_samples"
    cmd = calls[0]
    assert "--shots" in cmd
    assert str(123456) in cmd
    assert cmd.count("--device") == 1
    device_idx = cmd.index("--device")
    assert cmd[device_idx + 1] == "npu"
    assert not manifest
