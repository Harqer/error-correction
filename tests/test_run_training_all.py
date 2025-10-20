import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np


def _write_npz(path: Path) -> None:
    samples = np.zeros((4, 2, 3, 3), dtype=np.float32)
    labels = np.arange(4, dtype=np.float32)
    np.savez(path, data=samples, obs=labels)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_SPEC = importlib.util.spec_from_file_location(
    "_run_training_all", Path(__file__).resolve().parent.parent / "run_training_all.py"
)
assert _SPEC and _SPEC.loader
launcher = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(launcher)  # type: ignore[arg-type]


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


def test_discover_npu_devices_prefers_environment(monkeypatch):
    class _FakeNPU:
        def __init__(self, count: int) -> None:
            self._count = count

        def device_count(self) -> int:  # pragma: no cover - trivial
            return self._count

    class _FakeTorch:
        def __init__(self, count: int) -> None:
            self.npu = _FakeNPU(count)

    monkeypatch.setenv("ASCEND_VISIBLE_DEVICES", "2,3")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    original_torch = launcher.torch
    try:
        launcher.torch = _FakeTorch(1)
        devices = launcher._discover_npu_devices()
    finally:
        launcher.torch = original_torch

    assert devices == (2, 3)


def test_parse_visible_devices_handles_invalid_tokens():
    env = {"ASCEND_VISIBLE_DEVICES": "0,foo,1"}
    result = launcher._parse_visible_devices(env, ("ASCEND_VISIBLE_DEVICES",))
    assert result is None


def test_trains_all_pretrain_datasets_serial(monkeypatch, tmp_path):
    root = tmp_path / "pretrain_data"
    (root / "expA").mkdir(parents=True)
    (root / "nested" / "expB").mkdir(parents=True)

    paths = [
        root / "expA" / "samples_expA_bx.npz",
        root / "expA" / "samples_expA_bz.npz",
        root / "nested" / "expB" / "samples_expB.npz",
    ]
    for path in paths:
        _write_npz(path)

    launched = []

    def fake_run(cmd, check=False, env=None, **kwargs):
        launched.append((cmd, env))
        save_idx = cmd.index("--model-save-path") + 1
        out_path = Path(cmd[save_idx])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"ok")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    monkeypatch.setattr(launcher, "_discover_gpu_devices", lambda: (0,))
    monkeypatch.setattr(launcher, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setenv("PYTHONUNBUFFERED", "1")
    monkeypatch.setattr(sys, "argv", [
        "run_training_all.py",
        "--data-root",
        str(root),
        "--epochs",
        "1",
        "--batch-size",
        "2",
        "--max-samples",
        "2",
    ])

    launcher.main()

    assert len(launched) == len(paths)
    seen_npz = {
        cmd[cmd.index("--npz_file") + 1]
        for cmd, _ in launched
    }
    assert seen_npz == {str(p) for p in paths}
    for cmd, _ in launched:
        assert "--npu" not in cmd


def test_trains_all_pretrain_datasets_parallel_npu(monkeypatch, tmp_path):
    root = tmp_path / "pretrain_data"
    (root / "expA").mkdir(parents=True)
    (root / "expB").mkdir(parents=True)
    (root / "expC").mkdir(parents=True)

    paths = [
        root / "expA" / "samples_expA_bx.npz",
        root / "expB" / "samples_expB_bz.npz",
        root / "expC" / "samples_expC.npz",
    ]
    for path in paths:
        _write_npz(path)

    launched = []
    envs = []
    ports = iter(range(12345, 12355))

    class _Proc:
        def __init__(self, cmd, env):
            save_idx = cmd.index("--model-save-path") + 1
            out_path = Path(cmd[save_idx])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"ok")
            self.cmd = cmd
            self.env = env

        def wait(self):
            return 0

    def fake_popen(cmd, env=None, **kwargs):
        proc = _Proc(cmd, env or {})
        launched.append(cmd)
        envs.append(env or {})
        return proc

    def fake_alloc(allocated_ports):
        port = next(ports)
        allocated_ports.add(port)
        return port

    monkeypatch.setattr(launcher.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(launcher, "_discover_npu_devices", lambda: (0, 1))
    monkeypatch.setattr(launcher, "_allocate_master_port", fake_alloc)
    monkeypatch.setattr(launcher, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setenv("PYTHONUNBUFFERED", "1")
    monkeypatch.setattr(sys, "argv", [
        "run_training_all.py",
        "--npu",
        "--data-root",
        str(root),
        "--epochs",
        "1",
        "--batch-size",
        "2",
        "--max-samples",
        "2",
    ])

    launcher.main()

    assert len(launched) == len(paths)
    device_cycle = [0, 1, 0]
    for idx, cmd in enumerate(launched):
        assert "--npu" in cmd
        assert cmd[cmd.index("--device_index") + 1] == str(device_cycle[idx])
        npz_arg = cmd[cmd.index("--npz_file") + 1]
        assert npz_arg == str(paths[idx])
    for idx, env in enumerate(envs):
        device = str(device_cycle[idx])
        assert env["ASCEND_DEVICE_ID"] == device
        assert env["DEVICE_ID"] == device
        assert env["ASCEND_VISIBLE_DEVICES"] == device
        assert env["ASCEND_RT_VISIBLE_DEVICES"] == device
        assert env["NPU_VISIBLE_DEVICES"] == device
        assert env["MASTER_PORT"].isdigit()
