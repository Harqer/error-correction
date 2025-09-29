from argparse import Namespace
from importlib import util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "run_decode_all.py"
SPEC = util.spec_from_file_location("run_decode_all", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
_MODULE = util.module_from_spec(SPEC)
SPEC.loader.exec_module(_MODULE)

resolve_targets = _MODULE.resolve_targets


def make_args(**overrides):
    defaults = dict(
        targets=[],
        pattern=None,
        recursive=False,
        data_root=Path("output"),
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def test_directory_target_collects_np_files(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    npy_file = data_dir / "sample.npy"
    npy_file.write_bytes(b"binary")

    args = make_args(targets=[str(data_dir)])

    result = resolve_targets(args)

    assert result == [npy_file.resolve()]


def test_directory_target_respects_recursive_flag(tmp_path):
    parent = tmp_path / "parent"
    nested = parent / "nested"
    nested.mkdir(parents=True)
    file_in_nested = nested / "deep.npz"
    file_in_nested.write_bytes(b"content")

    # Without recursion the nested file should not be picked up.
    args_non_recursive = make_args(targets=[str(parent)], recursive=False)
    assert resolve_targets(args_non_recursive) == []

    # With recursion we collect the nested file.
    args_recursive = make_args(targets=[str(parent)], recursive=True)
    assert resolve_targets(args_recursive) == [file_in_nested.resolve()]
