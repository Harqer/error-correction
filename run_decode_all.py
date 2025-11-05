#!/usr/bin/env python3
"""Batch launcher for AlphaQubit decoding/evaluation."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Iterable, List, Sequence, Set

DEFAULT_PATTERNS: Sequence[str] = ("*.npy", "*.npz")
DEFAULT_DATA_ROOT = Path("output")
DEFAULT_EXPERIMENT_ROOT = Path.home() / "work/google_qec3v5_experiment_data"
FALLBACK_DATA_ROOTS: Sequence[Path] = (
    Path("simulated_data"),
    DEFAULT_EXPERIMENT_ROOT,
)
DECODE_SCRIPT = Path("ai_models/decode.py")
MODEL_SEARCH_DIRS: Sequence[Path] = (
    Path("."),
    Path("finetuned_models"),
    Path("ai_models/checkpoints"),
    Path("ai_models/models"),
    Path("checkpoints"),
    Path("models"),
)


def _discover_checkpoints(directory: Path) -> List[Path]:
    """Return all checkpoint files inside *directory* sorted by name."""

    if not directory.exists() or not directory.is_dir():
        return []

    candidates: List[Path] = []
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() == ".pth":
            resolved = path.resolve()
            if resolved not in candidates:
                candidates.append(resolved)

    return sorted(candidates)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decode every syndrome file in a directory using ai_models/decode.py. "
            "By default it scans the 'output' folder for .npy/.npz files, but "
            "you can override the search using positional glob patterns."
        )
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help=(
            "Path to the AlphaQubit checkpoint (.pth) used for decoding. "
            "When omitted the script attempts to auto-discover a single "
            "checkpoint in common directories.  Passing a directory causes "
            "the script to decode with every .pth file inside."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=(
            "Directory to scan for syndrome files when no positional targets are "
            "given (default: ./output)."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into sub-directories while collecting syndrome files.",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        default=None,
        help=(
            "Glob pattern(s) relative to --data-root.  When omitted the script "
            "searches for *.npy and *.npz files.  Provide multiple times to "
            "combine patterns."
        ),
    )
    parser.add_argument(
        "--basis",
        choices=("x", "z"),
        default=None,
        help="Force a measurement basis for all files (passed to decode.py).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size forwarded to decode.py (default: 512).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device forwarded to decode.py (default: auto).",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Hidden dimension forwarded to decode.py (must match training).",
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=8,
        help="Attention head count forwarded to decode.py.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=12,
        help="Transformer layer count forwarded to decode.py.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory to collect metrics JSON files.  When omitted, "
            "metrics are stored under ./results/ (grouped per model when "
            "multiple checkpoints are decoded).  When provided, each decode run "
            "writes RESULTS_DIR/<stem>_metrics.json."
        ),
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=None,
        help=(
            "If set, save per-shot probabilities to this directory as "
            "<stem>_probs.npy for each input file.  Defaults to no probability "
            "dumps unless explicitly requested."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Skip files whose metrics output already exists (only when "
            "--results-dir is provided or decode.py's default file is present)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without executing them.",
    )
    parser.add_argument(
        "targets",
        nargs="*",
        help=(
            "Optional explicit files or glob patterns.  When supplied the "
            "--data-root/--pattern options are ignored and only the resolved "
            "targets are decoded."
        ),
    )
    return parser.parse_args()


def _collect_from_root(
    root: Path,
    patterns: Sequence[str],
    recursive: bool,
) -> List[Path]:
    files: Set[Path] = set()
    for pattern in patterns:
        collector: Iterable[Path]
        if recursive:
            collector = root.rglob(pattern)
        else:
            collector = root.glob(pattern)
        for path in collector:
            if path.is_file():
                files.add(path.resolve())
    return sorted(files)


def resolve_targets(args: argparse.Namespace) -> List[Path]:
    if args.targets:
        paths: Set[Path] = set()
        patterns: Sequence[str] = tuple(args.pattern) if args.pattern else DEFAULT_PATTERNS

        for spec in args.targets:
            try:
                expanded = list(Path().glob(spec))
            except NotImplementedError:
                expanded = []
            if not expanded:
                # Treat as literal path (may include relative components)
                candidate = Path(spec)
                if candidate.exists():
                    expanded = [candidate]

            for path in expanded:
                if path.is_file() and path.suffix.lower() in {".npy", ".npz"}:
                    paths.add(path.resolve())
                elif path.is_dir():
                    for pattern in patterns:
                        iterator: Iterable[Path]
                        if args.recursive:
                            iterator = path.rglob(pattern)
                        else:
                            iterator = path.glob(pattern)
                        for candidate in iterator:
                            if candidate.is_file():
                                paths.add(candidate.resolve())
        return sorted(paths)

    root: Path = args.data_root
    patterns: Sequence[str] = tuple(args.pattern) if args.pattern else DEFAULT_PATTERNS

    if not root.exists():
        print(f"Warning: data root {root} does not exist.")
        files: List[Path] = []
    else:
        files = _collect_from_root(root, patterns, args.recursive)

    if files or args.targets:
        return files

    # No files were discovered in the requested root.  If the caller kept the
    # default root ("output"), try a couple of well-known fallback locations so
    # that bundled sample data can be decoded out-of-the-box.
    if root == DEFAULT_DATA_ROOT:
        for fallback in FALLBACK_DATA_ROOTS:
            if not fallback.exists():
                continue
            fallback_files = _collect_from_root(fallback, patterns, args.recursive)
            if fallback_files:
                print(
                    "Info: no syndrome files found under 'output/'. "
                    f"Falling back to '{fallback}/'."
                )
                return fallback_files

    return files


def make_metrics_path(
    data_file: Path,
    args: argparse.Namespace,
    model_identifier: str | None,
    multi_model: bool,
) -> Path:
    base_dir = args.results_dir if args.results_dir is not None else Path("results")
    if multi_model and model_identifier is not None:
        base_dir = base_dir / model_identifier
    return base_dir / f"{data_file.stem}_metrics.json"


def make_predictions_path(
    data_file: Path,
    args: argparse.Namespace,
    model_identifier: str | None,
    multi_model: bool,
) -> Path:
    base_dir = (
        args.predictions_dir if args.predictions_dir is not None else Path("results")
    )
    if multi_model and model_identifier is not None:
        base_dir = base_dir / model_identifier
    return base_dir / f"{data_file.stem}_probs.npy"


def resolve_model_paths(model: Path | None) -> List[Path]:
    """Resolve checkpoint paths, optionally searching common directories."""

    search_dirs: Sequence[Path] = MODEL_SEARCH_DIRS

    def _describe_search_dirs() -> str:
        return ", ".join(str(directory) for directory in search_dirs)

    if model is not None:
        if model.exists():
            resolved = model.resolve()
            if resolved.is_dir():
                candidates = _discover_checkpoints(resolved)
                if not candidates:
                    raise FileNotFoundError(
                        "Model directory provided but no .pth checkpoints were found: "
                        f"{resolved}"
                    )
                print(
                    f"Info: discovered {len(candidates)} model checkpoint(s) "
                    f"under '{resolved}'."
                )
                return candidates
            if resolved.is_file():
                return [resolved]
            raise FileNotFoundError(f"Model path is neither file nor directory: {resolved}")

        if not model.is_absolute() and model.parent == Path():
            for directory in search_dirs:
                candidate = (directory / model).resolve()
                if candidate.exists():
                    if candidate.is_dir():
                        candidates = _discover_checkpoints(candidate)
                        if not candidates:
                            raise FileNotFoundError(
                                "Model directory provided but no .pth checkpoints "
                                f"were found: {candidate}"
                            )
                        print(
                            f"Info: resolved model directory '{model}' to '{candidate}' "
                            f"and found {len(candidates)} checkpoint(s)."
                        )
                        return candidates
                    if candidate.is_file():
                        print(
                            f"Info: resolved model path '{model}' to '{candidate}'."
                        )
                        return [candidate]

        search_hint = _describe_search_dirs()
        raise FileNotFoundError(
            "Model checkpoint not found: "
            f"{model}. Provide the full path or place it in one of: {search_hint}"
        )

    candidates: List[Path] = []
    for directory in search_dirs:
        for path in _discover_checkpoints(directory):
            if path not in candidates:
                candidates.append(path)

    if not candidates:
        search_hint = _describe_search_dirs()
        raise FileNotFoundError(
            "No model checkpoint provided and none discovered. "
            "Use --model to specify the path explicitly or place a single .pth "
            f"file in one of: {search_hint}"
        )

    if len(candidates) > 1:
        formatted = "\n".join(f"  - {path}" for path in candidates)
        raise FileNotFoundError(
            "Multiple model checkpoints discovered. "
            "Use --model to choose one explicitly among:\n"
            f"{formatted}"
        )

    chosen = candidates[0]
    print(f"Info: auto-discovered model checkpoint at '{chosen}'.")
    return [chosen]


def main() -> None:
    args = parse_args()

    if not DECODE_SCRIPT.exists():
        raise FileNotFoundError(f"decode script not found: {DECODE_SCRIPT}")

    model_paths = resolve_model_paths(args.model)
    multi_model = len(model_paths) > 1

    data_files = resolve_targets(args)
    if not data_files:
        print(
            "No syndrome files found. Nothing to decode. "
            "Generate data first (e.g. via generate_data.py or run_create_all_samples.py) "
            "or adjust --data-root/targets."
        )
        return

    if args.results_dir is not None:
        args.results_dir.mkdir(parents=True, exist_ok=True)
    if args.predictions_dir is not None:
        args.predictions_dir.mkdir(parents=True, exist_ok=True)

    metrics_root = args.results_dir if args.results_dir is not None else Path("results")
    predictions_root = args.predictions_dir

    for model_path in model_paths:
        model_identifier = model_path.stem
        if multi_model:
            print(f"Info: decoding with model '{model_identifier}'.")

        model_metrics_dir = metrics_root
        if multi_model and model_identifier:
            model_metrics_dir = model_metrics_dir / model_identifier
        print(
            "Info: metrics JSON files will be written under "
            f"{model_metrics_dir}/"
        )

        if predictions_root is not None:
            model_predictions_dir = predictions_root
            if multi_model and model_identifier:
                model_predictions_dir = model_predictions_dir / model_identifier
            print(
                "Info: per-shot probability arrays will be written under "
                f"{model_predictions_dir}/"
            )

        for data_path in data_files:
            cmd: List[str] = [
                "python",
                str(DECODE_SCRIPT),
                "--model",
                str(model_path),
                "--data",
                str(data_path),
                "--batch-size",
                str(args.batch_size),
                "--hidden-dim",
                str(args.hidden_dim),
                "--heads",
                str(args.heads),
                "--layers",
                str(args.layers),
            ]
            if args.device is not None:
                cmd.extend(["--device", args.device])
            if args.basis is not None:
                cmd.extend(["--basis", args.basis])

            output_path = make_metrics_path(
                data_path, args, model_identifier, multi_model
            )
            predictions_path = None
            if args.skip_existing and output_path.exists():
                print(
                    f"Skipping {data_path} for model {model_identifier} "
                    f"(metrics already exist at {output_path})"
                )
                continue
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cmd.extend(["--output", str(output_path)])

            if args.predictions_dir is not None:
                predictions_path = make_predictions_path(
                    data_path, args, model_identifier, multi_model
                )
                predictions_path.parent.mkdir(parents=True, exist_ok=True)
                cmd.extend(["--predictions", str(predictions_path)])

            print(f"Running decode: {' '.join(cmd)}")
            if args.dry_run:
                continue
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
