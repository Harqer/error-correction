#!/usr/bin/env python3
"""
run_fine_tune_all.py

For each .pth checkpoint in the selected model directories, invoke
ai_models/fine_tune.py in-process, passing --dataset as the corresponding
folder (data_dir/<pth_stem>) and --model_path pointing to the fine-tuned copy
stored under finetuned_models/. No subprocesses are spawned.
"""

import sys
import runpy
import argparse
from pathlib import Path
import shutil

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune for all .pth files in cwd via ai_models/fine_tune.py"
    )
    parser.add_argument(
        "--data_dir", "-d",
        required=True,
        type=Path,
        help="Directory containing one subfolder per .pth (named by pth stem)"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=10,
        help="Number of epochs to pass to fine_tune"
    )
    parser.add_argument(
        "--batch_size", "-b",
        type=int,
        default=16,
        help="Batch size to pass to fine_tune"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing the base .pth checkpoints to fine-tune. "
            "Defaults to searching both the project root and ai_models/models."
        )
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=None,
        help=(
            "Directory where fine-tuned checkpoints will be written. "
            "Defaults to <repo>/finetuned_models."
        )
    )
    parser.add_argument("--npu", action="store_true", help="Use NPUs for training")
    args = parser.parse_args()

    project_root  = Path(__file__).resolve().parent
    ai_models_dir = project_root / "ai_models"
    ft_script     = ai_models_dir / "fine_tune.py"

    output_dir = args.output_dir or (project_root / "finetuned_models")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.data_dir.is_dir():
        print(f"Error: data_dir '{args.data_dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)
    if not ft_script.exists():
        print(f"Error: cannot find {ft_script}", file=sys.stderr)
        sys.exit(1)

    # Ensure imports inside fine_tune.py resolve correctly
    sys.path.insert(0, str(ai_models_dir))
    sys.path.insert(0, str(project_root))

    model_dirs = []
    if args.model_dir is not None:
        model_dirs.append(args.model_dir)
    else:
        model_dirs.extend([project_root, project_root / "ai_models" / "models"])

    pth_files = []
    seen = set()
    for directory in model_dirs:
        if not directory.exists():
            print(f"Warning: model directory not found, skipping: {directory}", file=sys.stderr)
            continue
        for path in sorted(directory.glob("*.pth")):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            pth_files.append(resolved)

    if not pth_files:
        print("No .pth files found in the searched model directories.", file=sys.stderr)
        return

    for pth in pth_files:
        # Dataset folder is data_dir / <pth_basename_without_ext>
        ds_folder = args.data_dir / pth.stem
        if not ds_folder.is_dir():
            print(f"Warning: dataset folder not found, skipping: {ds_folder}", file=sys.stderr)
            continue

        print(f"\n=== Fine-tuning on dataset {ds_folder} with model {pth.name} ===")
        finetuned_path = output_dir / pth.name
        if not finetuned_path.exists():
            finetuned_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pth, finetuned_path)
        else:
            print(f"    ↳ reusing existing checkpoint at {finetuned_path}")
        # Build argv as if calling:
        # python ai_models/fine_tune.py --dataset ds_folder --model_path finetuned_path --epochs X --batch-size Y
        sys.argv = [
            str(ft_script),
            "--dataset",    str(ds_folder),
            "--model_path", str(finetuned_path),
            "--epochs",     str(args.epochs),
            "--batch-size", str(args.batch_size),
        ]
        if args.npu:
            sys.argv.append("--npu")
        runpy.run_path(str(ft_script), run_name="__main__")

if __name__ == "__main__":
    main()
