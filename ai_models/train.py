"""Unified training entry point for AlphaQubit models.

This module bridges the gap between the configuration-driven workflow
advertised in the README and the lower level training utilities already
available in :mod:`ai_models.model_mla`.  Given a YAML configuration it
optionally synthesises a dataset with the appropriate noise model and
trains the MLA decoder on it.

Example
-------
    python ai_models/train.py --config configs/dem.yaml

The configuration may also provide a ``training`` section to override
hyperparameters.  Command line flags take precedence, followed by values
in the YAML file, with built-in defaults used as the final fallback.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import yaml

from simulator.dem_generator import generate_dem_data
from simulator.si1000_generator import si1000_noise_model
from simulator.pauli_plus_simulator import PauliPlusSimulator

from ai_models.model_mla import AlphaQubitDecoder, train as train_mla


MODEL_TYPES = {"dem", "si1000", "pauli_plus", "paper_aligned"}


class GeneratedSyndromeDataset(Dataset):
    """Dataset backed by in-memory syndrome arrays.

    Parameters
    ----------
    syndromes:
        Array of shape ``(N, S)`` or ``(N, R, S)`` or ``(N, R, S, F)``
        containing detection event measurements.
    logicals:
        Array of shape ``(N,)`` or ``(N, K)`` with logical observables.
    basis_id:
        Integer identifier for the measurement basis (``0`` for X,
        ``1`` for Z, ``-1`` for unknown/mixed).
    """

    def __init__(self, syndromes: np.ndarray, logicals: np.ndarray, basis_id: int) -> None:
        super().__init__()
        x = np.asarray(syndromes, dtype=np.float32)
        if x.ndim == 2:
            # (N, S) -> (N, R=1, S, F=1)
            x = x[:, None, :, None]
        elif x.ndim == 3:
            # (N, R, S) -> (N, R, S, F=1)
            x = x[:, :, :, None]
        elif x.ndim != 4:
            raise ValueError(
                "Syndrome array must have 2, 3 or 4 dimensions; "
                f"received shape {x.shape}"
            )

        N, R, S, F = x.shape
        # Embed the basis as an additional feature channel
        basis_feat = np.full((N, R, S, 1), float(basis_id), dtype=np.float32)
        x = np.concatenate([x, basis_feat], axis=-1)
        F = x.shape[-1]

        # Ensure that S+1 is a perfect square to respect the layout expected by
        # :class:`AlphaQubitDecoder`.  Pad dummy stabilisers if necessary.
        d = math.isqrt(S + 1)
        if d * d != S + 1:
            d += 1
            pad = d * d - 1 - S
            x = np.concatenate([x, np.zeros((N, R, pad, F), dtype=np.float32)], axis=2)
            S = x.shape[2]
        else:
            pad = 0

        # Final round mask: checkerboard pattern excluding the dummy stabiliser.
        final_mask = torch.tensor(
            [1 if (r + c) % 2 == 0 else 2 for r in range(d) for c in range(d)][1:],
            dtype=torch.int8,
        )
        if final_mask.numel() != S:
            raise RuntimeError(
                "Final mask construction mismatch: expected size "
                f"{S}, obtained {final_mask.numel()}"
            )

        y = np.asarray(logicals, dtype=np.float32)
        if y.ndim > 1:
            y = y[:, 0]

        self.inputs = torch.from_numpy(x)
        self.labels = torch.from_numpy(y)
        self.final_mask = final_mask
        self.basis_tensor = torch.tensor(int(basis_id), dtype=torch.int8)
        self.grid_size = d - 1
        self.pad = pad

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        xb = self.inputs[idx]
        return (xb, self.basis_tensor, self.final_mask), self.labels[idx]


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def infer_model_type(path: Path, config: Dict[str, Any]) -> str:
    model_type = str(config.get("model", "")).strip().lower()
    if model_type:
        return model_type
    return path.stem.lower()


def resolve_basis(args_basis: str | None, config: Dict[str, Any], model_type: str) -> Tuple[str, int]:
    basis = (args_basis or config.get("basis") or config.get("default_basis") or "z").lower()
    if basis not in {"x", "z"}:
        raise ValueError(f"Unsupported basis '{basis}'. Expected 'x' or 'z'.")
    basis_id = 0 if basis == "x" else 1
    if model_type == "paper_aligned" and basis not in {"x", "z"}:
        raise ValueError("Paper-aligned mode requires basis 'x' or 'z'.")
    return basis.upper(), basis_id


def generate_samples(model_type: str, config: Dict[str, Any], samples: int, basis: str) -> Tuple[np.ndarray, np.ndarray]:
    if model_type == "dem":
        syndromes, logicals = generate_dem_data(samples, config)
    elif model_type == "si1000":
        circuit = si1000_noise_model(config)
        sampler = circuit.compile_detector_sampler()
        syndromes, logicals = sampler.sample(samples, separate_observables=True)
    elif model_type in {"pauli_plus", "paper_aligned"}:
        simulator = PauliPlusSimulator(config, basis)
        if model_type == "paper_aligned":
            simulator.apply_paper_aligned_noise(config)
        sampler = simulator.circuit.compile_detector_sampler()
        syndromes, logicals = sampler.sample(samples, separate_observables=True)
    else:
        raise ValueError(
            f"Unknown model type '{model_type}'. Expected one of {sorted(MODEL_TYPES)}."
        )
    return syndromes, logicals


def choose(value: Any, config_section: Dict[str, Any], key: str, default: Any) -> Any:
    if value is not None:
        return value
    if config_section and key in config_section:
        return config_section[key]
    return default


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the AlphaQubit MLA decoder using a YAML config")
    parser.add_argument("--config", required=True, type=Path, help="Path to the noise model configuration YAML")
    parser.add_argument("--samples", type=int, default=None, help="Number of Monte-Carlo shots to generate")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--model-path", type=Path, default=None, help="Where to save the trained weights")
    parser.add_argument("--basis", type=str, default=None, help="Basis override ('x' or 'z')")
    parser.add_argument("--train-split", type=float, default=None, help="Fraction of samples used for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset splits")
    parser.add_argument("--npu", action="store_true", help="Use NPUs if available")
    args = parser.parse_args()

    config = load_config(args.config)
    model_type = infer_model_type(args.config, config)
    if model_type not in MODEL_TYPES:
        raise ValueError(
            f"Configuration '{args.config}' resolves to model '{model_type}', "
            f"which is not supported. Choose from {sorted(MODEL_TYPES)}."
        )

    training_cfg = config.get("training", {}) if isinstance(config, dict) else {}

    samples = int(choose(args.samples, training_cfg, "samples", 5000))
    epochs = int(choose(args.epochs, training_cfg, "epochs", 20))
    batch_size = int(choose(args.batch_size, training_cfg, "batch_size", 64))
    lr = float(choose(args.lr, training_cfg, "lr", 5e-4))
    train_split = float(choose(args.train_split, training_cfg, "train_split", 0.9))
    model_path = choose(args.model_path, training_cfg, "model_path", None)
    if model_path is None:
        model_path = Path(f"alphaqubit_{model_type}.pth")
    else:
        model_path = Path(model_path)

    basis, basis_id = resolve_basis(args.basis, config, model_type)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading config from {args.config} ({model_type}), generating {samples} samples in {basis}-basis")
    syndromes, logicals = generate_samples(model_type, config, samples, basis)

    dataset = GeneratedSyndromeDataset(syndromes, logicals, basis_id)
    total = len(dataset)
    if total < 2:
        raise RuntimeError("Need at least two samples to perform train/validation split")

    train_size = max(1, int(total * train_split))
    valid_size = total - train_size
    if valid_size == 0:
        valid_size = 1
        train_size = total - valid_size
    generator = torch.Generator().manual_seed(args.seed)
    train_ds, valid_ds = random_split(dataset, [train_size, valid_size], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size)

    (x0, _, mask0), _ = dataset[0]
    R, S, F = x0.shape
    grid_size = dataset.grid_size
    print(f"Dataset stats: rounds={R}, stabilisers={S}, features={F}, grid={grid_size + 1}x{grid_size + 1}, pad={dataset.pad}")

    model = AlphaQubitDecoder(F, 256, S, grid_size, num_heads=8, num_layers=12)

    if args.npu and hasattr(torch, "npu") and torch.npu.is_available():
        device = torch.device("npu")
    else:
        if args.npu:
            print("[warn] --npu requested but torch.npu is unavailable; falling back to CUDA/CPU")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_save_path = str(model_path)
    os.makedirs(os.path.dirname(model_save_path) or ".", exist_ok=True)

    train_mla(model, train_loader, valid_loader, epochs, lr, device, model_save_path)
    print(f"Training complete. Model saved to {model_save_path}")


if __name__ == "__main__":
    main()
