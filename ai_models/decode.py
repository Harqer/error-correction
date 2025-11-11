#!/usr/bin/env python3
"""Decode detection-event syndromes with a trained AlphaQubit model.

This utility mirrors the command advertised in the project README.  It loads
``*.npy``/``*.npz`` bundles containing detection events, feeds them through a
trained :class:`~ai_models.model_mla.AlphaQubitDecoder` checkpoint, and writes
summary metrics to ``results/``.

The loader is intentionally forgiving:

* ``.npz`` files are expected to contain a ``data`` array of shape ``(N, …)``
  plus optional label arrays such as ``obs`` or ``label``.
* ``.npy`` files may store raw syndrome tensors or a pickled dictionary with
  ``data``/``syndromes`` and optional ``logical`` keys.

Only a *single* measurement basis is supported per file.  If it cannot be
inferred from the filename or payload, provide ``--basis x`` or ``--basis z``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ai_models.model_mla import AlphaQubitDecoder


LabelArray = Optional[np.ndarray]
BasisArray = Optional[np.ndarray]


@dataclass
class LoadedSyndromes:
    """Container for data parsed from ``--data``."""

    syndromes: np.ndarray
    labels: LabelArray
    basis: BasisArray


class InferenceDataset(Dataset):
    """Minimal dataset yielding (inputs, basis, mask) tuples for decoding."""

    def __init__(
        self,
        inputs: torch.Tensor,
        basis: torch.Tensor,
        final_mask: torch.Tensor,
    ) -> None:
        super().__init__()
        if basis.ndim != 1:
            raise ValueError("Basis tensor must be 1D (one label per sample)")
        if inputs.shape[0] != basis.shape[0]:
            raise ValueError("Mismatch between number of samples and basis vector")

        self.inputs = inputs
        self.basis = basis
        self.final_mask = final_mask.to(torch.int8)

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.basis[idx], self.final_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Path to a saved AlphaQubit checkpoint (.pth)")
    parser.add_argument("--data", type=Path, required=True, help="Syndrome data (.npz/.npy) to decode")
    parser.add_argument("--basis", type=str, default=None, help="Override measurement basis: 'x' or 'z'")
    parser.add_argument("--batch-size", type=int, default=512, help="Mini-batch size for decoding")
    parser.add_argument("--device", type=str, default=None, help="Torch device to run on (default: auto)")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden width of the decoder (must match training)")
    parser.add_argument("--heads", type=int, default=8, help="Number of attention heads (must match training)")
    parser.add_argument("--layers", type=int, default=12, help="Number of transformer layers (must match training)")
    parser.add_argument("--output", type=Path, default=None, help="Where to store aggregated metrics (JSON)")
    parser.add_argument("--predictions", type=Path, default=None, help="Optional path to save per-shot probabilities (.npy)")
    parser.add_argument(
        "--cudaq",
        action="store_true",
        help="Enable CUDA-Q execution so inference can be dispatched to a quantum backend via NVQLink",
    )
    parser.add_argument(
        "--cudaq-target",
        type=str,
        default=None,
        help="Optional CUDA-Q target name (defaults to NVQLink-enabled target when --cudaq is set)",
    )
    return parser.parse_args()


def load_syndrome_file(path: Path) -> LoadedSyndromes:
    """Load a ``.npy``/``.npz`` bundle and return its contents."""

    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            if "data" not in data:
                raise KeyError(f"Expected 'data' array in {path}")
            syndromes = np.asarray(data["data"])
            labels = _pick_first_existing(data, ["obs", "label", "labels", "logical", "logical_error"])
            basis = _pick_first_existing(data, ["basis", "basis_id", "bases", "basis_ids"])
            return LoadedSyndromes(syndromes, labels, basis)

    array = np.load(path, allow_pickle=True)
    if isinstance(array, np.ndarray) and array.dtype == object:
        # Could be a pickled dictionary wrapped in an object array.
        if array.shape == ():
            array = array.item()
        elif array.ndim == 1 and array.size == 1:
            array = array[0]

    if isinstance(array, dict):
        syndromes = np.asarray(array.get("data") or array.get("syndromes") or array.get("detections"))
        if syndromes is None:
            raise KeyError(f"Could not locate syndrome array inside {path}")
        labels = array.get("obs") or array.get("label") or array.get("labels") or array.get("logical")
        basis = array.get("basis") or array.get("basis_id")
        return LoadedSyndromes(np.asarray(syndromes), _as_optional_array(labels), _as_optional_array(basis))

    return LoadedSyndromes(np.asarray(array), None, None)


def _pick_first_existing(store: Iterable[str], keys: Iterable[str]) -> Optional[np.ndarray]:
    for key in keys:
        if key in store:
            return np.asarray(store[key])
    return None


def _as_optional_array(value: object) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.size == 0:
        return None
    return arr


def infer_basis(basis_arg: Optional[str], loaded_basis: BasisArray, data_path: Path, num_samples: int) -> torch.Tensor:
    """Resolve the measurement basis for each sample."""

    if basis_arg is not None:
        basis_id = _basis_str_to_id(basis_arg)
        return torch.full((num_samples,), basis_id, dtype=torch.int8)

    if loaded_basis is not None:
        arr = np.asarray(loaded_basis)
        if arr.ndim == 0:
            basis_id = _validate_basis_id(int(arr))
            return torch.full((num_samples,), basis_id, dtype=torch.int8)
        if arr.ndim == 1 and arr.shape[0] == num_samples:
            vec = np.vectorize(_validate_basis_id, otypes=[np.int8])(arr)
            return torch.from_numpy(vec.astype(np.int8))
        raise ValueError(
            "Loaded basis metadata must be scalar or length-N array of {0,1}. "
            f"Received shape {arr.shape}"
        )

    inferred = infer_basis_from_name(data_path)
    if inferred is not None:
        return torch.full((num_samples,), inferred, dtype=torch.int8)

    raise ValueError(
        "Unable to determine measurement basis. Pass --basis (x/z) explicitly."
    )


def _basis_str_to_id(text: str) -> int:
    lowered = text.strip().lower()
    if lowered in {"x", "0", "bx"}:
        return 0
    if lowered in {"z", "1", "bz"}:
        return 1
    raise ValueError(f"Unsupported basis specifier '{text}'. Use 'x' or 'z'.")


def _validate_basis_id(value: int) -> int:
    if value in (0, 1):
        return int(value)
    raise ValueError(f"Basis IDs must be 0 (X) or 1 (Z); received {value}")


def infer_basis_from_name(path: Path) -> Optional[int]:
    name = path.name.lower()
    if any(tag in name for tag in ("_bx_", "-bx-", "_basisx", "_x_basis")):
        return 0
    if any(tag in name for tag in ("_bz_", "-bz-", "_basisz", "_z_basis")):
        return 1

    tokens = [token for token in re.split(r"[^a-z0-9]+", name) if token]
    for token in tokens:
        if token in {"bx", "x", "basisx", "xbasis"}:
            return 0
        if token in {"bz", "z", "basisz", "zbasis"}:
            return 1
    return None


def prepare_inputs(syndromes: np.ndarray, basis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Convert raw syndromes to tensors expected by the decoder."""

    x = np.asarray(syndromes, dtype=np.float32)
    if x.ndim == 2:
        x = x[:, None, :, None]
    elif x.ndim == 3:
        x = x[:, :, :, None]
    elif x.ndim != 4:
        raise ValueError(
            "Syndrome array must have 2, 3 or 4 dimensions; "
            f"received shape {x.shape}"
        )

    N, R, S, F = x.shape
    basis_feat = basis.float().numpy().reshape(N, 1, 1, 1)
    basis_feat = np.broadcast_to(basis_feat, (N, R, S, 1))
    x = np.concatenate([x, basis_feat.astype(np.float32)], axis=-1)

    d = math.isqrt(S + 1)
    if d * d != S + 1:
        d += 1
        pad = d * d - 1 - S
        x = np.concatenate([x, np.zeros((N, R, pad, x.shape[-1]), dtype=np.float32)], axis=2)
        S = x.shape[2]

    grid_size = d - 1
    final_mask = torch.tensor(
        [1 if (r + c) % 2 == 0 else 2 for r in range(d) for c in range(d)][1:],
        dtype=torch.int8,
    )
    if final_mask.numel() != S:
        raise RuntimeError(
            "Constructed final_mask has incorrect size: "
            f"expected {S}, got {final_mask.numel()}"
        )

    return torch.from_numpy(x), final_mask, grid_size


def select_device(requested: Optional[str]) -> torch.device:
    if requested is None:
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    try:
        torch.empty(1, device=device)
    except (RuntimeError, AssertionError):
        raise RuntimeError(f"Failed to initialise torch device '{requested}'")
    return device


def load_model(
    model_path: Path,
    num_features: int,
    num_stabilizers: int,
    grid_size: int,
    hidden_dim: int,
    heads: int,
    layers: int,
    device: torch.device,
) -> AlphaQubitDecoder:
    def _unwrap_state(obj):
        if isinstance(obj, torch.nn.Module):
            return obj.state_dict()
        if isinstance(obj, dict):
            for key in ("state_dict", "model_state", "model", "module"):
                if key in obj:
                    unwrapped = _unwrap_state(obj[key])
                    if isinstance(unwrapped, dict):
                        return unwrapped
            return obj
        return obj

    def _strip_module_prefix(state_dict: dict) -> dict:
        if not isinstance(state_dict, dict):
            return state_dict
        if not state_dict:
            return state_dict

        prefix = "module."
        if any(key.startswith(prefix) for key in state_dict):
            factory = state_dict.__class__
            stripped = factory()
            for key, value in state_dict.items():
                if key.startswith(prefix):
                    stripped[key[len(prefix):]] = value
                else:
                    stripped[key] = value
            return stripped
        return state_dict

    state = torch.load(model_path, map_location=device)
    state = _unwrap_state(state)
    state = _strip_module_prefix(state)

    model = AlphaQubitDecoder(
        num_features,
        hidden_dim,
        num_stabilizers,
        grid_size,
        num_heads=heads,
        num_layers=layers,
    )

    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        patched = _maybe_patch_feature_projections(state, num_features)
        if patched is None:
            raise RuntimeError(
                "Failed to load checkpoint and no compatible feature projection "
                "upgrade path was found."
            ) from exc
        model.load_state_dict(patched)

    model.to(device)
    model.eval()
    return model


def configure_cudaq(target: Optional[str]) -> str:
    """Initialise CUDA-Q with the requested target and NVQLink support."""

    try:
        import cudaq
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError(
            "CUDA-Q support requested via --cudaq, but the 'cudaq' package is not installed."
        ) from exc

    resolved_target = target or "nvq-link"
    try:
        cudaq.set_target(resolved_target)
    except Exception as exc:  # pragma: no cover - depends on external runtime
        raise RuntimeError(f"Failed to configure CUDA-Q target '{resolved_target}'") from exc

    if not _enable_nvqlink(cudaq):
        warnings.warn(
            "CUDA-Q runtime does not expose an NVQLink enablement hook; proceeding without explicit activation.",
            RuntimeWarning,
        )

    return resolved_target


def _enable_nvqlink(cudaq_module) -> bool:
    """Attempt to enable NVQLink within a CUDA-Q runtime module."""

    # Known entry points may vary between CUDA-Q versions; try a few options.
    candidates = (
        getattr(cudaq_module, "enable_nvqlink", None),
        getattr(cudaq_module, "enableNVQLink", None),
    )
    for fn in candidates:
        if callable(fn):
            fn()
            return True

    runtime = getattr(cudaq_module, "runtime", None)
    if runtime is not None:
        for name in ("enable_nvqlink", "enableNVQLink"):
            fn = getattr(runtime, name, None)
            if callable(fn):
                fn()
                return True

    # As a last resort, set a well-named environment toggle in case the runtime
    # inspects it on start-up.
    os.environ.setdefault("CUDAQ_ENABLE_NVQLINK", "1")
    return False


def _maybe_patch_feature_projections(state: dict, expected_features: int) -> Optional[dict]:
    """Back-fill missing ``embedder.feature_projs`` weights when possible."""

    feature_keys = []
    for key in state:
        if not key.startswith("embedder.feature_projs."):
            continue
        parts = key.split(".")
        if len(parts) < 4 or parts[3] not in {"weight", "bias"}:
            continue
        feature_keys.append(key)
    if not feature_keys:
        return None

    present_indices = sorted({int(key.split(".")[2]) for key in feature_keys})
    if present_indices == list(range(expected_features)):
        return None

    if present_indices == list(range(expected_features - 1)):
        template_idx = present_indices[-1]
        missing_idx = expected_features - 1

        patched_state = state.__class__(state)
        for suffix in ("weight", "bias"):
            template_key = f"embedder.feature_projs.{template_idx}.{suffix}"
            missing_key = f"embedder.feature_projs.{missing_idx}.{suffix}"
            if template_key not in state:
                return None
            patched_state[missing_key] = state[template_key].clone()
        return patched_state

    return None


def compute_metrics(probabilities: torch.Tensor, labels: Optional[torch.Tensor]) -> dict:
    probs = probabilities.detach().cpu()
    metrics = {
        "shots": int(probs.numel()),
        "mean_logical_probability": float(probs.mean().item()),
        "predicted_logical_error_rate": float((probs >= 0.5).float().mean().item()),
    }

    if labels is not None:
        labels = labels.detach().cpu().float()
        metrics["observed_logical_error_rate"] = float(labels.mean().item())
        preds = (probs >= 0.5).float()
        metrics["decoder_accuracy"] = float((preds == labels).float().mean().item())
        eps = 1e-7
        bce = -(labels * torch.log(probs + eps) + (1 - labels) * torch.log(1 - probs + eps)).mean()
        metrics["binary_cross_entropy"] = float(bce.item())

    return metrics


def main() -> None:
    args = parse_args()

    loaded = load_syndrome_file(args.data)
    num_samples = int(loaded.syndromes.shape[0])
    basis_vector = infer_basis(args.basis, loaded.basis, args.data, num_samples)
    inputs, final_mask, grid_size = prepare_inputs(loaded.syndromes, basis_vector)

    cudaq_target = None
    if args.cudaq:
        cudaq_target = configure_cudaq(args.cudaq_target)
        if args.device is None:
            args.device = "cuda"

    device = select_device(args.device)

    dataset = InferenceDataset(inputs, basis_vector, final_mask)
    loader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=(device.type == "cuda"))

    sample_input = inputs[0]
    R, S, F = sample_input.shape

    model = load_model(
        args.model,
        num_features=F,
        num_stabilizers=S,
        grid_size=grid_size,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        layers=args.layers,
        device=device,
    )

    probs: list[torch.Tensor] = []
    with torch.no_grad():
        for xb, basis, mask in loader:
            xb = xb.to(device)
            basis = basis.to(device)
            mask = mask.to(device)
            logits = model(xb, basis, mask)
            probs.append(torch.sigmoid(logits).cpu())

    probabilities = torch.cat(probs)

    labels_tensor = None
    if loaded.labels is not None:
        labels = np.asarray(loaded.labels)
        if labels.ndim > 1:
            labels = labels[:, 0]
        if labels.shape[0] != num_samples:
            raise ValueError(
                "Label array shape mismatch: "
                f"expected length {num_samples}, got {labels.shape}"
            )
        labels_tensor = torch.from_numpy(labels.astype(np.float32))

    metrics = compute_metrics(probabilities, labels_tensor)

    output_path = args.output
    if output_path is None:
        output_dir = Path("results")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{args.data.stem}_metrics.json"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_path": str(args.model.resolve()),
        "data_path": str(args.data.resolve()),
        "basis": "X" if basis_vector[0].item() == 0 else "Z",
        "rounds": int(R),
        "stabilizers": int(S),
        "features": int(F),
        **({"cudaq_target": cudaq_target} if cudaq_target is not None else {}),
        **metrics,
    }

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"Decoded {num_samples} shots. Metrics written to {output_path}.")
    if labels_tensor is None:
        print("(No labels provided – reported rates are model predictions only.)")

    if args.predictions is not None:
        args.predictions.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.predictions, probabilities.numpy())
        print(f"Saved per-shot probabilities to {args.predictions}")


if __name__ == "__main__":
    main()

