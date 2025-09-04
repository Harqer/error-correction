"""
Thin adapter that **delegates** to the paper's noise model implementation
bundled inside this repository under `google_qec_paper_noise_model/`.

We intentionally avoid re-implementing any physics here to ensure 100% fidelity
with the paper's method. The adapter searches for a reasonable "builder" entrypoint
and calls it to obtain a `stim.Circuit` for the requested basis.

Expected behavior and parameters follow the repo README (Paper-aligned noise model)
and the Nature paper (AlphaQubit). See:
 - Repo README section "Paper-aligned 噪声模型参数"
 - Nature: "Learning high-accuracy error decoding for quantum processors"
"""
from __future__ import annotations
from typing import Any, Callable

# `stim` is already a repo dependency
import stim  # type: ignore


def _resolve_builder() -> Callable[[dict, str], stim.Circuit]:
    """Locate the paper's builder entrypoint.

    We first look for module-level functions inside
    ``google_qec_paper_noise_model``.  If none are found, we fall back to
    instantiating ``SurfaceCodeCircuitBuilder`` directly.  This avoids
    re-implementing any of the physics while remaining resilient to minor
    refactors of the bundled paper code.

    Returns:
        A callable: ``(config_dict, basis) -> stim.Circuit``
    Raises:
        ImportError: if no usable entrypoint can be found.
    """

    # Known function-style entrypoints.
    candidate_imports = [
        "google_qec_paper_noise_model.build_circuit",
        "google_qec_paper_noise_model.builder.build_circuit",
        "google_qec_paper_noise_model.noise.build_circuit",
        "google_qec_paper_noise_model.paper_aligned.build_circuit",
        "google_qec_paper_noise_model.build_paper_aligned_circuit",
        "google_qec_paper_noise_model.builder.build_paper_aligned_circuit",
    ]

    for dotted in candidate_imports:
        try:
            module_path, func_name = dotted.rsplit(".", 1)
            mod = __import__(module_path, fromlist=[func_name])
            func = getattr(mod, func_name, None)
            if callable(func):
                return func  # type: ignore[return-value]
        except Exception:
            continue

    # Fallback: instantiate SurfaceCodeCircuitBuilder from the paper package.
    try:
        from google_qec_paper_noise_model.circuit_builder import (
            SurfaceCodeCircuitBuilder,
        )

        def _wrapper(cfg: dict, basis: str) -> stim.Circuit:
            distance = int(cfg.get("distance", 5))
            rounds = int(cfg.get("rounds", 25))
            processor = cfg.get("processor", "72_qubit_paper_aligned")
            builder = SurfaceCodeCircuitBuilder(
                distance=distance,
                rounds=rounds,
                basis=basis,
                processor=processor,
            )
            return builder.build_circuit()

        return _wrapper
    except Exception as exc:  # pragma: no cover - env dependent
        raise ImportError(
            "Could not find a paper-aligned circuit builder in "
            "`google_qec_paper_noise_model`."
        ) from exc


def build_paper_aligned_circuit(config: dict[str, Any], basis: str) -> stim.Circuit:
    """Return a Stim circuit using the **paper's** exact noise model implementation.

    Args:
        config: dict loaded from `configs/paper_aligned.yaml` (see repo README).
        basis:  'x' or 'z' (memory experiment basis).
    """
    if not isinstance(basis, str) or basis.lower() not in {"x", "z"}:
        raise ValueError("`basis` must be 'x' or 'z'.")
    builder = _resolve_builder()
    return builder(config, basis.lower())
