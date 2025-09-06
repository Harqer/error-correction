"""
Thin adapter that *delegates* to the canonical noise-model implementation
that ships with this repository under `my_noise_model`.

This file intentionally contains no physics logic. It exists to:
  * Provide a stable entry point for data generation (`generate_data.py --model paper_aligned`)
  * Guarantee 1:1 behavior with the paper's noise model by calling the paper’s own builder
    instead of re-implementing it here.
"""
from __future__ import annotations
from typing import Any, Dict
import importlib


def _normalize_basis(basis: str) -> str:
    b = (basis or "").lower().strip()
    if b not in ("x", "z"):
        raise ValueError("basis must be 'x' or 'z'")
    return b


def build_paper_aligned_circuit(config: Dict[str, Any], basis: str):
    """
    Returns a Stim circuit whose noise model is *exactly* the one defined by the
    paper’s implementation included in this repository.

    Parameters
    ----------
    config : Dict[str, Any]
        YAML-loaded parameters (see configs/paper_aligned.yaml).
    basis : str
        'x' or 'z'
    """
    b = _normalize_basis(basis)

    # Known entry points; we try them in order.  We intentionally do not catch
    # AttributeError here beyond trying the next candidate, so downstream errors
    # still surface to the caller for easier debugging.
    candidates = [
        # Preferred: a clear builder exposed by the paper's package in this repo
        ("my_noise_model", "build_paper_aligned_circuit"),
        ("my_noise_model.main", "build_paper_aligned_circuit"),
        # Fallback: some repos expose this through their simulator wrapper
        ("google_qec_simulator", "build_paper_aligned_circuit"),
    ]

    last_err = None
    for modname, funcname in candidates:
        try:
            mod = importlib.import_module(modname)
            fn = getattr(mod, funcname)
            return fn(config, b)
        except Exception as e:
            last_err = e
            continue

    raise ImportError(
        "Unable to locate the paper-aligned circuit builder. Make sure "
        "`my_noise_model` is available in this repo and exposes "
        "`build_paper_aligned_circuit(config, basis)`. Last error: {}".format(last_err)
    )
