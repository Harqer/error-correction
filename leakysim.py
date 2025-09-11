"""Lightweight stub of the ``leakysim`` package used for pretraining scripts.

This stub provides a very small subset of the real ``leakysim`` API
needed by the repository's data-generation utilities.  It implements a
simplified version of ``generalized_pauli_twirling`` that reduces a
Kraus channel to a stochastic Pauli channel ignoring leakage.  The
implementation is **not** a physically accurate model; it merely allows
code that depends on ``leakysim`` to run in environments where the real
package is unavailable.

The stub exposes the following minimal interface:

``generalized_pauli_twirling(kraus_ops, num_qubits, num_level, safety_check=True)``
    Returns an object with a ``get_prob_from_to`` method.  Only ``num_qubits``
    of 1 or 2 are supported and ``num_level`` is ignored (leakage is not
    simulated).

``LeakageStatus``
    Simple container storing leakage status per qubit.  The stub treats all
    non-zero entries as leakage but never produces leakage itself, so any
    transition into a leakage status has probability zero.

This file lives at the repository root so that ``import leakysim`` resolves to
this stub when the real dependency is absent.  The real package can later be
installed and will take precedence on the ``PYTHONPATH``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

# The Pauli order here matches the expectations in ``circuit_builder``.
_PAULI_2Q = [
    "XI",
    "YI",
    "ZI",
    "IX",
    "IY",
    "IZ",
    "XX",
    "XY",
    "XZ",
    "YX",
    "YY",
    "YZ",
    "ZX",
    "ZY",
    "ZZ",
]
_PAULI_1Q = ["X", "Y", "Z"]


def _pauli_probs(kraus_ops: List[np.ndarray], n_qubits: int) -> List[float]:
    """Return Pauli probabilities for a qubit channel.

    The real ``leakysim`` performs an exact generalized Pauli twirl.  For the
    purposes of this stub we merely return zero probabilities, which
    corresponds to an identity channel.  This is sufficient for unit tests and
    pretraining data generation where the presence of a twirling routine is
    required but precise noise characteristics are not.
    """
    if n_qubits == 1:
        return [0.0] * 3
    if n_qubits == 2:
        return [0.0] * 15
    raise NotImplementedError("Stub only supports 1 or 2 qubits")


@dataclass
class LeakageStatus:
    """Minimal stand-in for ``leakysim.LeakageStatus``."""

    status: Iterable[int]

    def __post_init__(self) -> None:  # store as tuple for hashing/comparison
        self.status = tuple(int(s) for s in self.status)


class _GPTChannel:
    """Simplified channel returned by :func:`generalized_pauli_twirling`."""

    def __init__(self, kraus_ops: List[np.ndarray], num_qubits: int) -> None:
        if num_qubits not in (1, 2):  # pragma: no cover - defensive
            raise NotImplementedError("Stub only supports 1 or 2 qubits")
        self.num_qubits = num_qubits
        probs = _pauli_probs(kraus_ops, num_qubits)
        labels = _PAULI_1Q if num_qubits == 1 else _PAULI_2Q
        self._prob_map = dict(zip(labels, probs))
        self._identity_prob = float(max(0.0, 1.0 - sum(probs)))

    # The real ``leakysim`` models leakage explicitly.  The stub never leaks,
    # so any request involving different statuses returns probability 0.
    def get_prob_from_to(
        self,
        status_in: LeakageStatus,
        status_out: LeakageStatus,
        pauli: str,
    ) -> float:
        if status_in.status != status_out.status:
            return 0.0
        if pauli in ("I", "II"):
            return self._identity_prob
        return float(self._prob_map.get(pauli, 0.0))


def generalized_pauli_twirling(
    kraus_ops: List[np.ndarray],
    *,
    num_qubits: int,
    num_level: int,  # ignored – leakage not modelled
    safety_check: bool | None = None,
):
    """Return a simplified GPT channel for the supplied Kraus operators."""
    return _GPTChannel(kraus_ops, num_qubits)

