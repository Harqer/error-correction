# -*- coding: utf-8 -*-
"""
Pauli Twirling for Quantum Channels, including Generalized Twirling for
channels with leakage.

This module implements the conversion of a quantum channel into a stochastic
Pauli channel. The primary method uses the Pauli Transfer Matrix (PTM).
"""
from typing import Literal
import numpy as np
from . import kraus_utils

# Pauli bases used for standard twirling over qubit channels.  These are kept
# here instead of in ``kraus_utils`` so that the tests can access them directly
# via ``pauli_twirl.PAULI_2Q_BASIS``.
PAULI_1Q_BASIS = [
    kraus_utils.PAULI_I,
    kraus_utils.PAULI_X,
    kraus_utils.PAULI_Y,
    kraus_utils.PAULI_Z,
]
PAULI_2Q_BASIS = [
    np.kron(p1, p2) for p1 in PAULI_1Q_BASIS for p2 in PAULI_1Q_BASIS
]


def _choi_from_kraus(kraus_ops: list[np.ndarray]) -> np.ndarray:
    """Computes the Choi matrix for a quantum channel."""
    d = kraus_ops[0].shape[0]
    choi = np.zeros((d**2, d**2), dtype=complex)
    for k in kraus_ops:
        vec_k = k.flatten('F').reshape(-1, 1)
        choi += vec_k @ vec_k.conj().T
    return choi


def _ptm_from_choi(choi: np.ndarray, basis: list[np.ndarray]) -> np.ndarray:
    """Computes the Pauli Transfer Matrix (PTM) from a Choi matrix."""
    d_sq = len(basis)
    d = int(np.sqrt(d_sq))
    ptm = np.zeros((d_sq, d_sq), dtype=float)

    for i, pi in enumerate(basis):
        for j, pj in enumerate(basis):
            term = np.kron(pi.T, pj)
            ptm[i, j] = np.real(np.trace(choi @ term))

    return ptm / d


def twirl_to_pauli_probs(
    kraus_ops: list[np.ndarray],
    n_qubits: int,
    method: Literal["ptm"] = "ptm",
) -> np.ndarray:
    """Return Pauli error probabilities for a qubit channel.

    The returned vector contains the probabilities of the non-identity Pauli
    operators after twirling.  The sum of the probabilities equals the total
    error rate of the channel.
    """
    if n_qubits not in [1, 2]:
        raise NotImplementedError("Only 1 and 2 qubit channels are supported.")

    basis = PAULI_1Q_BASIS if n_qubits == 1 else PAULI_2Q_BASIS
    d = 2 ** n_qubits

    chi_diag = np.zeros(len(basis), dtype=float)
    for k in kraus_ops:
        coeffs = np.array([np.trace(b.conj().T @ k) for b in basis]) / d
        chi_diag += np.abs(coeffs) ** 2

    # Numerical noise can produce tiny negative values; clip them and
    # renormalize so the probabilities sum to one.
    chi_diag = np.clip(chi_diag.real, 0.0, None)
    chi_diag /= chi_diag.sum()

    # Exclude the identity component (index 0) from the returned vector.
    return chi_diag[1:]


def twirl_to_generalized_pauli_probs(
    kraus_ops: list[np.ndarray], n_qutrits: int
) -> dict[str, float]:
    """Generalized Pauli Twirling via ``leakysim``.

    This delegates the heavy lifting to ``leakysim``'s reference implementation
    and extracts the probabilities for Pauli errors that keep the system within
    the computational subspace.
    """

    if n_qutrits not in [1, 2]:
        raise NotImplementedError("Only 1 and 2-qutrit twirling is implemented.")

    try:  # pragma: no cover - environment dependent
        import leakysim as _leaky
    except ModuleNotFoundError:  # pragma: no cover - fallback for renamed package
        import leaky as _leaky

    channel = _leaky.generalized_pauli_twirling(
        kraus_ops, num_qubits=n_qutrits, num_level=3, safety_check=False
    )
    status = _leaky.LeakageStatus(status=[0] * n_qutrits)

    import itertools

    paulis = ["".join(p) for p in itertools.product("IXYZ", repeat=n_qutrits)]
    return {p: channel.get_prob_from_to(status, status, p) for p in paulis}
