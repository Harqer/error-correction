from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple

# --------
# Pauli bases
# --------
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI_1Q = [I, X, Y, Z]

def _kron(*ops: np.ndarray) -> np.ndarray:
    out = np.array([[1.0 + 0j]])
    for op in ops:
        out = np.kron(out, op)
    return out

PAULI_2Q: List[np.ndarray] = [_kron(a, b) for a in PAULI_1Q for b in PAULI_1Q]
# order: {I,X,Y,Z} ⊗ {I,X,Y,Z}

# --------
# Helpers: project Kraus ops to qubit subspace
# --------

def _project_1q_to_qubit(K: np.ndarray) -> np.ndarray:
    return K if K.shape[0] == 2 else K[:2, :2]

def _project_2q_to_qubit(K: np.ndarray) -> np.ndarray:
    if K.shape[0] == 4:
        return K
    comp = [0, 1, 3, 4]
    K4 = np.zeros((4, 4), dtype=complex)
    for a, ia in enumerate(comp):
        for b, ib in enumerate(comp):
            K4[a, b] = K[ia, ib]
    return K4

# --------
# Leakage estimates
# --------

def _avg_leakage_1q(Ks: List[np.ndarray]) -> float:
    rho1_3 = np.zeros((3, 3), complex); rho1_3[1, 1] = 1.0
    Ks3: List[np.ndarray] = []
    for K in Ks:
        if K.shape[0] == 2:
            K3 = np.zeros((3, 3), complex)
            K3[:2, :2] = K
            K3[2, 2] = 1.0
            Ks3.append(K3)
        else:
            Ks3.append(K)
    E = sum(K @ rho1_3 @ K.conj().T for K in Ks3)
    return float(np.real(E[2, 2]))

def _avg_leakage_2q(Ks: List[np.ndarray]) -> float:
    if all(K.shape[0] == 4 for K in Ks):
        return 0.0
    comp = [0, 1, 3, 4]
    leaked = 0.0
    for basis_idx in comp:
        rho = np.zeros((9, 9), complex); rho[basis_idx, basis_idx] = 1.0
        E = sum(K @ rho @ K.conj().T for K in Ks)
        mask = np.zeros(9, float)
        for i in range(3):
            for j in range(3):
                if i == 2 or j == 2:
                    mask[3 * i + j] = 1.0
        leaked += float(np.real(np.sum(np.diag(E) * mask)))
    return leaked / 4.0

# --------
# PTM diagonals and conversions to Pauli probs
# --------

def _ptm_diag_from_kraus_1q(Ks: List[np.ndarray]) -> np.ndarray:
    Ks2 = [_project_1q_to_qubit(K) for K in Ks]
    lam = np.zeros(4, dtype=float)
    for idx, P in enumerate(PAULI_1Q):
        EP = sum(K @ P @ K.conj().T for K in Ks2)
        lam[idx] = 0.5 * np.real(np.trace(P.conj().T @ EP))
    lam[0] = 1.0
    return lam

def _ptm_diag_from_kraus_2q(Ks: List[np.ndarray]) -> np.ndarray:
    lam = np.zeros(16, dtype=float)
    for idx, P in enumerate(PAULI_2Q):
        EP = sum(K @ P @ K.conj().T for K in Ks)
        lam[idx] = 0.25 * np.real(np.trace(P.conj().T @ EP))
    lam[0] = 1.0
    return lam

def _hadamard4() -> np.ndarray:
    return np.array([[1, 1, 1, 1],
                     [1, 1, -1, -1],
                     [1, -1, 1, -1],
                     [1, -1, -1, 1]], dtype=float)

def _hadamard16() -> np.ndarray:
    H4 = _hadamard4()
    return np.kron(H4, H4)

def _lam_to_probs_1q(lam: np.ndarray) -> np.ndarray:
    H = _hadamard4()
    p = (H @ lam.reshape(4, 1)).ravel() / 4.0
    p[p < 0] = 0.0
    s = p.sum()
    return p / s if s > 0 else np.array([1, 0, 0, 0], float)

def _lam_to_probs_2q(lam: np.ndarray) -> np.ndarray:
    H = _hadamard16()
    p = (H @ lam.reshape(16, 1)).ravel() / 16.0
    p[p < 0] = 0.0
    s = p.sum()
    return p / s if s > 0 else np.eye(16, dtype=float)[0]

# --------
# Public API
# --------

def twirl_to_pauli_channel(Ks: List[np.ndarray], n_qubits: int) -> Tuple[np.ndarray, np.ndarray]:
    if n_qubits == 1:
        lam = _ptm_diag_from_kraus_1q(Ks)
        probs = _lam_to_probs_1q(lam)
        leak = _avg_leakage_1q(Ks)
        return probs, leak
    elif n_qubits == 2:
        if Ks and Ks[0].shape[0] in (2, 3):
            lam = _ptm_diag_from_kraus_1q(Ks)
            p1 = _lam_to_probs_1q(lam)
            probs = np.kron(p1, p1)
            leak = 1.0 - (1.0 - _avg_leakage_1q(Ks)) ** 2
            return probs, leak
        Ks4 = [_project_2q_to_qubit(K) for K in Ks]
        lam = _ptm_diag_from_kraus_2q(Ks4)
        probs = _lam_to_probs_2q(lam)
        leak = _avg_leakage_2q(Ks)
        return probs, leak
    else:
        raise NotImplementedError("Only 1 or 2 qubits supported")
