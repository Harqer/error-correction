from __future__ import annotations
import numpy as np
from typing import List, Tuple

def _pauli(name: str) -> np.ndarray:
    if name == "I":
        return np.array([[1, 0],[0, 1]], dtype=complex)
    if name == "X":
        return np.array([[0, 1],[1, 0]], dtype=complex)
    if name == "Y":
        return np.array([[0, -1j],[1j, 0]], dtype=complex)
    if name == "Z":
        return np.array([[1, 0],[0, -1]], dtype=complex)
    raise ValueError(name)

def kron(*ops: np.ndarray) -> np.ndarray:
    out = np.array([[1.0+0j]])
    for op in ops:
        out = np.kron(out, op)
    return out

def amplitude_damping_kraus(tau: float) -> List[np.ndarray]:
    """Single-qubit amplitude damping with strength gamma = 1 - exp(-tau)."""
    gamma = 1.0 - np.exp(-float(tau))
    g = float(gamma)
    K0 = np.array([[1, 0],[0, np.sqrt(1-g)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(g)],[0, 0]], dtype=complex)
    return [K0, K1]

def dephasing_kraus(p: float) -> List[np.ndarray]:
    """Single-qubit dephasing channel."""
    p = float(p)
    K0 = np.sqrt(1.0 - p) * _pauli("I")
    K1 = np.sqrt(p) * _pauli("Z")
    return [K0, K1]

def depolarizing_1q_kraus(p: float) -> List[np.ndarray]:
    """Single-qubit depolarizing channel."""
    p = float(p)
    K = [np.sqrt(1.0 - p) * _pauli("I")]
    for name in ("X","Y","Z"):
        K.append(np.sqrt(p/3.0) * _pauli(name))
    return K

def depolarizing_2q_kraus(p: float) -> List[np.ndarray]:
    """Two-qubit depolarizing channel with total error probability p."""
    p = float(p)
    I = _pauli("I"); X=_pauli("X"); Y=_pauli("Y"); Z=_pauli("Z")
    paulis = [I,X,Y,Z]
    K = [np.sqrt(1.0 - p) * kron(I, I)]
    rest = []
    for a in (I,X,Y,Z):
        for b in (I,X,Y,Z):
            if a is I and b is I:
                continue
            rest.append(kron(a,b))
    for op in rest:
        K.append(np.sqrt(p/15.0) * op)
    return K

def leakage_injection_kraus(p: float) -> List[np.ndarray]:
    """
    Single-qutrit leakage injection: with prob p, map |0>,|1>,|2> -> |2|;
    otherwise identity. Kraus set (trace-preserving).
      K0 = sqrt(1-p) * I_3
      K1 = sqrt(p) |2><0|
      K2 = sqrt(p) |2><1|
      K3 = sqrt(p) |2><2|
    """
    p = float(p)
    I3 = np.eye(3, dtype=complex)
    K0 = np.sqrt(1.0 - p) * I3
    K1 = np.zeros((3,3), complex); K1[2,0] = np.sqrt(p)
    K2 = np.zeros((3,3), complex); K2[2,1] = np.sqrt(p)
    K3 = np.zeros((3,3), complex); K3[2,2] = np.sqrt(p)
    return [K0, K1, K2, K3]

def lift_qubit_to_qutrit(Ks_2x2: List[np.ndarray]) -> List[np.ndarray]:
    """Embed 2x2 Kraus ops into 3x3 by acting on {|0>,|1>} and keeping |2> fixed."""
    out = []
    for K in Ks_2x2:
        K3 = np.zeros((3,3), complex)
        K3[:2,:2] = K
        K3[2,2] = 1.0
        out.append(K3)
    return out

def cz_induced_leakage_kraus(p_leak: float) -> List[np.ndarray]:
    """Two-ququart CZ-induced leakage channel.

    Models the dephasing-driven processes ``|11> -> |02>`` and ``|11> -> |20>``
    each occurring with probability ``p_leak/2``.  All other basis states,
    including those involving the fourth level used by leakage transport, are
    left unaffected.
    """

    p = float(p_leak)
    dim = 16
    I16 = np.eye(dim, dtype=complex)

    idx = lambda i, j: 4 * i + j
    K0 = I16.copy()
    idx11 = idx(1, 1)
    K0[idx11, idx11] = np.sqrt(max(0.0, 1.0 - p))

    K1 = np.zeros((dim, dim), complex)
    K2 = np.zeros((dim, dim), complex)
    K1[idx(0, 2), idx11] = np.sqrt(p / 2.0)  # |02><11|
    K2[idx(2, 0), idx11] = np.sqrt(p / 2.0)  # |20><11|

    return [K0, K1, K2]

def leakage_transport_kraus(p_move: float) -> List[np.ndarray]:
    """Four-level leakage transport channel.

    The paper describes a stochastic process where population in ``|12>`` (or
    ``|21>``) is transferred to the opposite qubit while the partner is reset to
    ``|0>``.  This is modelled explicitly in a two-ququart (four-level)
    Hilbert space with the transitions

    - ``|12> → |30>`` with probability ``p_move``
    - ``|21> → |03>`` with probability ``p_move``

    All other basis states are left unchanged.  The channel is trace preserving
    and reduces to identity when ``p_move = 0``.
    """

    p = float(p_move)
    dim = 16
    I16 = np.eye(dim, dtype=complex)

    # Identity branch with reduced amplitude on transported states.
    K0 = I16.copy()
    idx = lambda i, j: 4 * i + j
    idx12 = idx(1, 2)
    idx21 = idx(2, 1)
    K0[idx12, idx12] = np.sqrt(max(0.0, 1.0 - p))
    K0[idx21, idx21] = np.sqrt(max(0.0, 1.0 - p))

    # Transport branches |30><12| and |03><21|
    K1 = np.zeros((dim, dim), complex)
    K2 = np.zeros((dim, dim), complex)
    K1[idx(3, 0), idx12] = np.sqrt(p)  # |30><12|
    K2[idx(0, 3), idx21] = np.sqrt(p)  # |03><21|

    return [K0, K1, K2]


def dqlr_kraus(p_matrix: List[List[float]]) -> List[np.ndarray]:
    """Kraus operators for an imperfect DQLR reset channel.

    ``p_matrix`` is a 3x3 matrix where element ``p_matrix[i][j]`` gives the
    probability of resetting from state ``|j>`` to ``|i>``.  The resulting Kraus
    operators satisfy ``sum_i K_i† K_i = I`` provided each column of
    ``p_matrix`` sums to 1.
    """

    P = np.array(p_matrix, dtype=float)
    Ks: List[np.ndarray] = []
    for i in range(3):
        for j in range(3):
            amp = np.sqrt(max(P[i, j], 0.0))
            if amp == 0:
                continue
            K = np.zeros((3, 3), complex)
            K[i, j] = amp
            Ks.append(K)
    return Ks

def spectator_crosstalk_z_kraus(p: float) -> List[np.ndarray]:
    """
    Single-qubit Z-phase kick used to model spectator crosstalk during parallel CZs.
    """
    return dephasing_kraus(p)

def multi_level_reset_kraus(f_reset: float, rel_leak_after: float) -> List[np.ndarray]:
    """
    DQLR: reset {|0>,|1>,|2>} -> |0> with fidelity f_reset, leaving a small
    residual leakage probability rel_leak_after on |2>.
    """
    f = float(f_reset)
    r = float(rel_leak_after)
    # Kraus mapping everything to |0> with prob f, and to |2> with small r,
    # otherwise identity remainder.
    K0 = np.zeros((3,3), complex); K0[0,:] = np.sqrt(f)  # reset-to-0 branch
    K1 = np.zeros((3,3), complex); K1[2,:] = np.sqrt(r)  # residual leakage branch
    K2 = np.sqrt(max(0.0, 1.0 - f - r)) * np.eye(3, dtype=complex)
    return [K0, K1, K2]

