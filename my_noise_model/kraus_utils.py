# -*- coding: utf-8 -*-
"""
A library of common quantum channels (Kraus operators).

This module provides functions to generate Kraus operators for various
noise channels relevant to superconducting qubits. These channels can be
used to construct detailed noise models for quantum circuits.
"""

import numpy as np
from scipy.linalg import sqrtm

# Pauli matrices (2x2 for qubits)
PAULI_I = np.array([[1, 0], [0, 1]], dtype=complex)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)

# Qutrit operators (3x3 for handling leakage to |2>)
QUTRIT_I = np.eye(3, dtype=complex)
# Generalized Gell-Mann matrices can be used, but for simple transitions,
# projection operators are more direct.
P0 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=complex)
P1 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=complex)
P2 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=complex)

# Transition operators
P10 = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=complex) # |0><1|
P01 = P10.T.conj()                                             # |1><0|
P21 = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=complex) # |1><2|
P12 = P21.T.conj()                                             # |2><1|
P20 = np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=complex) # |0><2|
P02 = P20.T.conj()                                             # |2><0|

# Generalized Pauli Basis for a single qutrit
# This basis spans the space of 3x3 matrices and is used for Pauli twirling
# with leakage. The operators are chosen to be orthonormal w.r.t. Hilbert-Schmidt inner product.
# Tr(A.dag * B) = d * delta_AB
G1_I = P0 + P1  # Identity on computational subspace
G1_X = P01 + P10
G1_Y = -1j * P01 + 1j * P10
G1_Z = P0 - P1
G1_L02_X = P02 + P20 # Leakage <-> |0> (X-like)
G1_L12_X = P12 + P21 # Leakage <-> |1> (X-like)
G1_L02_Y = -1j * P02 + 1j * P20
G1_L12_Y = -1j * P12 + 1j * P21
G1_P2 = P2 # Projector onto leakage state |2>

# Normalize the basis
QUTRIT_BASIS = [
    G1_I, G1_X, G1_Y, G1_Z,
    G1_L02_X, G1_L12_X, G1_L02_Y, G1_L12_Y,
    G1_P2
]
for i in range(len(QUTRIT_BASIS)):
    op = QUTRIT_BASIS[i]
    norm = np.sqrt(np.real(np.trace(op.conj().T @ op)))
    if norm > 1e-9:
        QUTRIT_BASIS[i] = op / norm

QUTRIT_BASIS_LABELS = [
    'I_c', 'X_c', 'Y_c', 'Z_c',
    'L02_X', 'L12_X', 'L02_Y', 'L12_Y',
    'P_2'
]

# Two-qutrit basis (81 operators)
QUTRIT_2Q_BASIS = [np.kron(P1, P2) for P1 in QUTRIT_BASIS for P2 in QUTRIT_BASIS]
QUTRIT_2Q_LABELS = [L1 + L2 for L1 in QUTRIT_BASIS_LABELS for L2 in QUTRIT_BASIS_LABELS]


def kraus_amplitude_damping(time: float, t1: float) -> list[np.ndarray]:
    """
    Generates Kraus operators for an amplitude damping channel (T1 decay).

    Args:
        time: The duration of the noise channel.
        t1: The T1 relaxation time constant.

    Returns:
        A list of Kraus operators [E0, E1].
    """
    if t1 <= 0:
        return [PAULI_I]
    p = 1 - np.exp(-time / t1)
    e0 = np.array([[1, 0], [0, np.sqrt(1 - p)]], dtype=complex)
    e1 = np.array([[0, np.sqrt(p)], [0, 0]], dtype=complex)
    return [e0, e1]


def kraus_dephasing(time: float, t2: float, t1: float) -> list[np.ndarray]:
    """
    Generates Kraus operators for a pure dephasing channel (T_phi).

    The T_phi decay rate is calculated from T1 and T2 as:
    1/T_phi = 1/T2 - 1/(2*T1)

    Args:
        time: The duration of the noise channel.
        t2: The T2 coherence time.
        t1: The T1 relaxation time.

    Returns:
        A list of Kraus operators [E0, E1].
    """
    if t2 <= 0:
        return [PAULI_I]

    # Ensure T2 is not longer than the theoretical limit
    if t2 > 2 * t1:
        # In this case, dephasing is negligible or parameters are unphysical
        t_phi_inv = 0
    else:
        t_phi_inv = 1/t2 - 1/(2*t1)

    if t_phi_inv <= 0:
        return [PAULI_I]

    t_phi = 1 / t_phi_inv
    p = 1 - np.exp(-2 * time / t_phi) # Note the factor of 2 for T_phi

    e0 = np.sqrt(1 - p/2) * PAULI_I
    e1 = np.sqrt(p/2) * PAULI_Z
    return [e0, e1]


def kraus_depolarizing(prob: float, n_qubits: int = 1) -> list[np.ndarray]:
    """
    Generates Kraus operators for a depolarizing channel.

    Args:
        prob: The probability of a Pauli error occurring.
        n_qubits: The number of qubits (1 or 2).

    Returns:
        A list of Kraus operators.
    """
    if n_qubits == 1:
        paulis = [PAULI_I, PAULI_X, PAULI_Y, PAULI_Z]
        d = 4
    elif n_qubits == 2:
        paulis = [
            np.kron(P1, P2) for P1 in [PAULI_I, PAULI_X, PAULI_Y, PAULI_Z]
            for P2 in [PAULI_I, PAULI_X, PAULI_Y, PAULI_Z]
        ]
        d = 16
    else:
        raise ValueError("Only 1 or 2 qubits are supported.")

    kraus_ops = [np.sqrt(prob / (d - 1)) * P for P in paulis[1:]]
    kraus_ops.insert(0, np.sqrt(1 - prob) * paulis[0])
    return kraus_ops


# Ideal gate matrices
IDEAL_H = (1 / np.sqrt(2)) * np.array([
    [1, 1],
    [1, -1]
], dtype=complex)

IDEAL_CX = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=complex)

IDEAL_CZ = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, -1]
], dtype=complex)


def kraus_t1_t2_idle(time: float, t1: float, t2: float) -> list[np.ndarray]:
    """
    Generates Kraus operators for combined T1 and T2 thermal relaxation.
    Note: This assumes T_phi is calculated from T1 and T2.
    """
    if time == 0:
        return [PAULI_I]

    t1_channel = kraus_amplitude_damping(time, t1)
    t2_channel = kraus_dephasing(time, t2, t1)

    # Since T1 and T2 channels (in this formulation) commute,
    # we can combine them by composing their Kraus operators.
    return combine_kraus_channels(t1_channel, t2_channel)


def kraus_zz_interaction(strength: float, time: float) -> list[np.ndarray]:
    """
    Generates a Kraus operator for a ZZ stray interaction.
    This is a unitary evolution U = exp(-i * strength * Z @ Z * time).
    """
    zz = np.kron(PAULI_Z, PAULI_Z)
    # The interaction angle theta = strength * time
    # The scipy.linalg.expm function is suitable for matrix exponentials.
    from scipy.linalg import expm
    unitary = expm(-1j * strength * time * zz)
    return [unitary]


def combine_kraus_channels(
    channel1: list[np.ndarray], channel2: list[np.ndarray]
) -> list[np.ndarray]:
    """
    Combines two quantum channels by composing their Kraus operators.
    This represents applying channel2 after channel1.

    Args:
        channel1: List of Kraus operators for the first channel.
        channel2: List of Kraus operators for the second channel.

    Returns:
        A new list of Kraus operators for the combined channel.
    """
    return [np.dot(k2, k1) for k2 in channel2 for k1 in channel1]


def kraus_leakage_heating(p01: float, p12: float) -> list[np.ndarray]:
    """Full three-level heating channel.

    Implements a sequential heating process ``|0> -> |1>`` with probability
    ``p01`` and ``|1> -> |2>`` with probability ``p12`` while leaving ``|2>``
    fixed.  This follows the multi-step heating model described in the paper
    and is trace preserving for all ``0 <= p01, p12 <= 1``.

    Args:
        p01: Probability of the ``|0> -> |1>`` transition.
        p12: Probability of the ``|1> -> |2>`` transition.

    Returns:
        List of Kraus operators ``[K0, K1, K2]`` implementing the process.
    """

    p01 = float(p01)
    p12 = float(p12)

    # Identity branch with reduced amplitudes on states that can heat.
    K0 = np.eye(3, dtype=complex)
    K0[0, 0] = np.sqrt(max(0.0, 1.0 - p01))
    K0[1, 1] = np.sqrt(max(0.0, 1.0 - p12))

    # Heating branches |1><0| and |2><1|
    K1 = np.zeros((3, 3), complex)
    K2 = np.zeros((3, 3), complex)
    K1[1, 0] = np.sqrt(p01)
    K2[2, 1] = np.sqrt(p12)

    return [K0, K1, K2]


def kraus_heating_to_2(prob: float) -> list[np.ndarray]:
    """Compatibility wrapper for the legacy single-step model.

    Historically the project used a simplified heating channel that only
    implemented the ``|1> -> |2>`` transition.  This function now delegates to
    :func:`kraus_leakage_heating` with ``p01=0`` to retain backwards
    compatibility with existing code.

    Args:
        prob: Probability of the ``|1> -> |2>`` transition.

    Returns:
        List of Kraus operators implementing the heating process.
    """

    return kraus_leakage_heating(0.0, prob)


def kraus_cz_leakage(prob: float) -> list[np.ndarray]:
    """CZ-induced leakage channel.

    Models the dephasing-driven leakage processes ``|11> -> |02>`` and
    ``|11> -> |20>`` occurring during a faulty CZ gate.  Each branch occurs
    with probability ``prob/2`` and all other basis states are left unchanged,
    matching the stochastic description in the paper.

    Args:
        prob: Total probability that ``|11>`` leaks during the gate.

    Returns:
        A list of Kraus operators acting on two qutrits.
    """

    p = float(prob)
    dim = 9
    I9 = np.eye(dim, dtype=complex)

    # Identity branch with reduced amplitude on |11> to preserve trace.
    K0 = I9.copy()
    idx11 = 3 * 1 + 1
    K0[idx11, idx11] = np.sqrt(max(0.0, 1.0 - p))

    # Leakage branches |02><11| and |20><11|
    K1 = np.zeros((dim, dim), complex)
    K2 = np.zeros((dim, dim), complex)
    K1[3 * 0 + 2, idx11] = np.sqrt(p / 2.0)
    K2[3 * 2 + 0, idx11] = np.sqrt(p / 2.0)

    return [K0, K1, K2]


def lift_2q_kraus_to_qutrit(kraus_ops: list[np.ndarray], levels: int = 3) -> list[np.ndarray]:
    """Embed two-qubit Kraus operators into a two-``levels``-level space.

    The input operators act on the computational subspace spanned by
    {|00>, |01>, |10>, |11>} and are embedded into an ``(levels^2)x(levels^2)``
    matrix acting on two ``levels``-level systems (e.g. qutrits for
    ``levels=3`` or ququarts for ``levels=4``).  The remaining basis states are
    left untouched and an additional Kraus operator is appended to act as
    identity on the leaked subspace, keeping the channel trace preserving.

    Args:
        kraus_ops: List of 4x4 Kraus operators.
        levels: Number of levels per qubit (default 3).

    Returns:
        List of ``(levels^2)x(levels^2)`` Kraus operators.
    """

    dim = levels * levels
    comp_idx = [levels * i + j for i in range(2) for j in range(2)]
    leak_idx = [i for i in range(dim) if i not in comp_idx]

    lifted = []
    for k in kraus_ops:
        kN = np.zeros((dim, dim), dtype=complex)
        for a, ia in enumerate(comp_idx):
            for b, jb in enumerate(comp_idx):
                kN[ia, jb] = k[a, b]
        lifted.append(kN)

    # Identity on leaked subspace to keep channel trace preserving
    leak_eye = np.zeros((dim, dim), dtype=complex)
    for idx in leak_idx:
        leak_eye[idx, idx] = 1.0
    lifted.append(leak_eye)
    return lifted


def kraus_from_pauli_probs(probs: list[float], n_qubits: int) -> list[np.ndarray]:
    """
    Creates a Kraus representation of a Pauli channel from a list of probabilities.
    """
    if n_qubits == 1:
        basis = PAULI_1Q_BASIS
        d = 4
    elif n_qubits == 2:
        basis = PAULI_2Q_BASIS
        d = 16
    else:
        raise ValueError("Only 1 or 2 qubits are supported.")

    if len(probs) != d:
        raise ValueError(f"Length of probs should be {d}")

    kraus_ops = [np.sqrt(p) * op for p, op in zip(probs, basis) if p > 0]
    return kraus_ops


def kraus_dqlr(prob_matrix: np.ndarray) -> list[np.ndarray]:
    """
    Generates Kraus operators for the phenomenological DQLR channel.

    Args:
        prob_matrix: A 3x3 matrix where P_ij is the probability of
                     transitioning from state |j> to |i>.

    Returns:
        A list of Kraus operators.
    """
    kraus_ops = []
    k_sum = np.zeros((3, 3), dtype=complex)
    for i in range(3):
        for j in range(3):
            if prob_matrix[i, j] > 0:
                op = np.zeros((3, 3), dtype=complex)
                op[i, j] = 1.0 # |i><j|
                k = np.sqrt(prob_matrix[i, j]) * op
                kraus_ops.append(k)
                k_sum += k.conj().T @ k

    k0 = sqrtm(np.eye(3) - k_sum)
    kraus_ops.insert(0, k0)
    return kraus_ops
