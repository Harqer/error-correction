"""gpt.py —— 广义 Pauli 托恩 (GPT) 的核心实现。

广义 Pauli 托恩是论文中将任意噪声通道映射为 Pauli（含泄漏分支）通道的关键
步骤。通过先构造 Kraus 表示，再转换为 Choi 矩阵并在 Pauli 群上平均，可以
得到便于 Clifford 仿真的等效噪声模型。本文件详细注释各个线性代数步骤。"""

from __future__ import annotations
import math
import numpy as np
from typing import Dict, Iterable, List, Tuple
from .channels import amplitude_damping_kraus, dephasing_kraus
from .gpta import twirl_to_pauli_channel


def amp_phase_kraus(*, dt_us: float, T1_us: float, Tphi_us: float) -> List[np.ndarray]:
    """旧版接口：返回给定时间步的振幅+相位阻尼 Kraus 集。"""
    # amplitude damping strength parameter (tau)
    tau = dt_us / T1_us
    K_amp = amplitude_damping_kraus(tau)
    # pure dephasing probability mapped to dephasing channel parameter
    p_phase = 0.5 * (1 - math.exp(-dt_us / Tphi_us))
    K_phase = dephasing_kraus(p_phase)
    # Compose amplitude then phase: K = P * A
    Ks: List[np.ndarray] = []
    for A in K_amp:
        for P in K_phase:
            Ks.append(P @ A)
    return Ks


def gpt_single_qubit(Ks: List[np.ndarray]) -> Dict[str, float]:
    """旧版接口：调用 :func:`twirl_to_pauli_channel` 获取 Pauli 概率。"""
    probs, _ = twirl_to_pauli_channel(Ks, 1)
    return {"I": float(probs[0]), "X": float(probs[1]), "Y": float(probs[2]), "Z": float(probs[3])}

 
 
 

# Define Pauli matrices as numpy arrays for convenience
 
"""
Generalized Pauli Twirling (GPT) utilities.

Implements the per-channel twirling step described in the Supplementary Information
of "Quantum error correction below the surface code threshold":
> "After the circuit is dressed with all relevant error channels, we apply a
> Generalized Pauli Twirling Approximation to each noise channel. This converts
> each noise channel to a generalized Pauli channel that also includes leakage,
> thereby making it compatible with Clifford simulation methods." (SI, IV.A.1)
"""


# Single-qubit Pauli matrices in computational basis
 
 
PAULI_1Q = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}
 
 

def amp_phase_kraus(dt_us: float, T1_us: float, Tphi_us: float) -> List[np.ndarray]:
    """在时间步 ``dt_us`` 内组合振幅阻尼 (T₁) 与纯退相干 (Tφ) 的 Kraus 集。

    参数
    ----
    dt_us: float
        演化时间，单位微秒。
    T1_us, Tphi_us: float
        分别为振幅阻尼与纯退相干的时间常数。

    先构造振幅阻尼的 Kraus 算符 ``K0,K1``，再与纯退相干的 ``D0,D1`` 做矩阵乘积，
    得到所有可能的噪声路径 ``D·K``。"""
    gamma = 1.0 - np.exp(-dt_us / max(T1_us, 1e-12))
    lam = 1.0 - np.exp(-dt_us / max(Tphi_us, 1e-12))
    # Amplitude damping Kraus operators
    K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    # Pure dephasing Kraus operators
    D0 = np.sqrt(1 - lam) * np.eye(2, dtype=complex)
    D1 = np.sqrt(lam) * np.array([[1, 0], [0, -1]], dtype=complex)
    # Compose: apply amplitude then dephasing
    return [D @ K for K in (K0, K1) for D in (D0, D1)]


def _kraus_to_choi(kraus_ops: Iterable[np.ndarray]) -> np.ndarray:
    """将 Kraus 集转换为 Choi 矩阵，便于执行托恩和提取概率。"""
    choi = np.zeros((4, 4), dtype=complex)
    for K in kraus_ops:
        choi += np.kron(K, K.conj())
    return choi


def gpt_single_qubit(kraus_ops: Iterable[np.ndarray]) -> Dict[str, float]:
    """对单量子比特通道执行 GPT，返回 Pauli 概率。"""
    choi = _kraus_to_choi(kraus_ops)
    probs: Dict[str, float] = {}
    for label, P in PAULI_1Q.items():
        v = np.kron(P, P.conj())
        probs[label] = float(np.real(np.trace(choi @ v)))
 
def _kraus_to_choi(kraus_ops: Iterable[np.ndarray]) -> np.ndarray:
    """（冗余实现）通过 ``vec`` 运算构造 Choi 矩阵，保留是为文献对照。"""
    choi = None
    for K in kraus_ops:
        v = np.kron(K, np.eye(K.shape[0]))
        vv = v.reshape(-1, 1) @ v.conj().reshape(1, -1)  # vec(K) vec(K)†
 
        choi = vv if choi is None else choi + vv
    return choi

def _twirl_choi_1q(choi: np.ndarray) -> np.ndarray:
    """在 Pauli 群上对单量子比特 Choi 矩阵做平均，得到对角化形式。"""
    twirled = np.zeros_like(choi, dtype=complex)
 
    # P(E)(ρ) = (1/4) Σ_P P† E(P ρ P†) P  -> Choi mapping by conjugation
 
    for P in PAULI_1Q.values():
        U = np.kron(P.T, P.conj())
        twirled += U @ choi @ U.conj().T
    return twirled / 4.0

def _pauli_probs_from_twirled_choi_1q(choi: np.ndarray) -> Dict[str, float]:
    """从托恩后的 Choi 矩阵中提取 Pauli 传输概率。"""
    # Pauli transfer matrix entry R_{ab} = Tr[ (Pauli_a \otimes Pauli_b^T) Choi ] / 2
    R = {}
    labels = ["I", "X", "Y", "Z"]
    for a in labels:
        for b in labels:
            A = np.kron(PAULI_1Q[a], PAULI_1Q[b].T)
            R[(a, b)] = np.real_if_close(np.trace(A @ choi) / 2.0).item()
    # After twirl, the channel is diagonal in the Pauli basis: probabilities = PTM row of I
    # p_I = (1 + R[Z,Z] + R[X,X] + R[Y,Y]) / 4  (standard relation)
 
    rxx, ryy, rzz = R[("X", "X")], R[("Y", "Y")], R[("Z", "Z")]
    pI = (1.0 + rxx + ryy + rzz) / 4.0
    pX = (1.0 + rxx - ryy - rzz) / 4.0
    pY = (1.0 - rxx + ryy - rzz) / 4.0
    pZ = (1.0 - rxx - ryy + rzz) / 4.0
 
    probs = {"I": float(np.clip(pI, 0.0, 1.0)),
             "X": float(np.clip(pX, 0.0, 1.0)),
             "Y": float(np.clip(pY, 0.0, 1.0)),
             "Z": float(np.clip(pZ, 0.0, 1.0))}
    # Renormalize to sum to 1 on the computational subspace.
 
 
    s = sum(probs.values())
    if s > 0:
        for k in probs:
            probs[k] /= s
    return probs
 
 

def gpt_single_qubit(kraus_ops: Iterable[np.ndarray]) -> Dict[str, float]:
    """执行完整的 GPT 流程：Kraus → Choi → 托恩 → Pauli 概率。"""
    choi = _kraus_to_choi(list(kraus_ops))
    twirled = _twirl_choi_1q(choi)
    return _pauli_probs_from_twirled_choi_1q(twirled)

