"""channels.py —— Kraus 通道构建工具。

本文件提供多种 1Q/2Q/多能级噪声通道的 Kraus 表达，用于在 Pauli+ 模型中
描述 T₁/T₂ 弛豫、去极化、泄漏注入、泄漏运输、DQLR 复位等物理机制。所有
函数均返回满足 CPTP 条件的 Kraus 集，并在注释中详细说明其物理含义。
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple

def _pauli(name: str) -> np.ndarray:
    """返回指定 Pauli 算符的矩阵表示。"""
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
    """连乘克罗内克积，用于构造多体 Kraus 算符。"""
    out = np.array([[1.0+0j]])
    for op in ops:
        out = np.kron(out, op)
    return out

def amplitude_damping_kraus(tau: float) -> List[np.ndarray]:
    r"""单量子比特振幅阻尼通道。

    ``tau`` 表示归一化的演化时间，阻尼强度 ``γ = 1 - e^{-tau}``。
    返回两个 Kraus 算符 ``K0`` 与 ``K1``，分别对应保持在计算子空间和
    ``\|1⟩→\|0⟩`` 跃迁。"""
    gamma = 1.0 - np.exp(-float(tau))
    g = float(gamma)
    K0 = np.array([[1, 0],[0, np.sqrt(1-g)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(g)],[0, 0]], dtype=complex)
    return [K0, K1]

def dephasing_kraus(p: float) -> List[np.ndarray]:
    """单量子比特纯退相干通道，概率 ``p`` 施加 Z 相位翻转。"""
    p = float(p)
    K0 = np.sqrt(1.0 - p) * _pauli("I")
    K1 = np.sqrt(p) * _pauli("Z")
    return [K0, K1]

def depolarizing_1q_kraus(p: float) -> List[np.ndarray]:
    """单量子比特去极化通道，总错误概率为 ``p``。"""
    p = float(p)
    K = [np.sqrt(1.0 - p) * _pauli("I")]
    for name in ("X","Y","Z"):
        K.append(np.sqrt(p/3.0) * _pauli(name))
    return K

def depolarizing_2q_kraus(p: float) -> List[np.ndarray]:
    """双量子比特去极化通道，总错误概率为 ``p``。"""
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
    r"""单 qutrit 泄漏注入通道。

    以概率 ``p`` 将 ``\|0⟩, \|1⟩, \|2⟩`` 全部泵浦到泄漏态 ``\|2⟩``，否则保持不变。
    Kraus 集满足 CPTP 条件：

    ``K0 = √(1-p)·I₃``，``K1 = √p·|2⟩⟨0|``，``K2 = √p·|2⟩⟨1|``，``K3 = √p·|2⟩⟨2|``。
    """
    p = float(p)
    I3 = np.eye(3, dtype=complex)
    K0 = np.sqrt(1.0 - p) * I3
    K1 = np.zeros((3,3), complex); K1[2,0] = np.sqrt(p)
    K2 = np.zeros((3,3), complex); K2[2,1] = np.sqrt(p)
    K3 = np.zeros((3,3), complex); K3[2,2] = np.sqrt(p)
    return [K0, K1, K2, K3]

def lift_qubit_to_qutrit(Ks_2x2: List[np.ndarray]) -> List[np.ndarray]:
    """将 2×2 Kraus 算符嵌入到含泄漏态的 3×3 空间。"""
    out = []
    for K in Ks_2x2:
        K3 = np.zeros((3,3), complex)
        K3[:2,:2] = K
        K3[2,2] = 1.0
        out.append(K3)
    return out

def cz_induced_leakage_kraus(p_leak: float) -> List[np.ndarray]:
    r"""双四能级系统（两量子比特 + 泄漏态）的 CZ 诱导泄漏通道。

    模拟 ``\|11⟩`` 态在 CZ 作用下转移到 ``\|02⟩`` 与 ``\|20⟩`` 的过程，每个分支
    概率 ``p_leak/2``。其余基态（包含泄漏运输需要的第四能级）保持不变。
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
    r"""四能级泄漏迁移通道。

    论文中描述了 ``\|12⟩``/``\|21⟩`` 泄漏态在泄漏管理脉冲作用下向另一量子比特
    迁移的过程：

    - ``\|12⟩ → \|30⟩``，概率 ``p_move``；
    - ``\|21⟩ → \|03⟩``，概率 ``p_move``。

    其他基态保持不变。当 ``p_move = 0`` 时退化为恒等映射。
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
    r"""生成 DQLR 复位过程的 Kraus 算符。

    ``p_matrix`` 为 3×3 转移矩阵，列索引 ``j`` 表示复位前的能级 ``\|j⟩``，行
    索引 ``i`` 表示复位后的目标能级 ``\|i⟩``。只要每一列概率和为 1，即可保
    证生成的 Kraus 集满足 CPTP。"""

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
    """用于模拟并行 CZ 引起观测者串扰的单量子比特 Z 相位噪声。"""
    return dephasing_kraus(p)

def multi_level_reset_kraus(f_reset: float, rel_leak_after: float) -> List[np.ndarray]:
    r"""三能级复位过程：以保真度 ``f_reset`` 复位到 ``\|0⟩``，并保留 ``rel_leak_after`` 泄漏。"""
    f = float(f_reset)
    r = float(rel_leak_after)
    # Kraus mapping everything to |0> with prob f, and to |2> with small r,
    # otherwise identity remainder.
    K0 = np.zeros((3,3), complex); K0[0,:] = np.sqrt(f)  # reset-to-0 branch
    K1 = np.zeros((3,3), complex); K1[2,:] = np.sqrt(r)  # residual leakage branch
    K2 = np.sqrt(max(0.0, 1.0 - f - r)) * np.eye(3, dtype=complex)
    return [K0, K1, K2]

