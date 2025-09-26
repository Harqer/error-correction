"""iq_readout.py —— 软测量 (I/Q) 似然与后验模型。

根据论文方法，测量结果被视为一维高斯分布的 I/Q 样本：\|0⟩ 与 \|1⟩ 的均值由
信噪比 (SNR) 决定，泄漏态使用宽而居中的分布。该文件提供后验概率计算、
软检测器等功能，并附有中文注释解释物理含义。"""

import math
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np

def _gauss_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """标准高斯密度函数，返回 ``N(mu, sigma^2)`` 在 ``x`` 处的值。"""
    z = (x - mu) / sigma
    return np.exp(-0.5 * z * z) / (math.sqrt(2.0 * math.pi) * sigma)

@dataclass
class IQReadoutModel:
    """一维 I/Q 读出模型，输出 {0,1,L} 的后验概率。"""
    snr: float
    tau: float
    # Prior probability of leakage for the measurement under consideration.
    p_leak_prior: float = 1e-3
    sigma: float = 1.0
    leak_sigma_scale: float = 1.6

    def _means(self) -> Tuple[float, float]:
        """根据 SNR 与阻尼时间常数计算 \|0⟩/\|1⟩ 高斯分布的均值。"""
        mu = 0.5 * self.snr
        # amplitude damping collapses the |1> cloud toward |0|
        # We use alpha = exp(-tau) as the retained |1| amplitude component.
        alpha = math.exp(-self.tau)
        mu0 = +mu
        mu1 = -alpha * mu
        return mu0, mu1

    def posteriors(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """给定观测 ``x``（标量或数组），返回三种状态的后验概率。"""
        x = np.asarray(x, dtype=float)
        mu0, mu1 = self._means()
        s = self.sigma
        sL = self.leak_sigma_scale * s
        # Priors: split non-leakage evenly between 0/1
        piL = float(self.p_leak_prior)
        pi0 = 0.5 * (1.0 - piL)
        pi1 = 0.5 * (1.0 - piL)
        L0 = _gauss_pdf(x, mu0, s)
        L1 = _gauss_pdf(x, mu1, s)
        LL = _gauss_pdf(x, 0.0, sL)
        num0 = pi0 * L0
        num1 = pi1 * L1
        numL = piL * LL
        Z = num0 + num1 + numL + 1e-18
        return {
            "p0": num0 / Z,
            "p1": num1 / Z,
            "pl": numL / Z,
        }

    def soft_meas_probs(self, x: np.ndarray) -> np.ndarray:
        """以矩阵形式返回软测量概率 ``[p0, p1, pl]``。"""
        post = self.posteriors(x)
        return np.stack([post["p0"], post["p1"], post["pl"]], axis=-1)

    def soft_meas_prob1(self, x: np.ndarray) -> np.ndarray:
        """兼容旧接口，仅返回 ``P(m=1)``。"""
        return self.soft_meas_probs(x)[..., 1]

    def soft_vector(self, x: np.ndarray) -> np.ndarray:
        """``soft_meas_probs`` 的别名，方便旧代码调用。"""
        return self.soft_meas_probs(x)

    # Convenience generators for synthetic I/Q samples (useful for tests)
    def sample_states(self, n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        """根据先验采样状态标签与对应的高斯 I/Q 样本。"""
        piL = float(self.p_leak_prior)
        pi0 = 0.5 * (1.0 - piL)
        pi1 = 0.5 * (1.0 - piL)
        probs = np.array([pi0, pi1, piL], dtype=float)
        states = rng.choice(3, size=n, p=probs)
        mu0, mu1 = self._means()
        mus = np.array([mu0, mu1, 0.0], dtype=float)
        sigs = np.array([self.sigma, self.sigma, self.leak_sigma_scale * self.sigma], dtype=float)
        xs = rng.normal(loc=mus[states], scale=sigs[states])
        return states, xs

