"""paper_aligned.py —— 论文对齐的 Pauli+ 噪声高层封装。

该模块将 `my_noise_model` 中的底层 Kraus/GPT 工具整合，构建与论文参数一致的
噪声配置对象，并驱动 :class:`~simulator.pauli_plus_simulator.PauliPlusSimulator`
完成电路“穿衣”。"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np

from simulator.pauli_plus_simulator import PauliPlusSimulator
from .gpt import amp_phase_kraus
from .gpta import twirl_to_pauli_channel
from .channels import lift_qubit_to_qutrit, dqlr_kraus
from . import kraus_utils


@dataclass
class PaperAlignedNoiseConfig:
    """存储与论文一致的噪声参数。"""

    # Timing (ns)
    cycle_ns: float = 1100.0
    # Decoherence
    T1_us: float = 68.0
    Tphi_us: float = 89.0
    p_heat_01: float = 0.0  # |0> -> |1> heating
    p_heat_12: float = 0.0  # |1> -> |2> heating
    # Readout / reset (classical bit-flip rates)
    p_readout: float = 0.003
    p_reset: float = 0.003
    # DQLR imperfection matrix P_{j->i} on |0>,|1>,|2>
    dqlr_matrix: Tuple[Tuple[float, ...], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.05, 0.90, 0.05),
    )
    # CZ related mechanisms
    p_cz_leak_11_to_02: float = 0.001
    p_cz_crosstalk_ZZ: float = 0.0005
    p_cz_swap_like: float = 0.0005
    p_leak_transport_12_to_30: float = 0.0005
    # Residual Pauli noise
    p_1q_excess: float = 0.0005
    p_cz_excess: float = 0.0010
    p_idle_excess: float = 0.0005


class PaperAlignedNoiseModel:
    """高层封装：将论文噪声注入 PauliPlusSimulator。"""

    def __init__(self, config: Dict, basis: str = "z"):
        self.cfg = self._load_cfg(config)
        self.basis = basis.lower()
        if self.basis not in ("x", "z"):
            raise ValueError("basis must be 'x' or 'z'")

        # Build simulator and instrument with paper noise
        self.sim = PauliPlusSimulator(config, basis)
        self.sim.apply_paper_aligned_noise(config=self.cfg.__dict__)

        # Precompute GPT-twirled single-qubit channels
        self._ptm_1q_idle, self._p_idle_leak = self._build_idle_ptm()
        self._ptm_dqlr, self._p_dqlr_leak = self._build_dqlr_ptm()
        self._ptm_1q_excess = {
            "I": 1 - self.cfg.p_1q_excess,
            "X": self.cfg.p_1q_excess / 3,
            "Y": self.cfg.p_1q_excess / 3,
            "Z": self.cfg.p_1q_excess / 3,
        }

    def _load_cfg(self, raw: Dict) -> PaperAlignedNoiseConfig:
        """将字典配置映射到 :class:`PaperAlignedNoiseConfig` 对象。"""
        cfg = PaperAlignedNoiseConfig()
        for k, v in (raw or {}).items():
            if k == "p_heat":
                cfg.p_heat_12 = float(v)
                continue
            # Allow configs to explicitly set ``dqlr_matrix: null`` to use the
            # dataclass default instead of passing ``None`` into downstream
            # Kraus builders that expect a 3x3 matrix.
            if k == "dqlr_matrix" and v is None:
                continue
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

    def _build_idle_ptm(self) -> Tuple[Dict[str, float], float]:
        """构造空闲段的 GPT 后 Pauli 传输矩阵与泄漏概率。"""
        dt_us = self.cfg.cycle_ns / 1000.0
        K_amp_phase = amp_phase_kraus(dt_us, self.cfg.T1_us, self.cfg.Tphi_us)
        K = lift_qubit_to_qutrit(K_amp_phase)
        if self.cfg.p_heat_01 > 0 or self.cfg.p_heat_12 > 0:
            heat = kraus_utils.kraus_leakage_heating(self.cfg.p_heat_01, self.cfg.p_heat_12)
            K = kraus_utils.combine_kraus_channels(K, heat)
        probs, leak = twirl_to_pauli_channel(K, 1)
        ptm = {
            "I": float(probs[0]),
            "X": float(probs[1]),
            "Y": float(probs[2]),
            "Z": float(probs[3]),
        }
        return ptm, float(leak)

    def _build_dqlr_ptm(self) -> Tuple[Dict[str, float], float]:
        """基于 DQLR Kraus 集计算等效 Pauli 概率与泄漏。"""
        Ks = dqlr_kraus(self.cfg.dqlr_matrix)
        probs, leak = twirl_to_pauli_channel(Ks, 1)
        ptm = {
            "I": float(probs[0]),
            "X": float(probs[1]),
            "Y": float(probs[2]),
            "Z": float(probs[3]),
        }
        return ptm, float(leak)

    def sample(self, num_samples: int):
        """调用底层模拟器采样 ``num_samples`` 次检测事件。"""
        sampler = self.sim.circuit.compile_detector_sampler()
        return sampler.sample(num_samples, separate_observables=True)
