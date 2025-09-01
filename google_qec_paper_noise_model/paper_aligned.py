from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np

from simulator.pauli_plus_simulator import PauliPlusSimulator
from .gpt import amp_phase_kraus
from .gpta import twirl_to_pauli_channel
from .channels import passive_heating_kraus, lift_qubit_to_qutrit, dqlr_kraus
from . import kraus_utils


@dataclass
class PaperAlignedNoiseConfig:
    """Noise parameters following the paper's methodology."""

    # Timing (ns)
    cycle_ns: float = 1100.0
    # Decoherence
    T1_us: float = 68.0
    Tphi_us: float = 89.0
    p_heat: float = 0.0  # passive heating to |2>
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
    """High-level facade applying paper-aligned noise to Pauli+ simulator."""

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
        cfg = PaperAlignedNoiseConfig()
        for k, v in (raw or {}).items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

    def _build_idle_ptm(self) -> Tuple[Dict[str, float], float]:
        dt_us = self.cfg.cycle_ns / 1000.0
        K_amp_phase = amp_phase_kraus(dt_us, self.cfg.T1_us, self.cfg.Tphi_us)
        K = lift_qubit_to_qutrit(K_amp_phase)
        if self.cfg.p_heat > 0:
            heat = passive_heating_kraus(self.cfg.p_heat)
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
        """Sample using the underlying simulator."""
        sampler = self.sim.circuit.compile_detector_sampler()
        return sampler.sample(num_samples, separate_observables=True)
