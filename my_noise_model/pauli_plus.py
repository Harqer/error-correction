"""pauli_plus.py —— 生成 Pauli+ 噪声表的核心函数。

本模块根据 YAML/字典配置构造 GPTA 托恩后的 Pauli 概率与泄漏率，涵盖单比特、
CZ、测量空闲、串扰及 DQLR 等物理机制。"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import numpy as np

from .channels import (
    depolarizing_1q_kraus, depolarizing_2q_kraus, amplitude_damping_kraus,
    dephasing_kraus, leakage_injection_kraus, lift_qubit_to_qutrit,
    cz_induced_leakage_kraus, spectator_crosstalk_z_kraus,
    leakage_transport_kraus,
    multi_level_reset_kraus,
)
from .kraus_utils import combine_kraus_channels, lift_2q_kraus_to_qutrit
from .gpta import twirl_to_pauli_channel

@dataclass
class PauliPlusParams:
    """汇总 Pauli+ 噪声中用到的关键概率参数。"""

    oneq_excess: float
    cz_excess: float
    idle_during_meas: float
    cz_leak: float
    leak_transport: float
    crosstalk_z: float
    dqlr_enabled: bool
    dqlr_f_reset: float
    dqlr_rel_leak_after: float

def build_pauli_plus_channels(params: Dict[str, Any]) -> Dict[str, Any]:
    """根据输入参数构造 Pauli+ 噪声通道表。"""
    p = PauliPlusParams(
        oneq_excess = float(params["oneq_excess"]),
        cz_excess = float(params["cz_excess"]),
        idle_during_meas = float(params["idle_during_meas"]),
        cz_leak = float(params["cz_leak"]),
        leak_transport = float(params["leak_transport"]),
        crosstalk_z = float(params["crosstalk_z"]),
        dqlr_enabled = bool(params.get("dqlr", {}).get("enabled", False)) if "dqlr" in params else False,
        dqlr_f_reset = float(params.get("dqlr", {}).get("f_reset", 0.99)) if "dqlr" in params else 0.99,
        dqlr_rel_leak_after = float(params.get("dqlr", {}).get("rel_leak_after", 1e-4)) if "dqlr" in params else 1e-4,
    )
    out: Dict[str, Any] = {}
    # 1q gate 'excess' noise
    probs_1q, leak_1q = twirl_to_pauli_channel(
        lift_qubit_to_qutrit(depolarizing_1q_kraus(p.oneq_excess)), 1
    )
    out["1q"] = {"probs": probs_1q, "leak": leak_1q}
    # 2q CZ 'excess' + induced leakage and subsequent leakage transport.
    # Depolarizing part acts on qubits, so embed into two-qutrit space before
    # composing with the explicit CZ-leakage and transport channels from the
    # paper.
    K_dep = depolarizing_2q_kraus(p.cz_excess)
    K_dep = lift_2q_kraus_to_qutrit(K_dep, levels=4)
    K_cz = cz_induced_leakage_kraus(p.cz_leak)
    K_trans = leakage_transport_kraus(p.leak_transport)
    K_2q = combine_kraus_channels(K_dep, K_cz)
    K_2q = combine_kraus_channels(K_2q, K_trans)
    # Generalized Pauli twirling returns qubit Pauli probabilities and
    # the average leakage fraction produced by the full channel.
    probs_2q, leak_avg = twirl_to_pauli_channel(K_2q, 2)
    # twirl_to_pauli_channel reports leakage averaged over all computational
    # basis states.  Only |11> can leak in the CZ channel, so rescale to
    # recover the actual per-gate leakage probability specified by cz_leak.
    leak_2q = min(1.0, 4.0 * leak_avg)
    out["2q_cz"] = {"probs": probs_2q, "leak": leak_2q}
    # Idle during measurement/reset on data qubits
    probs_idle, leak_idle = twirl_to_pauli_channel(
        lift_qubit_to_qutrit(dephasing_kraus(p.idle_during_meas)), 1
    )
    out["idle_meas"] = {"probs": probs_idle, "leak": leak_idle}
    # Measurement-prep noise: small dephasing + possible pre-meas leakage injection
    probs_meas, leak_meas = twirl_to_pauli_channel(
        lift_qubit_to_qutrit(dephasing_kraus(0.0)), 1
    )
    out["meas_prep"] = {"probs": probs_meas, "leak": leak_meas}
    # Crosstalk during parallel CZ (modeled as extra spectator Z)
    probs_xt, leak_xt = twirl_to_pauli_channel(
        lift_qubit_to_qutrit(spectator_crosstalk_z_kraus(p.crosstalk_z)), 1
    )
    out["crosstalk_z"] = {"probs": probs_xt, "leak": leak_xt}
    # DQLR
    if p.dqlr_enabled:
        out["dqlr"] = {"kraus": [multi_level_reset_kraus(p.dqlr_f_reset, p.dqlr_rel_leak_after)]}
    else:
        out["dqlr"] = {"kraus": []}
    return out

