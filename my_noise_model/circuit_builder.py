"""my_noise_model.circuit_builder —— Pauli+ 噪声电路生成器。

本模块忠实复现 Google 论文《Quantum error correction below the surface code
threshold》中“Pauli+”噪声模型的建模策略：

* 使用 ``leakysim`` 对含泄漏的噪声通道执行 **广义 Pauli 托恩** (Generalized
  Pauli Twirling, GPT)，从而可在 Stim 框架下模拟泄漏/回归过程；
* 在门操作持续时间内折算 T₁/T₂ 弛豫，并与额外的去极化误差合成等效
  的 Pauli 概率；
* 对并行 CZ 门注入四体相关 ZZ 噪声，以模拟实验中观测到的相位串扰；
* 将泄漏诱导的三能级动力学与复位/测量的软 I/Q 读出模型进行解耦，
  便于后续联合生成离散与连续噪声数据。

代码内提供详细的中文注释，帮助读者理解各个物理机制如何映射到模拟。"""
from pathlib import Path
from typing import Any, Dict, List, Tuple
import yaml
import stim
import numpy as np
from . import kraus_utils
from .iq_readout import IQReadoutModel
try:
    import leakysim  # Pauli+ GPT w/ leakage
except ModuleNotFoundError:  # pragma: no cover
    import leaky as leakysim

CONFIG_DIR = Path(__file__).parent


def _targets_list(inst) -> List[int]:
    return [t.value for t in inst.targets_copy()]


def _twirl_pauli_probs_1q(depol_p: float, t_gate: float, t1: float, t2: float) -> np.ndarray:
    """将单量子比特噪声折算为等效的 Pauli 概率向量。

    参数
    ----
    depol_p: float
        去极化通道的总错误概率，来自实验标定的“额外误差”项。
    t_gate: float
        门操作的持续时间（秒），用于把 T₁/T₂ 时间常数换算成一次门操作
        所累积的退相干概率。
    t1, t2: float
        分别表示振幅弛豫 (T₁) 与相干时间 (T₂)。

    处理流程
    --------
    #. 通过 :func:`~my_noise_model.kraus_utils.kraus_t1_t2_idle` 构造一个在
       ``t_gate`` 内仅包含 T₁/T₂ 机制的 Kraus 集；
    #. 构造与 ``depol_p`` 对应的去极化通道并与上一步的 Kraus 集进行合并；
    #. 使用 :mod:`my_noise_model.pauli_twirl` 中的标准 Pauli 托恩得到等效 Pauli
       概率 ``[p_I, p_X, p_Y, p_Z]``。

    对单量子比特而言，广义托恩与传统托恩在计算子空间内的结果完全一致，
    因此无需额外考虑泄漏态。"""

    ch_idle = kraus_utils.kraus_t1_t2_idle(t_gate, t1, t2)
    ch_dep = kraus_utils.kraus_depolarizing(depol_p, 1)
    ch = kraus_utils.combine_kraus_channels(ch_idle, ch_dep)
    from .pauli_twirl import twirl_to_pauli_probs
    return twirl_to_pauli_probs(ch, 1)


def _twirl_pauli_probs_2q_with_leakage(
    depol_p: float,
    t_gate: float,
    t1: float,
    t2: float,
    p_cz_leak: float,
) -> Tuple[np.ndarray, float]:
    r"""针对双量子比特通道执行广义 Pauli 托恩并返回泄漏概率。

    组合了以下物理机制：

    1. 两个量子比特分别经历的 T₁/T₂ 弛豫，通过
       :func:`~my_noise_model.kraus_utils.kraus_t1_t2_idle` 构造 Kraus 集后做克罗内克积；
    2. 在实验中拟合得到的去极化误差 ``depol_p``；
    3. CZ 门导致的 ``\|11⟩→\|02⟩`` 与 ``\|11⟩→\|20⟩`` 泄漏过程，使用
       :func:`~my_noise_model.kraus_utils.kraus_cz_leakage` 描述；
    4. 通过 ``leakysim.generalized_pauli_twirling`` 在三能级空间内执行 GPT，
       得到含泄漏分支的广义 Pauli 通道。

    返回值
    ------
    Tuple[np.ndarray, float]
        第一个元素是长度为 15 的 Pauli 概率数组（XI、YI、…、ZZ），第二个元素为
        在一次门操作中离开计算子空间的泄漏概率 ``p_leak_out``。
    """
    # 2-qubit decoherence
    dec1 = kraus_utils.kraus_t1_t2_idle(t_gate, t1, t2)
    dec2 = kraus_utils.kraus_t1_t2_idle(t_gate, t1, t2)
    deco = [np.kron(k1, k2) for k1 in dec1 for k2 in dec2]
    dep = kraus_utils.kraus_depolarizing(depol_p, 2)
    cz_leak = kraus_utils.kraus_cz_leakage(p_cz_leak)  # acts in qutrit (leakage) space

    # Combine (order is irrelevant up to twirling; we follow paper approach).
    ch = kraus_utils.combine_kraus_channels(deco, dep)
    # Embed qubit channel into two-qutrit space before adding leakage.
    ch = kraus_utils.lift_2q_kraus_to_qutrit(ch)
    ch = kraus_utils.combine_kraus_channels(ch, cz_leak)

    # Generalized Pauli twirling (qutrit levels) via leakysim.
    gpt = leakysim.generalized_pauli_twirling(ch, num_qubits=2, num_level=3)

    # leakysim represents leakage status as integer arrays; 0=COMP, 1=LEAK
    comp = leakysim.LeakageStatus(status=[0, 0])
    leak = leakysim.LeakageStatus(status=[1, 1])
    probs = []
    for pa in ['XI','YI','ZI','IX','IY','IZ','XX','XY','XZ','YX','YY','YZ','ZX','ZY','ZZ']:
        probs.append(gpt.get_prob_from_to(comp, comp, pa))

    # Leakage probability is taken directly from GPT instead of inferred.
    p_leak = gpt.get_prob_from_to(comp, leak, 'II')

    # Guard clipping
    probs = np.clip(np.array(probs, dtype=float), 0.0, None)
    s = probs.sum()
    if s > 1e-12:
        probs = probs / s * (1.0 - min(max(p_leak, 0.0), 1.0))
    p_leak = float(np.clip(p_leak, 0.0, 1.0))
    return probs, p_leak


def _pairwise(lst):
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            yield lst[i], lst[j]


class SurfaceCodeCircuitBuilder:
    """根据论文参数构造带 Pauli+ 噪声的表面码 Stim 电路。"""

    def __init__(
        self,
        distance: int,
        rounds: int,
        basis: str = 'Z',
        processor: str = "72_qubit_paper_aligned",
    ):
        self.distance = distance
        self.rounds = rounds
        self.basis = basis.lower()
        self.processor_name = processor
        self.params = self._load_noise_params()
        self._iq_model = None  # created on demand

    def _load_noise_params(self) -> dict:
        """读取 `noise_params.yaml` 中对应处理器的噪声参数字典。"""
        p = CONFIG_DIR / "noise_params.yaml"
        with open(p, "r") as f:
            all_params = yaml.safe_load(f)
        return all_params["processors"][self.processor_name]

    @property
    def iq_model(self) -> IQReadoutModel:
        """延迟构造的软 I/Q 读出模型，用于生成连续测量结果。"""
        if self._iq_model is None:
            iq = self.params.get("readout_iq", {})
            self._iq_model = IQReadoutModel(
                snr=float(iq.get("snr", 10.0)),
                tau=float(iq.get("tau", 0.01)),
                p_leak_prior=float(iq.get("leak_prior", 1e-3)),
            )
        return self._iq_model

    def _append_cross_talk_for_tick(
        self,
        noisy: stim.Circuit,
        cz_ops: List[Tuple[int, Tuple[int, int]]],
        p_cz_crosstalk: float,
    ):
        """为同一 tick 内的并行 CZ 门注入四体相关噪声。

        Stim 中的 ``TICK`` 指令代表一个并行层。根据论文 SI 的描述，同时
        执行的 CZ 门之间会产生以 ZZ 形式出现的相位串扰；在 Pauli 托恩后可
        等效为概率 ``p_cz_crosstalk`` 的 ``Z⊗Z⊗Z⊗Z`` 相关误差。这里我们遍历
        该 tick 中的所有 CZ 对，并通过 ``CORRELATED_ERROR`` 指令注入噪声。
        """
        if p_cz_crosstalk <= 0 or len(cz_ops) < 2:
            return
        for (idx_a, (a1, a2)), (idx_b, (b1, b2)) in _pairwise(cz_ops):
            # Stim: CORRELATED_ERROR p Z(a1) Z(a2) Z(b1) Z(b2)
            noisy.append_operation(
                "CORRELATED_ERROR",
                [
                    stim.target_z(a1),
                    stim.target_z(a2),
                    stim.target_z(b1),
                    stim.target_z(b2),
                ],
                p_cz_crosstalk,
            )

    def build_circuit(self) -> stim.Circuit:
        """返回带噪声的 Stim 电路。

        步骤：
        1. 利用 Stim 生成理想表面码电路；
        2. 读取 ``noise_params.yaml`` 中的实验参数，将其换算成概率；
        3. 遍历理想指令，逐条追加到 ``noisy`` 电路，并插入对应噪声操作；
        4. 在每个 ``TICK`` 结束时注入并行 CZ 的 ZZ 串扰；
        5. 返回完成穿衣的电路以用于后续采样或导出。
        """
        ideal = stim.Circuit.generated(
            f"surface_code:rotated_memory_{self.basis}",
            rounds=self.rounds,
            distance=self.distance,
        ).flattened()
        t1 = float(self.params["decoherence"]["t1_us"]) * 1e-6
        t2 = float(self.params["decoherence"]["t2_cpmg_us"]) * 1e-6
        p_reset = float(self.params["readout_reset"]["reset"])
        p_readout = float(self.params["readout_reset"]["readout"])
        p_sq = float(self.params["gate_errors"]["sq_gates"])
        p_cz_leak = float(self.params["gate_errors"]["cz_leakage_prob"])
        p_xtalk = float(self.params["gate_errors"]["cz_crosstalk"])

        # 超导体系典型门时长（约 25 ns 单比特、50 ns 双比特）
        t_1q = 25e-9
        t_2q = 50e-9

        noisy = stim.Circuit()
        current_tick_cz: List[Tuple[int, Tuple[int, int]]] = []

        for inst in ideal:
            name = inst.name
            if name == "TICK":
                # 一个并行层结束，统一对当前 tick 记录的 CZ 对注入串扰噪声。
                self._append_cross_talk_for_tick(noisy, current_tick_cz, p_xtalk)
                current_tick_cz.clear()
                noisy.append(inst)
                continue

            tgts = _targets_list(inst)

            if name not in ("M", "MR"):
                # For operations without measurement, directly append them without modification
                noisy.append(inst)

                if name in ("H", "S", "S_DAG"):
                    probs = _twirl_pauli_probs_1q(depol_p=p_sq, t_gate=t_1q, t1=t1, t2=t2)
                    noisy.append_operation("PAULI_CHANNEL_1", tgts, probs)

                elif name in ("CX", "CZ"):
                    # 对双量子比特门执行 GPT，得到 Pauli 概率与泄漏概率。
                    probs2, _p_leak = _twirl_pauli_probs_2q_with_leakage(
                        depol_p=p_xtalk, t_gate=t_2q, t1=t1, t2=t2, p_cz_leak=p_cz_leak
                    )
                    noisy.append_operation("PAULI_CHANNEL_2", tgts, probs2)
                    # 记录下该 CZ 门，留待 tick 结束时统一添加 ZZ 串扰。
                    if len(tgts) == 2:
                        current_tick_cz.append((len(noisy) - 1, (tgts[0], tgts[1])))

                elif name == "R":
                    noisy.append_operation("X_ERROR", tgts, p_reset)

            # For operations with measurement, append their corresponding noisy operations
            elif name == "M":
                # 刻意保留硬翻转误差，软 I/Q 概率由 IQReadoutModel 在采样阶段处理。
                noisy.append_operation("M", tgts, p_readout)

            else:  # name == "MR"
                noisy.append_operation("MR", tgts, p_readout)
                noisy.append_operation("X_ERROR", tgts, p_reset)  # add reset noise

        # 若电路以 CZ 结束，最后再执行一次串扰注入。
        self._append_cross_talk_for_tick(noisy, current_tick_cz, p_xtalk)
        return noisy


def build_paper_aligned_circuit(config: Dict[str, Any], basis: str) -> stim.Circuit:
    """便捷封装：按照论文参数构造带 Pauli+ 噪声的表面码电路。

    参数
    ----
    config:
        字典形式的电路配置，允许覆写码距 ``distance``、循环次数 ``rounds`` 以及
        使用的处理器名称 ``processor``（对应 ``noise_params.yaml`` 中的条目）。
    basis:
        逻辑基选择，``'x'`` 或 ``'z'``。直接传递给 Stim 的电路生成器。
    """
    distance = int(config.get("distance", 3))
    rounds = int(config.get("rounds", 3))
    processor = config.get("processor", "72_qubit_paper_aligned")
    builder = SurfaceCodeCircuitBuilder(
        distance=distance, rounds=rounds, basis=basis, processor=processor
    )
    return builder.build_circuit()


if __name__ == '__main__':
    b = SurfaceCodeCircuitBuilder(distance=3, rounds=3)
    circ = b.build_circuit()
    print(circ.stats())
    # Example: access I/Q model for soft inputs
    iq = b.iq_model
    print("IQ model:", iq)

