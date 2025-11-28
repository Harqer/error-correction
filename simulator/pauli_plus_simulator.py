"""Stim-based surface code simulator with paper-aligned noise injection.

This module constructs a rotated memory surface-code circuit using Stim and
provides a method ``apply_paper_aligned_noise`` that instruments the circuit
with the detailed error mechanisms described in Google's "quantum error
correction below the surface code threshold" paper.  Each mechanism is modelled
via Kraus operators and converted to a generalized Pauli channel using the
generalized Pauli twirling approximation (GPTA) before being appended to the
Stim circuit.

The implementation supports the parameters defined in
``configs/paper_aligned.yaml``.  Key mappings are:

* ``T1_us`` / ``Tphi_us`` – single qubit amplitude/phase damping combined with
  passive heating and twirled to a Pauli+ channel applied after 1Q gates.
* ``p_cz_crosstalk_ZZ`` – correlated ZZ after parallel CZ windows.
* ``p_cz_swap_like`` – swap‑like correlated errors modelled as (XX+YY)/2.
* ``p_cz_leak_11_to_02`` – Kraus model of dephasing‑induced leakage twirled to
  a Pauli channel.
* ``p_leak_transport_12_to_30`` – leakage transport channel applied during CZs.
* ``p_readout`` / ``p_reset`` – classical flips preceding measurement/reset.
* ``dqlr_matrix`` – imperfect DQLR reset modelled with Kraus operators.
* ``p_1q_excess``, ``p_cz_excess`` and ``p_idle_excess`` – residual Pauli noise
  around gates and idles.

The resulting circuit can be sampled using Stim's detector sampler to generate
synthetic syndromes and logical observables.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import stim

from my_noise_model.gpta import twirl_to_pauli_channel
from my_noise_model.gpt import amp_phase_kraus
from my_noise_model.channels import (
    lift_qubit_to_qutrit,
    dqlr_kraus,
    leakage_injection_kraus,     # kept for completeness
    cz_induced_leakage_kraus,    # new: two‑qutrit CZ‑leakage model
    leakage_transport_kraus,     # new: two‑qutrit leakage transport
)
from my_noise_model.kraus_utils import (
    combine_kraus_channels,
    kraus_leakage_heating,
)


ONE_Q_GATES = {
    "H",
    "X",
    "Y",
    "Z",
    "S",
    "SQRT_X",
    "SQRT_Y",
    "RX",
    "RY",
    "RZ",
    "H_XZ",
    "H_YZ",
    "T",
    "SQRT_Z",
}
TWO_Q_GATES = {"CZ", "CNOT", "CX"}


class PauliPlusSimulator:
    """Constructs a base surface-code circuit and attaches noise channels."""

    def __init__(self, config: Dict, basis: str) -> None:
        self.config = config
        self.distance = config.get("distance")
        self.rounds = config.get("rounds")
        if self.distance is None or self.rounds is None:
            raise ValueError("config must include 'distance' and 'rounds'")

        self.depolarization = config.get("depolarization", 0.001)
        self.leakage_rate = config.get("leakage_rate", 0.01)
        self.cross_talk = config.get("cross_talk", 0.002)

        self.basis = basis.upper()
        if self.basis not in ("X", "Z"):
            raise ValueError("basis must be 'X' or 'Z'")

        # Build the base circuit and attach simple two-qubit noise for backwards
        # compatibility.  Paper-aligned noise is injected later via
        # ``apply_paper_aligned_noise``.
        self.circuit = self._build_base_circuit(self.basis)
        self.circuit = self._attach_noise_to_two_qubit_gates(self.circuit)

    # ------------------------------------------------------------------
    # Circuit construction helpers
    # ------------------------------------------------------------------
    def _build_base_circuit(self, basis: str) -> stim.Circuit:
        return stim.Circuit.generated(
            f"surface_code:rotated_memory_{basis.lower()}",
            rounds=self.rounds,
            distance=self.distance,
        )

    def _attach_noise_to_two_qubit_gates(self, circuit: stim.Circuit) -> stim.Circuit:
        """Insert simple leakage and depolarization after each two-qubit gate."""
        noisy = stim.Circuit()
        for inst in circuit:
            noisy.append(inst)
            if inst.name in ("CX", "CZ"):
                targets = [t.value for t in inst.targets_copy()]
                noisy.append_operation(
                    "PAULI_CHANNEL_2",
                    targets,
                    [
                        self.leakage_rate / 3,
                        self.leakage_rate / 3,
                        self.leakage_rate / 3,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ],
                )
                noisy.append_operation("DEPOLARIZE2", targets, self.cross_talk)
        return noisy

    # ------------------------------------------------------------------
    # Paper aligned noise instrumentation
    # ------------------------------------------------------------------
    def apply_paper_aligned_noise(self, config: Dict) -> None:
        """Instrument the circuit with paper-aligned noise channels.

        Parameters are read from ``config`` and correspond directly to the
        fields in ``configs/paper_aligned.yaml``.  Each physical mechanism is
        converted to an equivalent Pauli channel before being appended to the
        Stim circuit.
        """

        cfg = config or {}

        cycle_ns = float(cfg.get("cycle_ns", 1100.0))
        dt_us = cycle_ns / 1000.0
        T1_us = float(cfg.get("T1_us", 68.0))
        Tphi_us = float(cfg.get("Tphi_us", 89.0))
        p_heat_01 = float(cfg.get("p_heat_01", 0.0))
        p_heat_12 = float(cfg.get("p_heat_12", cfg.get("p_heat", 0.0)))

        p_readout = float(cfg.get("p_readout", 3e-3))
        p_reset = float(cfg.get("p_reset", 3e-3))
        dqlr_matrix: List[List[float]] = cfg.get(
            "dqlr_matrix", ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        )

        p_1q_excess = float(cfg.get("p_1q_excess", 0.0))
        p_idle_excess = float(cfg.get("p_idle_excess", 0.0))
        p_cz_excess = float(cfg.get("p_cz_excess", 0.0))
        p_cz_zz = float(cfg.get("p_cz_crosstalk_ZZ", 0.0))
        p_cz_swap = float(cfg.get("p_cz_swap_like", 0.0))
        p_cz_leak = float(cfg.get("p_cz_leak_11_to_02", 0.0))
        p_leak_transport = float(cfg.get("p_leak_transport_12_to_30", 0.0))

        twirl_idles_each_tick = bool(cfg.get("twirl_idles_each_tick", False))
        twirl_after_1q_gates = bool(cfg.get("twirl_after_1q_gates", True))

        # Build single-qubit idle channel via Kraus ops and GPTA.
        K_amp = amp_phase_kraus(dt_us=dt_us, T1_us=T1_us, Tphi_us=Tphi_us)
        K = lift_qubit_to_qutrit(K_amp)
        if p_heat_01 > 0 or p_heat_12 > 0:
            K = combine_kraus_channels(K, kraus_leakage_heating(p_heat_01, p_heat_12))
        idle_probs, _ = twirl_to_pauli_channel(K, 1)
        px, py, pz = float(idle_probs[1]), float(idle_probs[2]), float(idle_probs[3])
        # Fold excess single-qubit errors evenly into XYZ.
        px += p_1q_excess / 3.0
        py += p_1q_excess / 3.0
        pz += p_1q_excess / 3.0

        # DQLR reset imperfections via Kraus operators.
        dqlr_probs, _ = twirl_to_pauli_channel(dqlr_kraus(dqlr_matrix), 1)
        dqlr_px, dqlr_py, dqlr_pz = (
            float(dqlr_probs[1]),
            float(dqlr_probs[2]),
            float(dqlr_probs[3]),
        )

        # ----- Exact two-qubit GPT for CZ-related channels (paper method) -----
        # Build composed two‑qutrit channel for (leakage during CZ) ∘ (leakage transport)
        cz_Ks: List = []
        if p_cz_leak > 0:
            cz_Ks = cz_induced_leakage_kraus(p_cz_leak)
        if p_leak_transport > 0:
            Ks_move = leakage_transport_kraus(p_leak_transport)
            cz_Ks = Ks_move if not cz_Ks else combine_kraus_channels(cz_Ks, Ks_move)
        # Twirl to exact 2‑qubit Pauli channel (16 probs including II)
        cz_probs16 = None
        if cz_Ks:
            cz_probs16, _ = twirl_to_pauli_channel(cz_Ks, 2)
        # Helper: map 16‑probs (II,IX,IY,IZ, XI,XX,XY,XZ, YI,YX,YY,YZ, ZI,ZX,ZY,ZZ)
        # to Stim PAULI_CHANNEL_2 args (15 probs for everything except II).
        def _probs16_to_stim_args(p16):
            order = [1,2,3, 4,5,6,7, 8,9,10,11, 12,13,14,15]  # exclude 0=II
            return [float(p16[i]) for i in order]
        # Uniform 2‑qubit depolarizing "excess" to be added on top if configured.
        def _add_cz_excess(args15: List[float]) -> List[float]:
            if p_cz_excess <= 0:
                return args15
            u = p_cz_excess / 15.0
            scale = 1.0 - p_cz_excess
            return [a * scale + u for a in args15]

        new_circuit = stim.Circuit()
        current_cz_pairs: List[Tuple[int, int]] = []
        seen_qubits: set[int] = set()

        def add_pauli_ch1(q: int, px_: float, py_: float, pz_: float) -> None:
            if px_ <= 0 and py_ <= 0 and pz_ <= 0:
                return
            new_circuit.append_operation("PAULI_CHANNEL_1", [q], [px_, py_, pz_])

        def add_pauli_ch2(a: int, b: int, probs16) -> None:
            if probs16 is None:
                return
            args15 = _probs16_to_stim_args(probs16)
            args15 = _add_cz_excess(args15)
            if all(v <= 0 for v in args15):
                return
            new_circuit.append_operation("PAULI_CHANNEL_2", [a, b], args15)

        def inject_cz_pair(a: int, b: int) -> None:
            if p_cz_zz > 0:
                new_circuit.append_operation(
                    "CORRELATED_ERROR",
                    [stim.target_z(a), stim.target_z(b)],
                    p_cz_zz,
                )
            if p_cz_swap > 0:
                new_circuit.append_operation(
                    "CORRELATED_ERROR",
                    [stim.target_x(a), stim.target_x(b)],
                    0.5 * p_cz_swap,
                )
                new_circuit.append_operation(
                    "CORRELATED_ERROR",
                    [stim.target_y(a), stim.target_y(b)],
                    0.5 * p_cz_swap,
                )
            # Inject exact 2‑qubit Pauli channel from GPT of two‑qutrit CZ noise
            add_pauli_ch2(a, b, cz_probs16)

        # Flatten the circuit to handle REPEAT blocks
        for inst in self.circuit.flattened():
            name = inst.name
            targs = inst.targets_copy()
            gargs = inst.gate_args_copy()

            if name == "M":
                if p_readout > 0:
                    for t in targs:
                        if t.is_qubit_target:
                            new_circuit.append_operation("X_ERROR", [t], [p_readout])
                new_circuit.append_operation(name, targs, gargs)
                continue

            if name in ("R", "RX", "RY", "RZ"):
                new_circuit.append_operation(name, targs, gargs)
                for t in targs:
                    if not t.is_qubit_target:
                        continue
                    if p_reset > 0:
                        new_circuit.append_operation("X_ERROR", [t], [p_reset])
                    add_pauli_ch1(t.value, dqlr_px, dqlr_py, dqlr_pz)
                continue

            if name in ONE_Q_GATES:
                new_circuit.append_operation(name, targs, gargs)
                if twirl_after_1q_gates:
                    for t in targs:
                        if t.is_qubit_target:
                            q = t.value
                            seen_qubits.add(q)
                            add_pauli_ch1(q, px, py, pz)
                continue

            if name in TWO_Q_GATES:
                qs = [t.value for t in targs if t.is_qubit_target]
                # Stim encodes multiple two-qubit gates in one instruction; process in pairs.
                for i in range(0, len(qs), 2):
                    if i + 1 < len(qs):
                        a, b = qs[i], qs[i + 1]
                        current_cz_pairs.append((a, b))
                        seen_qubits.update((a, b))
                new_circuit.append_operation(name, targs, gargs)
                continue

            if name == "TICK":
                for a, b in current_cz_pairs:
                    inject_cz_pair(a, b)
                current_cz_pairs.clear()
                if twirl_idles_each_tick and seen_qubits:
                    s = p_idle_excess / 3.0
                    for q in sorted(seen_qubits):
                        add_pauli_ch1(q, px + s, py + s, pz + s)
                new_circuit.append_operation(name, targs, gargs)
                continue

            new_circuit.append_operation(name, targs, gargs)

        for a, b in current_cz_pairs:
            inject_cz_pair(a, b)
        self.circuit = new_circuit

