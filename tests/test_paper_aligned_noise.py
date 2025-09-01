"""Unit tests for the paper-aligned noise instrumentation.

The goal of these tests is not to validate the physical accuracy of the noise
model but to ensure that configuration parameters are plumbed through to the
Stim circuit.  When a non-zero probability is specified for a mechanism the
resulting circuit should contain the corresponding Stim operations.
"""

import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import simulator.pauli_plus_simulator as ps


def _build_sim():
    cfg = {"distance": 3, "rounds": 2}
    return ps.PauliPlusSimulator(cfg, "Z")


def test_readout_and_reset_noise_injected():
    sim = _build_sim()
    noise = {
        "p_readout": 0.1,
        "p_reset": 0.1,
    }
    sim.apply_paper_aligned_noise(noise)
    text = str(sim.circuit)
    # readout and reset errors are modeled as X_ERROR preceding M/R
    assert "X_ERROR" in text


def test_correlated_cz_and_leakage_channels():
    sim = _build_sim()
    noise = {
        "p_cz_crosstalk_ZZ": 0.2,
        "p_cz_swap_like": 0.3,
        "p_cz_leak_11_to_02": 0.4,
        "p_leak_transport_12_to_30": 0.5,
        "p_cz_excess": 0.6,
    }
    sim.apply_paper_aligned_noise(noise)
    text = str(sim.circuit)
    # correlated mechanisms use CORRELATED_ERROR ("E") and PAULI_CHANNEL_1
    assert "E(" in text
    assert text.count("PAULI_CHANNEL_1") >= 1

