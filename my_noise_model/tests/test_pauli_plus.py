import numpy as np
from my_noise_model.pauli_plus import build_pauli_plus_channels


def test_cz_leakage_included():
    params = {
        "oneq_excess": 0.0,
        "cz_excess": 0.0,
        "idle_during_meas": 0.0,
        "cz_leak": 0.02,
        "leak_transport": 0.05,
        "crosstalk_z": 0.0,
    }
    ch = build_pauli_plus_channels(params)["2q_cz"]
    probs = ch["probs"]
    assert np.isclose(probs.sum(), 1.0)
    # With no depolarizing noise the identity should dominate.
    assert probs[0] > 0.99
    # Leakage transport should not change overall leakage probability
    assert abs(ch["leak"] - params["cz_leak"]) < 1e-12
