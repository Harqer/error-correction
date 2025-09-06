import numpy as np
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from my_noise_model.gpta import twirl_to_pauli_channel
from my_noise_model.channels import amplitude_damping_kraus, dephasing_kraus, lift_qubit_to_qutrit

def test_twirl_1q_probs_valid():
    Ks = lift_qubit_to_qutrit(amplitude_damping_kraus(0.02) + dephasing_kraus(0.01))
    p, leak = twirl_to_pauli_channel(Ks, 1)
    assert p.shape == (4,)
    assert np.all(p >= -1e-12)
    assert abs(p.sum() - 1.0) < 1e-9
    assert 0.0 <= leak <= 1.0

def test_twirl_2q_probs_valid():
    Ks = lift_qubit_to_qutrit(amplitude_damping_kraus(0.03) + dephasing_kraus(0.02))
    p, leak = twirl_to_pauli_channel(Ks, 2)
    assert p.shape == (16,)
    assert np.all(p >= -1e-12)
    assert abs(p.sum() - 1.0) < 1e-9
    assert 0.0 <= leak <= 1.0

