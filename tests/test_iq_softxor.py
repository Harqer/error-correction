import numpy as np
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from google_qec_paper_noise_model.iq_readout import IQReadoutModel
from google_qec_paper_noise_model.softxor import soft_xor, soft_detection_sequence

def test_softxor_identity_cases():
    p = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    q = p.copy()
    r = soft_xor(p, q)
    # XOR(p,p) == 2p - 2p^2; equals 0 when p in {0,1}
    assert np.allclose(r[[0,-1]], [0.0, 0.0])

def test_soft_detection_sequence_shape():
    seq = [
        np.array([[0.9, 0.05, 0.05], [0.8, 0.1, 0.1]]),
        np.array([[0.85, 0.1, 0.05], [0.75, 0.15, 0.1]]),
        np.array([[0.8, 0.15, 0.05], [0.7, 0.2, 0.1]]),
    ]
    de, leak = soft_detection_sequence(seq)
    assert de.shape == (2, 2)
    assert leak.shape == (3, 2)
    assert np.allclose(leak, [[0.05, 0.1], [0.05, 0.1], [0.05, 0.1]])

def test_iq_posteriors_sum_to_one():
    model = IQReadoutModel(snr=10.0, tau=0.01, p_leak_prior=1e-3)
    rng = np.random.default_rng(1)
    _, xs = model.sample_states(1000, rng)
    post = model.posteriors(xs)
    s = post["p0"] + post["p1"] + post["pl"]
    assert np.allclose(s, 1.0, atol=1e-8)

