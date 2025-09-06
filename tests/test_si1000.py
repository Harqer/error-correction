import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from my_noise_model.si1000 import make_si1000_weights

def test_si1000_weights_exact():
    p = 1e-3
    w = make_si1000_weights(p)
    assert abs(w["meas_bitflip"] - 5.0*p) < 1e-15
    assert abs(w["reset_bitflip"] - 2.0*p) < 1e-15
    assert abs(w["resonator_idle"] - 2.0*p) < 1e-15
    assert abs(w["twoq_depol"] - 1.0*p) < 1e-15
    assert abs(w["oneq_depol"] - 0.1*p) < 1e-15
    assert abs(w["idle"] - 0.1*p) < 1e-15

