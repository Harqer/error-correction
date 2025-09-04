import numpy as np

from .. import channels


def _apply(Ks, idx):
    dim = Ks[0].shape[0]
    rho = np.zeros((dim, dim), complex)
    rho[idx, idx] = 1.0
    return sum(K @ rho @ K.conj().T for K in Ks)


def test_leakage_transport_trace_preserving():
    p = 0.2
    Ks = channels.leakage_transport_kraus(p)
    acc = sum(K.conj().T @ K for K in Ks)
    np.testing.assert_allclose(acc, np.eye(16), atol=1e-9)


def test_leakage_transport_moves_12_to_30():
    p = 0.3
    Ks = channels.leakage_transport_kraus(p)
    idx = lambda i, j: 4 * i + j
    idx12 = idx(1, 2)
    idx30 = idx(3, 0)
    E = _apply(Ks, idx12)
    assert np.isclose(E[idx30, idx30].real, p, atol=1e-9)
    assert np.isclose(E[idx12, idx12].real, 1 - p, atol=1e-9)


def test_leakage_transport_moves_21_to_03():
    p = 0.4
    Ks = channels.leakage_transport_kraus(p)
    idx = lambda i, j: 4 * i + j
    idx21 = idx(2, 1)
    idx03 = idx(0, 3)
    E = _apply(Ks, idx21)
    assert np.isclose(E[idx03, idx03].real, p, atol=1e-9)
    assert np.isclose(E[idx21, idx21].real, 1 - p, atol=1e-9)
