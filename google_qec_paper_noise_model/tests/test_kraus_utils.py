# -*- coding: utf-8 -*-
"""
Unit tests for the kraus_utils module.
"""

import numpy as np
import pytest

from .. import kraus_utils


def check_kraus_is_trace_preserving(kraus_ops: list[np.ndarray]):
    """
    Helper function to check if a set of Kraus operators for a
    single-qubit channel is trace-preserving (sum K_dag * K = I).
    """
    d = kraus_ops[0].shape[0]
    identity = np.identity(d, dtype=complex)
    sum_k_dag_k = np.zeros((d, d), dtype=complex)
    for k in kraus_ops:
        sum_k_dag_k += k.conj().T @ k

    np.testing.assert_allclose(sum_k_dag_k, identity, atol=1e-9)


def test_amplitude_damping_is_trace_preserving():
    """
    Tests that the amplitude damping channel is trace-preserving.
    """
    kraus_ops = kraus_utils.kraus_amplitude_damping(time=50e-9, t1=20e-6)
    check_kraus_is_trace_preserving(kraus_ops)


def test_dephasing_is_trace_preserving():
    """
    Tests that the dephasing channel is trace-preserving.
    """
    kraus_ops = kraus_utils.kraus_dephasing(time=50e-9, t2=30e-6, t1=20e-6)
    check_kraus_is_trace_preserving(kraus_ops)


def test_depolarizing_1q_is_trace_preserving():
    """
    Tests that the 1Q depolarizing channel is trace-preserving.
    """
    kraus_ops = kraus_utils.kraus_depolarizing(prob=0.01, n_qubits=1)
    check_kraus_is_trace_preserving(kraus_ops)


def _apply(Ks, idx):
    d = Ks[0].shape[0]
    rho = np.zeros((d, d), complex)
    rho[idx, idx] = 1.0
    return sum(K @ rho @ K.conj().T for K in Ks)


def test_leakage_heating_three_level():
    """Multi-step heating conserves trace and applies correct transitions."""
    p01 = 0.1
    p12 = 0.2
    Ks = kraus_utils.kraus_leakage_heating(p01, p12)
    check_kraus_is_trace_preserving(Ks)

    E0 = _apply(Ks, 0)
    assert np.isclose(E0[1, 1].real, p01)
    assert np.isclose(E0[0, 0].real, 1 - p01)

    E1 = _apply(Ks, 1)
    assert np.isclose(E1[2, 2].real, p12)
    assert np.isclose(E1[1, 1].real, 1 - p12)
