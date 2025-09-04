# -*- coding: utf-8 -*-
"""
Integration tests for the circuit_builder module using the paper-aligned
physical noise model.
"""

import pytest
import stim
from .. import circuit_builder


def test_circuit_builder_builds_a_circuit():
    """
    Tests that the builder creates a stim.Circuit object without errors.
    """
    builder = circuit_builder.SurfaceCodeCircuitBuilder(
        distance=3, rounds=2, processor="72_qubit_paper_aligned"
    )
    circuit = builder.build_circuit()

    assert circuit is not None
    assert isinstance(circuit, stim.Circuit)


def test_circuit_contains_expected_instructions():
    """
    Tests that the generated circuit contains the expected types of instructions
    for the paper-aligned physical noise model.
    """
    builder = circuit_builder.SurfaceCodeCircuitBuilder(
        distance=3, rounds=2, processor="72_qubit_paper_aligned"
    )
    circuit = builder.build_circuit()

    circuit_str = str(circuit)

    # Check for core surface code instructions
    assert "H" in circuit_str
    assert "CX" in circuit_str
    assert "M" in circuit_str
    assert "R" in circuit_str
    assert "DETECTOR" in circuit_str
    assert "OBSERVABLE_INCLUDE" in circuit_str

    # Check for our injected noise channels
    assert "PAULI_CHANNEL_1" in circuit_str
    assert "PAULI_CHANNEL_2" in circuit_str
    assert "X_ERROR" in circuit_str


def test_circuit_stats_are_reasonable():
    """
    Checks the stats of a small generated circuit to ensure the number of
    noise operations is reasonable with the paper-aligned physical noise model.
    """
    distance = 3
    rounds = 2
    builder = circuit_builder.SurfaceCodeCircuitBuilder(
        distance=distance, rounds=rounds, processor="72_qubit_paper_aligned"
    )
    circuit = builder.build_circuit()

    ideal_circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=rounds,
        distance=distance,
    )

    def count_ops(c, name):
        return sum(1 for op in c if op.name == name)

    # Check reset and measurement noise
    assert count_ops(circuit, "X_ERROR") == (
        count_ops(ideal_circuit, "R") + count_ops(ideal_circuit, "M")
    )

    # Check two-qubit gate noise
    num_two_qubit_gates = count_ops(ideal_circuit, "CX") + count_ops(ideal_circuit, "CZ")
    # Each two-qubit gate is followed by a single PAULI_CHANNEL_2 noise operation.
    assert count_ops(circuit, "PAULI_CHANNEL_2") == num_two_qubit_gates

    # Check single-qubit gate noise
    num_single_qubit_gates = sum(
        1 for op in ideal_circuit if op.name in ["H", "S", "S_DAG"]
    )
    assert count_ops(circuit, "PAULI_CHANNEL_1") == num_single_qubit_gates
