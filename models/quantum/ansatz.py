"""Parameterized quantum circuits (ansatz) for Variational Quantum Classifiers."""

from __future__ import annotations

from collections.abc import Sequence
import numpy as np

try:
    import pennylane as qml
except ImportError:
    qml = None


def create_ansatz(
    weights: np.ndarray,
    wires: Sequence[int],
    *,
    entanglement: str = "circular",
) -> None:
    """Construct a parameterized variational ansatz over the given wires.

    Each layer applies single-qubit rotations Ry and Rz followed by
    entangling CNOT gates connecting adjacent qubits.

    Args:
        weights: Array of shape (n_layers, n_qubits, 2) containing variational rotation angles.
        wires: Sequence of wire indices.
        entanglement: Entanglement topology, either 'circular' or 'linear'.

    Raises:
        ImportError: If pennylane is not installed.
        ValueError: If weights shape does not match wires or layers.
    """
    if qml is None:
        raise ImportError("pennylane is required to construct quantum ansatz circuits.")

    n_wires = len(wires)
    weights_arr = np.asarray(weights)
    if weights_arr.ndim != 3 or weights_arr.shape[1] != n_wires or weights_arr.shape[2] != 2:
        raise ValueError(
            f"Expected weights of shape (n_layers, {n_wires}, 2); received shape {weights_arr.shape}"
        )

    n_layers = weights_arr.shape[0]

    for layer in range(n_layers):
        # Parametric single-qubit rotations
        for i, wire in enumerate(wires):
            qml.RY(weights_arr[layer, i, 0], wires=wire)
            qml.RZ(weights_arr[layer, i, 1], wires=wire)

        # Entanglement layer
        if n_wires > 1:
            for i in range(n_wires - 1):
                qml.CNOT(wires=[wires[i], wires[i + 1]])
            if entanglement == "circular" and n_wires > 2:
                qml.CNOT(wires=[wires[-1], wires[0]])


def get_ansatz_weight_shape(n_layers: int, n_qubits: int) -> tuple[int, int, int]:
    """Return the expected parameter weight shape for the ansatz."""
    if n_layers < 1:
        raise ValueError("n_layers must be >= 1")
    if n_qubits < 1:
        raise ValueError("n_qubits must be >= 1")
    return (n_layers, n_qubits, 2)
