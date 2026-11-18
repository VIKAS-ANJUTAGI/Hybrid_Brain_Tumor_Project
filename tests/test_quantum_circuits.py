"""Unit tests for quantum angle embedding and ansatz circuits."""

from __future__ import annotations

import unittest
import numpy as np
import pennylane as qml

from models.quantum.ansatz import create_ansatz, get_ansatz_weight_shape
from models.quantum.embeddings import create_angle_embedding


class QuantumCircuitTests(unittest.TestCase):
    def test_angle_embedding_validates_dimensions(self) -> None:
        with self.assertRaises(ValueError):
            create_angle_embedding([0.1, 0.2], wires=[0, 1, 2])

    def test_ansatz_weight_shape(self) -> None:
        self.assertEqual(get_ansatz_weight_shape(2, 4), (2, 4, 2))
        self.assertEqual(get_ansatz_weight_shape(3, 8), (3, 8, 2))
        self.assertEqual(get_ansatz_weight_shape(1, 16), (1, 16, 2))

    def test_circuit_execution_all_qubit_configurations(self) -> None:
        for n_qubits in (2, 4, 8, 16):
            with self.subTest(n_qubits=n_qubits):
                dev = qml.device("default.qubit", wires=n_qubits)

                @qml.qnode(dev)
                def full_circuit(features, weights):
                    create_angle_embedding(features, wires=range(n_qubits), rotation="Y")
                    create_ansatz(weights, wires=range(n_qubits), entanglement="circular")
                    return qml.expval(qml.PauliZ(0))

                sample_features = np.zeros(n_qubits)
                sample_weights = np.zeros((1, n_qubits, 2))

                result = full_circuit(sample_features, sample_weights)
                # For zero rotation angles, state is |00...0>, expval(PauliZ(0)) must be exactly 1.0
                self.assertAlmostEqual(float(result), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
