"""Variational Quantum Classifier (VQC) with angle embedding and balanced loss."""

from __future__ import annotations

import math
from typing import Any
import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

try:
    import pennylane as qml
    from pennylane import numpy as pnp
except ImportError:
    qml = None
    pnp = None

from models.quantum.ansatz import create_ansatz, get_ansatz_weight_shape
from models.quantum.embeddings import create_angle_embedding


class VariationalQuantumClassifier(BaseEstimator, ClassifierMixin):
    """Variational Quantum Classifier for binary slice-level tumor detection.

    Implements a Parameterized Quantum Circuit (PQC) using PennyLane on a
    simulated quantum device ('default.qubit').

    Features are encoded using Angle Embedding into the rotation angles of single-qubit
    gates. A layered ansatz of single-qubit rotations and entangling CNOT gates
    produces expressive quantum state representations. The decision boundary is
    derived from the expectation value of Pauli-Z on the readout qubit:
        expval = <psi(x, theta)| Z_0 |psi(x, theta)> in [-1, 1]
        P(y = 1 | x) = sigmoid(expval + bias)

    Handles class imbalance using weighted binary cross-entropy loss.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        *,
        learning_rate: float = 0.05,
        max_epochs: int = 20,
        batch_size: int = 32,
        class_weight: str | None = "balanced",
        random_state: int = 42,
    ) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.class_weight = class_weight
        self.random_state = random_state

        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0
        self.scale_: float = 1.0
        self.classes_: np.ndarray = np.array([0, 1])
        self.training_loss_history_: list[float] = []
        self._qnode = None

    def _init_circuit(self) -> None:
        if qml is None:
            raise ImportError("pennylane is required to run VariationalQuantumClassifier.")

        dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(dev, interface="autograd", diff_method="parameter-shift")
        def circuit(features: np.ndarray, weights: np.ndarray) -> Any:
            create_angle_embedding(features, wires=range(self.n_qubits), rotation="Y")
            create_ansatz(weights, wires=range(self.n_qubits), entanglement="circular")
            return qml.expval(qml.PauliZ(0))

        self._qnode = circuit

    def _sigmoid(self, z: np.ndarray | float) -> np.ndarray | float:
        z_clipped = np.clip(z, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def _forward_sample(
        self,
        features: np.ndarray,
        weights: np.ndarray,
        bias: float,
        scale: float,
    ) -> Any:
        raw_expval = self._qnode(features, weights)
        logit = scale * raw_expval + bias
        # Autograd-compatible sigmoid:
        clipped = pnp.clip(logit, -30.0, 30.0)
        return 1.0 / (1.0 + pnp.exp(-clipped))

    def fit(self, X: np.ndarray, y: np.ndarray) -> VariationalQuantumClassifier:
        """Fit the variational quantum parameters using mini-batch gradient descent."""
        if qml is None or pnp is None:
            raise ImportError("pennylane is required to train the VariationalQuantumClassifier.")

        features = np.asarray(X, dtype=np.float64)
        targets = np.asarray(y, dtype=np.int64)

        if features.ndim != 2 or features.shape[1] != self.n_qubits:
            raise ValueError(
                f"Expected features shape (N, {self.n_qubits}); received {features.shape}"
            )
        if targets.ndim != 1 or len(targets) != len(features):
            raise ValueError("Target dimension must match number of samples.")

        unique_labels = np.unique(targets)
        if not np.isin(unique_labels, (0, 1)).all():
            raise ValueError("VariationalQuantumClassifier only supports binary targets in {0, 1}.")

        self._init_circuit()

        # Class weights calculation
        if self.class_weight == "balanced":
            n_samples = len(targets)
            n_0 = np.sum(targets == 0) or 1
            n_1 = np.sum(targets == 1) or 1
            w_0 = float(n_samples / (2.0 * n_0))
            w_1 = float(n_samples / (2.0 * n_1))
        else:
            w_0, w_1 = 1.0, 1.0

        # Parameter initialization
        rng = np.random.RandomState(self.random_state)
        weight_shape = get_ansatz_weight_shape(self.n_layers, self.n_qubits)
        init_weights = rng.uniform(low=-np.pi, high=np.pi, size=weight_shape)

        weights = pnp.tensor(init_weights, requires_grad=True)
        bias = pnp.tensor(0.0, requires_grad=True)
        scale = pnp.tensor(1.0, requires_grad=True)

        opt = qml.AdamOptimizer(stepsize=self.learning_rate)
        n_samples = len(features)
        batch_size = min(self.batch_size, n_samples)
        self.training_loss_history_ = []

        for epoch in range(self.max_epochs):
            indices = rng.permutation(n_samples)
            epoch_loss = 0.0
            n_batches = 0

            for start_idx in range(0, n_samples, batch_size):
                batch_indices = indices[start_idx : start_idx + batch_size]
                x_batch = features[batch_indices]
                y_batch = targets[batch_indices]

                def cost_fn(w: Any, b: Any, s: Any) -> Any:
                    loss = 0.0
                    eps = 1e-7
                    for xi, yi in zip(x_batch, y_batch):
                        prob = self._forward_sample(xi, w, b, s)
                        if yi == 1:
                            loss = loss - w_1 * pnp.log(prob + eps)
                        else:
                            loss = loss - w_0 * pnp.log(1.0 - prob + eps)
                    return loss / len(x_batch)

                (weights, bias, scale), loss_val = opt.step_and_cost(cost_fn, weights, bias, scale)
                epoch_loss += float(loss_val)
                n_batches += 1

            avg_loss = epoch_loss / max(1, n_batches)
            self.training_loss_history_.append(avg_loss)

        self.weights_ = np.asarray(weights, dtype=np.float64)
        self.bias_ = float(bias)
        self.scale_ = float(scale)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute class probabilities for samples."""
        if self.weights_ is None:
            raise RuntimeError("Classifier must be fitted before calling predict_proba().")

        features = np.asarray(X, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != self.n_qubits:
            raise ValueError(
                f"Expected features shape (N, {self.n_qubits}); received {features.shape}"
            )

        if self._qnode is None:
            self._init_circuit()

        probs_1 = []
        for xi in features:
            raw_expval = float(self._qnode(xi, self.weights_))
            prob = float(self._sigmoid(self.scale_ * raw_expval + self.bias_))
            probs_1.append(prob)

        p1 = np.array(probs_1, dtype=np.float64)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary class labels for samples."""
        probabilities = self.predict_proba(X)
        return (probabilities[:, 1] >= 0.5).astype(np.int64)

    def save(self, filepath: str) -> None:
        """Serialize model parameters and metadata to disk."""
        data = {
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "learning_rate": self.learning_rate,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
            "weights": self.weights_,
            "bias": self.bias_,
            "scale": self.scale_,
            "training_loss_history": self.training_loss_history_,
        }
        joblib.dump(data, filepath)

    @classmethod
    def load(cls, filepath: str) -> VariationalQuantumClassifier:
        """Load model parameters and reinitialize the classifier."""
        data = joblib.load(filepath)
        clf = cls(
            n_qubits=data["n_qubits"],
            n_layers=data["n_layers"],
            learning_rate=data["learning_rate"],
            max_epochs=data["max_epochs"],
            batch_size=data["batch_size"],
            class_weight=data["class_weight"],
            random_state=data["random_state"],
        )
        clf.weights_ = data["weights"]
        clf.bias_ = data["bias"]
        clf.scale_ = data.get("scale", 1.0)   # backward-compatible default for legacy artifacts
        clf.training_loss_history_ = data["training_loss_history"]
        clf._init_circuit()
        return clf
