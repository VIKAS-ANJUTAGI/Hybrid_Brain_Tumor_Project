"""Unit tests for VariationalQuantumClassifier training and inference."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

import pennylane as qml
from pennylane import numpy as pnp

from models.quantum.vqc_model import VariationalQuantumClassifier
from models.quantum.ansatz import create_ansatz, get_ansatz_weight_shape
from models.quantum.embeddings import create_angle_embedding


class VariationalQuantumClassifierTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.RandomState(42)
        # Small synthetic 2-qubit dataset
        self.X_train = rng.uniform(-np.pi, np.pi, size=(12, 2))
        self.y_train = (self.X_train[:, 0] > 0).astype(int)
        self.X_test = rng.uniform(-np.pi, np.pi, size=(4, 2))

    def test_vqc_fit_and_predict(self) -> None:
        vqc = VariationalQuantumClassifier(
            n_qubits=2,
            n_layers=1,
            max_epochs=2,
            batch_size=4,
            learning_rate=0.1,
            random_state=42,
        )
        vqc.fit(self.X_train, self.y_train)

        self.assertIsNotNone(vqc.weights_)
        self.assertEqual(len(vqc.training_loss_history_), 2)

        probs = vqc.predict_proba(self.X_test)
        self.assertEqual(probs.shape, (4, 2))
        # Probabilities must sum to 1.0 and be in [0, 1]
        np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(4), atol=1e-5)
        self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0))

        preds = vqc.predict(self.X_test)
        self.assertEqual(preds.shape, (4,))
        self.assertTrue(set(preds).issubset({0, 1}))

    def test_vqc_rejects_mismatched_dimensions(self) -> None:
        vqc = VariationalQuantumClassifier(n_qubits=4)
        with self.assertRaises(ValueError):
            # Provided 2 features instead of 4
            vqc.fit(self.X_train, self.y_train)

    def test_vqc_serialization_roundtrip(self) -> None:
        vqc = VariationalQuantumClassifier(
            n_qubits=2,
            n_layers=1,
            max_epochs=2,
            batch_size=4,
            random_state=42,
        )
        vqc.fit(self.X_train, self.y_train)
        probs_orig = vqc.predict_proba(self.X_test)

        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = Path(tmp_dir) / "vqc.joblib"
            vqc.save(str(artifact_path))

            loaded_vqc = VariationalQuantumClassifier.load(str(artifact_path))
            probs_loaded = loaded_vqc.predict_proba(self.X_test)

            np.testing.assert_allclose(probs_orig, probs_loaded, atol=1e-5)

    # ------------------------------------------------------------------
    # Regression tests for Defect 3: VQC classical readout scaling
    # ------------------------------------------------------------------

    def _make_vqc_and_data(self, n_qubits: int, n_samples: int = 10):
        """Helper: build a minimal VQC and synthetic dataset for n_qubits."""
        rng = np.random.RandomState(0)
        X = rng.uniform(-np.pi, np.pi, size=(n_samples, n_qubits))
        y = (X[:, 0] > 0).astype(int)
        vqc = VariationalQuantumClassifier(
            n_qubits=n_qubits,
            n_layers=1,
            max_epochs=1,
            batch_size=n_samples,
            learning_rate=0.01,
            random_state=0,
        )
        return vqc, X, y

    def test_forward_pass_output_is_finite(self) -> None:
        """Forward pass must produce finite probabilities in [0, 1]."""
        vqc, X, y = self._make_vqc_and_data(n_qubits=2)
        vqc.fit(X, y)
        probs = vqc.predict_proba(X)
        self.assertTrue(np.all(np.isfinite(probs)), "predict_proba returned non-finite values")
        self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0 + 1e-9),
                        f"Probabilities out of [0,1]: min={probs.min():.6f} max={probs.max():.6f}")

    def test_predict_proba_two_column_shape(self) -> None:
        """predict_proba must return a (N, 2) array."""
        vqc, X, y = self._make_vqc_and_data(n_qubits=2)
        vqc.fit(X, y)
        probs = vqc.predict_proba(X)
        self.assertEqual(probs.ndim, 2)
        self.assertEqual(probs.shape[1], 2)

    def test_probabilities_sum_to_one(self) -> None:
        """Each row of predict_proba must sum to 1."""
        vqc, X, y = self._make_vqc_and_data(n_qubits=2)
        vqc.fit(X, y)
        probs = vqc.predict_proba(X)
        np.testing.assert_allclose(
            probs.sum(axis=1), np.ones(len(X)), atol=1e-6,
            err_msg="predict_proba rows do not sum to 1",
        )

    def test_scale_attribute_initialized_to_one(self) -> None:
        """scale_ must be initialized to 1.0 before training."""
        vqc = VariationalQuantumClassifier(n_qubits=2)
        self.assertAlmostEqual(
            vqc.scale_, 1.0,
            msg="scale_ must be initialized to 1.0 on construction",
        )

    def test_scale_receives_nonzero_finite_gradient(self) -> None:
        """scale must receive a finite, non-zero gradient via parameter-shift.

        Regression test for Defect 3: verifies that the new scale parameter is
        connected to the computational graph and receives a valid gradient during
        a single backward pass.
        """
        n_qubits = 2
        rng = np.random.RandomState(7)
        X_batch = rng.uniform(-np.pi, np.pi, size=(4, n_qubits))
        y_batch = np.array([1, 0, 1, 0], dtype=np.int64)

        vqc = VariationalQuantumClassifier(n_qubits=n_qubits, n_layers=1, random_state=7)
        vqc._init_circuit()

        weight_shape = get_ansatz_weight_shape(1, n_qubits)
        weights = pnp.tensor(
            rng.uniform(-np.pi, np.pi, size=weight_shape), requires_grad=True
        )
        bias = pnp.tensor(0.0, requires_grad=True)
        scale = pnp.tensor(1.0, requires_grad=True)

        w_0 = w_1 = 1.0
        eps = 1e-7

        def cost_fn(w, b, s):
            loss = 0.0
            for xi, yi in zip(X_batch, y_batch):
                prob = vqc._forward_sample(xi, w, b, s)
                if yi == 1:
                    loss = loss - w_1 * pnp.log(prob + eps)
                else:
                    loss = loss - w_0 * pnp.log(1.0 - prob + eps)
            return loss / len(X_batch)

        grad_fn = qml.grad(cost_fn, argnums=[0, 1, 2])
        grad_w, grad_b, grad_s = grad_fn(weights, bias, scale)

        # scale gradient must be finite and non-zero
        self.assertTrue(
            np.isfinite(float(grad_s)),
            f"Gradient of scale is not finite: {float(grad_s)}",
        )
        self.assertNotAlmostEqual(
            float(grad_s), 0.0, places=10,
            msg="Gradient of scale is exactly zero — scale is detached from the graph",
        )

        # bias gradient must be finite
        self.assertTrue(
            np.isfinite(float(grad_b)),
            f"Gradient of bias is not finite: {float(grad_b)}",
        )

        # circuit weights gradients must all be finite
        grad_w_arr = np.asarray(grad_w)
        self.assertTrue(
            np.all(np.isfinite(grad_w_arr)),
            f"Circuit weight gradients contain non-finite values: {grad_w_arr}",
        )

    def test_serialization_preserves_scale(self) -> None:
        """save/load round-trip must preserve the trained scale_ value.

        Regression test for Defect 3: verifies that scale_ is included in
        the serialized artifact and restored correctly on load.
        """
        vqc, X, y = self._make_vqc_and_data(n_qubits=2)
        vqc.fit(X, y)
        scale_before = vqc.scale_

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "vqc_scale.joblib"
            vqc.save(str(path))
            loaded = VariationalQuantumClassifier.load(str(path))

        self.assertAlmostEqual(
            loaded.scale_, scale_before, places=10,
            msg=f"scale_ changed after save/load: {scale_before} -> {loaded.scale_}",
        )
        # Predictions must also be identical after reload
        probs_orig = vqc.predict_proba(X)
        probs_loaded = loaded.predict_proba(X)
        np.testing.assert_allclose(probs_orig, probs_loaded, atol=1e-6,
                                   err_msg="predict_proba differs after serialization roundtrip")

    def test_legacy_load_defaults_scale_to_one(self) -> None:
        """Loading an artifact without a 'scale' key must default scale_ to 1.0.

        Regression test for Defect 3: backward compatibility — artifacts saved
        before the fix must still load without error.
        """
        import joblib

        vqc, X, y = self._make_vqc_and_data(n_qubits=2)
        vqc.fit(X, y)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "vqc_legacy.joblib"
            # Simulate a legacy artifact (no 'scale' key)
            legacy_data = {
                "n_qubits": vqc.n_qubits,
                "n_layers": vqc.n_layers,
                "learning_rate": vqc.learning_rate,
                "max_epochs": vqc.max_epochs,
                "batch_size": vqc.batch_size,
                "class_weight": vqc.class_weight,
                "random_state": vqc.random_state,
                "weights": vqc.weights_,
                "bias": vqc.bias_,
                # NOTE: 'scale' key intentionally omitted
                "training_loss_history": vqc.training_loss_history_,
            }
            joblib.dump(legacy_data, str(path))

            loaded = VariationalQuantumClassifier.load(str(path))

        self.assertAlmostEqual(
            loaded.scale_, 1.0, places=10,
            msg=f"Legacy artifact load did not default scale_ to 1.0, got {loaded.scale_}",
        )

    def test_training_runs_for_multiple_iterations(self) -> None:
        """Training must complete multiple epochs without error and record loss."""
        vqc, X, y = self._make_vqc_and_data(n_qubits=2, n_samples=8)
        vqc.max_epochs = 3
        vqc.fit(X, y)
        self.assertEqual(len(vqc.training_loss_history_), 3)
        for loss in vqc.training_loss_history_:
            self.assertTrue(np.isfinite(loss), f"Non-finite training loss: {loss}")

    # ------------------------------------------------------------------
    # Smoke tests: forward pass for all supported qubit configurations
    # ------------------------------------------------------------------

    def test_smoke_all_qubit_configurations(self) -> None:
        """Smoke test: 2, 4, 8, 16-qubit VQC forward pass + 1 training epoch.

        Verifies that the full pipeline (fit → predict_proba → predict) works
        for every supported qubit count using small synthetic batches.
        No assertion on accuracy — only finiteness and shape are checked.
        """
        for n_qubits in (2, 4, 8, 16):
            with self.subTest(n_qubits=n_qubits):
                rng = np.random.RandomState(n_qubits)
                # Need enough samples for mini-batch; use 2*n_qubits
                n_samples = max(2 * n_qubits, 8)
                X = rng.uniform(-np.pi, np.pi, size=(n_samples, n_qubits))
                y = (X[:, 0] > 0).astype(int)
                # Ensure at least one of each class
                y[0] = 0
                y[1] = 1

                vqc = VariationalQuantumClassifier(
                    n_qubits=n_qubits,
                    n_layers=1,
                    max_epochs=1,
                    batch_size=n_samples,
                    learning_rate=0.05,
                    random_state=n_qubits,
                )
                vqc.fit(X, y)

                # scale_ must exist and be finite
                self.assertTrue(
                    np.isfinite(vqc.scale_),
                    f"n_qubits={n_qubits}: scale_ is not finite after training",
                )
                self.assertTrue(
                    np.isfinite(vqc.bias_),
                    f"n_qubits={n_qubits}: bias_ is not finite after training",
                )

                probs = vqc.predict_proba(X)
                self.assertEqual(probs.shape, (n_samples, 2),
                                 f"n_qubits={n_qubits}: wrong proba shape {probs.shape}")
                self.assertTrue(
                    np.all(np.isfinite(probs)),
                    f"n_qubits={n_qubits}: non-finite probabilities",
                )
                self.assertTrue(
                    np.all(probs >= 0.0) and np.all(probs <= 1.0 + 1e-9),
                    f"n_qubits={n_qubits}: probabilities outside [0,1]",
                )
                np.testing.assert_allclose(
                    probs.sum(axis=1), np.ones(n_samples), atol=1e-6,
                    err_msg=f"n_qubits={n_qubits}: proba rows don't sum to 1",
                )

                preds = vqc.predict(X)
                self.assertEqual(preds.shape, (n_samples,))
                self.assertTrue(set(preds).issubset({0, 1}))

                # Verify gradients for scale on a single batch
                vqc._init_circuit()
                weight_shape = get_ansatz_weight_shape(1, n_qubits)
                rng2 = np.random.RandomState(99 + n_qubits)
                w = pnp.tensor(rng2.uniform(-np.pi, np.pi, weight_shape), requires_grad=True)
                b = pnp.tensor(0.0, requires_grad=True)
                s = pnp.tensor(1.0, requires_grad=True)

                x0 = pnp.tensor(X[0], requires_grad=False)

                def single_cost(ww, bb, ss):
                    prob = vqc._forward_sample(x0, ww, bb, ss)
                    return -pnp.log(prob + 1e-7)

                grad_fn = qml.grad(single_cost, argnums=[0, 1, 2])
                gw, gb, gs = grad_fn(w, b, s)

                self.assertTrue(
                    np.isfinite(float(gs)),
                    f"n_qubits={n_qubits}: scale gradient is not finite",
                )
                self.assertTrue(
                    np.isfinite(float(gb)),
                    f"n_qubits={n_qubits}: bias gradient is not finite",
                )
                self.assertTrue(
                    np.all(np.isfinite(np.asarray(gw))),
                    f"n_qubits={n_qubits}: weight gradients contain non-finite values",
                )


if __name__ == "__main__":
    unittest.main()
