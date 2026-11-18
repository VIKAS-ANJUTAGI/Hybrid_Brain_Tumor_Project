"""Unit tests for quantum feature preprocessing and dimensionality reduction."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from models.quantum.dim_reduction import QuantumDimensionalityReducer


class QuantumDimensionalityReducerTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.RandomState(42)
        # 30 samples, 16 features
        self.X_train = rng.normal(loc=5.0, scale=2.0, size=(30, 16))
        self.X_test = rng.normal(loc=5.0, scale=2.0, size=(10, 16))

    def test_supports_all_target_qubit_configurations(self) -> None:
        for n_qubits in (2, 4, 8, 16):
            with self.subTest(n_qubits=n_qubits):
                reducer = QuantumDimensionalityReducer(n_qubits=n_qubits, random_state=42)
                transformed_train = reducer.fit_transform(self.X_train)
                transformed_test = reducer.transform(self.X_test)

                self.assertEqual(transformed_train.shape, (30, n_qubits))
                self.assertEqual(transformed_test.shape, (10, n_qubits))

                # Values must be bounded in [-pi, pi]
                self.assertTrue(np.all(transformed_train >= -np.pi - 1e-6))
                self.assertTrue(np.all(transformed_train <= np.pi + 1e-6))

    def test_rejects_unsupported_qubit_count(self) -> None:
        for invalid_q in (1, 3, 5, 12, 32):
            with self.subTest(n_qubits=invalid_q):
                with self.assertRaises(ValueError):
                    QuantumDimensionalityReducer(n_qubits=invalid_q)

    def test_rejects_non_16_feature_input(self) -> None:
        reducer = QuantumDimensionalityReducer(n_qubits=4)
        with self.assertRaises(ValueError):
            reducer.fit(np.zeros((10, 15)))

    def test_transform_before_fit_raises(self) -> None:
        reducer = QuantumDimensionalityReducer(n_qubits=4)
        with self.assertRaises(RuntimeError):
            reducer.transform(self.X_test)

    def test_serialization_and_deserialization(self) -> None:
        reducer = QuantumDimensionalityReducer(n_qubits=4, random_state=42)
        angles_orig = reducer.fit_transform(self.X_train)

        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = Path(tmp_dir) / "reducer.joblib"
            reducer.save(artifact_path)
            loaded = QuantumDimensionalityReducer.load(artifact_path)

            angles_loaded = loaded.transform(self.X_train)
            np.testing.assert_allclose(angles_orig, angles_loaded)

    # ------------------------------------------------------------------
    # Regression tests for Defect 2: quantum angle bounding (clip=True)
    # ------------------------------------------------------------------

    def test_out_of_range_samples_clipped_to_pi(self) -> None:
        """Out-of-training-range samples must stay strictly within [-pi, pi].

        Regression test for Defect 2: MinMaxScaler must be constructed with
        clip=True so that unseen validation/test samples whose projected values
        fall outside the training range are clamped rather than extrapolated.
        """
        for n_qubits in (2, 4, 8, 16):
            with self.subTest(n_qubits=n_qubits):
                reducer = QuantumDimensionalityReducer(n_qubits=n_qubits, random_state=42)
                reducer.fit(self.X_train)

                # Extreme positive values — far outside any training range
                X_extreme_pos = np.full((5, 16), 1e6)
                angles_pos = reducer.transform(X_extreme_pos)
                self.assertTrue(
                    np.all(angles_pos >= -np.pi - 1e-9),
                    f"n_qubits={n_qubits}: min angle {angles_pos.min():.6f} < -pi",
                )
                self.assertTrue(
                    np.all(angles_pos <= np.pi + 1e-9),
                    f"n_qubits={n_qubits}: max angle {angles_pos.max():.6f} > pi",
                )

                # Extreme negative values — far outside any training range
                X_extreme_neg = np.full((5, 16), -1e6)
                angles_neg = reducer.transform(X_extreme_neg)
                self.assertTrue(
                    np.all(angles_neg >= -np.pi - 1e-9),
                    f"n_qubits={n_qubits}: min angle {angles_neg.min():.6f} < -pi",
                )
                self.assertTrue(
                    np.all(angles_neg <= np.pi + 1e-9),
                    f"n_qubits={n_qubits}: max angle {angles_neg.max():.6f} > pi",
                )

    def test_in_range_samples_remain_within_pi(self) -> None:
        """Normal in-distribution samples must also stay within [-pi, pi]."""
        reducer = QuantumDimensionalityReducer(n_qubits=4, random_state=42)
        reducer.fit(self.X_train)
        angles = reducer.transform(self.X_test)
        self.assertTrue(
            np.all(angles >= -np.pi - 1e-9),
            f"Min angle {angles.min():.6f} < -pi for in-range samples",
        )
        self.assertTrue(
            np.all(angles <= np.pi + 1e-9),
            f"Max angle {angles.max():.6f} > pi for in-range samples",
        )

    def test_transform_does_not_refit_scaler_or_pca(self) -> None:
        """transform() must not mutate the fitted StandardScaler or MinMaxScaler.

        Regression test for Defect 2: confirms that training/test isolation is
        preserved — transform() must never call fit() internally.
        """
        reducer = QuantumDimensionalityReducer(n_qubits=4, random_state=42)
        reducer.fit(self.X_train)

        # Snapshot scaler state before any transform calls
        mean_before = reducer.scaler_.mean_.copy()
        var_before = reducer.scaler_.var_.copy()
        angle_scale_before = reducer.angle_scaler_.scale_.copy()
        angle_min_before = reducer.angle_scaler_.data_min_.copy()

        # Transform both in-range and out-of-range data
        reducer.transform(self.X_test)
        reducer.transform(np.full((3, 16), 1e6))

        # Scaler state must be identical after transform calls
        np.testing.assert_array_equal(
            reducer.scaler_.mean_, mean_before,
            err_msg="StandardScaler mean changed after transform() — possible refit!",
        )
        np.testing.assert_array_equal(
            reducer.scaler_.var_, var_before,
            err_msg="StandardScaler var changed after transform() — possible refit!",
        )
        np.testing.assert_array_equal(
            reducer.angle_scaler_.scale_, angle_scale_before,
            err_msg="MinMaxScaler scale changed after transform() — possible refit!",
        )
        np.testing.assert_array_equal(
            reducer.angle_scaler_.data_min_, angle_min_before,
            err_msg="MinMaxScaler data_min changed after transform() — possible refit!",
        )

    def test_serialization_preserves_clip_behavior(self) -> None:
        """A reloaded reducer must still clip out-of-range values to [-pi, pi].

        Regression test for Defect 2: verifies that joblib serialization/
        deserialization preserves the clip=True attribute of MinMaxScaler.
        """
        reducer = QuantumDimensionalityReducer(n_qubits=4, random_state=42)
        reducer.fit(self.X_train)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "reducer_clip.joblib"
            reducer.save(path)
            loaded = QuantumDimensionalityReducer.load(path)

        # Confirm the loaded scaler has clip=True
        self.assertTrue(
            loaded.angle_scaler_.clip,
            "Loaded MinMaxScaler does not have clip=True — serialization lost the attribute.",
        )

        # Confirm extreme values are still clamped after reload
        X_extreme = np.full((3, 16), 1e6)
        angles = loaded.transform(X_extreme)
        self.assertTrue(
            np.all(angles >= -np.pi - 1e-9),
            f"Reloaded reducer: min angle {angles.min():.6f} < -pi",
        )
        self.assertTrue(
            np.all(angles <= np.pi + 1e-9),
            f"Reloaded reducer: max angle {angles.max():.6f} > pi",
        )


if __name__ == "__main__":
    unittest.main()
