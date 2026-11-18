"""Regression tests for Defect 1: models.quantum package-level imports.

Verifies that all four public symbols declared in __all__ are importable
directly from the models.quantum package namespace, and that individual
submodule imports continue to work alongside package-level imports.
"""

from __future__ import annotations

import unittest


class QuantumPackageImportTests(unittest.TestCase):
    """Verify that models.quantum exposes all __all__ symbols correctly."""

    def test_import_quantum_dimensionality_reducer_from_package(self) -> None:
        """QuantumDimensionalityReducer must be importable from models.quantum."""
        from models.quantum import QuantumDimensionalityReducer  # noqa: F401
        self.assertTrue(callable(QuantumDimensionalityReducer))

    def test_import_create_angle_embedding_from_package(self) -> None:
        """create_angle_embedding must be importable from models.quantum."""
        from models.quantum import create_angle_embedding  # noqa: F401
        self.assertTrue(callable(create_angle_embedding))

    def test_import_create_ansatz_from_package(self) -> None:
        """create_ansatz must be importable from models.quantum."""
        from models.quantum import create_ansatz  # noqa: F401
        self.assertTrue(callable(create_ansatz))

    def test_import_variational_quantum_classifier_from_package(self) -> None:
        """VariationalQuantumClassifier must be importable from models.quantum."""
        from models.quantum import VariationalQuantumClassifier  # noqa: F401
        self.assertTrue(callable(VariationalQuantumClassifier))

    def test_all_symbols_match_dunder_all(self) -> None:
        """Every name in __all__ must be resolvable in the package namespace."""
        import models.quantum as qpkg
        for name in qpkg.__all__:
            with self.subTest(symbol=name):
                self.assertTrue(
                    hasattr(qpkg, name),
                    f"models.quantum.__all__ lists '{name}' but it is not in the namespace",
                )

    def test_submodule_imports_still_work(self) -> None:
        """Direct submodule imports must continue to work after adding package imports."""
        from models.quantum.dim_reduction import QuantumDimensionalityReducer as QDR  # noqa
        from models.quantum.embeddings import create_angle_embedding as cae  # noqa
        from models.quantum.ansatz import create_ansatz as ca  # noqa
        from models.quantum.vqc_model import VariationalQuantumClassifier as VQC  # noqa
        self.assertTrue(callable(QDR))
        self.assertTrue(callable(cae))
        self.assertTrue(callable(ca))
        self.assertTrue(callable(VQC))

    def test_package_and_submodule_symbols_are_same_objects(self) -> None:
        """Package-level names must be the identical objects as submodule names."""
        from models.quantum import QuantumDimensionalityReducer as pkg_QDR
        from models.quantum.dim_reduction import QuantumDimensionalityReducer as sub_QDR
        self.assertIs(pkg_QDR, sub_QDR,
                      "models.quantum.QuantumDimensionalityReducer is not the same object "
                      "as models.quantum.dim_reduction.QuantumDimensionalityReducer")

        from models.quantum import VariationalQuantumClassifier as pkg_VQC
        from models.quantum.vqc_model import VariationalQuantumClassifier as sub_VQC
        self.assertIs(pkg_VQC, sub_VQC,
                      "models.quantum.VariationalQuantumClassifier is not the same object "
                      "as models.quantum.vqc_model.VariationalQuantumClassifier")


if __name__ == "__main__":
    unittest.main()
