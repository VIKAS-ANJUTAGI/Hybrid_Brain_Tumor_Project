"""Classical feature preprocessing and dimensionality reduction for quantum circuits."""

from __future__ import annotations

from pathlib import Path
from typing import Literal
import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

SUPPORTED_QUBIT_COUNTS = (2, 4, 8, 16)


class QuantumDimensionalityReducer(BaseEstimator, TransformerMixin):
    """Reduces 16 classical MRI features to target qubit rotation angles in [-pi, pi].

    For n_qubits < 16 (e.g. 2, 4, 8), standardizes features, applies PCA to
    extract the top n_qubits orthogonal components, and bounds the values
    to [-pi, pi] for angle embedding.

    For n_qubits == 16, retains all 16 features, standardizes them, and scales
    them directly to [-pi, pi] without losing individual modality-statistic identity.

    Guarantees strict train/test isolation: fit() must only be called on training
    data, and transform() applies the learned transformations without leakage.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        *,
        random_state: int = 42,
        angle_range: tuple[float, float] = (-np.pi, np.pi),
    ) -> None:
        if n_qubits not in SUPPORTED_QUBIT_COUNTS:
            raise ValueError(
                f"Unsupported n_qubits: {n_qubits}. Must be one of {SUPPORTED_QUBIT_COUNTS}."
            )
        self.n_qubits = n_qubits
        self.random_state = random_state
        self.angle_range = angle_range

        self.scaler_: StandardScaler | None = None
        self.pca_: PCA | None = None
        self.angle_scaler_: MinMaxScaler | None = None
        self.is_fitted_: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> QuantumDimensionalityReducer:
        """Fit preprocessing transformations strictly on training features."""
        data = np.asarray(X, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != 16:
            raise ValueError(f"Expected input with 16 features; received shape {data.shape}")
        if data.shape[0] < self.n_qubits:
            raise ValueError(
                f"Number of samples ({data.shape[0]}) must be >= n_qubits ({self.n_qubits})"
            )

        self.scaler_ = StandardScaler()
        standardized = self.scaler_.fit_transform(data)

        if self.n_qubits < 16:
            self.pca_ = PCA(n_components=self.n_qubits, random_state=self.random_state)
            projected = self.pca_.fit_transform(standardized)
        else:
            self.pca_ = None
            projected = standardized

        self.angle_scaler_ = MinMaxScaler(feature_range=self.angle_range, clip=True)
        self.angle_scaler_.fit(projected)
        self.is_fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features into angle embeddings in [-pi, pi]."""
        if not self.is_fitted_ or self.scaler_ is None or self.angle_scaler_ is None:
            raise RuntimeError("QuantumDimensionalityReducer must be fitted before calling transform().")

        data = np.asarray(X, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != 16:
            raise ValueError(f"Expected input with 16 features; received shape {data.shape}")

        standardized = self.scaler_.transform(data)
        if self.pca_ is not None:
            projected = self.pca_.transform(standardized)
        else:
            projected = standardized

        angles = self.angle_scaler_.transform(projected)
        return angles.astype(np.float64)

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Fit to data, then transform it."""
        return self.fit(X, y).transform(X)

    def save(self, filepath: str | Path) -> None:
        """Serialize the fitted reducer artifact to disk."""
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str | Path) -> QuantumDimensionalityReducer:
        """Load a serialized reducer artifact from disk."""
        loaded = joblib.load(filepath)
        if not isinstance(loaded, cls):
            raise TypeError(f"Loaded object is not a {cls.__name__}")
        return loaded
