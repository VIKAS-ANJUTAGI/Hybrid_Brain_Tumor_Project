"""Quantum feature encoding and angle embedding circuits."""

from __future__ import annotations

from collections.abc import Sequence
import numpy as np

try:
    import pennylane as qml
except ImportError:
    qml = None  # Handled gracefully if loaded before installation completes


def create_angle_embedding(
    features: Sequence[float] | np.ndarray,
    wires: Sequence[int],
    *,
    rotation: str = "Y",
) -> None:
    """Encode classical continuous features as rotation angles on specified qubit wires.

    Args:
        features: 1D array of angles (typically in [-pi, pi]), one per wire.
        wires: Target qubit wire indices.
        rotation: Axis of rotation: "X", "Y", or "Z" (defaults to "Y").

    Raises:
        ImportError: If pennylane is not installed.
        ValueError: If length of features does not match length of wires.
    """
    if qml is None:
        raise ImportError("pennylane is required for quantum angle embedding.")

    if len(features) != len(wires):
        raise ValueError(
            f"Feature dimension ({len(features)}) must match number of wires ({len(wires)})."
        )

    rot_upper = rotation.upper()
    if rot_upper == "X":
        gate = qml.RX
    elif rot_upper == "Y":
        gate = qml.RY
    elif rot_upper == "Z":
        gate = qml.RZ
    else:
        raise ValueError(f"Unsupported rotation axis: {rotation}. Choose 'X', 'Y', or 'Z'.")

    for feature_val, wire in zip(features, wires):
        gate(feature_val, wires=wire)
