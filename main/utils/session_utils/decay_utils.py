# TODO migrate sknn to use this module
# TODO migrate the tests from sknn to standalone

from typing import Optional

import numpy as np


def precompute_decay_arrays(
    decay: Optional[str], max_length: int
) -> dict[int, np.ndarray]:
    """Precomputes decay vectors for sessions of length up to max_length parameter.

    Args:
        decay: A string indicating the decay type.
        max_length: An integer indicating

    Returns:
        A dictionary that maps session lengths decay arrays stored as 1d ndarrays.

    Raises:
        ValueError: If the given decay function is not recognized.
    """
    precomputed_decay_arrays_dict: dict[int, np.ndarray] = {}
    for i in range(1, max_length + 1):
        precomputed_decay_arrays_dict[i] = compute_decay_array(decay, i)
    return precomputed_decay_arrays_dict


def compute_decay_array(decay: Optional[str], length: int) -> np.ndarray:
    """Computes a decay array for the given decay type and session length.

    Args:
        decay: A string indicating the decay type. The options are linear, log,
            harmonic, and quadratic. Passing None indicates no decay, in which case
            the method returns an array of ones.
        length: A string indicating the session length.

    Returns:
        A 1d ndarray representing the decay array.

    Raises:
        ValueError: If the given decay function is not recognized.
    """
    descending_ints: list[int] = list(range(length, 0, -1))
    if decay is None:
        return np.ones(length, dtype=int)
    elif decay == "constant_linear":
        return np.array([max(0.00001, 1.0 - (i - 1) * 0.1) for i in descending_ints])
    elif decay == "scaling_linear":
        return np.array(
            [max(0.00001, 1.0 - (i - 1) * (1 / length)) for i in descending_ints]
        )
    elif decay == "scaling_quadratic":
        return compute_decay_array("scaling_linear", length) ** 2
    elif decay == "scaling_cubic":
        return compute_decay_array("scaling_linear", length) ** 3
    elif decay == "log":
        return np.array([1 / (np.log(i) + 1) for i in descending_ints])
    elif decay == "harmonic":
        return np.array([1 / i for i in descending_ints])
    elif decay == "harmonic_squared":
        return np.array([1 / i**2 for i in descending_ints])
    else:
        raise ValueError(f"The given decay function of {decay} is not recognized.")
