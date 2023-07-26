"""This module contains methods that compute various similarity measures."""

from typing import Callable

import numpy as np


def get_available_similarity_measures() -> list[str]:
    """Returns the names of available similarity measures in a list.

    Returns:
        A list of strings containing the similarity measures.
    """
    measures: list[str] = [
        "jaccard",
        "stiles",
        "anderberg",
        "ample",
        "peirce",
        "yule_w",
        "tarantula",
        "simpson",
        "eyraud",
        "fager_mcgowan",
        "driver_kroeber",
    ]
    return measures


def get_similarity_func(similarity: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Returns the function that computes the similarity measure with the given name.

    Args:
        similarity: A string representing the name of the similarity measure.

    Returns:
        A callable that takes 2 binary vectors as input and returns a float representing
        the similarity score.
    """
    # Case matching would be really nice here. We can refactor when we migrate to 3.10.
    if similarity == "jaccard":
        return _jaccard
    elif similarity == "stiles":
        return _stiles
    elif similarity == "anderberg":
        return _anderberg
    elif similarity == "ample":
        return _ample
    elif similarity == "peirce":
        return _peirce
    elif similarity == "yule_w":
        return _yule_w
    elif similarity == "tarantula":
        return _tarantula
    elif similarity == "simpson":
        return _simpson
    elif similarity == "eyraud":
        return _eyraud
    elif similarity == "fager_mcgowan":
        return _fager_mcgowan
    elif similarity == "driver_kroeber":
        return _driver_kroeber
    else:
        raise ValueError(f"{similarity} is not a supported similarity measure.")


def _jaccard(first: np.ndarray, second: np.ndarray) -> float:
    """Computes the Jaccard similarity between the given vectors.

    Args:
        first: A binary numpy array representing the first vector.
        second: A binary numpy array representing the second vector.

    Returns:
        A float representing the Jaccard similarity.
    """
    a, b, c, d = _get_operational_units(first, second)
    numerator: int = a
    denominator: int = a + b + c

    return numerator / denominator


def _stiles(first: np.ndarray, second: np.ndarray) -> float:
    """Computes the Stiles similarity between the given vectors.

    Args:
        first: A binary numpy array representing the first vector.
        second: A binary numpy array representing the second vector.

    Returns:
        A float representing the Stiles similarity.
    """
    a, b, c, d = _get_operational_units(first, second)
    n: int = first.shape[0]
    numerator: float = n * (np.abs(a * d - b * c) - n / 2) ** 2
    denominator: int = (a + b) * (a + c) * (b + d) * (c + d)

    return np.log10(numerator / denominator)


def _anderberg(first: np.ndarray, second: np.ndarray) -> float:
    """Computes the Anderberg similarity between the given vectors.

    Args:
        first: A binary numpy array representing the first vector.
        second: A binary numpy array representing the second vector.

    Returns:
        A float representing the Anderberg similarity.
    """
    a, b, c, d = _get_operational_units(first, second)
    n: int = first.shape[0]
    numerator: int = _get_sigma(a, b, c, d) - _get_sigma_prime(a, b, c, d)
    denominator: int = 2 * n

    return numerator / denominator


def _ample(first: np.ndarray, second: np.ndarray) -> float:
    """Computes the Ample similarity between the given vectors.

    Args:
        first: A binary numpy array representing the first vector.
        second: A binary numpy array representing the second vector.

    Returns:
        A float representing the Ample similarity.
    """
    a, b, c, d = _get_operational_units(first, second)
    numerator: int = a * (c + d)
    denominator: int = c * (a + b)
    if denominator == 0:
        return 0.0

    return np.abs(numerator / denominator)


def _peirce(first: np.ndarray, second: np.ndarray) -> float:
    """Computes the Peirce similarity between the given vectors.

    Args:
        first: A binary numpy array representing the first vector.
        second: A binary numpy array representing the second vector.

    Returns:
        A float representing the Peirce similarity.
    """
    a, b, c, d = _get_operational_units(first, second)
    numerator: int = (a * b) + (b * c)
    denominator: int = (a * b) + (2 * b * c) + (c * d)
    if denominator == 0:
        return 0.0

    return numerator / denominator


def _yule_w(first: np.ndarray, second: np.ndarray) -> float:
    """Computes the YULEw similarity between the given vectors.

    Args:
        first: A binary numpy array representing the first vector.
        second: A binary numpy array representing the second vector.

    Returns:
        A float representing the YULEw similarity.
    """
    a, b, c, d = _get_operational_units(first, second)
    numerator: float = np.sqrt(a * d) - np.sqrt(b * c)
    denominator: float = np.sqrt(a * d) + np.sqrt(b * c)

    return numerator / denominator


def _tarantula(first: np.ndarray, second: np.ndarray) -> float:
    """Computes the Tarantula similarity between the given vectors.

    Args:
        first: A binary numpy array representing the first vector.
        second: A binary numpy array representing the second vector.

    Returns:
        A float representing the Tarantula similarity.
    """
    a, b, c, d = _get_operational_units(first, second)
    numerator: int = a * (c + d)
    denominator: int = c * (a + b)
    if denominator == 0:
        return 0.0

    return numerator / denominator


def _simpson(first: np.ndarray, second: np.ndarray) -> float:
    """Computes the Simpson similarity between the given vectors.

    Args:
        first: A binary numpy array representing the first vector.
        second: A binary numpy array representing the second vector.

    Returns:
        A float representing the Simpson similarity.
    """
    a, b, c, d = _get_operational_units(first, second)
    numerator: int = a
    denominator: int = min(a + b, a + c)

    return numerator / denominator


def _eyraud(first: np.ndarray, second: np.ndarray) -> float:
    """Computes the Eyraud similarity between the given vectors.

    Args:
        first: A binary numpy array representing the first vector.
        second: A binary numpy array representing the second vector.

    Returns:
        A float representing the Eyraud similarity.
    """
    a, b, c, d = _get_operational_units(first, second)
    n: int = first.shape[0]
    numerator: int = n**2 * ((n * a) - (a + b) * (a + c))
    denominator: int = (a + b) * (a + c) * (b + d) * (c + d)

    return numerator / denominator


def _fager_mcgowan(first: np.ndarray, second: np.ndarray) -> float:
    """Computes the Fager & McGowan similarity between the given vectors.

    Args:
        first: A binary numpy array representing the first vector.
        second: A binary numpy array representing the second vector.

    Returns:
        A float representing the Fager & McGowan similarity.
    """
    a, b, c, d = _get_operational_units(first, second)
    minuend: float = a / np.sqrt((a + b) * (a + c))
    subtrahend: float = max(a + b, a + c) / 2

    return minuend - subtrahend


def _driver_kroeber(first: np.ndarray, second: np.ndarray) -> float:
    """Computes the Driver & Kroeber similarity between the given vectors.

    Args:
        first: A binary numpy array representing the first vector.
        second: A binary numpy array representing the second vector.

    Returns:
        A float representing the Driver & Kroeber similarity.
    """
    a, b, c, d = _get_operational_units(first, second)

    return (a / 2) * (1 / (a + b) + 1 / (a + c))


def _get_operational_units(
    first: np.ndarray, second: np.ndarray
) -> tuple[int, int, int, int]:
    """Returns a tuple containing the operational units of similarity measures.

    Args:
        first: A binary numpy array representing the first vector.
        second: A binary numpy array representing the second vector.

    Returns:
        A tuple of 4 integers representing the operational units for the given vectors.
    """
    a: int = np.sum(first & second)
    b: int = np.sum((1 - first) & second)
    c: int = np.sum(first & (1 - second))
    d: int = np.sum((1 - first) & (1 - second))
    return a, b, c, d


def _get_sigma(a: int, b: int, c: int, d: int) -> int:
    """Returns the sigma value of the given operational units, which is defined by the
    formula given as code.

    Args:
        a: An integer representing the operational unit a.
        b: An integer representing the operational unit b.
        c: An integer representing the operational unit c.
        d: An integer representing the operational unit d.

    Returns:
        An integer representing the sigma value.
    """
    sigma: int = max(a, b) + max(c, d) + max(a, c) + max(b, d)
    return sigma


def _get_sigma_prime(a: int, b: int, c: int, d: int) -> int:
    """Returns the sigma prime value of the given operational units, which is defined by
    the formula given as code.

    Args:
        a: An integer representing the operational unit a.
        b: An integer representing the operational unit b.
        c: An integer representing the operational unit c.
        d: An integer representing the operational unit d.

    Returns:
        An integer representing the sigma prime value.
    """
    sigma_prime: int = max(a + c, b + d) + max(a + b, c + d)
    return sigma_prime
