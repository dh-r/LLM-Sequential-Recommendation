from typing import TypedDict

import numpy as np


class SideInformation(TypedDict):
    """A TypedDict class that encapsulates side information."""

    features: np.ndarray

    num_non_categorical_features: int
    num_categorical_features: int
    total_features: int

    category_sizes: list[int]


def create_side_information(
    features: np.ndarray,
    category_sizes: list[int] = None,
) -> SideInformation:
    """Returns a SideInformation instance with the given features and category sizes.

    Args:
        features: A numpy ndarray containing the features.
        category_sizes: A list of integers specifying the category sizes of the
            categorical features.
    """
    if category_sizes is None:
        category_sizes = []
    side_information: SideInformation = {
        "features": features,
        "total_features": features.shape[1],
        "num_categorical_features": len(category_sizes),
        "num_non_categorical_features": features.shape[1] - len(category_sizes),
        "category_sizes": category_sizes,
    }
    return side_information
