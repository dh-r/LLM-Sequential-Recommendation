import numpy as np
from typing import Iterable

INT_INF = 1_000_000_000


def to_dense_encoding(
    sparse_representation: Iterable[Iterable], space_size: int, ignore_oob: bool = False
) -> np.ndarray:
    """Converts an iterable of iterables consisting of sparse encodings to a dense
    representation.

    We simply increment the index for each value in the sparse representations,
    so note that we do allow for duplicates.

    Args:
        sparse_representation (Iterable[Iterable]): The sparse data. The sparse data
            is assumed to be in the range [0, space_size)
        space_size (int): The dimension of the sparse encodings.
        ignore_oob (bool): Whether to ignore out-of-bound indices.

    Returns:
        np.ndarray: The dense representation.

    Example:
            sparse_representation: [[1, 3], [0, 0]]
            space_size : 4
        results in:
            [
                [0, 1, 0, 1],
                [2, 0, 0, 0]
            ]
    """
    result = np.zeros((len(sparse_representation), space_size))

    for i in range(len(sparse_representation)):
        for value in sparse_representation[i]:
            if ignore_oob:
                if value >= space_size:
                    continue
            result[i, value] += 1

    return result
