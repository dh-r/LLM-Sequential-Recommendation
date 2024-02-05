# TODO add documentation
# TODO migrate sknn to use this module
# TODO migrate the tests from sknn to standalone

from typing import Optional

import numpy as np

from main.utils.session_utils import decay_utils


def combine_embeddings(
    embeddings: np.ndarray,
    strategy: str,
    combination_decay: Optional[str] = None,
    decay_arrays: Optional[dict[int, np.ndarray]] = None,
) -> np.ndarray:
    if strategy == "last":
        return embeddings[-1]
    else:
        if combination_decay is None:
            if strategy == "mean":
                return np.mean(embeddings, axis=0)
            elif strategy == "concat":
                return embeddings.ravel()
            else:
                raise ValueError
        else:
            n_embeddings: int = len(embeddings)
            if decay_arrays is not None and n_embeddings in decay_arrays:
                decay_arr: np.ndarray = decay_arrays[n_embeddings]
            else:
                decay_arr: np.ndarray = decay_utils.compute_decay_array(
                    combination_decay, n_embeddings
                )
            if strategy == "mean":
                return np.average(embeddings, axis=0, weights=decay_arr)
            elif strategy == "concat":
                decay_arr = np.repeat(decay_arr, embeddings[0].size)
                return embeddings.ravel() * decay_arr
            else:
                raise ValueError
