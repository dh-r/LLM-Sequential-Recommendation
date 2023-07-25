import pandas as pd
import numpy as np
from typing import Any, Union


class TopKComputer:
    """A class to compute the indices of the top k values from different types of input."""

    @staticmethod
    def compute_top_k(
        data: Any, top_k: int, filter_zero_predictions: bool = False
    ) -> Any:
        if type(data) == np.ndarray:
            return TopKComputer.__compute_top_k_ndarray(
                data, top_k, filter_zero_predictions
            )

        return None

    @staticmethod
    def __compute_top_k_ndarray(
        data: np.ndarray, top_k: int, filter_zero_predictions: bool = False
    ) -> Union[np.ndarray, list[np.ndarray]]:
        """Compute top k indices on the last axis of data.

        Args:
            data (np.ndarray): The data from which the top_k values on the last axis
                must be computed.

            top_k (int): The number of indices that correspond to the highest values
                that need to be returned. If -1, returns complete ordering. If
                -1 and filter_zero_predictions it does filter the zero scores.

            filter_zero_predictions (bool): Remove indices of predictions that are zero.
                Defaults to False.

        Returns:
            np.ndarray: The indices of the highest values in data along the last axis,
                sorted.
        """
        original_shape = data.shape
        if len(original_shape) > 2:
            data = np.reshape(data, (-1, data.shape[-1]))

        # We can not the top k indices on data with less than k entries in its last
        # axis, so set top_k to the number of entries in the last axis.
        if top_k > data.shape[-1]:
            top_k = data.shape[-1]

        # Similarly if top_k is -1, we will provide an ordering for _all_
        # entries in the last axis.
        if top_k == -1:
            top_k = data.shape[-1]

        top_k_indices = np.argpartition(data, -top_k)[:, -top_k:]
        rows = np.indices((len(data), top_k))[0]
        top_k_values = data[rows, top_k_indices]
        top_k_indices_sorted = top_k_indices[rows, np.argsort(-top_k_values)]

        if len(original_shape) > 2:
            new_shape = [original_shape[i] for i in range(len(original_shape))]
            new_shape[-1] = top_k
            top_k_indices_sorted = np.reshape(top_k_indices_sorted, new_shape)

        if filter_zero_predictions:
            # Find predictions in top-k that are zeros.
            mask = data[rows, top_k_indices_sorted] == 0

            # Find the first index in the top-k that is zero.
            # We can remove all indices after this first occurrence.
            zeros = np.where(mask.any(1), mask.argmax(1), -1).reshape(-1)

            # Keep only the rows for which we have predictions that are zero.
            zero_idx = np.where(zeros != -1)[0]

            # Turn matrix into list of vectors. Due to variable length we cannot stick
            # to matrices anymore.
            recs = list(top_k_indices_sorted)

            # Remove predictions from the rows with those 'zero' predictions.
            for i, row, nz in zip(
                zero_idx, top_k_indices_sorted[zero_idx], zeros[zero_idx]
            ):
                recs[i] = row.flatten()[:nz]

            return recs
        return top_k_indices_sorted
