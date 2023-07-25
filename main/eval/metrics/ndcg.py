from main.eval.metrics.metric import RankingMetric
import numpy as np
from typing import List
import numba as nb


class NormalizedDiscountedCumulativeGain(RankingMetric):
    @staticmethod
    def get_utilities(
        predictions: List[np.ndarray],  # e.g. [[1, 2, 3], [3, 2], []]
        ground_truth: List[np.ndarray],  # e.g. [[1, 2], [], [0]]
        intersect: List[np.ndarray],  # e.g. [[2, 3], [3], []]
        top_k: int,
    ):
        pred_utility = np.zeros((len(predictions), top_k))
        test_utility = np.zeros((len(ground_truth), top_k))

        for i in range(len(predictions)):
            if len(intersect[i]) == 0:
                continue

            test_utility[i, : min(np.count_nonzero(ground_truth[i] >= 0), top_k)] = 1

            # This is faster than using intersect1d.
            inter = set(intersect[i])
            for j in range(len(predictions[i])):
                if predictions[i][j] in inter:
                    pred_utility[i, j] = 1

        return pred_utility, test_utility

    def eval_bulk(
        self,
        predictions: List[np.ndarray],
        ground_truth: List[np.ndarray],
        intersect: List[np.ndarray],
        sample_ids: np.ndarray,
    ) -> np.ndarray:
        pred_utility, test_utility = NormalizedDiscountedCumulativeGain.get_utilities(
            predictions, ground_truth, intersect, top_k=self.top_k
        )

        achieved_dcg = NormalizedDiscountedCumulativeGain._compute_dcg(pred_utility)

        # For ideal DCG, we can just use the true relevance as utility.
        ideal_dcg = NormalizedDiscountedCumulativeGain._compute_dcg(test_utility)

        return np.divide(
            achieved_dcg,
            ideal_dcg,
            out=np.zeros_like(achieved_dcg),
            where=ideal_dcg != 0,
        )

    def eval_sample(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        intersect: np.ndarray,
        sample_id: int,
    ) -> float:
        # Utility vector with all irrelevant recs set to 0 and
        # all relevant recs set to one.
        pred_utility = np.pad(
            np.isin(predictions, intersect, assume_unique=True).astype(np.int32),
            (0, self.top_k - predictions.shape[0]),
        )
        test_utility = np.ones(ground_truth.shape[0])[: self.top_k]

        if np.count_nonzero(pred_utility) == 0 or np.count_nonzero(test_utility) == 0:
            return 0

        achieved_dcg = NormalizedDiscountedCumulativeGain._compute_dcg(pred_utility)
        # For ideal DCG, we can just use the true relevance as utility.
        ideal_dcg = NormalizedDiscountedCumulativeGain._compute_dcg(test_utility)
        return achieved_dcg / ideal_dcg

    @staticmethod
    @nb.njit(cache=True)
    def _compute_dcg(utility: np.ndarray) -> float:
        """Compute discounted cumulative gain (DCG) based on a utility.

        Args:
            utility (np.ndarray): A top-k prediction list, where only the relevant items
                are non-zero.

        For example, when we have the following relevant/ground-truth/test items {1, 9},
        and the top-5 recommended indices are [3, 7, 9, 2, 4], then utility should be
        [0, 0, predicted_rating_of(9), 0, 0]. One can also pass the ground-truth items
        to this function, i.e. [9, 4, 3, 1, 1] for top-5 or [3] when there is only one
        relevant item.

        Returns:
            np.float64: The DCG for the utility vector.
        """
        rows, cols = utility.shape if utility.ndim > 1 else (1, utility.size)

        # Create a 2D array of ranks with the same shape as the utility array
        ranks = np.empty((rows, cols), dtype=np.float64)
        for i in range(rows):
            for j in range(cols):
                ranks[i, j] = j + 1

        # Compute the discount
        discount = np.log2(ranks + 1)

        # Compute 1 / discount.
        discount = np.reciprocal(discount)

        # Multiply with utilities and sum.
        dcg: float = np.sum(utility * discount, axis=1)

        return dcg

    def name(self) -> str:
        return f"NDCG@{self.top_k}"
