from main.eval.metrics.metric import RankingMetric
import numpy as np
from typing import List


class MeanReciprocalRank(RankingMetric):
    def eval_sample(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        intersect: np.ndarray,
        sample_id: int,
    ) -> float:
        reciprocal_top_k = np.reciprocal(np.arange(1, self.top_k + 1, dtype=np.float64))

        if len(intersect) == 0:
            return 0

        # Build top-k ratings where all irrelevant items are set to zero and all relevant to 1.
        intersect_mask = np.pad(
            np.in1d(predictions, intersect, assume_unique=True).astype(np.int32),
            (0, self.top_k - len(predictions)),
        )

        # Compute reciprocal rank multiplied with the intersect mask and sum it.
        return np.sum(reciprocal_top_k * intersect_mask) / len(intersect)

    def eval_bulk(
        self,
        predictions: List[np.ndarray],
        ground_truth: List[np.ndarray],
        intersect: List[np.ndarray],
        sample_ids: np.ndarray,
    ) -> np.ndarray:
        reciprocal_top_k_all = np.repeat(
            [np.reciprocal(np.arange(1, self.top_k + 1, dtype=np.float64))],
            len(predictions),
            axis=0,
        )

        intersect_mask_all = np.vstack(
            (
                np.pad(
                    np.isin(pred, inter, assume_unique=True).astype(np.int32),
                    (0, self.top_k - len(pred)),
                )
                for pred, inter in zip(predictions, intersect)
            )
        )

        intersect_count = np.array([len(inter) for inter in intersect])

        return np.divide(
            np.sum(reciprocal_top_k_all * intersect_mask_all, axis=1),
            intersect_count,
            out=np.zeros_like(intersect_count, dtype=np.float64),
            where=intersect_count != 0,
        )

    def name(self) -> str:
        return f"MRR@{self.top_k}"
