from main.eval.metrics.metric import RankingMetric
import numpy as np
from typing import Union, Any


class HitRate(RankingMetric):
    def eval_sample(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        intersect: np.ndarray,
        sample_id: int,
    ) -> float:
        return int(len(intersect) > 0)

    def name(self) -> str:
        return f"HitRate@{self.top_k}"
