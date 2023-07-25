from main.eval.metrics.metric import RankingMetric
import numpy as np
from typing import List, Set


class CatalogCoverage(RankingMetric):
    def eval_sample(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        intersect: np.ndarray,
        sample_id: int,
    ) -> Set:
        return set(predictions.tolist())

    def eval_bulk(
        self,
        predictions: List[np.ndarray],
        ground_truth: List[np.ndarray],
        intersect: List[np.ndarray],
        sample_ids: np.ndarray,
    ) -> Set:
        return set(np.concatenate(predictions).tolist())

    def state_init(self) -> Set:
        return set()

    def state_merge(self, current, to_add) -> Set:
        return current.union(to_add)

    def state_merge_bulk(self, current: Set, to_add: Set) -> Set:
        return self.state_merge(current, to_add)

    def state_finalize(self, current) -> float:
        if self.num_items == 0:
            return 0

        return len(current) / self.num_items

    def per_sample(self) -> bool:
        return False

    def name(self) -> str:
        return f"Catalog coverage@{self.top_k}"
