from main.eval.metrics.metric import (
    RankingMetric,
    MetricDependency,
)
import numpy as np
from typing import List, Dict, Optional


class Novelty(RankingMetric):
    def __init__(self):
        super().__init__()

        self.item_popularity: Optional[Dict[str, int]] = None
        self.sum_popularity: Optional[int] = None

    def state_init(self) -> float:
        self.item_popularity: Dict[str, int] = self.get_dependency(
            MetricDependency.ITEM_COUNT
        )
        self.sum_popularity: int = sum(self.item_popularity.values())

        return super().state_init()

    def eval_sample(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        intersect: np.ndarray,
        sample_id: int,
    ) -> float:
        """Compute 'novelty' metric of a sample.

        Based on the paper:
            Zhou, T., Kuscsik, Z., Liu, J. G., Medo, M., Wakeling, J. R.,
            & Zhang, Y. C. (2010). Solving the apparent diversity-accuracy dilemma of
            recommender systems. Proceedings of the National Academy of Sciences,
            107(10), 4511-4515.

        We compute novely for each user as the negative log of the
        relative item popularity. This is also known as 'self-information` of an
        item (Zhou et al.). Intuitively, popular items get 'punished' by a low
        self-information whereas novel items have high self-information.

        More formally:
        $$p_i = \dsum_u^|U|{r\_{ui}} * \dfraq{1}{|U|}$$
        where $\dsum_u^|U|{r\_{ui}}$ sums the amount of times item $i$ was bought.
        Then $-log_2{(p_i)}$ is self-information. To compute overall novelty, we compute
        the self-information for each recommendation and average it.
        """

        novelty = 0

        # For each recommendation get the negative log of the relative item popularity.
        for i in predictions:
            if i in self.item_popularity:
                novelty += -np.log2(self.item_popularity[i] / self.sum_popularity)

        if len(predictions) == 0:
            return 0

        return novelty / len(predictions)

    def get_required_dependencies(self) -> List[MetricDependency]:
        return super().get_required_dependencies() + [MetricDependency.ITEM_COUNT]

    def name(self):
        return f"Novelty@{self.top_k}"
