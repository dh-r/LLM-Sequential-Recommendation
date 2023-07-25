from main.eval.metrics.metric import (
    RankingMetric,
    MetricDependency,
)
from typing import List, Optional, Dict
import numpy as np


class Serendipity(RankingMetric):
    def __init__(self):
        super().__init__()
        self.expected_items: Optional[List[int]] = None

    def state_init(self) -> float:
        item_count: Dict[int, int] = self.get_dependency(MetricDependency.ITEM_COUNT)
        self.expected_items: List[int] = list(
            map(
                int,
                map(
                    float,
                    sorted(item_count, key=item_count.get, reverse=True)[: self.top_k],
                ),
            )
        )
        return super().state_init()

    def eval_sample(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        intersect: np.ndarray,
        sample_id: int,
    ) -> float:
        """Compute 'serendipity' metric of a sample.

        Based on the paper:
            Ge, M., Delgado-Battenfeld, C., & Jannach, D. (2010, September).
            Beyond accuracy: evaluating recommender systems by coverage and serendipity.
            In Proceedings of the fourth ACM conference on Recommender systems
            (pp. 257-260).

        We compute serendipity for each user as the average unexpected relevant
        recommendations. Unexpected items are items _not_ recommend by the
        base recommender. In our case, we opted for the popularity baseline which
        recommends the top-k most popular items to each user.

        First we compute, for user $u$, what is the difference between the ground truths
        and the expected items. This results in a set of unexpected 'test' items:
        $$UNEXP_u = T_u \ PM_u$$
        where T is the ground_truths and PM is a primitive prediction model.
        Then we compute,
        $$SERENDIPITY_u = \dfrac{\sum^{|UNEXP_u|}_i{Pred(i)}}{{|UNEXP_u|}}$$
        where $Pred()$ indicates if a recommendation has been predicted by the
        recommender model such that $Pred(i) = 1$ is a relevant recommendation for
        user $u$ and $Pred(i) = 0$ is irrelevant.

        In other words, this metric answers the following question:
        "of all the unexpected items in the ground truth/test data,
        how many did we recommend?"

        Notes:
        - An advantage of this measure is that it incorporates the relevance
            of a serendipitous recommendation.
        - A related disadvantage, is that it assumes the user actually bought
            a serendipitous item (i.e. it is available in the test set).
        - The usefulness of this metric is directly related to the underlying
            primitive model.
        - Serendipity is a value between 0 and 1. 1 is high serendipity and 0 is low.
        """
        # Compute the set of unexpected recommendations.
        unexpected = np.setdiff1d(ground_truth, self.expected_items)

        # Find unexpected but relevant items.
        unexpected_relevant = 0

        for item in predictions:
            if item in unexpected:
                unexpected_relevant += 1

        # Sometimes there are less predicitons than ground truths,
        # than pick the min length.
        denom = min(len(unexpected), len(predictions))

        # Prevent division by zero.
        if denom == 0:
            return 0

        return unexpected_relevant / denom

    def get_required_dependencies(self) -> List[MetricDependency]:
        return [MetricDependency.ITEM_COUNT]

    def name(self):
        return f"Serendipity@{self.top_k}"
