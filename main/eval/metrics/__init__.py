from main.eval.metrics.mrr import MeanReciprocalRank
from main.eval.metrics.hitrate import HitRate
from main.eval.metrics.serendipity import Serendipity
from main.eval.metrics.novelty import Novelty
from main.eval.metrics.ndcg import (
    NormalizedDiscountedCumulativeGain,
)
from main.eval.metrics.catalog_coverage import CatalogCoverage
from main.eval.metrics.metric import (
    RankingMetric,
    MetricDependency,
)

ALL_DEFAULT = [
    NormalizedDiscountedCumulativeGain(),
    HitRate(),
    MeanReciprocalRank(),
    CatalogCoverage(),
    Serendipity(),
    Novelty(),
]

BEYOND_ACCURACY = [Serendipity(), Novelty(), CatalogCoverage()]

ALL_RANKING = [
    NormalizedDiscountedCumulativeGain(),
    HitRate(),
    MeanReciprocalRank(),
]

ALL = [
    NormalizedDiscountedCumulativeGain(),
    HitRate(),
    MeanReciprocalRank(),
    CatalogCoverage(),
    Serendipity(),
    Novelty(),
]
