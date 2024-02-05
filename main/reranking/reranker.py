from abc import ABC, abstractmethod
import numpy as np
from typing import Any


class Reranker(ABC):
    """Abstract implementation representing a re-ranker.

    A re-ranker takes a list of candidates from a model as input and reorders those.
    This reordering might include a filtering step. The goal of the reranker is to
    improve the quality of the recommendations by taking into account additional factors
    that were not considered by the model.
    """

    def __init__(self, is_verbose: bool, cores: int):
        self.is_verbose: bool = is_verbose
        self.cores: int = cores
        self.is_trained: bool = False

    def train(self, train_data: Any):
        """
        Abstract method that trains the reranking model if necessary.

        Args:
            train_data: The training data.
        """
        self.is_trained = True

    @abstractmethod
    def rerank(
        self, candidates: dict[Any, np.ndarray], top_k: int = 20
    ) -> dict[Any, np.ndarray]:
        """
        Abstract method that reranks the candidates.

        Args:
            candidates: A dictionary mapping candidate id's to their candidates.
            top_k: The number of top-k reranked candidates to return.

        Returns:
             A dictionary mapping the candidate id's to their top-k reranked candidates.

        """
        pass

    @abstractmethod
    def name(self) -> str:
        pass
