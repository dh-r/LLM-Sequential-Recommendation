from abc import abstractmethod
from main.reranking.reranker import Reranker
import numpy as np
from typing import Any, Union
from main.utils.split_dict import split_dict
from main.utils.multiprocessing import execute_function_on_threads
import logging
from tqdm import tqdm


class BatchReranker(Reranker):
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
        if self.cores == 1:
            reranked_candidates, no_candidates_count = self._rerank_batch(
                candidates, top_k
            )
        else:
            candidate_chunks: list[dict[Any, Union[np.ndarray, list]]] = split_dict(
                candidates, self.cores
            )
            reranked_candidates, no_candidates_count = execute_function_on_threads(
                num_threads=self.cores,
                function=self._rerank_batch,
                function_args=[[c, top_k] for c in candidate_chunks],
                results_aggregate_init=({}, 0),
                results_aggregate_function=lambda x, y: (x[0] | y[0], x[1] + y[1]),
            )

        logging.warning(
            f"For {no_candidates_count} out of {len(candidates)} we could not rerank "
            f"due to the lack of reranker candidates. "
        )

        return reranked_candidates

    def _rerank_single(
        self, candidate_id: Any, model_candidates: np.ndarray, top_k: int
    ) -> tuple[np.ndarray, int]:
        pass

    def _rerank_batch(
        self, model_candidates: dict[Any, Union[np.ndarray, list]], top_k: int
    ) -> tuple[dict[Any, np.ndarray], int]:
        """ """
        reranked_candidates: dict[Any, np.ndarray] = {}
        no_new_candidates_all = 0

        for candidate_id, candidates in tqdm(
            model_candidates.items(), disable=not self.is_verbose, mininterval=10
        ):
            # Sometimes candidates are in list format, then we need to
            # convert to ndarray.
            if isinstance(candidates, list):
                candidates = np.array(candidates)

            # If the model candidates are empty, we cannot rerank.
            if len(candidates) == 0:
                reranked_candidates[candidate_id] = candidates
                continue

            # Rerank for this candidate.
            new_candidates, no_new_candidates = self._rerank_single(
                candidate_id, candidates, top_k=top_k
            )

            # Update state.
            no_new_candidates_all += no_new_candidates
            reranked_candidates[candidate_id] = new_candidates

        return reranked_candidates, no_new_candidates_all

    @abstractmethod
    def name(self) -> str:
        pass
