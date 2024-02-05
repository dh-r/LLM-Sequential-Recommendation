from scipy import sparse
import numpy as np
from typing import Any, Union, Optional
import logging
from main.reranking.reranker_batch import BatchReranker
from tqdm import tqdm


class PairwiseSimilarityFilterReranker(BatchReranker):
    def __init__(
        self,
        similarity_matrix: Union[np.ndarray, sparse.csr_matrix],
        similarity_threshold: float = 0.9,
        is_verbose: bool = False,
        cores: int = 1,
    ):
        """The diversity reranker is initialized to enhance the diversity of
            recommendations by excluding pairwise item similarities that exceed a
            specified threshold in the given similarity matrix.
            This filtering process is applied to all the recommendations in the list,
            following the order of the original model candidates.

        Arguments:
            similarity_matrix (Union[np.ndarray, sparse.csr_matrix]): a (sparse)
                item x items matrix representing pair-wise similarity. We assume this
                matrix is symmetric.
            similarity_threshold (float, optional): specifies the maximum similarity
                score between two items allowed in the list of recommendations. Ensure
                this threshold is between 0 and 1.0.
            is_verbose (bool, optional): A boolean indicating whether the console output
                of this reranker will be verbose or not.
            cores (int, optional): The amount of cores to use, if applicable.
        """
        super().__init__(is_verbose, cores)
        self.similarity_matrix: Union[np.ndarray, sparse.csr_matrix] = similarity_matrix
        self.similarity_threshold: float = similarity_threshold

        if not 0 < similarity_threshold < 1.0:
            raise ValueError("Similarity threshold needs to be between 0 and 1.")

    def _rerank_single(
        self, user_id: Any, model_candidates: np.ndarray, top_k: int
    ) -> tuple[np.ndarray, int]:
        """In this step, we apply the diversity reranker to a single list of candidate
        items and limit it to the top-k items. To achieve this, we iterate through the
        `model_candidates` and add each recommendation to the `new_candidates` list only
        if its similarity score to all previously added items is below the specified
        threshold. This ensures that the final list of recommended items includes
        diverse options with minimal similarity to each other.

        Arguments:
            user_id (Any): The identifier of the current candidate.
            model_candidates (np.ndarray): An ordered list of model candidates.
                These are reranked.
            top_k: The top-k re-ranked candidates to keep.

        Returns: Top-k reranked candidates and the amount of users/session for which
            we don't have any (new) reranked candidates.
        """
        new_candidates: int = [model_candidates[0]]
        for rec in model_candidates[1:]:
            # We only have to filter until we found top_k recommendations.
            if len(new_candidates) >= top_k:
                break

            # Select similarities of this recommendation compared to all other recs
            # that we already picked.
            all_similarities: np.ndarray = self.similarity_matrix[rec, new_candidates]

            if np.max(all_similarities) >= self.similarity_threshold:
                # Skip this recommendation because it is too similar to at least 1
                # other recommendation.
                continue

            new_candidates.append(rec)

        # It might be that we don't have enough recommendations after this filter.
        # For now, we don't have any backfill strategy.
        new_candidates: np.ndarray = np.array(new_candidates)

        # Track for how many users/sessions we don't find any reranked candidates.
        no_new_candidates = 0
        if len(new_candidates) > 0:
            no_new_candidates = 1

        # Keep only top-k candidates.
        return new_candidates[:top_k], no_new_candidates

    def name(self) -> str:
        return f"item diversity"
