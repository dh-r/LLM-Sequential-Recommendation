"""This module contains the Session-Based kNN recommender."""

import math
import time
from collections import defaultdict
from typing import Callable, Optional

import numpy as np
import pandas as pd
from multiprocess.pool import Pool, AsyncResult
from scipy.sparse import csr_matrix
from sklearn.metrics import pairwise
from tqdm import tqdm

from main.utils import similarity
from main.abstract_model import Model
from main.utils.id_reducer import IDReducer

global_item_original_to_reduced: Optional[dict[int, int]] = None
global_item_reduced_to_original: Optional[np.ndarray] = None
global_session_item_index: Optional[csr_matrix] = None

global_time_ordered_session_ids: Optional[np.ndarray] = None
global_sequential_items_dict: Optional[dict[int, np.ndarray]] = None
global_idf_dict: Optional[dict[int, float]] = None
global_session_times: Optional[np.ndarray] = None
global_predict_data: Optional[dict[int, np.ndarray]] = None

global_rng: Optional[np.random.Generator] = None


class SessionBasedCF(Model):
    """A class representing the Session-Based kNN recommender. This class relies on the
    SessionDataset class to manage training and prediction data.

    Attributes:
        k: An integer representing the number of closest neighbors to use. Must be a
            positive integer smaller than or equal to sample_size.
        sample_size: An integer representing the number of potential neighbors sampled
            to calculate the nearest neighbors. Must be a positive integer.
        sampling: A string indicating the sampling approach for the potential neighbor
            sessions. The options are random and recent.
        sample_random_state: An integer indicating the random seed to use for random
            sampling. Defaults to None, in which case the seed is selected randomly.
        sequential_weighting: A boolean indicating whether the items that are common
            between a prompt session and a candidate session are weighted by how recent
            the common item appears in the prompt session.
        sequential_filter: A boolean indicating whether the candidate items should be
            filtered based on whether they appeared, in any training session, after the
            last item of the prompt session.
        decay: A string indicating what type of decay to apply to the items of the
            prompt sessions for V-SKNN variant. Defaults to None, in which a decay is
            not applied.
        similarity_measure: A string indicating the type of similarity measure to use.
        filter_prompt_items: A boolean indicating whether the items of the prompt
            session should be filtered out from the recommendations.
        idf_weighting: A boolean indicating whether the scores of candidate items are
            to be discounted by their inverse document frequency.
        session_time_discount: A float indicating the discount applied to the influence
            of the neighbor sessions on candidate item scores based on how old a
            neighboring session is.
        item_original_to_reduced: An integer that maps original item ids to reduced item
            ids. Reduced item ids are consecutive integers that prevent original ids
            from resulting in empty columns in the session-item index. It also allows
            the algorithm to know whether a prompt session contains items that are not
            learned.
        item_reduced_to_original: A numpy array that maps reduced ids, through the index
            of the array, to original ids.
        session_item_index: A sparse csr_matrix used to keep track of what sessions
            contain what items, and vice versa. The rows are sessions, the columns are
            items, and the data are binary indicators.
    """

    def __init__(
        self,
        is_verbose: bool = False,
        cores: int = 1,
        k: int = 100,
        sample_size: int = 500,
        sampling: str = "random",
        sample_random_state: int = None,
        sequential_weighting: bool = False,
        sequential_filter: bool = False,
        decay: str = None,
        similarity_measure: str = "cosine",
        filter_prompt_items: bool = True,
        idf_weighting: bool = False,
        session_time_discount: float = None,
    ) -> None:
        if cores < 1:
            raise ValueError(
                f"The number of cores must be a positive integer. Got {cores}"
            )
        if k < 1 or k > sample_size:
            raise ValueError(
                "k must be a positive integer smaller than or equal to sample_size."
                f"Got k: {k} and sample_size: {sample_size}"
            )
        if sample_size < 1:
            raise ValueError(
                f"sample_size must be a positive integer. Got {sample_size}"
            )
        if sampling not in ["random", "recent"]:
            raise ValueError(
                "The supported sampling strategies are random and recent. "
                f"Got {sampling}"
            )
        if decay is not None and similarity_measure != "dot":
            raise ValueError(
                f"A decay requires the similarity measure to be the dot product. Got {decay}"
            )
        super().__init__(is_verbose, cores)
        # Model settings.
        self.k: int = k
        self.sample_size: int = sample_size
        self.sampling: str = sampling
        self.sample_random_state: Optional[int] = sample_random_state
        self.sequential_weighting: bool = sequential_weighting
        self.sequential_filter: bool = sequential_filter
        self.decay: Optional[str] = decay
        self.similarity_measure: str = similarity_measure
        self.filter_prompt_items: bool = filter_prompt_items
        self.idf_weighting: bool = idf_weighting
        self.session_time_discount: Optional[float] = session_time_discount

        # Lookups
        self.item_original_to_reduced: Optional[dict[int, int]] = None
        self.item_reduced_to_original: Optional[np.ndarray] = None
        self.session_item_index: Optional[csr_matrix] = None

    def train(self, train_data: pd.DataFrame) -> None:
        """Trains the model with the session data given in the DataFrame.

        Args:
            train_data: A pandas DataFrame containing the session data.
        """
        train_time_start: float = time.perf_counter()

        global global_rng
        global_rng = np.random.default_rng(self.sample_random_state)

        session_id_reducer: IDReducer = IDReducer(train_data, "SessionId")
        train_data = session_id_reducer.to_reduced(train_data)

        item_id_reducer: IDReducer = IDReducer(train_data, "ItemId")
        train_data = item_id_reducer.to_reduced(train_data)
        self.item_original_to_reduced = item_id_reducer.id_reverse_lookup
        self.item_reduced_to_original = item_id_reducer.get_to_original_array()
        global global_item_original_to_reduced, global_item_reduced_to_original
        global_item_original_to_reduced = self.item_original_to_reduced
        global_item_reduced_to_original = self.item_reduced_to_original

        # Configuration dependant training follows.
        if self.idf_weighting:
            global global_idf_dict
            global_idf_dict = SessionBasedCF.compute_idf_dict(train_data)

        if self.sampling == "recent":
            session_times: np.ndarray = SessionBasedCF.get_session_times(train_data)
            ordered_session_ids: np.ndarray = np.argsort(session_times)
            global global_time_ordered_session_ids
            global_time_ordered_session_ids = ordered_session_ids

        if self.session_time_discount is not None:
            session_times: np.ndarray = SessionBasedCF.get_session_times(train_data)
            global global_session_times
            global_session_times = session_times

        if self.sequential_filter:
            global global_sequential_items_dict
            global_sequential_items_dict = SessionBasedCF.get_sequential_items_dict(
                train_data
            )

        # Construct the session-item index.
        self.session_item_index = SessionBasedCF.construct_session_item_index(
            train_data
        )
        global global_session_item_index
        global_session_item_index = self.session_item_index

        # Training is done.
        self.is_trained = True
        train_time_finish: float = time.perf_counter()
        print(f"Training took {(train_time_finish - train_time_start):.3f} seconds.")

    @staticmethod
    def get_session_times(df: pd.DataFrame) -> np.ndarray:
        """Returns the timestamps of the sessions found in the given DataFrame.

        Args:
            df: A pandas DataFrame containing the session data.

        Returns:
            A numpy array containing the timestamps of sessions.
        """
        # The time value at index i corresponds to session with reduced id of i, as the
        # reduced ids are just order of appearances.
        return df.groupby("SessionId")["Time"].max().values

    @staticmethod
    def compute_idf_dict(df: pd.DataFrame) -> dict[int, float]:
        """Computes the inverse document frequency values of the items found in the
        given DataFrame. The idf for an item in this context is the log of the number of
        unique sessions divided by the number of sessions in which the said item
        appears.

        Args:
            df: A pandas DataFrame containing session data.

        Returns:
            A python dictionary that maps item ids to idf values.
        """
        session_count: int = df["SessionId"].nunique()
        item_counts: dict[int, int] = (
            df.drop_duplicates(subset=["SessionId", "ItemId"])["ItemId"]
            .value_counts()
            .to_dict()
        )
        idf_dict: dict[int, float] = {
            k: 1 + math.log(session_count / v) for k, v in item_counts.items()
        }
        return idf_dict

    @staticmethod
    def get_sequential_items_dict(df: pd.DataFrame) -> dict[int, np.ndarray]:
        """Returns a dictionary that maps each item, via its id as key, to an array
        containing ids of all the items that follow the given item in any of the
        training sessions.

        Args:
            df: A pandas DataFrame containing the session data.

        Returns:
            A python dictionary with integer keys and numpy array values.
        """
        item_seq_dict: dict[int, set] = defaultdict(set)
        for _, session_df in df.groupby("SessionId"):
            item_ids: np.ndarray = session_df["ItemId"].values
            for i, item_id in enumerate(item_ids[:-1]):
                item_seq_dict[item_id].add(item_ids[i + 1])
        return {k: np.fromiter(v, int, len(v)) for k, v in item_seq_dict.items()}

    @staticmethod
    def construct_session_item_index(df: pd.DataFrame) -> csr_matrix:
        """Constructs csr_matrix that contains the training sessions as rows and
        training items as columns, where the data is binary indicator of item appearance
        in the session.

        Args:
            df: A pandas DataFrame containing the session data.

        Returns:
            A csr_matrix containing the session-item index.
        """
        rows: np.ndarray = np.array(df["SessionId"].to_numpy(), dtype=int)
        cols: np.ndarray = np.array(df["ItemId"].to_numpy(), dtype=int)
        data: np.ndarray = np.array(df["Reward"].to_numpy(), dtype=int)
        n_sessions: int = df["SessionId"].nunique()
        n_items: int = df["ItemId"].nunique()
        return csr_matrix((data, (rows, cols)), shape=(n_sessions, n_items))

    def predict(
        self, predict_data: dict[int, np.ndarray], top_k: int = 10
    ) -> dict[int, np.ndarray]:
        """Generates predictions, i.e. recommendations, for the sessions in the given
        dictionary.

        Args:
            predict_data: A dictionary that maps session ids to items in the session
                represented as a numpy array.
            top_k: An integer representing the top k.

        Returns:
            A dictionary that maps the same session ids to item recommendations
            represented as numpy arrays.
        """
        predict_time_start: float = time.perf_counter()
        if not self.is_trained:
            raise Exception("The model has not been trained yet.")
        global global_predict_data
        global_predict_data = predict_data

        # Distribute prompts among cores.
        prompt_sessions: list[int] = list(predict_data.keys())
        q, r = divmod(len(prompt_sessions), self.cores)
        core_state_index: list[int] = [0]
        for i in range(self.cores):
            start: int = core_state_index[i]
            chunk_size: int = q + (i < r)
            end: int = start + chunk_size
            core_state_index.append(end)

        with Pool(self.cores) as pool:
            chunk_results: list[AsyncResult] = []
            for i in range(self.cores):
                start: int = core_state_index[i]
                end: int = core_state_index[i + 1]
                chunk_results.append(
                    pool.apply_async(
                        SessionBasedCF._predict_chunk,
                        (
                            prompt_sessions[start:end],
                            self.k,
                            self.sample_size,
                            self.sampling,
                            self.sequential_weighting,
                            self.sequential_filter,
                            self.decay,
                            self.similarity_measure,
                            self.filter_prompt_items,
                            self.idf_weighting,
                            top_k,
                        ),
                    )
                )

            predictions: dict[int, np.ndarray] = {}
            for res in tqdm(
                chunk_results, desc="Predicted Chunks", disable=not self.is_verbose
            ):
                chunk_predictions: dict[int, np.ndarray] = res.get()
                predictions |= chunk_predictions

        predict_time_finish: float = time.perf_counter()
        print(
            f"Prediction took {(predict_time_finish - predict_time_start):.3f} seconds."
        )
        return predictions

    @staticmethod
    def _predict_chunk(
        chunk_sessions: list[int],
        k: int,
        sample_size: int,
        sampling: str,
        sequential_weighting: bool,
        sequential_filter: bool,
        decay: Optional[str],
        similarity_measure: str,
        filter_prompt_items: bool,
        idf_weighting: bool,
        top_k: int,
    ) -> dict[int, np.ndarray]:
        """Generates predictions, i.e. recommendations, for the sessions contained in
        the given list.

        Args:
            chunk_sessions: A list that contains the sessions that belong to this chunk.
            k: An integer indicating the choice of k in k-nearest neighbors.
            sample_size: An integer indicating the number of sampled neighbors.
            sampling: A string indicating the sampling approach for fetching the
                potential set of neighbor sessions.
            sequential_weighting: A boolean indicating whether sequential weighting
                is applied to candidate items.
            sequential_filter: A boolean indicating whether sequential filtering is
                applied to the candidate items.
            decay: A string representing the decay type.
            similarity_measure: A string indicating the type of similarity measure to
                use.
            filter_prompt_items: A boolean indicating whether the items of the prompt
                session are filtered out from the candidate items.
            idf_weighting: A boolean indicating the use of idf weights on candidate item
                scoring.
            top_k: An integer representing the top k.

        Returns:
            A dictionary that maps the same session ids to item recommendations
            represented as numpy arrays.
        """
        chunk_predictions: dict[int, np.ndarray] = {}
        for session_id in chunk_sessions:
            session: np.ndarray = global_predict_data[session_id]
            chunk_predictions[session_id] = SessionBasedCF._predict_session(
                session,
                k,
                sample_size,
                sampling,
                sequential_weighting,
                sequential_filter,
                decay,
                similarity_measure,
                filter_prompt_items,
                idf_weighting,
                top_k,
            )
        return chunk_predictions

    @staticmethod
    def _predict_session(
        session: np.ndarray,
        k: int,
        sample_size: int,
        sampling: str,
        sequential_weighting: bool,
        sequential_filter: bool,
        decay: Optional[str],
        similarity_measure: str,
        filter_prompt_items: bool,
        idf_weighting: bool,
        top_k: int,
    ) -> np.ndarray:
        """Generates a prediction, i.e. recommendations, for the given session.

        Args:
            session: A numpy array that contains the prompt session's original item ids.
            k: An integer indicating the choice of k in k-nearest neighbors.
            sample_size: An integer indicating the number of sampled neighbors.
            sampling: A string indicating the sampling approach for fetching the
                potential set of neighbor sessions.
            sequential_weighting: A boolean indicating whether sequential weighting
                is applied to candidate items.
            sequential_filter: A boolean indicating whether sequential filtering is
                applied to the candidate items.
            decay: A string representing the decay type.
            similarity_measure: A string indicating the type of similarity measure to
                use.
            filter_prompt_items: A boolean indicating whether the items of the prompt
                session are filtered out from the candidate items.
            idf_weighting: A boolean indicating the use of idf weights on candidate item
                scoring.
            top_k: An integer representing the top k.

        Returns:
            A numpy array containing the original ids of the recommended items.
        """
        reduced_session: np.ndarray = SessionBasedCF._get_reduced_session(session)
        # In case the prompt session includes only non-trained items.
        if reduced_session.size == 0:
            return np.array([])

        potential_neighbors_matrix: csr_matrix = (
            SessionBasedCF._get_potential_neighbors(
                reduced_session, sample_size, sampling
            )
        )
        kn_matrix, kn_similarities = SessionBasedCF._get_k_neighbors(
            potential_neighbors_matrix,
            reduced_session,
            k,
            decay,
            similarity_measure,
        )

        candidate_items: np.ndarray = np.unique(kn_matrix.nonzero()[1])
        if filter_prompt_items:
            candidate_items: np.ndarray = candidate_items[
                ~np.isin(candidate_items, reduced_session)
            ]

        sequential_weights: Optional[np.ndarray] = (
            SessionBasedCF._get_sequential_weights(kn_matrix, reduced_session)
            if sequential_weighting
            else None
        )

        if sequential_filter:
            candidate_items = SessionBasedCF._apply_sequential_filter(
                candidate_items, reduced_session
            )
            if len(candidate_items) == 0:
                return np.array([])

        scores: np.ndarray = SessionBasedCF._compute_candidate_scores(
            candidate_items,
            kn_matrix,
            kn_similarities,
            sequential_weights,
            idf_weighting,
        )
        sorted_idx: np.ndarray = np.argsort(scores)[::-1]
        sorted_items: np.ndarray = candidate_items[sorted_idx]
        return global_item_reduced_to_original[sorted_items[:top_k]]

    @staticmethod
    def _get_reduced_session(original_session: np.ndarray) -> np.ndarray:
        """Returns the reduced id encoding of the given session.

        Args:
            original_session: A numpy array containing the original ids of a session.

        Returns:
            A numpy array containing the reduced ids of the given session.
        """
        reduced_item_ids: list[int] = [
            global_item_original_to_reduced[item]
            # We apply pd.unique to filter out duplicate items in the prompt session.
            for item in pd.unique(original_session)
            if item in global_item_original_to_reduced
        ]
        return np.array(reduced_item_ids)

    @staticmethod
    def _get_potential_neighbors(
        session: np.ndarray, sample_size: int, sampling: str
    ) -> csr_matrix:
        """Returns a sample of the sessions that contain at least one item from the
        given session.

        Args:
            session: A numpy array of reduced item ids representing the prompt session.
            sample_size: An integer indicating the number of sampled neighbors.
            sampling: A string indicating the sampling approach for fetching the
                potential set of neighbor sessions.

        Returns:
            A sparse csr_matrix containing the potential sessions.
        """
        possible_neighbors: np.ndarray = SessionBasedCF._get_possible_neighbors(session)
        sampled_neighbors_index: np.ndarray = SessionBasedCF._sample_neighbors(
            session, possible_neighbors, sample_size, sampling
        )
        return global_session_item_index[sampled_neighbors_index, :]

    @staticmethod
    def _get_possible_neighbors(session: np.ndarray) -> np.ndarray:
        """Returns all sessions that contain at least one item from the given session.

        Args:
            session: A numpy array containing reduced ids of a session.

        Returns:
            A numpy array containing the ids of the possible neighbor sessions.
        """
        return np.unique(global_session_item_index[:, session].nonzero()[0])

    @staticmethod
    def _sample_neighbors(
        session: np.ndarray,
        possible_neighbors: np.ndarray,
        sample_size: int,
        sampling: str,
    ) -> np.ndarray:
        """Returns a sample of the given possible neighbors based on the given sampling
        strategy. There are currently 2 options: random and recent.

        Args:
            session: A numpy array of reduced item ids representing the prompt session.
            possible_neighbors: A numpy array containing the reduced ids of the possible
                neighbors.
            sample_size: An integer indicating the sample size.
            sampling: A string indicating the sampling approach for fetching the
                potential set of neighbor sessions.

        Returns:
            A numpy array containing the reduced ids of the sampled sessions.

        Raises:
            ValueError: If the given sampling strategy is not supported.
        """
        if sampling == "random":
            return SessionBasedCF._sample_neighbors_random(
                possible_neighbors, sample_size
            )
        elif sampling == "recent":
            return SessionBasedCF._sample_neighbors_recent(
                possible_neighbors, sample_size
            )
        else:
            raise ValueError(
                f"The given sampling strategy of {sampling} is not supported."
            )

    @staticmethod
    def _sample_neighbors_random(
        possible_neighbors: np.ndarray, sample_size: int
    ) -> np.ndarray:
        """Returns a random sample from the given possible neighbors.

        Args:
            possible_neighbors: A numpy array containing the reduced ids of the possible
                neighbors.
            sample_size: An integer indicating the sample size.

        Returns:
            A numpy array containing the ids of the sampled sessions.
        """
        return global_rng.choice(
            possible_neighbors,
            size=min(sample_size, len(possible_neighbors)),
            replace=False,
        )

    @staticmethod
    def _sample_neighbors_recent(
        possible_neighbors: np.ndarray, sample_size: int
    ) -> np.ndarray:
        """Returns a sample from the given possible neighbors based on recency, where
        more recent sessions are added to the sample.

        Args:
            possible_neighbors: A numpy array containing the reduced ids of the possible
                neighbors.
            sample_size: An integer indicating the sample size.

        Returns:
            A numpy array containing the ids of the sampled sessions.
        """
        return global_time_ordered_session_ids[
            np.isin(global_time_ordered_session_ids, possible_neighbors)
        ][-sample_size:]

    @staticmethod
    def _get_k_neighbors(
        potential_neighbors_matrix: csr_matrix,
        session: np.ndarray,
        k: int,
        decay: Optional[str],
        similarity_measure: str,
    ) -> tuple[csr_matrix, np.ndarray]:
        """Returns a matrix containing the k nearest neighbor session for the given
        session and the similarity scores of each of these neighbor session with the
        given session.

        Args:
            potential_neighbors_matrix: A csr_matrix containing the k-nearest sessions.
            session: A numpy array of reduced item ids representing a session.
            k: An integer representing the k in k-nearest.
            decay: A string representing the decay type.
            similarity_measure: A string indicating the type of similarity measure to
                use.

        Returns:
            A tuple that contains the k-nearest neighbors matrix as first element and
            a numpy array containing the similarity scores as the second element. The
            indices of the session between these two objects are the same.
        """
        session_row_matrix: csr_matrix = SessionBasedCF._get_session_row_matrix(
            session, decay
        )
        similarities: np.ndarray = SessionBasedCF._compute_similarity(
            potential_neighbors_matrix, session_row_matrix, similarity_measure
        )
        k_neighbors_index: np.ndarray = np.argpartition(
            -similarities, min(k, len(similarities) - 1)
        )[:k]
        kn_matrix: csr_matrix = potential_neighbors_matrix[k_neighbors_index, :]
        kn_similarities: np.ndarray = similarities[k_neighbors_index]
        return kn_matrix, kn_similarities

    @staticmethod
    def _get_session_row_matrix(
        session: np.ndarray, decay: Optional[str]
    ) -> csr_matrix:
        """Returns the given session as a sparse matrix of a single row, where the
        columns are the all reduced ids training items. The values of the matrix are all
        set to 1 if there is no decay. Else, the values are determined by the decay
        applied to the items based on their position in the session.

        Args:
            session: A numpy array containing the reduced ids of the session items.
            decay: A string indicating whether a decay is applied to the session items.

        Returns:
            A sparse matrix with a single row containing the given session.
        """
        data: np.ndarray = SessionBasedCF._decay(decay, len(session))
        indices: np.ndarray = np.array(session)
        indptr: np.ndarray = np.array([0, len(data)])
        session_row_matrix: csr_matrix = csr_matrix(
            (data, indices, indptr), shape=(1, global_session_item_index.shape[1])
        )
        return session_row_matrix

    @staticmethod
    def _decay(decay: Optional[str], length: int) -> np.ndarray:
        """Generates a numpy array whose values decay by the given decay function. The
        values decay from right to left where the starting value is 1.

        Args:
            decay: A string indicating the decay function. If None, an array of ones is
                returned.
            length: An integer indicating the length of the returned array.

        Returns:
            A numpy array holding the decaying values.

        Raises:
            ValueError: If the given decay function is not recognized.
        """
        if decay is None:
            return np.ones(length, dtype=int)

        descending_ints: list[int] = list(range(length, 0, -1))
        if decay == "linear":
            return np.array([max(0.0, 1.0 - (i - 1) * 0.1) for i in descending_ints])
        elif decay == "log":
            return np.array([1 / (np.log(i) + 1) for i in descending_ints])
        elif decay == "harmonic":
            return np.array([1 / i for i in descending_ints])
        elif decay == "quadratic":
            return np.array([1 / i**2 for i in descending_ints])
        else:
            raise ValueError(f"The given decay function of {decay} is not recognized.")

    @staticmethod
    def _compute_similarity(
        candidates: csr_matrix,
        target: csr_matrix,
        similarity_measure: str,
    ) -> np.ndarray:
        """Computes the similarity of each row in candidates to the target matrix, which
        consists of only a single row representing the target session.

        Args:
            candidates: A csr_matrix representing the candidates session to compute
                similarity for.
            target: A csr_matrix with a single row that represents the target session.
            similarity_measure: A string indicating what similarity measure to use.

        Returns:
            A numpy array of shape (n_candidates,) containing the similarity score of
            each respective candidate.
        """
        if similarity_measure == "cosine":
            return pairwise.cosine_similarity(candidates, target).ravel()
        elif similarity_measure == "dot":
            return pairwise.linear_kernel(candidates, target).ravel()
        else:
            vec_sim_func: Callable = np.vectorize(
                similarity.get_similarity_func(similarity_measure),
                signature="(n),(n)->()",
            )
            return vec_sim_func(candidates.toarray(), target.toarray().ravel())

    @staticmethod
    def _get_sequential_weights(
        kn_matrix: csr_matrix, session: np.ndarray
    ) -> np.ndarray:
        """Computes the sequential weights for each session in the kn_matrix, i.e. the
        neighboring sessions. The sequential weights is the following: for a given
        prompt session s and candidate session c, if item s_x at position x (1-indexed)
        is the most recent, in s, common item between s and c, then the sequential
        weight of c is (x / |s|).

        Args:
            kn_matrix: A csr_matrix containing the neighbor sessions, in rows, and the
                items of these sessions, in columns, and binary data representing an
                item's presence at a given session. Note that the session indices do not
                correspond to reduced session ids.
            session: A numpy array of reduced item ids representing a session.

        Returns:
            A numpy array containing the sequential weights of the sessions in the order
            that they appear in the matrix.
        """
        session_length: int = session.size
        neighbors: np.ndarray = np.unique(kn_matrix.nonzero()[0])
        sequential_weights: np.ndarray = np.zeros_like(neighbors, dtype=float)
        for i, neighbor in enumerate(neighbors):
            neighbor_items: np.ndarray = kn_matrix.getrow(neighbor).nonzero()[1]
            common_idx: np.ndarray = np.intersect1d(
                session, neighbor_items, assume_unique=True, return_indices=True
            )[1]
            sequential_weights[i] = (np.max(common_idx) + 1) / session_length
        return sequential_weights

    @staticmethod
    def _apply_sequential_filter(candidate_items, session: np.ndarray) -> np.ndarray:
        """Applies sequential filtering to the given candidate items. Sequential
        filtering filters out candidate items that do not follow the last item of the
        given session in any of the historic sessions.

        Args:
            candidate_items: A numpy array containing the reduced ids of the candidate
                items.
            session: A numpy array containing the reduced item ids of the session to use
                for filtering.

        Returns:
            A numpy array containing the ids of the items that remained after filtering.
        """
        last_item: int = session[-1]
        if last_item not in global_sequential_items_dict:
            return np.array([])
        following_items: np.ndarray = global_sequential_items_dict[last_item]
        filtered_items: np.ndarray = candidate_items[
            np.isin(candidate_items, following_items)
        ]
        return filtered_items

    @staticmethod
    def _compute_candidate_scores(
        candidate_items: np.ndarray,
        kn_matrix: csr_matrix,
        kn_similarities: np.ndarray,
        sequential_weights: Optional[np.ndarray],
        idf_weighting: bool,
    ) -> np.ndarray:
        """Computes the item score for each given candidate item.

        Args:
            candidate_items: A numpy array containing the ids of candidate items.
            kn_matrix: A csr_matrix containing k-nearest neighbor sessions as rows and
                their items as columns where the data is a binary indicator of item
                appearance.
            kn_similarities: A numpy array containing the similarity score of the
                k-nearest neighboring sessions. The indexing corresponds to the order of
                the rows in the kn_matrix parameter.
            sequential_weights: An optional numpy array containing the sequential
                weights of the k-nearest neighbors. The indexing corresponds to the
                order of the rows in the kn_matrix parameter.
            idf_weighting: A boolean indicating whether the candidate items are to be
                weighted by their inverse document frequency.

        Returns:
            A numpy array containing the scores of candidate items, where the indexing
            corresponds to the order of appearance in the candidate_items parameter.
        """
        scores: list[float] = []
        for item in candidate_items:
            containing_sessions: np.ndarray = kn_matrix.getcol(item).nonzero()[0]
            similarities: np.ndarray = kn_similarities[containing_sessions]
            if sequential_weights is not None:
                similarities = np.multiply(
                    similarities, sequential_weights[containing_sessions]
                )
            # TODO session time weighting for neighbor sessions

            # noinspection PyTypeChecker
            score: float = np.sum(similarities)
            if idf_weighting:
                score *= global_idf_dict[item]
            scores.append(score)
        return np.array(scores)

    def name(self) -> str:
        """Returns the name of the model.

        Returns:
            A string representing the name of the model.
        """
        if self.decay is not None:
            return "V-SKNN"
        elif self.sequential_filter:
            return "SF-SKNN"
        elif self.sequential_weighting:
            return "S-SKNN"
        else:
            return "SKNN"
