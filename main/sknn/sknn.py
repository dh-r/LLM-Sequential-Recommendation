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

from main.abstract_model import Model
from main.dim_reducer.dim_reducer import DimReducer
from main.exceptions import InvalidStateError
from main.utils import similarity
from main.utils.id_reducer import IDReducer

global_item_original_to_reduced: Optional[dict[int, int]] = None
global_item_reduced_to_original: Optional[np.ndarray] = None
global_item_embeddings: Optional[np.ndarray] = None
global_session_embeddings: Optional[np.ndarray | dict[int, np.ndarray]] = None
global_session_item_index: Optional[csr_matrix] = None
global_precomputed_decay_arrays: Optional[dict[int, np.ndarray]] = None

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
            sessions. The options random, recent, idf, and idf_greedy.
        sample_random_state: An integer indicating the random seed to use for random
            sampling. Defaults to None, in which case the seed is selected randomly.
        use_item_embeddings: A boolean indicating whether to use item embeddings for
            session similarity computation.
        prompt_session_emb_comb_strategy: A string indicating the embedding combination
            strategy for prompt sessions.
        training_session_emb_comb_strategy: A string indicating the embedding
            combination strategy for training sessions.
        dim_reduction_config: A dictionary specifying the dimensionality reduction
            config for item embeddings.
        sequential_weighting: A boolean indicating whether the items that are common
            between a prompt session and a candidate session are weighted by how recent
            the common item appears in the prompt session.
        sequential_filter: A boolean indicating whether the candidate items should be
            filtered based on whether they appeared, in any training session, after the
            last item of the prompt session.
        decay: A string indicating what type of decay to apply to the items of the
            prompt sessions for V-SKNN variant. Defaults to None, in which a decay is
            not applied. Decay options are linear, log, harmonic, and quadratic.
        training_session_decay: A string indicating the type of decay to use for the
            training sessions.
        max_session_length_for_decay_precomputation: An integer indicating the maximum
            session length for which the training precomputes a decay vector.
        similarity_measure: A string indicating the type of similarity measure to use.
        filter_prompt_items: A boolean indicating whether the items of the prompt
            session should be filtered out from the recommendations.
        last_n_items: An integer indicating the number of last items of a prompt session
            to base recommendations on.
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
        use_item_embeddings: bool = False,
        prompt_session_emb_comb_strategy: str = "last",
        training_session_emb_comb_strategy: str = "last",
        dim_reduction_config: dict = None,
        sequential_weighting: bool = False,
        sequential_filter: bool = False,
        decay: str = None,
        training_session_decay: str = None,
        max_session_length_for_decay_precomputation: int = 500,
        similarity_measure: str = "cosine",
        filter_prompt_items: bool = True,
        last_n_items: int = None,
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
        if sampling not in ["random", "recent", "idf", "idf_greedy"]:
            raise ValueError(
                "The supported sampling strategies are random, recent, idf, and"
                f" idf_greedy. Got {sampling}"
            )
        if decay is not None and (
            similarity_measure != "dot" and similarity_measure != "cosine"
        ):
            raise ValueError(
                "A decay requires the similarity measure to be either dot product or"
                f" cosine. Got {decay} as decay."
            )
        if max_session_length_for_decay_precomputation < 1:
            raise ValueError(
                f"Maximum session length for decay precomputation must be an integer"
                f" greater than 0."
            )
        if (
            prompt_session_emb_comb_strategy == "concat"
            and training_session_emb_comb_strategy != "concat"
        ) or (
            prompt_session_emb_comb_strategy != "concat"
            and training_session_emb_comb_strategy == "concat"
        ):
            raise ValueError(
                "If one of the embedding strategies is concat, the other has to be"
                " concat as well."
            )

        super().__init__(is_verbose, cores)
        # Model settings.
        self.k: int = k
        self.sample_size: int = sample_size
        self.sampling: str = sampling
        self.sample_random_state: Optional[int] = sample_random_state
        self.use_item_embeddings: bool = use_item_embeddings
        self.prompt_session_emb_comb_strategy: str = prompt_session_emb_comb_strategy
        self.training_session_emb_comb_strategy: str = (
            training_session_emb_comb_strategy
        )
        self.dim_reduction_config: Optional[dict] = dim_reduction_config
        self.sequential_weighting: bool = sequential_weighting
        self.sequential_filter: bool = sequential_filter
        self.decay: Optional[str] = decay
        self.training_session_decay: Optional[str] = training_session_decay
        self.max_session_length_for_decay_precomputation: int = (
            max_session_length_for_decay_precomputation
        )
        self.similarity_measure: str = similarity_measure
        self.filter_prompt_items: bool = filter_prompt_items
        self.last_n_items: Optional[int] = last_n_items
        self.idf_weighting: bool = idf_weighting
        self.session_time_discount: Optional[float] = session_time_discount

        # Lookups
        self.item_original_to_reduced: Optional[dict[int, int]] = None
        self.item_reduced_to_original: Optional[np.ndarray] = None
        self.item_embeddings: Optional[np.ndarray] = None
        self.session_embeddings: Optional[np.ndarray | dict[int, np.ndarray]] = None
        self.precomputed_decay_arrays: Optional[dict[int, np.ndarray]] = None
        self.idf_dict: Optional[dict[int, float]] = None
        self.time_ordered_session_ids: Optional[np.ndarray] = None
        self.sequential_items_dict: Optional[dict[int, np.ndarray]] = None
        self.session_item_index: Optional[csr_matrix] = None

        self.dim_reducer: Optional[DimReducer] = None

    def train(self, train_data: pd.DataFrame, item_data: pd.DataFrame = None) -> None:
        """Trains the model with the session data given in the DataFrame.

        Args:
            train_data: A pandas DataFrame containing the session data.
            item_data: A pandas DataFrame containing item metadata such as embedding and
                class. Defaults to None.
        """
        if self.use_item_embeddings and item_data is None:
            raise ValueError(
                "use_item_embeddings was set to True, but no item side information is"
                " given."
            )

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

        self.precomputed_decay_arrays = SessionBasedCF._precompute_decay_arrays(
            self.decay, self.max_session_length_for_decay_precomputation
        )
        global global_precomputed_decay_arrays
        global_precomputed_decay_arrays = self.precomputed_decay_arrays

        # Configuration dependant training follows.
        if self.use_item_embeddings:
            # We do not want to alter the original df.
            item_data = item_data.copy()
            # Map the keys from original to reduced.
            item_data["ItemId"] = item_data["ItemId"].map(self.item_original_to_reduced)
            # Some items may not exist because the mapping is based on the training set.
            item_data = item_data[item_data["ItemId"] != -1]
            item_data = item_data.sort_values(by="ItemId")
            (
                self.item_embeddings,
                self.session_embeddings,
            ) = self.compute_embedding_lookups(train_data, item_data)
            global global_item_embeddings, global_session_embeddings
            global_item_embeddings, global_session_embeddings = (
                self.item_embeddings,
                self.session_embeddings,
            )

        if (
            self.sampling == "idf"
            or self.sampling == "idf_greedy"
            or self.idf_weighting
        ):
            self.idf_dict = SessionBasedCF.compute_idf_dict(train_data)
            global global_idf_dict
            global_idf_dict = self.idf_dict

        if self.sampling == "recent":
            session_times: np.ndarray = SessionBasedCF.get_session_times(train_data)
            self.time_ordered_session_ids = np.argsort(session_times)
            global global_time_ordered_session_ids
            global_time_ordered_session_ids = self.time_ordered_session_ids

        if self.session_time_discount is not None:
            session_times: np.ndarray = SessionBasedCF.get_session_times(train_data)
            global global_session_times
            global_session_times = session_times

        if self.sequential_filter:
            self.sequential_items_dict = SessionBasedCF.get_sequential_items_dict(
                train_data
            )
            global global_sequential_items_dict
            global_sequential_items_dict = self.sequential_items_dict

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

    def compute_embedding_lookups(
        self, train_data: pd.DataFrame, item_data: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray | dict[int, np.ndarray]]:
        """Computes the embedding lookups for sessions and items. Session embeddings are
        computed from the given item embeddings. Item embeddings are optionally reduced
        in dimensionality.

        Args:
            train_data: A pandas DataFrame containing the session item interaction data.
            item_data: A pandas DataFrame containing the item embeddings along with
                other metadata that might be needed for dimensionality reduction.

        Returns:
            A 2-tuple containing a numpy ndarray containing the item embeddings and
            either a numpy ndarray or a dictionary containing the session embeddings.
            A dictionary is only returned if the training session combination strategy
            was set to concatenation, which results in varying length of session
            embeddings.
        """
        n_sessions: int = train_data["SessionId"].nunique()
        embedding_dim: int = item_data["embedding"].iloc[0].size

        item_data["category_size"].iloc[0] = []

        item_embeddings: np.ndarray = np.stack(item_data["embedding"].values)
        if self.dim_reduction_config is not None:
            self.dim_reducer = DimReducer(**self.dim_reduction_config)
            item_embeddings = self.dim_reducer.reduce(item_data, "embedding")
            embedding_dim = item_embeddings.shape[1]

        if self.training_session_emb_comb_strategy == "concat":
            session_embeddings: dict[int, np.ndarray] = {}
        else:
            session_embeddings: np.ndarray = np.empty(shape=(n_sessions, embedding_dim))
        for session_id, session_df in train_data.groupby("SessionId"):
            session_item_ids: np.ndarray = session_df["ItemId"].to_numpy()
            session_embeddings[session_id] = SessionBasedCF._compute_session_embedding(
                session_item_ids=session_item_ids,
                item_embeddings=item_embeddings,
                combination_strategy=self.training_session_emb_comb_strategy,
                combination_decay=self.training_session_decay,
            )

        return item_embeddings, session_embeddings

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

    @staticmethod
    def _precompute_decay_arrays(
        decay: Optional[str], max_length: int
    ) -> dict[int, np.ndarray]:
        """Precomputes decay vectors for sessions of length up to max_length parameter.

        Args:
            decay: A string indicating the decay type.
            max_length: An integer indicating

        Returns:
            A dictionary that maps session lengths decay arrays stored as 1d ndarrays.

        Raises:
            ValueError: If the given decay function is not recognized.
        """
        precomputed_decay_arrays_dict: dict[int, np.ndarray] = {}
        for i in range(1, max_length + 1):
            precomputed_decay_arrays_dict[i] = SessionBasedCF._compute_decay_array(
                decay, i
            )
        return precomputed_decay_arrays_dict

    @staticmethod
    def _compute_decay_array(decay: Optional[str], length: int) -> np.ndarray:
        """Computes a decay array for the given decay type and session length.

        Args:
            decay: A string indicating the decay type. The options are linear, log,
                harmonic, and quadratic. Passing None indicates no decay, in which case
                the method returns an array of ones.
            length: A string indicating the session length.

        Returns:
            A 1d ndarray representing the decay array.

        Raises:
            ValueError: If the given decay function is not recognized.
        """
        descending_ints: list[int] = list(range(length, 0, -1))
        if decay is None:
            return np.ones(length, dtype=int)
        elif decay == "linear":
            return np.array([max(0.0001, 1.0 - (i - 1) * 0.1) for i in descending_ints])
        elif decay == "log":
            return np.array([1 / (np.log(i) + 1) for i in descending_ints])
        elif decay == "harmonic":
            return np.array([1 / i for i in descending_ints])
        elif decay == "quadratic":
            return np.array([1 / i**2 for i in descending_ints])
        else:
            raise ValueError(f"The given decay function of {decay} is not recognized.")

    def predict_single(
        self,
        prompt: np.ndarray,
        top_k: int = 20,
        item_mask: np.ndarray = None,
        return_scores: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Makes predictions for a single prompt.

        Args:
            prompt: Either an integer indicating the user/sample id or a numpy ndarray
                containing prompt data, i.e. session items or a user's historically
                interacted items.
            top_k: An integer indicating the number of item recommendations to return.
            item_mask: A numpy ndarray containing the item ids that we allow the model
                to recommend. Note: Not supported in this model.
            return_scores: A boolean indicating whether to also return the scores of the
                recommended items.

        Returns:
            A numpy ndarray containing the items ids of the recommended items. If
            return_scores is set to True, the return value is a 2-tuple containing two
            ndarrays: item ids and scores.
        """
        return SessionBasedCF._predict_session(
            prompt,
            self.k,
            self.sample_size,
            self.sampling,
            self.use_item_embeddings,
            self.prompt_session_emb_comb_strategy,
            self.sequential_weighting,
            self.sequential_filter,
            self.decay,
            self.similarity_measure,
            self.filter_prompt_items,
            self.last_n_items,
            self.idf_weighting,
            top_k,
            return_scores,
        )

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
            raise InvalidStateError("The model has not been trained yet.")
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
                            self.use_item_embeddings,
                            self.prompt_session_emb_comb_strategy,
                            self.sequential_weighting,
                            self.sequential_filter,
                            self.decay,
                            self.similarity_measure,
                            self.filter_prompt_items,
                            self.last_n_items,
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
        use_item_embeddings: bool,
        embedding_combination_strategy: str,
        sequential_weighting: bool,
        sequential_filter: bool,
        decay: Optional[str],
        similarity_measure: str,
        filter_prompt_items: bool,
        last_n_items: Optional[int],
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
            use_item_embeddings: A boolean indicating whether to use item embeddings for
                similarity computation.
            embedding_combination_strategy: A string indicating the embedding
                combination strategy for the prompt sessions.
            sequential_weighting: A boolean indicating whether sequential weighting
                is applied to candidate items.
            sequential_filter: A boolean indicating whether sequential filtering is
                applied to the candidate items.
            decay: A string representing the decay type.
            similarity_measure: A string indicating the type of similarity measure to
                use.
            filter_prompt_items: A boolean indicating whether the items of the prompt
                session are filtered out from the candidate items.
            last_n_items: An integer indicating the number of last items to use for
                making a recommendation. If None, all items are used.
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
                use_item_embeddings,
                embedding_combination_strategy,
                sequential_weighting,
                sequential_filter,
                decay,
                similarity_measure,
                filter_prompt_items,
                last_n_items,
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
        use_item_embeddings: bool,
        embedding_combination_strategy: str,
        sequential_weighting: bool,
        sequential_filter: bool,
        decay: Optional[str],
        similarity_measure: str,
        filter_prompt_items: bool,
        last_n_items: Optional[int],
        idf_weighting: bool,
        top_k: int,
        return_scores: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Generates a prediction, i.e. recommendations, for the given session.

        Args:
            session: A numpy array that contains the prompt session's original item ids.
            k: An integer indicating the choice of k in k-nearest neighbors.
            sample_size: An integer indicating the number of sampled neighbors.
            sampling: A string indicating the sampling approach for fetching the
                potential set of neighbor sessions.
            use_item_embeddings: A boolean indicating whether to use item embeddings for
                similarity computation.
            embedding_combination_strategy: A string indicating the embedding
                combination strategy for the prompt sessions.
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
            return_scores: A boolean indicating whether to also return the scores of the
                recommended items.

        Returns:
            A numpy array containing the original ids of the recommended items. If
            return_scores is set to True, a 2-tuple is returned where the first element
            is the array containing original ids and the second is an array containing
            scores.
        """
        if last_n_items is not None:
            session = session[-last_n_items:]

        session: np.ndarray = SessionBasedCF._get_reduced_session(session)
        # In case the prompt session includes only non-trained items.
        if session.size == 0:
            return np.array([])

        (
            potential_neighbor_matrix,
            potential_neighbor_session_indices,
        ) = SessionBasedCF._get_potential_neighbors(
            session,
            sample_size,
            sampling,
        )

        if use_item_embeddings:
            session_embedding: np.ndarray = SessionBasedCF._compute_session_embedding(
                session_item_ids=session,
                item_embeddings=global_item_embeddings,
                combination_strategy=embedding_combination_strategy,
                combination_decay=decay,
            )
            if embedding_combination_strategy == "concat":
                similarities: np.ndarray = (
                    SessionBasedCF._compute_embedding_based_similarities_concat(
                        session_embedding,
                        potential_neighbor_session_indices,
                        similarity_measure,
                    )
                )
            else:
                similarities: np.ndarray = (
                    SessionBasedCF._compute_embedding_based_similarities(
                        session_embedding,
                        potential_neighbor_session_indices,
                        similarity_measure,
                    )
                )
        else:
            similarities: np.ndarray = SessionBasedCF._compute_item_based_similarities(
                session, potential_neighbor_matrix, similarity_measure, decay
            )

        kn_matrix, kn_similarities = SessionBasedCF._get_k_neighbors(
            potential_neighbor_matrix,
            similarities,
            k,
        )

        candidate_items: np.ndarray = np.unique(kn_matrix.nonzero()[1])
        if filter_prompt_items:
            candidate_items: np.ndarray = candidate_items[
                ~np.isin(candidate_items, session)
            ]

        sequential_weights: Optional[np.ndarray] = (
            SessionBasedCF._get_sequential_weights(kn_matrix, session)
            if sequential_weighting
            else None
        )

        if sequential_filter:
            candidate_items = SessionBasedCF._apply_sequential_filter(
                candidate_items, session
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

        if top_k < len(scores):
            partitioned_idx: np.ndarray = np.argpartition(-scores, top_k)
            scores = scores[partitioned_idx][:top_k]
            candidate_items = candidate_items[partitioned_idx][:top_k]

        sorted_idx: np.ndarray = np.argsort(scores)[::-1]
        sorted_items: np.ndarray = candidate_items[sorted_idx]
        sorted_original_items: np.ndarray = global_item_reduced_to_original[
            sorted_items
        ]

        if return_scores:
            sorted_scores: np.ndarray = scores[sorted_idx]
            return sorted_original_items, sorted_scores
        else:
            return sorted_original_items

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
    ) -> tuple[csr_matrix, np.ndarray]:
        """Returns a sample of the sessions that contain at least one item from the
        given session.

        Args:
            session: A numpy array of reduced item ids representing the prompt session.
            sample_size: An integer indicating the number of sampled neighbors.
            sampling: A string indicating the sampling approach for fetching the
                potential set of neighbor sessions.

        Returns:
            A 2-tuple containing a sparse csr_matrix containing the potential sessions
            and a numpy ndarray containing the indices of the return sessions.
        """
        possible_neighbors: np.ndarray = SessionBasedCF._get_possible_neighbors(session)
        sampled_neighbors_index: np.ndarray = SessionBasedCF._sample_neighbors(
            session, possible_neighbors, sample_size, sampling
        )
        potential_neighbors: csr_matrix = global_session_item_index[
            sampled_neighbors_index, :
        ]
        return potential_neighbors, sampled_neighbors_index

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
        strategy. There are 4 options: random, recent, idf, and idf_greedy.

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
        elif sampling == "idf":
            return SessionBasedCF._sample_neighbors_idf(
                session, possible_neighbors, sample_size
            )
        elif sampling == "idf_greedy":
            return SessionBasedCF._sample_neighbors_idf_greedy(session, sample_size)
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
    def _sample_neighbors_idf(
        session: np.ndarray,
        possible_neighbors: np.ndarray,
        sample_size: int,
    ) -> np.ndarray:
        """Samples a set of sessions of size sample_size from the given
        possible_neighbors. The sampling is done by the following approach:

        For each neighbor, we compute the sum of the idf scores of the common items
        between it and the prompt session. Then, we return the sample_size neighbors
        with the highest sums.

        Args:
            session: A numpy array containing the reduced ids of the prompt session.
            possible_neighbors: A numpy array containing the reduced ids of the possible
                neighbor sessions.
            sample_size: An integer indicating the sample size.

        Returns:
            A numpy array containing the reduced ids of the sampled sessions.
        """
        idf_sums: np.ndarray = np.empty(len(possible_neighbors))
        for i, neighbor_id in enumerate(possible_neighbors):
            neighbor_row: csr_matrix = global_session_item_index.getrow(neighbor_id)
            common_items: set[int] = set(neighbor_row.nonzero()[1]).intersection(
                set(session)
            )
            idf_sums[i] = sum([global_idf_dict[item] for item in common_items])

        sampled_neighbors_idx: np.ndarray = np.argpartition(
            -idf_sums, min(sample_size, len(idf_sums) - 1)
        )[:sample_size]
        return possible_neighbors[sampled_neighbors_idx]

    @staticmethod
    def _sample_neighbors_idf_greedy(
        session: np.ndarray,
        sample_size: int,
    ) -> np.ndarray:
        """Samples a set of sessions of size sample_size from the given
        possible_neighbors. The sampling is done by the following approach:

        1. Sort the items of the prompt session by decreasing idf.
        2. Iterate over these items, and fill the sample with the sessions that contain
            the item of iteration until the sample is full.

        If any iteration overfills the sample, the method randomly selects sessions from
        this iteration until sample is full.

        Args:
            session: A numpy array containing the reduced ids of the prompt session.
            sample_size: An integer indicating the sample size.

        Returns:
            A numpy array containing the reduced ids of the sampled sessions.
        """
        session_idf_values: list[float] = [global_idf_dict[item] for item in session]
        sorted_session_items: np.ndarray = session[np.argsort(session_idf_values)][::-1]

        sample: set[int] = set()
        for item in sorted_session_items:
            n_needed_samples: int = sample_size - len(sample)
            new_samples: set[int] = set(
                global_session_item_index.getcol(item).nonzero()[0]
            ).difference(sample)
            if n_needed_samples >= len(new_samples):
                sample |= set(new_samples)
            else:
                sample |= set(
                    global_rng.choice(
                        list(new_samples),
                        size=n_needed_samples,
                        replace=False,
                    )
                )

        return np.array(list(sample))

    @staticmethod
    def _compute_embedding_based_similarities(
        session_embedding: np.ndarray,
        neighbor_session_indices: np.ndarray,
        similarity_measure: str,
    ) -> np.ndarray:
        """Computes an embedding based similarity between the given session and its
        neighbors based on the given similarity measure.

        Args:
            session_embedding: A numpy ndarray containing the embedding of the prompt
                session.
            neighbor_session_indices: A numpy ndarray containing the indices of the
                neighboring sessions.
            similarity_measure: A string indicating the similarity measure to use.

        Returns:
            A numpy ndarray containing the similarity values in the same order as the
            given indices.
        """
        neighbor_embeddings: np.ndarray = np.empty(
            shape=(neighbor_session_indices.size, session_embedding.size)
        )
        for i, neighbor_idx in enumerate(neighbor_session_indices):
            neighbor_embeddings[i] = global_session_embeddings[neighbor_idx]

        if similarity_measure == "cosine":
            return pairwise.cosine_similarity(
                neighbor_embeddings, session_embedding.reshape(1, -1)
            ).ravel()
        elif similarity_measure == "dot":
            return pairwise.linear_kernel(
                neighbor_embeddings, session_embedding.reshape(1, -1)
            ).ravel()
        else:
            raise ValueError(
                "The only supported similarity measures for embeddings are cosine and"
                " dot product."
            )

    @staticmethod
    def _compute_embedding_based_similarities_concat(
        session_embedding: np.ndarray,
        neighbor_session_indices: np.ndarray,
        similarity_measure: str,
    ) -> np.ndarray:
        """Computes an embedding based similarity between the given session and its
        neighbors based on the given similarity measure. This method is intended for use
        with the concatenation strategy for embedding combination due to the varying
        length of the session embeddings when concatenation is used.

        Args:
            session_embedding: A numpy ndarray containing the embedding of the prompt
                session.
            neighbor_session_indices: A numpy ndarray containing the indices of the
                neighboring sessions.
            similarity_measure: A string indicating the similarity measure to use.

        Returns:
            A numpy ndarray containing the similarity values in the same order as the
            given indices.
        """
        neighbor_embeddings: list[np.ndarray] = [
            global_session_embeddings[neighbor_idx]
            for neighbor_idx in neighbor_session_indices
        ]
        similarities: np.ndarray = np.empty(len(neighbor_session_indices))
        if similarity_measure == "cosine":
            for i, neighbor_embedding in enumerate(neighbor_embeddings):
                min_length: int = min(neighbor_embedding.size, session_embedding.size)
                similarities[i] = pairwise.cosine_similarity(
                    neighbor_embedding[-min_length:].reshape(1, -1),
                    session_embedding[-min_length:].reshape(1, -1),
                ).ravel()[0]
        elif similarity_measure == "dot":
            for i, neighbor_embedding in enumerate(neighbor_embeddings):
                min_length: int = min(neighbor_embedding.size, session_embedding.size)
                similarities[i] = pairwise.linear_kernel(
                    neighbor_embedding[-min_length:].reshape(1, -1),
                    session_embedding[-min_length:].reshape(1, -1),
                ).ravel()[0]
        else:
            raise ValueError(
                "The only supported similarity measures for embeddings are cosine and"
                " dot product."
            )
        return similarities

    @staticmethod
    def _compute_item_based_similarities(
        session: np.ndarray,
        neighbor_matrix: csr_matrix,
        similarity_measure: str,
        decay: Optional[str],
    ) -> np.ndarray:
        """Computes a common item based similarity between the given session and its
        neighbors based on the given similarity measure and decay. This is the
        similarity computation method that is used when item embeddings are not used.

        Args:
            session: A numpy ndarray containing the internal item ids of the prompt
                session.
            neighbor_matrix: A csr_matrix containing the neighbor sessions.
            similarity_measure: A string indicating the similarity measure.
            decay: A string indicating the decay function to apply to the prompt
                session.

        Returns:
            A numpy ndarray containing the similarity values in the order of appearance
            in the neighbor matrix.
        """
        session_row_matrix: csr_matrix = SessionBasedCF._get_session_row_matrix(
            session, decay
        )

        if similarity_measure == "cosine":
            return pairwise.cosine_similarity(
                neighbor_matrix, session_row_matrix
            ).ravel()
        elif similarity_measure == "dot":
            return neighbor_matrix.dot(session_row_matrix.T).data
        else:
            vec_sim_func: Callable = np.vectorize(
                similarity.get_similarity_func(similarity_measure),
                signature="(n),(n)->()",
            )
            return vec_sim_func(
                neighbor_matrix.toarray(),
                session_row_matrix.toarray().ravel(),
            )

    @staticmethod
    def _get_k_neighbors(
        potential_neighbors_matrix: csr_matrix,
        similarities: np.ndarray,
        k: int,
    ) -> tuple[csr_matrix, np.ndarray]:
        """Returns the k-nearest sessions from the given set of neighbor sessions based
        on the given similarity scores.

        Args:
            potential_neighbors_matrix: A csr_matrix containing a set of sessions
                containing potentially k-nearest neighbor sessions.
            similarities: A numpy ndarray containing the similarity score assigned to
                each neighbor.
            k: An integer representing the k in k-nearest.

        Returns:
            A tuple that contains the k-nearest neighbors matrix as first element and
            a numpy array containing the similarity scores as the second element.
        """
        k_neighbors_index: np.ndarray = np.argpartition(
            -similarities, min(k, len(similarities) - 1)
        )[:k]
        kn_matrix: csr_matrix = potential_neighbors_matrix[k_neighbors_index, :]
        kn_similarities: np.ndarray = similarities[k_neighbors_index]
        return kn_matrix, kn_similarities

    @staticmethod
    def _compute_session_embedding(
        session_item_ids: np.ndarray,
        item_embeddings: np.ndarray,
        combination_strategy: str,
        combination_decay: Optional[str],
    ) -> np.ndarray:
        """Computes a session embedding based on the embeddings of the items it
        contains.

        Args:
            session_item_ids: A numpy ndarray containing the ids of the items found in
                session.
            item_embeddings: A numpy ndarray containing embeddings of all items.
            combination_strategy: A string indicating the item embedding combination
                strategy to get a session embedding.
            combination_decay: A string indicating the type of decay to use when
                combining the embeddings of the session items.

        Returns:
            A numpy ndarray representing the session embedding.
        """
        session_item_embeddings: np.ndarray = np.empty(
            shape=(session_item_ids.size, item_embeddings.shape[1])
        )
        for i, item_id in enumerate(session_item_ids):
            session_item_embeddings[i] = item_embeddings[item_id]
        session_embedding: np.ndarray = SessionBasedCF._combine_embeddings(
            session_item_embeddings, combination_strategy, combination_decay
        )
        return session_embedding

    @staticmethod
    def _combine_embeddings(
        embeddings: np.ndarray, strategy: str, decay: Optional[str]
    ) -> np.ndarray:
        """Combines the given embeddings using the givens strategy and decay type. Decay
        mean a weight vector that we multiply with the embeddings before combining them
        such that their influence over the combined embedding is not equal.

        Args:
            embeddings: A numpy ndarray containing embeddings to combine.
            strategy: A string indicating the combination strategy.
            decay: A string indicating the decay type.

        Returns:
            A numpy ndarray representing the combined embedding.
        """
        if strategy == "mean":
            if decay is None:
                return np.mean(embeddings, axis=0)
            else:
                if (n_embeddings := len(embeddings)) in global_precomputed_decay_arrays:
                    decay_arr: np.ndarray = global_precomputed_decay_arrays[
                        n_embeddings
                    ]
                else:
                    decay_arr: np.ndarray = SessionBasedCF._compute_decay_array(
                        decay, n_embeddings
                    )
                return np.average(embeddings, axis=0, weights=decay_arr)
        elif strategy == "last":
            return embeddings[-1]
        elif strategy == "concat":
            if decay is None:
                return embeddings.ravel()
            else:
                if (n_embeddings := len(embeddings)) in global_precomputed_decay_arrays:
                    decay_arr: np.ndarray = global_precomputed_decay_arrays[
                        n_embeddings
                    ]
                else:
                    decay_arr: np.ndarray = SessionBasedCF._compute_decay_array(
                        decay, n_embeddings
                    )
                decay_arr = np.repeat(decay_arr, embeddings[0].size)
                return embeddings.ravel() * decay_arr
        else:
            raise ValueError(
                f"The given embedding combination strategy of {strategy} is not"
                f" supported."
            )

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
        if (n_items := len(session)) in global_precomputed_decay_arrays:
            data: np.ndarray = global_precomputed_decay_arrays[n_items]
        else:
            data: np.ndarray = SessionBasedCF._compute_decay_array(decay, n_items)
        indices: np.ndarray = np.array(session)
        indptr: np.ndarray = np.array([0, len(data)])
        session_row_matrix: csr_matrix = csr_matrix(
            (data, indices, indptr), shape=(1, global_session_item_index.shape[1])
        )
        return session_row_matrix

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
        if self.use_item_embeddings:
            return "SKNN_EMB"
        else:
            if self.decay is not None:
                return "V-SKNN"
            elif self.sequential_filter:
                return "SF-SKNN"
            elif self.sequential_weighting:
                return "S-SKNN"
            else:
                return "SKNN"
