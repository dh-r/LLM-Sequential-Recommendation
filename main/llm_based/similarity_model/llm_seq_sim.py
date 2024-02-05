# TODO implement saving/loading related methods
# TODO implement online prediction related methods

import math
import time
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise
from tqdm import tqdm

from main.exceptions import InvalidStateError
from main.abstract_model import Model
from main.utils.dim_reducer import DimReducer
from main.utils.session_utils import decay_utils
from main.utils.session_utils import session_embedding_utils
from main.utils.top_k_computer import TopKComputer
from main.utils.split_dict import split_dict


class LLMSeqSim(Model):
    """A class that implements LLMSeqSim, a recommender that leverages item embeddings
    coming from Large Language Models.

    Attributes:
        similarity_measure: A string indicating the similarity measure to use. Options
            are cosine, dot, and euclidian.
        embedding_combination_strategy: A string indicating the strategy to use when
            combining the embeddings of items that make up a session in order to obtain
            a session embedding. Options are mean and last.
        combination_decay: A string indicating the type of decay to apply to item
            embeddings when combining them.
        max_session_length_for_decay_precomputation: An integer indicating the maximum
            session length for which we precompute the decay vectors.
        filter_prompt_items: A boolean indicating whether to filter the prompt items
            when making recommendations.
        batch_size: An integer indicating the batch size to use for generating
            predictions for a given test set.
        dim_reduction_config: A dictionary containing the configuration of the
            dimensionality reduction to apply to the embeddings. If None, no reduction
            is performed.
    """

    def __init__(
        self,
        is_verbose: bool = False,
        cores: int = 1,
        similarity_measure: str = "cosine",
        embedding_combination_strategy: str = "last",
        combination_decay: Optional[str] = None,
        max_session_length_for_decay_precomputation: int = 500,
        filter_prompt_items: bool = True,
        batch_size: int = 500,
        dim_reduction_config: dict = None,
    ) -> None:
        if not (
            similarity_measure == "cosine"
            or similarity_measure == "dot"
            or similarity_measure == "euclidean"
        ):
            raise ValueError(
                f"The given similarity measure of {similarity_measure} is not"
                " supported."
            )
        if not (
            embedding_combination_strategy == "mean"
            or embedding_combination_strategy == "last"
        ):
            raise ValueError(
                "The given embedding combination strategy of"
                f" {embedding_combination_strategy} is not supported."
            )

        super().__init__(is_verbose, cores)

        self.similarity_measure: str = similarity_measure
        self.embedding_combination_strategy: str = embedding_combination_strategy
        self.combination_decay: Optional[str] = combination_decay
        self.max_session_length_for_decay_precomputation: int = (
            max_session_length_for_decay_precomputation
        )
        self.filter_prompt_items: bool = filter_prompt_items
        self.batch_size: int = batch_size
        self.dim_reduction_config: Optional[dict] = dim_reduction_config

        self.precomputed_decay_arrays: Optional[dict[int, np.ndarray]] = None
        self.item_original_to_reduced: Optional[dict[int, int]] = None
        self.item_reduced_to_original: Optional[np.ndarray] = None
        self.item_embeddings: Optional[np.ndarray] = None
        self.n_items: Optional[int] = None
        self.embedding_dim: Optional[int] = None
        self.dim_reducer: Optional[DimReducer] = None

    def train(self, train_data: pd.DataFrame, item_data: pd.DataFrame = None) -> None:
        """Trains the model with the given data.

        Args:
            train_data: A pandas DataFrame containing the session-item interaction data.
            item_data: A pandas DataFrame containing the item metadata.
        """
        item_data = item_data.copy()

        self.precomputed_decay_arrays = decay_utils.precompute_decay_arrays(
            self.combination_decay, self.max_session_length_for_decay_precomputation
        )

        item_ids: np.ndarray = item_data["ItemId"].values
        self.n_items = item_ids.size
        self.item_original_to_reduced = {k: i for i, k in enumerate(item_ids)}
        self.item_reduced_to_original = item_ids

        self.item_embeddings = np.stack(item_data["embedding"].values)
        if self.dim_reduction_config is not None:
            self.dim_reducer = DimReducer(**self.dim_reduction_config)
            self.item_embeddings = self.dim_reducer.reduce(item_data, "embedding")
        self.embedding_dim = self.item_embeddings.shape[1]

        self.is_trained = True

    def predict(
        self, predict_data: dict[int, np.ndarray], top_k: int = 10
    ) -> dict[int, np.ndarray]:
        """Generates top_k recommendations for the given sessions.

        Args:
            predict_data: A dictionary that maps session ids to items in the session
                represented as a numpy array.
            top_k: An integer representing the top k.

        Returns:
            A dictionary that maps the same session ids to item recommendations
            represented as numpy arrays.
        """
        if not self.is_trained:
            raise InvalidStateError("The model has not been trained yet.")

        predict_time_start: float = time.perf_counter()

        predictions: dict[int, np.ndarray] = {}
        n_batches: int = math.ceil(len(predict_data) / self.batch_size)
        batches: list[dict] = split_dict(predict_data, n_batches)
        for i, batch in enumerate(tqdm(batches, desc="Processed batches")):
            batch_predictions: dict = self._predict_batch(batch, top_k)
            predictions |= batch_predictions

        predict_time_finish: float = time.perf_counter()
        print(
            f"Prediction took {(predict_time_finish - predict_time_start):.3f} seconds."
        )
        return predictions

    def _predict_batch(
        self,
        batch_predict_data: dict[int, np.ndarray],
        top_k: int = 10,
    ) -> dict[int, np.ndarray]:
        """Predicts for the given batch.

        Args:
            batch_predict_data: A dictionary that maps session ids to items in the
                session represented as a numpy array.
            top_k: An integer representing the top k.

        Returns:
            A dictionary that maps the same session ids to item recommendations
            represented as numpy arrays.
        """
        n_sessions: int = len(batch_predict_data)
        session_embeddings: np.ndarray = np.empty(
            shape=(n_sessions, self.embedding_dim)
        )

        if self.filter_prompt_items:
            interacted_items: np.ndarray = np.zeros(shape=(n_sessions, self.n_items))

        for i, session in enumerate(batch_predict_data.values()):
            reduced_session: np.ndarray = np.array(
                [self.item_original_to_reduced[item] for item in session]
            )

            n_session_items: int = reduced_session.size
            session_item_embeddings: np.ndarray = np.empty(
                shape=(n_session_items, self.embedding_dim)
            )
            for j, item in enumerate(reduced_session):
                session_item_embeddings[j] = self.item_embeddings[item]

            session_embeddings[i] = session_embedding_utils.combine_embeddings(
                session_item_embeddings,
                self.embedding_combination_strategy,
                self.combination_decay,
                self.precomputed_decay_arrays,
            )

            if self.filter_prompt_items:
                interacted_items[i][reduced_session] = 1

        if self.similarity_measure == "cosine":
            similarities: np.ndarray = pairwise.cosine_similarity(
                session_embeddings, self.item_embeddings
            )
        elif self.similarity_measure == "dot":
            similarities: np.ndarray = pairwise.linear_kernel(
                session_embeddings, self.item_embeddings
            )
        else:  # Euclidean similarity
            similarities: np.ndarray = LLMSeqSim._euclidean_similarity(
                session_embeddings, self.item_embeddings
            )

        if self.filter_prompt_items:
            # Remove items already in batch from predictions.
            allowed_items = 0 - interacted_items * 1_000_000_000
            similarities = np.add(similarities, allowed_items)

        # Get top-k items.
        top_k_batch = TopKComputer.compute_top_k(similarities, top_k)
        top_k_batch_shape = top_k_batch.shape
        top_k_batch = np.ndarray.flatten(top_k_batch)
        top_k_batch = [self.item_reduced_to_original[item] for item in top_k_batch]
        top_k_batch = np.reshape(top_k_batch, top_k_batch_shape)

        return dict(zip(batch_predict_data.keys(), top_k_batch))

    def name(self) -> str:
        """Returns the name of the model with its configuration.

        Returns:
            A string representing the name.
        """
        return "LLMSeqSim"

    @staticmethod
    def _euclidean_similarity(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Computes the euclidian similarity between the given vectors. The similarity
        is defined as:

        1 / (1 + euclidian_distance(x, y))

        Args:
            x: A numpy ndarray.
            y: A numpy ndarray.

        Returns:
            A numpy ndarray containing the similarity values between the vectors in x
            and y.
        """
        # This method may result in catastrophic cancellation on very close vectors.
        # However, it has very low memory usage. I would argue that catastrophic
        # cancellation is not a problem for us. If two vectors are very close, for our
        # purposes, it is fine if our approximation is off in relative terms. In
        # absolute terms, the distance will still result in two close vectors being
        # considered close.
        distances: np.ndarray = pairwise.euclidean_distances(x, y)
        # TODO we can try different measures here
        similarities: np.ndarray = 1 / (1 + distances)
        return similarities
