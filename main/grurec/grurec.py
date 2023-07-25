from main.neural_model import NeuralModel
from main.grurec.grurec_model import GRURecModel

from main.utils.neural_utils.custom_generators.next_item_train_generator import (
    NextItemTrainGenerator,
)
from main.utils.neural_utils.custom_generators.next_item_test_generator import (
    NextItemTestGenerator,
)
from main.utils.top_k_computer import TopKComputer
from main.utils.config_util import extract_config
from main.utils.utils import INT_INF
from main.utils.utils import to_dense_encoding
from main.utils.id_reducer import IDReducer
from main.utils.neural_utils.custom_preprocessors.data_description import *

from tensorflow import keras
import numpy as np
import pandas as pd

from typing import Any
import logging


class GRURec(NeuralModel):
    def __init__(
        self,
        N: int = None,
        infer_N_percentile: int = 95,
        emb_dim: int = 64,
        hidden_dim: int = 128,
        drop_rate: float = 0.2,
        optimizer_kwargs: dict = {},
        pred_seen: bool = False,
        **neural_model_kwargs: dict,
    ):
        """The GRU4Rec model. 
        
        The implementation mostly follows the original
        [the paper introducing GRU to the recommendation domain](https://arxiv.org/pdf/1511.06939.pdf).

        Args:
            N (int): The length of the input sequence. Defaults to None, in which case
                it is inferred from the training data using N_infer_percentile.
            infer_N_percentile (float): When N is None, then the model has to choose a
                suitable value for N itself. The infer_N_percentile specifies
                the percentile (of the distribution of session lengths) which is used
                to choose N. Defaults to 0.95.
            emb_dim (int, optional): The dimension of the embedding of each item.
                Defaults to 64.
            hidden_dim (int, optional): The dimension of the hidden state of the GRU
                layer. Defaults to 128.
            drop_rate (float, optional): The drop rate on the input embedding and the output 
                of the GRU layer. Defaults to 0.2.
            optimizer_kwargs (dict, optional): Keyword arguments for the Adam optimizer.
                Defaults to {}.
            pred_seen (bool, optional): Whether the model is allowed to recommend
                items that were already present in the session. Defaults to False.
        """
        super().__init__(**neural_model_kwargs)

        # Save model parameters.
        self.N = N
        self.infer_N_percentile = infer_N_percentile
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.drop_rate = drop_rate
        self.optimizer_kwargs = optimizer_kwargs

        self.N = N

        # Save prediction parameters.
        self.pred_seen = pred_seen

        logging.info(
            f"Instantiating GRU4Rec with configuration: {extract_config(self)}"
        )

    def train(self, input_data: pd.DataFrame) -> None:
        """Trains the GRU4Rec model given the training data.

        Args:
            input_data (pd.DataFrame): A pandas DataFrame containing the session data,
                containing at least a SessionId and ItemId column. It also assumed that
                this dataframe is already sorted based on time.
        """
        self.data_description: DataDescription = get_data_description(input_data)

        # Set N to appropriate value if necessary.
        if self.N is None:
            self.N = int(
                self.data_description["session_length_description"][
                    f"{self.infer_N_percentile}%"
                ]
            )

        # Reset preprocessors.
        self.id_reducer = IDReducer(input_data, "ItemId")

        # Convert ItemIds to a reduced range.
        input_data = self.id_reducer.to_reduced(input_data)

        # Train the model.
        super().train(input_data)

    def get_recommendations_batched(
        self, predict_data: Dict[int, np.ndarray], top_k: int = 10
    ) -> Dict[int, np.ndarray]:
        """Generates predictions for the test sessions given.

        This method assumes that predict_data is already properly batched.

        Args:
            predict_data (Dict[int, np.ndarray]): A dictionary representing test
                sessions, where a key represents a SessionId, and the value represents a
                list of itemIds. This list is assumed to be sorted based on time.
            top_k (int, optional): An integer representing the number of items to
                recommend. Defaults to 10.

        Returns:
            Dict[int, np.ndarray]: A dictionary representing the predictions, where a
                key represents the id of the test session, and the value represents a
                list of top_k recommendations sorted on confidence descendingly.
        """
        # Convert ItemIDs to a reduced range.
        predict_data = self.id_reducer.to_reduced(predict_data)

        test_generator = self.get_test_generator(
            predict_data, for_prediction=True, batch_size=self.pred_batch_size
        )

        # Get prediction data from TestGenerator.
        # The first index is for accessing the first batch of testing data,
        # the second index is for getting the tensor with the prompts.
        # Note that predict data should have already been batched, so there should only
        # be one batch present.
        assert len(test_generator) == 1
        test_data_tensor = test_generator[0][0]

        # Generate predictions.
        predictions: np.ndarray = self.model.predict_on_batch(test_data_tensor)

        # Use a dense representation of the sessions to filter out seen items.
        space_size = self.data_description["num_items"]
        dense_sessions = to_dense_encoding(
            test_data_tensor, space_size, ignore_oob=True
        )
        if not (self.pred_seen):
            allowed_items = 0 - dense_sessions * INT_INF
            predictions = np.add(predictions, allowed_items)

        predictions_top_k = TopKComputer.compute_top_k(predictions, top_k)
        key_to_predictions = dict(zip(predict_data.keys(), predictions_top_k))

        # Convert ItemIDs back to their original IDs.
        recommendations = self.id_reducer.to_original(key_to_predictions)
        return recommendations

    def get_keras_model(self, data: dict[int, np.ndarray]) -> keras.Model:
        num_items = self.data_description["num_items"]
        return GRURecModel(
            num_items, self.emb_dim, self.hidden_dim, self.drop_rate, self.optimizer_kwargs, self.activation
        )

    def get_train_generator(
        self, data: pd.DataFrame, batch_size: int
    ) -> keras.utils.Sequence:
        return NextItemTrainGenerator(data, self.N, batch_size)

    def get_test_generator(
        self, data: pd.DataFrame, for_prediction: bool, batch_size: int
    ) -> keras.utils.Sequence:
        return NextItemTestGenerator(
            data, self.N, batch_size, for_prediction=for_prediction
        )    

    def name(self) -> str:
        return "GRU4Rec"