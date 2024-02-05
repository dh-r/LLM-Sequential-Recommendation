import logging
from typing import Dict, Any

import numpy as np
import pandas as pd

from main.neural_model import NeuralModel
from main.utils.config_util import extract_config
from main.utils.id_reducer import IDReducer
from main.utils.neural_utils.custom_losses.masked_sparse_categorical_crossentropy import (
    masked_sparse_categorical_crossentropy,
)
from main.utils.neural_utils.custom_preprocessors.data_description import (
    DataDescription,
    get_data_description,
)
from main.utils.top_k_computer import TopKComputer
from main.utils.utils import INT_INF
from main.utils.utils import to_dense_encoding


class Transformer(NeuralModel):
    """The base Transformer class.

    It has the following responsibilities:
        1. Before training the model, it extracts properties from the dataset,
        including the distribution of session lengths in order to choose a suitable
        value for N if necessary.
        2. It reduces the IDs before training and prediction.
        3. It implements the get_recommendations_batched method.
    """

    def __init__(
        self,
        N: int = None,
        L: int = 2,
        h: int = 2,
        emb_dim: int = 64,
        trans_dim_scale: int = 4,
        transformer_layer_kwargs: dict = {},
        drop_rate: float = 0.2,
        optimizer_kwargs: dict = {},
        infer_N_percentile: int = 95,
        pred_seen: bool = False,
        **neural_model_kwargs,
    ) -> None:
        """Instantiates a transformer model.

        Args:
            N (int): The length of the input sequence. Defaults to None, in which case
                it is inferred from the training data using N_infer_percentile.
            L (int): The number of transformer layers. Defaults to 2.
            h (int): The number of heads in the multi-head attention mechanism of each
                transformer layer. Defaults to 2.
            emb_dim (int): The dimension of the embedding of each item. Defaults to 64.
            trans_dim_scale (int). The intermediate dimension of the transformer layer
                as a multiple of the embedding dimension. Defaults to 4.
            transformer_layer_kwargs (dict, optional): Keyword arguments for the
                transformer encoder layer. Defaults to {}.
            drop_rate (float): The drop rate inside the transformer layer.
                Defaults to 0.2.
            optimizer_kwargs (dict, optional): Keyword arguments for the Adam optimizer.
                Defaults to {}.
            infer_N_percentile (float): When N is None, then the transformer has to
                choose a suitable value for N itself. The infer_N_percentile specifies
                the percentile (of the distribution of session lengths) which is used
                to choose N. Defaults to 0.95.
            pred_seen (bool): Whether to predict items that are already in the session.
                Defaults to False.
        """
        super().__init__(**neural_model_kwargs)

        # Variables necessary to instantiate any transformer.
        self.N: int = N
        self.L: int = L
        self.h: int = h
        self.emb_dim: int = emb_dim
        self.trans_dim_scale: int = trans_dim_scale
        self.transformer_layer_kwargs = transformer_layer_kwargs
        self.drop_rate: float = drop_rate
        self.optimizer_kwargs = optimizer_kwargs
        self.infer_N_percentile = infer_N_percentile
        self.pred_seen = pred_seen

        # Initialize the remaining member variables with None.
        # These will be instantiated during training.
        self.id_reducer = None
        self.data_description = None
        self.model = None

        logging.info(
            f"Instantiating {self.name()} with configuration: {extract_config(self)}"
        )

    def train(self, input_data: pd.DataFrame) -> None:
        """Trains the transformer model given the training data.

        Args:
            input_data (pd.DataFrame): A pandas DataFrame containing the session data,
                containing at least a SessionId and ItemId column. It also assumed that
                this datafrarme is already sorted based on time.
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

        # Get prediction data from TestGenerator.
        # The first index is for accessing the first batch of testing data,
        # the second index is for getting the tensor with the prompts.
        test_data_tensor = self.get_test_generator(
            predict_data, for_prediction=True, batch_size=self.pred_batch_size
        )[0][0]

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

    @classmethod
    def get_custom_keras_objects(cls: type["Transformer"]) -> dict[str, Any]:
        """Get the custom Keras objects that are necessary to load the model
        back into memory.

        For transformers, we have a custom loss function, so we need to add this to
        our custom Keras objects.

        Args:
            cls (type[NeuralModel]): The class of the model for which to return the
                custom Keras objects. In this case, it will simply be the BERT class.

        Returns:
            dict[str, Any]: The custom Keras objects in the form of a dictionary that
                maps the name of the object to the actual object.
        """
        return {
            "masked_sparse_categorical_crossentropy": masked_sparse_categorical_crossentropy
        }
