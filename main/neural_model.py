import logging
import numpy as np
import math
import pandas as pd
import itertools
import tensorflow as tf

from tensorflow import keras
from scipy.sparse import csr_matrix
from tenacity import retry, stop_after_attempt, wait_random
from typing import Tuple, Any, Union
from abc import abstractmethod

from main.abstract_model import Model
from main.utils.neural_utils.custom_callbacks.metric_callback import (
    MetricCallback,
)
from main.eval.metrics.ndcg import NormalizedDiscountedCumulativeGain


class NeuralModel(Model):
    def __init__(
        self,
        num_epochs: int = 100,
        fit_batch_size: int = 256,
        pred_batch_size: int = 1024,
        train_val_fraction: float = 0.1,
        early_stopping_patience: int = 2,
        activation: str = "gelu",
        filepath_weights=None,
        **model_kwargs,
    ) -> None:
        """NeuralModel is the base class intended for neural models.

        Args:
            num_epochs (int, optional): The number of epochs to train. If the model
                uses validation, then this acts as the maximum number of epochs.
                Defaults to 1000.
            fit_batch_size (int, optional): The batch size for training.
                Defaults to 256.
            pred_batch_size (int, optional): The batch size for prediction.
                Defaults to 1024.
            train_val_fraction (float, optional): The fraction of users that will be
                used for validation instead of training. To also make use of this data
                for training, we calculate for 2 more epochs after the validation is
                done. Defaults to 0.1.
            early_stopping_patience (int, optional): The amount of epochs where the
                validation loss has not decreased for the model to early-stop training.
                Defaults to 2.
            activation (str): The string representation of the activation function used
                in the network. Defaults to GELU.
            filepath_weights: The filepath to the weights of the model. If not None, we
                skip training and return a keras model with these weights.
        """
        super().__init__(**model_kwargs)

        # Save all parameters used for training and predicting with neural models.
        self.num_epochs = num_epochs
        self.fit_batch_size = fit_batch_size
        self.pred_batch_size = pred_batch_size
        self.train_val_fraction = train_val_fraction
        self.early_stopping_patience = early_stopping_patience
        self.activation = activation

        self.filepath_weights = filepath_weights

        # Initialize the remaining member variables with None.
        # These will be instantiated during training.
        self.input_data = None

        # Merge of all training histories of the model. Note that this is a dict.
        self.history: dict[str, float] = {}

        self.active_callbacks = []

        # Do a fancy print of the devices.
        devices = [str(device) for device in tf.config.list_physical_devices()]
        line_sep = "\n\t"
        logging.info(
            "Initializing neural model. The following devices will be "
            f"available to TensorFlow:{line_sep}{line_sep.join(devices)}"
        )

    def train(self, input_data: Union[csr_matrix, pd.DataFrame]) -> None:
        """Trains the model.

        We split the input data into a new train and validation matrix, where
        the latter is used to evaluate the accuracy of the model after each epoch. This
        allows us to use early stopping. After validation is done, we calculate 2
        more epochs to also train on the validation data before prediction.

        Args:
            input_data (csr_matrix): The input data for training.
        """
        self.input_data: Union[csr_matrix, pd.DataFrame] = input_data

        if isinstance(self.input_data, csr_matrix):
            self.num_samples = input_data.shape[0]
        else:
            self.num_samples = len(input_data)

        if self.fit_batch_size > self.num_samples:
            logging.warning(
                f"Batch size {self.fit_batch_size} larger than the number of samples "
                f"{self.num_samples}. Using number of samples as batch size instead."
            )
            self.fit_batch_size = self.num_samples

        # Get the Keras model from implementation.
        self.model = self.get_keras_model(self.input_data)

        if self.filepath_weights is not None:
            self.model.built = True
            self.model.load_weights(self.filepath_weights)
            self.is_trained = True
            return

        # Split training and validation data.
        train_data, val_data = self.split_session_data(
            self.input_data, self.train_val_fraction
        )

        # Define training generator.
        train_generator = self.get_train_generator(train_data, self.fit_batch_size)

        @retry(stop=stop_after_attempt(2), wait=wait_random(min=1, max=2))
        def fit_model(*args, **kwargs):
            return self.model.fit(*args, **kwargs)

        # Right now, this is hardcoded for session models with an id reducer.
        # In other words, this is for BERT4Rec, SASRec and GRU4Rec.
        ground_truths = (
            val_data.copy()
            .drop_duplicates(subset=["SessionId"], keep="last")
            .groupby("SessionId")["ItemId"]
            .apply(np.array)
            .to_dict()
        )
        predict_data = (
            val_data.copy()[val_data.duplicated(["SessionId"], keep="last")]
            .groupby("SessionId")["ItemId"]
            .apply(np.array)
            .to_dict()
        )
        ground_truths = self.id_reducer.to_original(ground_truths)
        predict_data = self.id_reducer.to_original(predict_data)

        metric_callback = MetricCallback(
            self,
            NormalizedDiscountedCumulativeGain,
            predict_data,
            ground_truths,
            top_k=20,
            prefix="inner_val",
        )
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="inner_val_NDCG@20",
            patience=self.early_stopping_patience,
            verbose=self.is_verbose,
            mode="max",
            restore_best_weights=True,
        )
        # Train using early stopping.
        fit_model(
            x=train_generator,
            epochs=self.num_epochs,
            callbacks=[metric_callback, early_stop, *self.active_callbacks],
            verbose=2 if self.is_verbose else 0,
        )

        self.is_trained = True

    def predict(
        self,
        predict_data: Union[np.ndarray, dict[int, np.ndarray]] = None,
        top_k: int = 10,
    ) -> dict[Any, np.ndarray]:
        """Predicts the top_k recommendations for a set of samples.

        We chunk the users to predict for into batches so that we do not
        run into any memory issues.

        The exact functionality for prediction is up to the models to define through
        get_recommendations_batched.

        Args:
            predict_data (np.ndarray): The users/samples to predict for.
                Defaults to all samples in the training data if set to None.
            top_k (int, optional): The amount of predictions per sample.
                Defaults to 10.

        Returns:
            dict[int, np.ndarray]: An array of item predictions per sample.
        """
        assert self.is_trained

        if predict_data is None:
            predict_data = np.arange(self.num_samples)

        # Get number of samples to predict.
        num_samples = len(predict_data)

        num_batches = math.ceil(num_samples / self.pred_batch_size)

        # Create batches
        i = itertools.cycle(range(num_batches))
        batches = [dict() for _ in range(num_batches)]
        for k, v in predict_data.items():
            batches[next(i)][k] = v

        recommendations = {}

        # Calculate recommendations for batches.
        for i, cur_batch in enumerate(batches):
            # Log if verbose and enough time has passed since last log.
            logging.info(f"Predicting batch {i} out of {len(batches)}")
            recommendations_batch = self.get_recommendations_batched(cur_batch, top_k)
            recommendations.update(recommendations_batch)

        return recommendations

    @abstractmethod
    def get_recommendations_batched(
        self, user_batch: np.ndarray, top_k
    ) -> dict[int, np.ndarray]:
        pass

    @abstractmethod
    def get_keras_model(self, data: Union[csr_matrix, pd.DataFrame]) -> keras.Model:
        pass

    @abstractmethod
    def get_train_generator(
        self, data: Union[csr_matrix, pd.DataFrame], batch_size: int
    ) -> keras.utils.Sequence:
        pass

    def split_session_data(
        self, data: pd.DataFrame, train_val_fraction: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Splits session train data into a train DataFrame and a validation DataFrame
        based on a train-validation split conveyed as a fraction of the overall train
        data.

        We use a random subset of rows of the data for validation.

        Args:
            data (pd.DataFrame): The training data.
            train_val_fraction (float): Fraction of train data to be used for validation.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (train_data, val_data).
        """
        unique_sessions = data["SessionId"].unique()

        num_sessions_val = min(
            math.ceil(train_val_fraction * len(unique_sessions)), 500
        )

        sessions_val = set(
            np.random.choice(unique_sessions, size=num_sessions_val, replace=False)
        )
        sessions_train = set([id for id in unique_sessions if id not in sessions_val])

        data_1 = data[data["SessionId"].isin(sessions_train)]
        data_2 = data[data["SessionId"].isin(sessions_val)]

        return (data_1, data_2)
