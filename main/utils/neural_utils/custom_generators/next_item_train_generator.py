from main.utils.neural_utils.custom_preprocessors.cloze import (
    Cloze,
)
from main.utils.neural_utils.custom_preprocessors.tensor_factory import (
    TensorFactory,
)
from main.utils.neural_utils.custom_preprocessors.data_description import (
    DataDescription,
)

import tensorflow as tf

import numpy as np
import pandas as pd
from tensorflow import keras

from typing import Union, Tuple
import math


class NextItemTrainGenerator(keras.utils.Sequence):
    def __init__(
        self,
        train_data: Union[pd.DataFrame, dict[int, np.ndarray]],
        N: int,
        batch_size: int,
    ) -> None:
        """Data generator for training purposes. A TrainGenerator object can be used
        to infinitely iterate over the training data.

        The NextItemTrainGenerator is used by models that are tasked with predicting
        the next item in the sequence, like GRU4Rec and SASRec. As such, train_true is
        a shifted version of train_input.

        Args:
            train_data (pd.DataFrame): The input training data.
            N (int): The sequence length.
            batch_size (int): The batch size for training.
        """
        self.train_data = train_data
        self.N = N
        self.batch_size = batch_size

        self.on_epoch_end()

    def __len__(self):
        return math.ceil(self.train_input.shape[0] / self.batch_size)

    def __getitem__(self, batch_index) -> Tuple[tf.Tensor, tf.Tensor]:
        """Method to support iteration over the data.

        Args:
            batch_index (int): Index of the batch, but unnecessary as of now.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: A tuple (train_input, train_true).
        """
        # Get indices for this batch
        start_index = batch_index * self.batch_size
        end_index = (batch_index + 1) * self.batch_size
        indices = self.indices[start_index:end_index]

        result = (
            tf.gather(self.train_input, indices),
            tf.gather(self.train_true, indices),
        )
        return result

    def on_epoch_end(self):
        """Method called at the end of an epoch. In this case, we want to regenerate
        the tensor from the TensorFactory, and shuffle the data.
        """
        sequence_tensor = TensorFactory.to_sequence_tensor(
            sessions=self.train_data,
            sequence_length=self.N + 1,
        )

        # The goal in training is to predict the next item in the sequence,
        # so train_true is a shifted version of train_input.
        self.train_input = sequence_tensor[:, :-1]
        self.train_true = sequence_tensor[:, 1:]

        # However, it is impossible to predict for the first item.
        # So we replace the first non-padding target in the true data
        # with the TensorFactory.PADDING_TARGET.
        padding = tf.equal(self.train_input, TensorFactory.PADDING_TARGET)
        self.train_true = tf.where(
            padding,
            TensorFactory.PADDING_TARGET,
            self.train_true,
        )

        self.indices = np.arange(len(self.train_input))
        np.random.shuffle(self.indices)
