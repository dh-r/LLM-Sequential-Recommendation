import math
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from main.utils.neural_utils.custom_preprocessors.cloze import (
    Cloze,
)
from main.utils.neural_utils.custom_preprocessors.data_description import (
    DataDescription,
)
from main.utils.neural_utils.custom_preprocessors.tensor_factory import (
    TensorFactory,
)


class TrainGenerator(keras.utils.Sequence):
    def __init__(
        self,
        train_data: pd.DataFrame,
        N: int,
        batch_size: int,
        mask_prob: float,
        data_description: DataDescription,
    ) -> None:
        """Data generator for training purposes. A TrainGenerator object can be used
        to infinitely iterate over the training data, where some items are randomly
        masked.

        Args:
            train_data (pd.DataFrame): The input training data.
            N (int): The sequence length.
            batch_size (int): The batch size for training.
            mask_prob (float): The probability of randomly masking an item.
            data_description (DataDescription): The description of the data, containing
                some properties that are necessary to instantiate a TrainGenerator.
        """
        self.train_data = train_data
        self.N = N
        self.batch_size = batch_size
        self.mask_prob = mask_prob

        num_items = data_description["num_items"]
        self.cloze = Cloze(num_items)

        self.on_epoch_end()

    def __len__(self):
        return math.ceil(self.train_input.shape[0] / self.batch_size)

    def __getitem__(self, batch_index):
        """Method to support iteration over the data."""
        # Get indices for this batch
        start_index = batch_index * self.batch_size
        end_index = (batch_index + 1) * self.batch_size
        indices = self.indices[start_index:end_index]

        return (
            tf.gather(self.train_input, indices),
            tf.gather(self.train_true, indices),
        )

    def on_epoch_end(self):
        """Method called at the end of an epoch. In this case, we want to regenerate
        the tensor from the TensorFactory (for the case we have a non-deterministic
        component in the session preprocessing) and subsequently regenerate training
        data.
        """
        self.__generate_train_data()

        self.indices = np.arange(len(self.train_input))
        np.random.shuffle(self.indices)

    def __generate_train_data(self):
        """Generates the input and true data by setting the x and y member variables."""
        sequences = TensorFactory.to_sequence_tensor(
            sessions=self.train_data,
            sequence_length=self.N,
        )

        train_data: list[tf.Tensor] = []
        true_data: list[tf.Tensor] = []

        # Produce train cases with the last item masked.
        loo_data: Tuple[tf.Tensor, tf.Tensor] = self.cloze.mask_last(sequences)
        (train_loo_data, true_loo_data) = loo_data
        train_data.append(train_loo_data)
        true_data.append(true_loo_data)

        # Produce random train cases with random masks.
        for _ in range(10):
            random_data: Tuple[tf.Tensor, tf.Tensor] = self.cloze.mask_random(
                sequences, self.mask_prob
            )
            (train_random_data, true_random_data) = random_data
            train_data.append(train_random_data)
            true_data.append(true_random_data)

        # Concatenate train and true into two single tensors.
        (self.train_input, self.train_true) = (
            tf.concat(train_data, 0),
            tf.concat(true_data, 0),
        )
