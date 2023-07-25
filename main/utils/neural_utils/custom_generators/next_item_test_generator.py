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
import tensorflow as tf
from tensorflow import keras

from typing import Tuple, Union
import math


class NextItemTestGenerator(keras.utils.Sequence):
    def __init__(
        self,
        sessions: Union[pd.DataFrame, dict[int, np.ndarray]],
        N: int,
        batch_size: int,
        for_prediction: bool,
    ) -> None:
        """Data generator for testing purposes. A DataGenerator object can be used
        to infinitely iterate over the testing sessions.

        This generator can be used for both validation and prediction. In the case of
        validation, we hold-out the last item from the input. In the case of prediction,
        we simply want to use all data as the input.

        Args:
            sessions (Union[pd.DataFrame, dict]): The input sessions.
            N (int): The sequence length.
            batch_size (int): The batch size for training.
            for_prediction (bool): Whether the generator is for training or prediction.
        """
        self.N = N
        self.batch_size = batch_size
        self.for_prediction = for_prediction

        self.__generate_test_data(sessions)

    def __len__(self) -> int:
        return math.ceil(self.test_input.shape[0] / self.batch_size)

    def __getitem__(self, batch_index: int) -> Tuple[tf.Tensor, tf.Tensor]:
        """Method to support iteration over the data. Using the batch index we take
        a slice of the test_input and test_true member variables. For the content
        of these variables, refer to the documentation of __generate_test_data.

        In case this generator is used for prediction, we actually set cur_test_true
        to None because it will not be used.

        Args:
            batch_index (int): Index of the batch.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: The tuple (cur_test_input, cur_test_true),
                which are both slices of the total test_input and test_true variables.
                In case this generator is used for prediction, cur_test_true is None.
        """
        start_index = batch_index * self.batch_size
        end_index = (batch_index + 1) * self.batch_size

        cur_test_input = self.test_input[start_index:end_index]

        if self.test_true is not None:
            cur_test_true = self.test_true[start_index:end_index]
        else:
            cur_test_true = None

        return (cur_test_input, cur_test_true)

    def on_epoch_end(self) -> None:
        pass

    def __generate_test_data(
        self, sessions: Union[pd.DataFrame, dict[int, np.ndarray]]
    ) -> None:
        """Generates the test data by setting it to self.test_input and self.test_true.
        Note that we set member variables instead of returning the data
        (which would be more conventional) because __get_item__ only has access
        to self.

        Args:
            sessions (Union[pd.DataFrame, dict]): The input sessions, either in
                DataFrame or in dictionary form.
        """
        self.sequence_tensor = TensorFactory.to_sequence_tensor(
            sessions=sessions,
            sequence_length=self.N + 1,
        )

        if self.for_prediction:
            # Prediction phase.
            # Simply take the last N items.
            self.test_input = self.sequence_tensor[:, 1:]
            self.test_true = None  # unused.
        else:
            # Validation phase.
            # Leave out last item for validation
            self.test_input = self.sequence_tensor[:, :-1]

            # Fill true with only the last item.
            self.test_true_column = self.sequence_tensor[:, -1:]

            padding_shape = self.test_input.shape.as_list()
            padding_shape[-1] = padding_shape[-1] - 1
            self.test_true_padding = tf.fill(padding_shape, -1)

            self.test_true = tf.concat(
                [self.test_true_padding, self.test_true_column], axis=-1
            )
