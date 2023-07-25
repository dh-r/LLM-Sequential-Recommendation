from main.utils.neural_utils.custom_preprocessors.cloze import (
    Cloze,
)
from main.utils.neural_utils.custom_preprocessors.tensor_factory import (
    TensorFactory,
)
from main.utils.neural_utils.custom_preprocessors.data_description import (
    DataDescription,
)
from main.utils.utils import INT_INF

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from typing import Dict, Tuple

from typing import Union, Dict


class TestGenerator(keras.utils.Sequence):
    def __init__(
        self,
        test_data: Union[pd.DataFrame, Dict[int, np.ndarray]],
        N: int,
        data_description: DataDescription,
        for_prediction: bool,
    ) -> None:
        """Data generator for testing purposes. A TestGenerator object can be used
        to infinitely iterate over the test data, where only the last items are
        masked.

        Args:
            test_data (Union[pd.DataFrame, Dict[int, np.ndarray]]): The input data.
            N (int): The sequence length.
            data_description (DataDescription): Description of the data.
            for_prediction (bool): Whether the TestGenerator is used for prediction or
                training.

        Raises:
            Exception: If an unknown data type is passed to the TestGenerator.
        """
        self.test_data = test_data
        self.N = N
        self.data_description = data_description
        self.for_prediction = for_prediction

        self.__generate_test_data()

    def __len__(self) -> int:
        return 1

    def __getitem__(self, batch_index: int) -> Tuple[tf.Tensor, tf.Tensor]:
        """Method to support iteration over the data.

        Args:
            batch_index (int): Index of the batch, but unnecessary as of now.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: A tuple (input_data, true_data). input_data
                consists of one copy of each original session, with the last item
                masked. true_data is of the same form, which contains the true
                labels of the items on the positions at which the items are masked in
                input_data.
        """
        return (self.test_input, self.test_true)

    def on_epoch_end(self) -> None:
        pass

    def __generate_test_data(self):
        """Generates the input and true data by setting the x and y member variables.

        Args:
            data (tf.Tensor): The input data.
        """
        test_sequences = TensorFactory.to_sequence_tensor(self.test_data, self.N)

        if self.for_prediction:
            test_sequences = tf.pad(
                test_sequences,
                [[0, 0], [0, 1]],
                mode="constant",
                constant_values=INT_INF,
            )[:, 1:]

        num_items = self.data_description["num_items"]
        self.cloze = Cloze(num_items)

        test_data: list[tf.Tensor] = []
        true_data: list[tf.Tensor] = []

        # Produce test cases with the last item masked.
        loo_data: Tuple[tf.Tensor, tf.Tensor] = self.cloze.mask_last(test_sequences)
        (train_loo_data, true_loo_data) = loo_data
        test_data.append(train_loo_data)
        true_data.append(true_loo_data)

        # Concatenate test and true into two single tensors.
        (self.test_input, self.test_true) = (
            tf.concat(test_data, 0),
            tf.concat(true_data, 0),
        )
