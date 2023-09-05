from main.utils.neural_utils.custom_preprocessors.tensor_factory import (
    TensorFactory,
)

import numpy as np
import tensorflow as tf

from typing import Tuple


class Cloze:
    """Cloze is a class to encapsulate all functionality needed to convert sequences
    into actual training instances by masking particular items."""

    def __init__(self, mask_target: int) -> None:
        self.mask_target = mask_target

    def mask_random(
        self, data: tf.Tensor, mask_prob: float
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Randomly masks items in the sequence, where the probability of an item
        getting masked is equal to mask_prob.

        Args:
            data (tf.Tensor): The input sequences.
            mask_prob (float): The probability that a particular item in the sequence
                is masked.

        Raises:
            Exception: Raise exception if MASK_TARGET has not yet been set.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: Tuple of the form (train_data, true_data),
                where the former represents the sequences with masked items, and the
                latter represents the sequences with the true label of the masked items.
        """
        mask: np.ndarray = np.random.binomial(1, mask_prob, data.shape)
        return self.__convert_train(data, mask)

    def mask_last(self, data: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Masks the last item in the sequence.

        Args:
            data (tf.Tensor): The input sequences.

        Raises:
            Exception: Raise exception if MASK_TARGET has not yet been set.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: Tuple of the form (train_data, true_data),
                where the former represents the sequences with masked items, and the
                latter represents the sequences with the true label of the masked items.
        """
        mask: np.ndarray = np.zeros(data.shape)
        mask[:, -1] = 1
        return self.__convert_train(data, mask)

    def __convert_train(
        self, data: tf.Tensor, mask: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Performs the mask on the data.

        Args:
            data (tf.Tensor): The input sequences.
            mask (tf.Tensor): The mask that needs to be applied to the data. This tensor
            is binary, where a one represents that the entry in the data should be
            masked.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: Tuple of the form (train_data, true_data),
                where the former represents the sequences with masked items, and the
                latter represents the sequences with the true label of the masked items.
            Example:
                data = [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                ]
                mask = [
                    [0, 0, 0, 1],
                    [0, 0, 0, 1],
                    [1, 0, 0, 1],
                ]

                train_data = [
                    [1, 2, 3, MASK_TARGET],
                    [5, 6, 7, MASK_TARGET],
                    [MASK_TARGET, 10, 11, MASK_TARGET],
                ]

                true_data = [
                    [PADDING_TARGET, PADDING_TARGET, PADDING_TARGET, 4]
                    [PADDING_TARGET, PADDING_TARGET, PADDING_TARGET, 8]
                    [9, PADDING_TARGET, PADDING_TARGET, 12]
                ]

            Note that we use PADDING_TARGET for masking in true_data out of convenience,
            because then all the entries that do not need to be considered are set to
            PADDING_TARGET.
        """
        non_padding_entries = tf.cast(
            tf.not_equal(data, TensorFactory.PADDING_TARGET), dtype=tf.int32
        )
        mask = tf.multiply(mask, non_padding_entries)
        new_train_targets = mask * self.mask_target
        new_true_targets = (1 - mask) * TensorFactory.PADDING_TARGET
        train_masked = tf.multiply(data, (1 - mask)) + new_train_targets
        true_masked = tf.multiply(data, mask) + new_true_targets
        return (train_masked, true_masked)
