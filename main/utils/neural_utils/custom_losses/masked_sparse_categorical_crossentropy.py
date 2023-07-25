import tensorflow as tf
from main.utils.neural_utils.custom_preprocessors.tensor_factory import (
    TensorFactory,
)

from keras import backend as K


def masked_sparse_categorical_crossentropy(y_true: tf.Tensor, y_pred: tf.Tensor):
    """The masked sparse categorical cross-entropy method calculates the
    cross-entropy loss on the predictions for true label of the masked items.

    Args:
        y_true (tf.Tensor): A tensor containing the true identities of the masked
            items. The tensor looks as follows:
            true_data = [
                [PADDING_TARGET, PADDING_TARGET, PADDING_TARGET, 4]
                [PADDING_TARGET, PADDING_TARGET, PADDING_TARGET, 8]
                [9, PADDING_TARGET, PADDING_TARGET, 12]
            ]
            We use boolean_mask to convert this tensor to the tensor [4, 8, 9, 12]
            containing just the true identities.
        y_pred (tf.Tensor): The predicted probability distributions for each of the
            masked items.
            So, the shape of this tensor is (num_masked_items_in_batch, num_items)
            because the probability distributions have dimension (num_items).

    Returns:
        tf.Tensor: A tensor containing the cross-entropy losses.
    """
    y_true_masked = tf.boolean_mask(
        y_true, tf.not_equal(y_true, TensorFactory.PADDING_TARGET)
    )

    return K.mean(K.sparse_categorical_crossentropy(y_true_masked, y_pred))
