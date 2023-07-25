from main.utils.neural_utils.custom_layers.bias_layer import (
    BiasLayer,
)
from main.utils.neural_utils.custom_activations import to_activation

import tensorflow as tf
from tensorflow import keras
from keras import layers

from keras.activations import softmax


class ProjectionHead(layers.Layer):
    """The output layer implementation as described in the original
    [BERT4REC paper](https://dl.acm.org/doi/abs/10.1145/3357384.3357895).

    It accepts a batch of the relevant transformations (the transformations that need to
     be converted into a probability distribution for predicting the next item).
    """

    def __init__(self, emb_dim, item_embedder, activation) -> None:
        super(ProjectionHead, self).__init__()

        self.dense: layers.Dense = layers.Dense(
            emb_dim,
            activation=to_activation(activation),
            )

        self.item_embedder: layers.Embedding = item_embedder

        self.bias: BiasLayer = BiasLayer()
        self.softmax: layers.Softmax = layers.Softmax()

    def call(self, x: tf.Tensor, training=None) -> tf.Tensor:
        result: tf.Tensor = self.dense(x)

        # Reuse the item embedding matrix to convert the transformation back to a value
        # for each item.
        # Multiplying an embedding with the transpose of the embedding matrix is
        # actually the inverse of the embedding function.
        # Note that we remove the last column from the transpose, since this last
        # column corresponds to the embedding of the mask item ID, which can not be the
        # true identity of the item.
        self.transpose_embedding_matrix: tf.Variable = tf.linalg.matrix_transpose(
            self.item_embedder.embeddings
        )[:, :-1]
        result: tf.Tensor = tf.linalg.matmul(result, self.transpose_embedding_matrix)

        # Add bias.
        result = self.bias(result)

        # Convert to a probability distribution.
        result = self.softmax(result)
        return result
