import tensorflow as tf
from tensorflow import keras
from keras import layers


class BiasLayer(layers.Layer):
    """Bias layer. This layer is part of the output projection layer.
    Given that the output projection layer is the same for all transformations of the
    last output layer, the shape of the bias layer is input_shape[2:], because we skip
    the batch_size dimension and the N dimension. This ensures the bias layer dimension
    matches the dimensions used in the output layer as a whole.
    """

    def __init__(self, *args, **kwargs) -> None:
        super(BiasLayer, self).__init__(*args, **kwargs)

    def build(self, input_shape) -> None:
        self.bias = self.add_weight(
            "bias", shape=input_shape[2:], initializer="zeros", trainable=True
        )

    def call(self, x) -> tf.Tensor:
        return x + self.bias
