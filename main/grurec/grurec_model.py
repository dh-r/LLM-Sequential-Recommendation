import tensorflow as tf
from keras import layers
from tensorflow import keras

from main.utils.neural_utils.custom_layers.projection_head import (
    ProjectionHead,
)
from main.utils.neural_utils.custom_losses.masked_sparse_categorical_crossentropy import (
    masked_sparse_categorical_crossentropy,
)
from main.utils.neural_utils.custom_preprocessors.tensor_factory import (
    TensorFactory,
)


class GRURecModel(keras.Model):
    def __init__(
        self,
        num_items: int,
        emb_dim: int,
        hidden_dim: int,
        drop_rate: float,
        optimizer_kwargs: dict,
        activation: str,
    ) -> None:
        """The GRU4Rec Model.

        It defines how the batches should be propagated through the network, and how
        the network should be trained.

        Args:
            num_items (int): The number of items in the input data. This is used
                to choose the correct dimension for some of the layers.
            emb_dim (int): The dimension of the embeddings used throughout the model.
                Note that for GRU4Rec this is a vital parameter, since the hidden
                state defines how much can be remembered from previous timesteps.
                Essentially, it defines the size of the "memory" of the model.
            hidden_dim (int): The dimension of the hidden state of the GRU
                layer.
            drop_rate (float): The drop rate on the input embedding and the output
                of the GRU layer. Defaults to 0.2.
            optimizer_kwargs (dict): The keyword arguments for the Adam optimizer.
            activation (str): The string representation of the activation function used
                in the network. In GRU4Rec we only use the activation in the projection
                head.
        """
        super().__init__()

        self.num_items = num_items
        self.mask_target_used = num_items  # for clarity
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.drop_rate = drop_rate
        self.optimizer_kwargs = optimizer_kwargs

        # Create Embedding layer.
        self.embedding_layer = layers.Embedding(
            input_dim=num_items + 1, output_dim=emb_dim
        )
        self.embedding_dropout = layers.Dropout(rate=drop_rate)

        # Create GRU layer.
        self.gru_layer: keras.Sequential = keras.Sequential(
            [
                layers.GRU(hidden_dim, return_sequences=True),
                layers.Dropout(rate=drop_rate),
            ]
        )

        # Create output dense layer.
        self.proj_head = ProjectionHead(emb_dim, self.embedding_layer, activation)

        self.compile_model()

    def call(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:
        """Define how input is transformed into the output.

        Args:
            inputs (tf.Tensor): An input tensor.
            training (bool): Whether the call is made for training or prediction.

        Returns:
            tf.Tensor: The output tensor.
        """
        # Get a boolean matrix indicating where the paddings are.
        padding = tf.equal(inputs, TensorFactory.PADDING_TARGET)

        # We replace the padding with the mask target, which is interpretable
        # by the embedding layer.
        inputs: tf.Tensor = tf.where(
            padding,
            self.mask_target_used,
            inputs,
        )

        embeddings = self.embedding_layer(inputs)
        embeddings = self.embedding_dropout(embeddings)

        transformations = self.gru_layer(embeddings)

        if training:
            # We want to produce a probability vector for all positions that
            # do not correspond to padding.
            relevant = tf.logical_not(padding)
            relevant_transformations = tf.boolean_mask(transformations, relevant)
        else:
            # During prediction/validation, we just want the last transformation to
            # predict the next-item.
            relevant_transformations = transformations[:, -1]

        result = self.proj_head(relevant_transformations)
        return result

    def compile_model(self):
        # Construct optimizer and compile.
        adam_w = keras.optimizers.experimental.AdamW(**self.optimizer_kwargs)
        self.compile(adam_w, loss=masked_sparse_categorical_crossentropy)
