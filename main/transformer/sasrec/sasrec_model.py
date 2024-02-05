import tensorflow as tf
from tensorflow import keras

from main.transformer.custom_layers.transformer_encoder_layer import (
    TransformerEncoderLayer,
)
from main.transformer.transformer_model import (
    TransformerModel,
)
from main.utils.neural_utils.custom_preprocessors.tensor_factory import (
    TensorFactory,
)


class SASRecModel(TransformerModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # SASRec additionally uses a dropout layer on the embeddings.
        self.embedding_dropout = keras.layers.Dropout(self.drop_rate)

    def call(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:
        """The call method defines how the input tensor is propagated through the model
        to form the output tensor.

        Args:
            inputs (tf.Tensor): An input tensor of the shape (batch_size,
                sequence_length,) containing masked sessions.
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

        transformations = self.transformer(embeddings)
        if training:
            # We want to produce a probability vector for all positions that
            # do not correspond to padding.
            relevant = tf.logical_not(padding)
            relevant_transformations: tf.Tensor = tf.boolean_mask(
                transformations, relevant
            )
        else:
            # During prediction/validation, we just want the last transformation to
            # predict the next-item.
            relevant_transformations = transformations[:, -1]

        predictions = self.projection_head(relevant_transformations)
        return predictions

    def get_transformer_layer(self):
        return TransformerEncoderLayer(
            self.emb_dim * self.trans_dim_scale,
            self.h,
            self.drop_rate,
            use_causal_mask=True,  # important for SASRec!
            activation=self.activation,
            **self.transformer_layer_kwargs,
        )
