import tensorflow as tf

from main.transformer.custom_layers.transformer_encoder_layer import (
    TransformerEncoderLayer,
)
from main.transformer.transformer_model import (
    TransformerModel,
)
from main.utils.neural_utils.custom_preprocessors.tensor_factory import (
    TensorFactory,
)


class BERTModel(TransformerModel):
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """The call method defines how the input tensor is propagated through the model
        to form the output tensor.

        Args:
            inputs (tf.Tensor): An input tensor of the shape (batch_size,
                sequence_length,) containing masked sessions.

        Returns:
            tf.Tensor: The output tensor.
        """
        relevant: tf.Tensor = tf.equal(inputs, self.mask_target_used)
        inputs: tf.Tensor = tf.where(
            tf.equal(inputs, TensorFactory.PADDING_TARGET),
            self.mask_target_used,
            inputs,
        )
        embeddings: tf.Tensor = self.embedding_layer(inputs)
        transformations: tf.Tensor = self.transformer(embeddings)
        relevant_transformations: tf.Tensor = tf.boolean_mask(transformations, relevant)
        predictions: tf.Tensor = self.projection_head(relevant_transformations)
        return predictions

    def get_transformer_layer(self):
        return TransformerEncoderLayer(
            self.emb_dim * self.trans_dim_scale,
            self.h,
            self.drop_rate,
            use_causal_mask=False,  # important for BERT4Rec!
            activation=self.activation,
            **self.transformer_layer_kwargs,
        )
