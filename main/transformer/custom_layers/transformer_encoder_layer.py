# Copyright 2022 The KerasNLP Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transformer encoder block implementation based on `keras.layers.Layer`."""

from main.utils.neural_utils.custom_activations import to_activation
from tensorflow import keras

from enum import Enum

# from keras_nlp.layers.transformer_layer_utils import (  # isort:skip
#     merge_padding_and_attention_mask,
# )

class TransformerEncoderLayerLayout(str, Enum):
    """There are some discrepancies between transformer architectures in the literature.
    We accomodate both by letting users pass the layout as a parameter. 

    Currently we support 2 layouts from different papers:
        0. [BERT4REC](https://dl.acm.org/doi/abs/10.1145/3357384.3357895), originally from 
                [Attention is All You Need](https://arxiv.org/abs/1706.03762)
        1. [SASRec paper](https://arxiv.org/abs/1808.09781).

    We use the ordering of the components as the abbreviation of the corresponding 
    layout. 
    """
    FDRN = 0 # function (mha, ffn), dropout, residual connection, then normalization 
    NFDR = 1 # normalization, function, dropout, residual connection 

    @staticmethod
    def from_str(label):
        if label in ["FDRN", "fdrn"]:
            return TransformerEncoderLayerLayout.FDRN
        elif label in ["NFDR", "nfdr"]:
            return TransformerEncoderLayerLayout.NFDR
        else:
            raise ValueError(f"Unknown transformer encoder layout {label}")


class TransformerEncoderLayer(keras.layers.Layer):
    """Transformer encoder layer.

    Args:
        intm_dim: int, the hidden size of feedforward network.
        num_heads: int, the number of heads in the
            `keras.layers.MultiHeadAttention` layer.
        dropout: float, defaults to 0. the dropout value, shared by
            `keras.layers.MultiHeadAttention` and feedforward network.
        activation: The string representation of the activation function used
                in the network. In the transformer layer we use this for the 
                feedforward network inside the layer. 
        layer_norm_epsilon: float, defaults to 1e-5. The epsilon value in layer
            normalization components.
        kernel_initializer: string or `keras.initializers` initializer,
            defaults to "glorot_uniform". The kernel initializer for
            the dense and multiheaded attention layers.
        bias_initializer: string or `keras.initializers` initializer,
            defaults to "zeros". The bias initializer for
            the dense and multiheaded attention layers.
        name: string, defaults to None. The name of the layer.
        **kwargs: other keyword arguments.

    References:
     - [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)
    """

    def __init__(
        self,
        intm_dim: int,
        num_heads: int,
        dropout: float,
        use_causal_mask: bool,
        activation : str,
        layer_norm_epsilon: float = 1e-05,
        layout : TransformerEncoderLayerLayout = TransformerEncoderLayerLayout.FDRN,
        name: str = None,
        **kwargs,
    ):
        """The Transformer Encoder Layer.

        This class has been adopted and adapted from the Keras NLP library to fit
        our use-cases.

        Args:
            intm_dim (int): The intermediate embedding dimension.
            num_heads (int): The number of heads.
            dropout (float): The dropout rate.
            use_causal_mask (bool): Whether to use a causal mask. A causal mask means
                that the attention scores under the diagonal of the attention matrix
                are set to zero. This prevents information leak from the "future".
                Essentially, this makes a model unidirectional.
            layer_norm_epsilon (float, optional): The epsilon used in layer
                normalization. Defaults to 1e-05.
            layout (TransformerEncoderLayerLayout): The transformer encoder layout.
            name (str, optional): Name of the layer. Defaults to None.
        """
        super().__init__(name=name, **kwargs)
        self.intm_dim = intm_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = to_activation(activation)
        self.layer_norm_epsilon = layer_norm_epsilon
        self.kernel_initializer = keras.initializers.GlorotUniform(seed=0)
        self.bias_initializer = keras.initializers.Zeros()
        self._built = False
        self.supports_masking = False

        self.use_causal_mask = use_causal_mask
        self.layout = layout if isinstance(layout, TransformerEncoderLayerLayout) else TransformerEncoderLayerLayout.from_str(layout)

    def build(self, input_shape):
        # Create layers based on input shape.
        self._built = True
        feature_size = input_shape[-1]
        self._attention_head_size = int(feature_size // self.num_heads)
        self._mha_layer = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self._attention_head_size,
            value_dim=self._attention_head_size,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
        )

        self._att_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
        )
        self._ff_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
        )

        self._att_do = keras.layers.Dropout(rate=self.dropout)

        self._intm_dense = keras.layers.Dense(
            self.intm_dim,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
        )

        # The last layer never has an activation; identity is always used.
        self._output_dense = keras.layers.Dense(
            feature_size,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
        )
        self._ff_do = keras.layers.Dropout(rate=self.dropout)

    def _mha(self, inputs): 
        # Self attention.
        # The MHA layer performs three separate linear transformations on the
        # inputs, one for the queries of the attention, one for the keys of the
        # attention, and one for the values of the attention. This is the reason
        # why `inputs` occurs three times as an argument to this layer.
        return self._mha_layer(
            inputs, inputs, inputs, use_causal_mask=self.use_causal_mask
        )
    
    def _ff(self, input):
        x = self._intm_dense(input)
        x = self._output_dense(x)
        return x

    def call(self, inputs):
        """Forward pass of the TransformerEncoder.

        Args:
            inputs: a Tensor. The input data to TransformerEncoder, should be
                of shape [batch_size, sequence_length, feature_dim].

        Returns:
            A Tensor of the same shape as the `inputs`.
        """
        if not self._built:
            self._build(inputs.shape)

        if self.layout == TransformerEncoderLayerLayout.FDRN: 
            attended = self._att_norm(inputs + self._att_do(self._mha(inputs)))
            result = self._ff_norm(attended + self._ff_do(self._ff(attended)))
        elif self.layout == TransformerEncoderLayerLayout.NFDR: 
            attended = inputs + self._att_do(self._mha(self._att_norm(inputs)))
            result = attended + self._ff_do(self._ff(self._ff_norm(attended)))

        return result

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "intm_dim": self.intm_dim,
                "num_heads": self.num_heads,
                "dropout": self.dropout,
                "activation": self.activation,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            }
        )
        return config
