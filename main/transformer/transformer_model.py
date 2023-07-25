from main.utils.config_util import extract_config
from main.utils.neural_utils.custom_losses.masked_sparse_categorical_crossentropy import (
    masked_sparse_categorical_crossentropy,
)
from main.utils.neural_utils.custom_layers.projection_head import (
    ProjectionHead,
)
from main.transformer.custom_layers.embedding_layer import (
    EmbeddingLayer,
)

import tensorflow as tf
from tensorflow import keras
from abc import abstractmethod


class TransformerModel(keras.Model):
    """The BERT4REC implementation in Keras. This class is responsible for defining the
        actual model described in the original
        [BERT4REC paper](https://dl.acm.org/doi/abs/10.1145/3357384.3357895).

    Args:
        keras.Model: The base implementation of a Keras model. It defines the abstract
            methods that BERTModel should implement.
    """

    def __init__(
        self,
        N: int,
        L: int,
        h: int,
        emb_dim: int,
        trans_dim_scale: int,
        drop_rate: float,
        activation : str,
        optimizer_kwargs: dict,
        transformer_layer_kwargs : dict,
        num_items: int,
    ) -> None:
        """The base Transformer model.

        Args:
            N (int): The length of the input sequence.
            L (int): The number of transformer layers.
            h (int): The number of heads in the multi-head attention mechanism of each
                transformer layer.
            emb_dim (int): The dimension of the embedding of each item.
            trans_dim_scale (int). The intermediate dimension of the transformer layer
                as a multiple of the embedding dimension.
            drop_rate (float): The drop rate inside the transformer layer.
            activation (str): The string representation of the activation function used
                in the network. In the transformer models we use this for the 
                feedforward networks inside the transformer encoder layers and for 
                the projection head.
            optimizer_kwargs (dict, optional): Keyword arguments for the Adam optimizer.
            transformer_layer_kwargs (dict, optional): Keyword arguments for the 
                transformer encoder layer.
            num_items (int): The number of unique items in the data.
        """
        super(TransformerModel, self).__init__()

        # Variables set through configuration
        self.N: int = N
        self.L: int = L
        self.h: int = h
        self.emb_dim: int = emb_dim
        self.trans_dim_scale = trans_dim_scale
        self.drop_rate: float = drop_rate
        self.activation : str = activation
        self.optimizer_kwargs : dict = optimizer_kwargs
        self.transformer_layer_kwargs : dict = transformer_layer_kwargs
        self.num_items: int = num_items
        self.mask_target_used = self.num_items  # for clarity

        self.embedding_layer = self.get_embedding_layer()

        self.transformer: keras.Sequential = keras.Sequential()
        for _ in range(self.L):
            self.transformer.add(self.get_transformer_layer())

        self.projection_head = self.get_projection_head()

        self.compile_model()

    def compile_model(self) -> None:
        """Compiles the model."""
        optimizer = self.get_optimizer()
        self.compile(optimizer=optimizer, loss=masked_sparse_categorical_crossentropy)

    def get_config(self):
        return extract_config(self)

    def get_embedding_layer(self):
        return EmbeddingLayer(self.N, self.num_items, self.emb_dim)

    @abstractmethod
    def get_transformer_layer(self, **transformer_layer_kwargs):
        pass

    def get_projection_head(self):
        return ProjectionHead(self.emb_dim, self.embedding_layer.item_emb, self.activation)

    def get_optimizer(self):
        return tf.keras.optimizers.experimental.AdamW(**self.optimizer_kwargs)
