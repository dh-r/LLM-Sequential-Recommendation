import tensorflow as tf
from tensorflow import keras
from keras import layers


class EmbeddingLayer(layers.Layer):
    """The embedding layer implementation as described in the original
    [BERT4REC paper](https://dl.acm.org/doi/abs/10.1145/3357384.3357895).

    Its output is the sum of the item embedding and the position embedding.
    """

    def __init__(self, N: int, num_items: int, emb_dim: int) -> None:
        super(EmbeddingLayer, self).__init__()
        self.N: int = N

        # The item embedding needs to be able to convert IDs in the range
        # [0, num_items], where the ID num_items corresponds to the ID used for masking.
        # Hence, the input dimension of the Embedding is num_items + 1.
        self.item_emb: layers.Embedding = layers.Embedding(
            input_dim=num_items + 1,
            output_dim=emb_dim,
        )
        self.pos_emb: layers.Embedding = layers.Embedding(
            input_dim=N,
            output_dim=emb_dim,
        )

    def call(self, x) -> tf.Tensor:
        # Get position embeddings
        pos_embs: tf.Tensor = self.pos_emb(tf.range(start=0, limit=self.N, delta=1))

        # Get item embeddings
        item_embs: tf.Tensor = self.item_emb(x)

        # Element-wise sum to get positional item embeddings
        final_embs: tf.Tensor = pos_embs + item_embs

        return final_embs
