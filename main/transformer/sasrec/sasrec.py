from main.transformer.transformer import Transformer
from main.transformer.sasrec.sasrec_model import (
    SASRecModel,
)
from main.utils.neural_utils.custom_generators.next_item_train_generator import (
    NextItemTrainGenerator,
)
from main.utils.neural_utils.custom_generators.next_item_test_generator import (
    NextItemTestGenerator,
)
from typing import Any

import pandas as pd
import tensorflow as tf
from tensorflow import keras


class SASRec(Transformer):
    """SASRec implementation for sequential recommendation.

    The implementation mostly follows the original
    [SASRec paper](https://arxiv.org/abs/1808.09781).
    """

    def __init__(self, **transformer_kwargs) -> None:
        transformer_kwargs["trans_dim_scale"] = 1
        super().__init__(**transformer_kwargs)

    def get_keras_model(self, data: pd.DataFrame) -> keras.Model:
        return SASRecModel(
            N=self.N,
            L=self.L,
            h=self.h,
            emb_dim=self.emb_dim,
            trans_dim_scale=self.trans_dim_scale,
            drop_rate=self.drop_rate,
            activation=self.activation,
            optimizer_kwargs=self.optimizer_kwargs,
            transformer_layer_kwargs=self.transformer_layer_kwargs,
            num_items=self.data_description["num_items"],
        )

    def get_train_generator(
        self, data: pd.DataFrame, batch_size: int
    ) -> keras.utils.Sequence:
        return NextItemTrainGenerator(data, self.N, batch_size)

    def get_test_generator(
        self, data: pd.DataFrame, for_prediction: bool, batch_size: int
    ) -> keras.utils.Sequence:
        return NextItemTestGenerator(
            data, self.N, batch_size, for_prediction=for_prediction
        )

    def name(self) -> str:
        return "SASRec"
