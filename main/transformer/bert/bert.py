import pandas as pd
from tensorflow import keras

from main.transformer.bert.bert_model import (
    BERTModel,
)
from main.transformer.bert.custom_generators.test_generator import (
    TestGenerator,
)
from main.transformer.bert.custom_generators.train_generator import (
    TrainGenerator,
)
from main.transformer.transformer import Transformer


class BERT(Transformer):
    """BERT4REC implementation for sequential recommendation.

    The implementation mostly follows the original
    [BERT4REC paper](https://dl.acm.org/doi/abs/10.1145/3357384.3357895).
    """

    def __init__(
        self,
        mask_prob: float = 0.4,
        **transformer_kwargs: dict,
    ) -> None:
        """Instantiates BERT.

        Args:
            mask_prob (float): The probability that an item in the sequence is masked,
                so that the model must predict its true value.
            transformer_kwargs (dict): All variables necessary to instantiate the
                parent Transformer class. For documentation on all transformer
                parameters, refer to the Transformer class.
        """
        # Variables necessary for training BERTModel
        self.mask_prob: float = mask_prob

        # We fix trans_dim_scale to accord with the original paper.
        transformer_kwargs["trans_dim_scale"] = 4

        super().__init__(**transformer_kwargs)

    def get_keras_model(self, data: pd.DataFrame) -> keras.Model:
        return BERTModel(
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
        return TrainGenerator(
            data,
            self.N,
            batch_size,
            self.mask_prob,
            self.data_description,
        )

    def get_test_generator(
        self, data: pd.DataFrame, for_prediction: bool, batch_size: int
    ) -> keras.utils.Sequence:
        return TestGenerator(
            data, self.N, self.data_description, for_prediction=for_prediction
        )

    def name(self) -> str:
        return "BERT4REC"
