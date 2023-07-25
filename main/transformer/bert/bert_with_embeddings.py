from typing import Any

from tensorflow import keras
from main.abstract_model import Model
from main.transformer.bert.bert import BERT
from sklearn.decomposition import PCA
import shutil
import os
import pandas as pd
import json
import numpy as np


class BERTWithEmbeddings(BERT):
    product_embeddings = None

    def __init__(self, product_embeddings_location, **bert_config) -> None:
        self.bert_config = bert_config
        self.product_embeddings_location = product_embeddings_location

        super().__init__(**bert_config)

        if BERTWithEmbeddings.product_embeddings is None:
            BERTWithEmbeddings.product_embeddings = pd.read_csv(
                product_embeddings_location,
                compression="gzip",
                usecols=["global_product_id", "name", "ada_embedding"],
            )
            BERTWithEmbeddings.product_id_to_name = (
                BERTWithEmbeddings.product_embeddings[["global_product_id", "name"]]
                .set_index("global_product_id")
                .to_dict()["name"]
            )
            BERTWithEmbeddings.product_index_to_embedding = (
                BERTWithEmbeddings.product_embeddings[
                    ["global_product_id", "ada_embedding"]
                ]
                .set_index("global_product_id")
                .to_dict()["ada_embedding"]
            )
            BERTWithEmbeddings.product_index_to_embedding = {
                k: json.loads(v)
                for k, v in BERTWithEmbeddings.product_index_to_embedding.items()
            }
            BERTWithEmbeddings.product_index_to_embedding = np.array(
                list(BERTWithEmbeddings.product_index_to_embedding.values())
            )

        BERTWithEmbeddings.product_index_to_id = list(
            BERTWithEmbeddings.product_id_to_name.keys()
        )
        BERTWithEmbeddings.product_id_to_index = {
            id: index for index, id in enumerate(BERTWithEmbeddings.product_index_to_id)
        }

    def train(self, train_data: Any) -> None:
        # If we use pre-computed weights, we do not have to do anything special 
        # in this class, and can just use the functionality from BERT.
        if self.filepath_weights is not None:
            super().train(train_data)
            return 

        temp_config = self.bert_config.copy()

        temp_config["num_epochs"] = 0

        self.temp_model = BERT(**temp_config)

        # We do not actually train here, this is just for initializing the variables 
        # we need. 
        self.temp_model.train(train_data)

        pca = PCA(n_components=self.emb_dim)
        BERTWithEmbeddings.product_index_to_embedding_pca = pca.fit_transform(
            BERTWithEmbeddings.product_index_to_embedding
        )

        # Inject the embeddings in the correct order, defined by the ID reducer.
        ordering = list(
            dict(sorted(self.temp_model.id_reducer.id_lookup.items())).values()
        )
        ordering = [BERTWithEmbeddings.product_id_to_index[item] for item in ordering]
        reduced_embeddings = np.array(
            [BERTWithEmbeddings.product_index_to_embedding_pca[ordering]]
        )

        # We do not have a pre-computed embedding for the mask, so we take the original
        # and merge it with the pre-computed openAI embeddings.
        mask_embedding = np.array(
            [[self.temp_model.model.embedding_layer.item_emb.embeddings[-1]]]
        )
        reduced_embeddings = np.concatenate(
            [reduced_embeddings, mask_embedding], axis=1
        )

        # Set the weights
        self.temp_model.model.embedding_layer.item_emb.set_weights(reduced_embeddings)
        self.temp_model.model.embedding_layer.item_emb.trainable = True
        self.temp_model.model.compile_model()

        super().train(train_data)

    def get_keras_model(self, data: pd.DataFrame) -> Model:
        # If we use pre-computed weights, we do not have to do anything special 
        # in this class, and can just use the functionality from BERT.
        if self.filepath_weights is not None:
            return super().get_keras_model(data)

        return self.temp_model.model


    def predict(self, predict_data: Any, top_k: int = 10) -> dict[int, np.ndarray]:
        # If we use pre-computed weights, we do not have to do anything special 
        # in this class, and can just use the functionality from BERT.
        if self.filepath_weights is not None:
            return super().predict(predict_data, top_k)
        
        return self.temp_model.predict(predict_data, top_k)

    def name(self) -> str:
        return "LLM2BERT4Rec"
