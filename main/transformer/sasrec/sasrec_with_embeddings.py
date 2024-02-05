import ast
import inspect
import json
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.random_projection import GaussianRandomProjection

from main.abstract_model import Model
from main.data.side_information import (
    create_side_information,
    SideInformation,
)
from main.transformer.sasrec.sasrec import SASRec
from main.utils.side_encoder import SideEncoder


class SASRecWithEmbeddings(SASRec):
    product_embeddings = None

    def __init__(
        self,
        product_embeddings_location,
        red_method: Literal["PCA", "RANDOM", "AE", "LDA"],
        red_params: dict,
        **sasrec_config,
    ) -> None:
        """Inits an instance of SASRecWithEmbeddings.

        Args:
            product_embeddings_location: A string indicating the path to the embedding
                file.
            red_method: A string indicating the dimensionality reduction method to use.
            red_params: A dictionary containing the configuration of the reduction.
            **sasrec_config: The rest of the parameters that belong to the SASRec model.
        """
        self.product_embeddings_location = product_embeddings_location
        self.red_method = red_method
        self.red_params = {k: ast.literal_eval(v) for k, v in red_params.items()}
        self.sasrec_config: dict = sasrec_config
        super().__init__(**sasrec_config)

        if SASRecWithEmbeddings.product_embeddings is None:
            SASRecWithEmbeddings.product_embeddings = pd.read_csv(
                product_embeddings_location,
            )
            SASRecWithEmbeddings.product_index_to_embedding = (
                SASRecWithEmbeddings.product_embeddings[["ItemId", "embedding"]]
                .set_index("ItemId")
                .to_dict()["embedding"]
            )
            SASRecWithEmbeddings.product_index_to_embedding = {
                k: json.loads(v)
                for k, v in SASRecWithEmbeddings.product_index_to_embedding.items()
            }
            SASRecWithEmbeddings.product_index_to_embedding = np.array(
                list(SASRecWithEmbeddings.product_index_to_embedding.values())
            )

        SASRecWithEmbeddings.product_index_to_id = (
            SASRecWithEmbeddings.product_embeddings["ItemId"].tolist()
        )
        SASRecWithEmbeddings.product_id_to_index = {
            id: index
            for index, id in enumerate(SASRecWithEmbeddings.product_index_to_id)
        }

    def train(self, train_data: Any) -> None:
        # If we use pre-computed weights, we do not have to do anything special
        # in this class, and can just use the functionality from SASRec.
        if self.filepath_weights is not None:
            super().train(train_data)
            return

        temp_config = self.sasrec_config.copy()
        temp_config["num_epochs"] = 0
        if self.red_method == "LDA":
            if self.emb_dim > (
                max_reduced_dim_size := min(
                    SASRecWithEmbeddings.product_index_to_embedding.shape[1],
                    len(np.unique(SASRecWithEmbeddings.product_embeddings["class"]))
                    - 1,
                )
            ):
                temp_config["emb_dim"] = max_reduced_dim_size
        self.temp_model = SASRec(**temp_config)
        # We do not actually train here, this is just for initializing the variables
        # we need.
        self.temp_model.train(train_data)

        if self.red_method == "PCA":
            pca = PCA(n_components=self.emb_dim)
            SASRecWithEmbeddings.product_index_to_embedding_red = pca.fit_transform(
                SASRecWithEmbeddings.product_index_to_embedding
            )
        elif self.red_method == "RANDOM":
            grp = GaussianRandomProjection(n_components=self.emb_dim)
            SASRecWithEmbeddings.product_index_to_embedding_red = grp.fit_transform(
                SASRecWithEmbeddings.product_index_to_embedding
            )
        elif self.red_method == "AE":
            side_information: SideInformation = create_side_information(
                SASRecWithEmbeddings.product_index_to_embedding, []
            )

            side_encoder_param_names: list = [
                param.name
                for param in inspect.signature(SideEncoder.__init__).parameters.values()
                if param.name != "self"
            ]
            side_encoder_params: dict = {
                k: v
                for k, v in self.red_params.items()
                if k in side_encoder_param_names
            }
            side_encoder: SideEncoder = SideEncoder(
                side_information=side_information,
                encoder_dimension=self.emb_dim,
                **side_encoder_params,
            )

            pretrain_param_names: list = [
                param.name
                for param in inspect.signature(SideEncoder.pretrain).parameters.values()
                if param.name != "self"
            ]
            pretrain_params: dict = {
                k: v for k, v in self.red_params.items() if k in pretrain_param_names
            }
            side_encoder.pretrain(**pretrain_params)

            SASRecWithEmbeddings.product_index_to_embedding_red = (
                side_encoder.get_encodings(
                    SASRecWithEmbeddings.product_index_to_embedding
                )
            )
        elif self.red_method == "LDA":
            class_labels = SASRecWithEmbeddings.product_embeddings["class"]
            if self.emb_dim > (
                max_reduced_dim_size := min(
                    SASRecWithEmbeddings.product_index_to_embedding.shape[1],
                    len(np.unique(class_labels)) - 1,
                )
            ):
                n_components: int = max_reduced_dim_size
            else:
                n_components: int = self.emb_dim
            lda: LinearDiscriminantAnalysis = LinearDiscriminantAnalysis(
                n_components=n_components, **self.red_params
            )
            SASRecWithEmbeddings.product_index_to_embedding_red = lda.fit_transform(
                SASRecWithEmbeddings.product_index_to_embedding, class_labels
            )
        else:
            raise Exception(
                f"Unknown reduce method for embeddings. Got {self.red_method}"
            )

        # Inject the embeddings in the correct order, defined by the ID reducer.
        ordering = list(
            dict(sorted(self.temp_model.id_reducer.id_lookup.items())).values()
        )
        ordering = [SASRecWithEmbeddings.product_id_to_index[item] for item in ordering]
        reduced_embeddings = np.array(
            [SASRecWithEmbeddings.product_index_to_embedding_red[ordering]]
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
        # in this class, and can just use the functionality from SASRec.
        if self.filepath_weights is not None:
            return super().get_keras_model(data)

        return self.temp_model.model

    def predict(self, predict_data: Any, top_k: int = 10) -> dict[int, np.ndarray]:
        # If we use pre-computed weights, we do not have to do anything special
        # in this class, and can just use the functionality from SASRec.
        if self.filepath_weights is not None:
            return super().predict(predict_data, top_k)

        return self.temp_model.predict(predict_data, top_k)

    def name(self) -> str:
        return "LLM2SASRec"
