# TODO add documentation
# TODO: techniques -> gzip
# TODO refactor the implementation such that we always work on a deep copy of the input
#  data

import inspect
import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.random_projection import GaussianRandomProjection

from main.data.side_information import (
    create_side_information,
    SideInformation,
)
from main.utils.side_encoder import SideEncoder


class DimReducer:
    def __init__(
        self,
        reduced_dim_size: int,
        reduction_config: dict,
        normalize: bool = True,
    ) -> None:
        self.reduced_dim_size: int = reduced_dim_size
        self.reduction_config: dict[str, Any] = reduction_config
        self.normalize: bool = normalize

        self.features: Optional[np.ndarray] = None
        self.reduced_features: Optional[np.ndarray] = None

    def reduce(self, data: pd.DataFrame, col_to_reduce) -> np.ndarray:
        self.features = np.stack(data[col_to_reduce].values)
        if self.reduction_config["reduction_technique"] == "random":
            self.reduced_features = DimReducer._reduce_random(
                features=self.features,
                reduced_dim_size=self.reduced_dim_size,
                config=self.reduction_config["config"],
            )
        elif self.reduction_config["reduction_technique"] == "pca":
            self.reduced_features = DimReducer._reduce_pca(
                features=self.features,
                reduced_dim_size=self.reduced_dim_size,
                config=self.reduction_config["config"],
            )
        elif self.reduction_config["reduction_technique"] == "lda":
            self.reduced_features = DimReducer._reduce_lda(
                features=self.features,
                class_labels=data["class"].values,
                reduced_dim_size=self.reduced_dim_size,
                config=self.reduction_config["config"],
            )
        elif self.reduction_config["reduction_technique"] == "autoencoder":
            self.reduced_features = DimReducer._reduce_ae(
                features=self.features,
                category_sizes=data["category_size"][0],
                reduced_dim_size=self.reduced_dim_size,
                config=self.reduction_config["config"],
            )
        else:
            raise ValueError(
                "The given reduction technique of"
                f" {self.reduction_config['reduction_technique']} is not supported."
            )

        if self.normalize:
            self.reduced_features = DimReducer._normalize(self.reduced_features)
        return self.reduced_features

    def get_features(self) -> np.ndarray:
        return self.features

    def get_reduced_features(self) -> np.ndarray:
        return self.reduced_features

    def get_configuration(self) -> dict:
        return {
            "reduced_dim_size": self.reduced_dim_size,
            "normalize": self.normalize,
            **self.reduction_config,
        }

    @staticmethod
    def _reduce_random(
        features: np.ndarray, reduced_dim_size: int, config: dict
    ) -> np.ndarray:
        grp: GaussianRandomProjection = GaussianRandomProjection(
            n_components=reduced_dim_size, **config
        )
        return grp.fit_transform(features)

    @staticmethod
    def _reduce_pca(
        features: np.ndarray, reduced_dim_size: int, config: dict
    ) -> np.ndarray:
        pca: PCA = PCA(n_components=reduced_dim_size, **config)
        return pca.fit_transform(features)

    @staticmethod
    def _reduce_lda(
        features: np.ndarray,
        class_labels: np.ndarray,
        reduced_dim_size: int,
        config: dict,
    ) -> np.ndarray:
        if reduced_dim_size > (
            max_reduced_dim_size := min(
                features.shape[1], len(np.unique(class_labels)) - 1
            )
        ):
            logging.warning(
                "LDA cannot reduce to a dimension that is larger than"
                " min(n_features, n_classes - 1). Reducing to"
                " min(n_features, n_classes - 1) instead."
            )
            n_components: int = max_reduced_dim_size
        else:
            n_components: int = reduced_dim_size
        lda: LinearDiscriminantAnalysis = LinearDiscriminantAnalysis(
            n_components=n_components, **config
        )
        return lda.fit_transform(features, class_labels)

    @staticmethod
    def _reduce_ae(
        features: np.ndarray,
        category_sizes: list[int],
        reduced_dim_size: int,
        config: dict,
    ) -> np.ndarray:
        side_information: SideInformation = create_side_information(
            features, category_sizes
        )

        side_encoder_param_names: list = [
            param.name
            for param in inspect.signature(SideEncoder.__init__).parameters.values()
            if param.name != "self"
        ]
        side_encoder_params: dict = {
            k: v for k, v in config.items() if k in side_encoder_param_names
        }
        side_encoder: SideEncoder = SideEncoder(
            side_information=side_information,
            encoder_dimension=reduced_dim_size,
            **side_encoder_params,
        )

        pretrain_param_names: list = [
            param.name
            for param in inspect.signature(SideEncoder.pretrain).parameters.values()
            if param.name != "self"
        ]
        pretrain_params: dict = {
            k: v for k, v in config.items() if k in pretrain_param_names
        }
        side_encoder.pretrain(**pretrain_params)

        return side_encoder.get_encodings(features)

    @staticmethod
    def _normalize(features: np.ndarray) -> np.ndarray:
        norms: np.ndarray = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return features / norms
