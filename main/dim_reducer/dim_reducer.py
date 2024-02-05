import inspect
import logging
from typing import Any

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
    """A class for managing the dimensionality reduction of data. Can be used by
    recommender classes for dimensionality reduction of embeddings.
    """

    def __init__(
        self,
        reduced_dim_size: int,
        reduction_config: dict,
        normalize: bool = True,
    ) -> None:
        """Inits a DimReducer instance with the given config.

        Args:
            reduced_dim_size: An integer indicating the target dimension to reduce to.
            reduction_config: A dictionary containing the configuration of the
                reduction. Should contain the following keys:
                    - reduction_technique: A string indicating the reduction method.
                        Options are:
                            - random: Gaussian random projection
                            - pca: Principal Component Analysis
                            - lda: Linear Discriminant Analysis
                            - autoencoder: Autoencoder neural network
                    - config: A dictionary that contains the parameters of the chosen
                        reduction method.
            normalize: A boolean indicating whether the vectors should be normalized
                (dividing by l1 norm) after the reduction.
        """
        self.reduced_dim_size: int = reduced_dim_size
        self.reduction_config: dict[str, Any] = reduction_config
        self.normalize: bool = normalize

        # self.features: Optional[np.ndarray] = None
        # self.reduced_features: Optional[np.ndarray] = None

    def reduce(self, data: pd.DataFrame, col_to_reduce: str) -> np.ndarray:
        """Reduces the dimensionality of the column with the given name in the given
        DataFrame.

        Args:
            data: A DataFrame containing the target column.
            col_to_reduce: A string indicating the name of the target column.

        Returns:
            A numpy array containing the reduced vectors.
        """
        features = np.stack(data[col_to_reduce].values)
        if self.reduction_config["reduction_technique"] == "random":
            reduced_features = DimReducer._reduce_random(
                features=features,
                reduced_dim_size=self.reduced_dim_size,
                config=self.reduction_config["config"],
            )
        elif self.reduction_config["reduction_technique"] == "pca":
            reduced_features = DimReducer._reduce_pca(
                features=features,
                reduced_dim_size=self.reduced_dim_size,
                config=self.reduction_config["config"],
            )
        elif self.reduction_config["reduction_technique"] == "lda":
            reduced_features = DimReducer._reduce_lda(
                features=features,
                class_labels=data["class"].values,
                reduced_dim_size=self.reduced_dim_size,
                config=self.reduction_config["config"],
            )
        elif self.reduction_config["reduction_technique"] == "autoencoder":
            reduced_features = DimReducer._reduce_ae(
                features=features,
                category_sizes=data["category_size"].iloc[0],
                reduced_dim_size=self.reduced_dim_size,
                config=self.reduction_config["config"],
            )
        else:
            raise ValueError(
                "The given reduction technique of"
                f" {self.reduction_config['reduction_technique']} is not supported."
            )

        if self.normalize:
            reduced_features = DimReducer._normalize(reduced_features)
        return reduced_features

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
        """Reduces the given features using Gaussian random projection.

        Args:
            features: A numpy nd array containing the features to reduce.
            reduced_dim_size: An integer indicating the target dimension.
            config: A dictionary containing the config of the reduction.

        Returns:
            A numpy ndarray containing the reduced vectors.
        """
        grp: GaussianRandomProjection = GaussianRandomProjection(
            n_components=reduced_dim_size, **config
        )
        return grp.fit_transform(features)

    @staticmethod
    def _reduce_pca(
        features: np.ndarray, reduced_dim_size: int, config: dict
    ) -> np.ndarray:
        """Reduces the given features using Principal Component Analysis.

        Args:
            features: A numpy nd array containing the features to reduce.
            reduced_dim_size: An integer indicating the target dimension.
            config: A dictionary containing the config of the reduction.

        Returns:
            A numpy ndarray containing the reduced vectors.
        """
        pca: PCA = PCA(n_components=reduced_dim_size, **config)
        return pca.fit_transform(features)

    @staticmethod
    def _reduce_lda(
        features: np.ndarray,
        class_labels: np.ndarray,
        reduced_dim_size: int,
        config: dict,
    ) -> np.ndarray:
        """Reduces the given features using Linear Discriminant Analysis. Note that LDA
        as a dimensionality reduction method is limited in the size of the target
        dimension: max_reduced_dim_size = min(n_features, n_classes - 1). If the given
        reduced_dim_size is larger than this value, the vectors are reduced to this
        value.

        Args:
            features: A numpy nd array containing the features to reduce.
            class_labels: A numpy ndarray containing the labels of the vectors to
                reduce.
            reduced_dim_size: An integer indicating the target dimension.
            config: A dictionary containing the config of the reduction.

        Returns:
            A numpy ndarray containing the reduced vectors.
        """
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
        """Reduces the given features using a Autoencoder neural network's bottleneck
        layer.

        Args:
            features: A numpy nd array containing the features to reduce.
            category_sizes: A list of integers indicating the sizes of the categorical
                features.
            reduced_dim_size: An integer indicating the target dimension.
            config: A dictionary containing the config of the reduction.

        Returns:
            A numpy ndarray containing the reduced vectors.
        """
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
        """Normalizes the given vectors.

        Args:
            features: A numpy ndarray containing the vectors to normalize.

        Returns:
            A numpy ndarray containing the normalized vectors.
        """
        norms: np.ndarray = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return features / norms
