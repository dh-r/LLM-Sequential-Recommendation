"""This module contains the abstract Model class."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import logging

# Initialize the logger.
logging.basicConfig(level=logging.INFO)


class Model(ABC):
    """An abstract class representing a model object.

    Algorithm-specific model implementations must inherit from this class.

    Attributes:
        is_trained: A boolean indicating whether the Model instance has been trained or
            not.
        is_verbose: A boolean indicating whether the console output of this model will
            be verbose or not.
        cores: An integer indicating the number of cores available for computation.
    """

    @abstractmethod
    def __init__(self, is_verbose: bool = False, cores: int = 1) -> None:
        """Inits a Model instance."""
        self.is_trained: bool = False
        self.is_verbose: bool = is_verbose
        self.cores: int = cores

    @abstractmethod
    def train(self, train_data: Any) -> None:
        """Trains the object with the given Dataset.

        Args:
            train_data: The data to be used for training. The type of this parameter can
                differ per Model implementation.
        """
        pass

    @abstractmethod
    def predict(self, predict_data: Any, top_k: int = 10) -> dict[int, np.ndarray]:
        """Generates predictions for the given dataset.

        Args:
            predict_data: The data for which recommendations will be generated. The type
                of this parameter can differ per Model implementation.
            top_k: An integer representing the chosen top-k value, i.e. the number of
                items in a recommendation slate. Defaults to 10.

        Returns:
            A dictionary containing identifiers to test cases as keys and recommendation
            slates as values, where a slate is represented as an array of item ids. Test
            case identifiers can be user ids, session ids, etc.
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """Name of the model.

        Returns:
            str: The name of the model.
        """
        pass
