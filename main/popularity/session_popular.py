"""This file contains the popularity recommender baselines for sessions."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from main.abstract_model import Model


class SessionBasedPopular(Model):
    def __init__(self, is_verbose: bool = False, cores: int = 1) -> None:
        """Inits an instance."""
        super().__init__(is_verbose, cores)
        self.items: Optional[np.ndarray] = None
        self.n_items: int = 0

    def train(self, train_data: pd.DataFrame) -> None:
        """Trains the recommender with the given DataFrame.

        Args:
            train_data: A pandas DataFrame containing session data.
        """
        self.items = train_data["ItemId"].value_counts().index.values.astype(int)
        self.n_items = len(self.items)

    def predict(
        self, predict_data: Dict[int, List[str]], top_k: int = 10
    ) -> Dict[int, np.ndarray]:
        """Generates predictions for the given test cases.

        Args:
            predict_data: A dictionary containing as values lists of integers that
                represent sessions and as keys integer identifiers of these sessions.
                The identifiers are used to match test cases with ground truths during
                testing.
            top_k: An integer representing the chosen top-k value, i.e. the number of
                items in a recommendation slate. Defaults to 10.

        Returns:
            A dictionary containing identifiers to test cases as keys and recommendation
            slates as values, where a slate is represented as a numpy array of item ids.
            Test case identifiers can be user ids, session ids, etc.

        Raises:
            ValueError: If top_k argument exceeds the number of possible items to
                recommend.
        """
        if top_k > self.n_items:
            raise ValueError(
                f"The given top_k value of {top_k} exceeds the number of possible"
                f" items to recommend, which is {self.n_items}. top_k must be less than"
                f" or equal to this number."
            )

        predictions: Dict[int, np.ndarray] = {}
        for case_id in tqdm(
            predict_data.keys(), mininterval=5, disable=not self.is_verbose
        ):
            predictions[case_id] = self.items[:top_k]

        return predictions

    def name(self) -> str:
        return "Popularity"
