import pandas as pd
import numpy as np
from typing import Optional

from sklearn.model_selection import KFold

from main.data.mece_split import MECESplit


class RandomSplit(MECESplit):
    """A class representing a randomized split of sessions.

    Attributes:
        random_state: An integer representing the random state seed.
    """

    def __init__(
        self,
        test_frac: float = 0.2,
        num_folds: int = 0,
        filter_non_trained_test_items: bool = False,
        random_state: int = None,
    ) -> None:
        """Inits a RandomSplit instance with the given arguments.

        Args:
            test_frac: A float representing the test fraction. Defaults to 0.2.
            num_folds: An integer representing the number of folds. Defaults to 0.
            filter_non_trained_test_items: A boolean indicating whether to filter out
                non-trained test items. Defaults to False.
            random_state: An integer representing the random seed to use for the splits.
                Defaults to None, in which case the seed is selected randomly.

        Raises:
            ValueError: If test_frac is not a float between 0 and 1 inclusive.
            ValueError: If num_folds is not a positive integer.
        """
        super().__init__(test_frac, num_folds, filter_non_trained_test_items)
        self.random_state: Optional[int] = random_state

    def _split_train_test_ids(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Splits the session ids in the given DataFrame randomly. The fraction of the
        split is determined by the test_frac attribute of the object.

        Args:
            df: A pandas DataFrame containing the session data.

        Returns:
            A tuple of numpy arrays where the first element contains the train session
            ids and the second element contains the test session ids.
        """
        session_ids: np.ndarray = df["SessionId"].unique()
        rng: np.random.Generator = np.random.default_rng(self.random_state)
        rng.shuffle(session_ids)
        train_ids, test_ids = np.array_split(
            session_ids, [int((1.0 - self.test_frac) * len(session_ids))]
        )
        return train_ids, test_ids

    def _split_k_fold_ids(
        self, df: pd.DataFrame
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Splits the session ids in the given DataFrame into folds of train and
        validation sets randomly. The number of folds is determined by the num_folds
        attribute of the object.

        Args:
            df: A pandas DataFrame containing the session data.

        Returns:
            A list of tuples of numpy arrays where the first element contains the train
            session ids and the second element contains the validation session ids.
        """
        k_fold: KFold = KFold(
            n_splits=self.num_folds, shuffle=True, random_state=self.random_state
        )
        session_ids: np.ndarray = df["SessionId"].unique()
        fold_ids: list[tuple[np.ndarray, np.ndarray]] = []
        for train_idx, val_idx in k_fold.split(session_ids):
            train_ids: np.ndarray = session_ids[train_idx]
            val_ids: np.ndarray = session_ids[val_idx]
            fold_ids.append((train_ids, val_ids))
        return fold_ids
