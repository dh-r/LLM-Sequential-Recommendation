import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

from main.data.abstract_dataset import SplitStrategy


class MECESplit(SplitStrategy, ABC):
    """A class representing a Mutually Exclusive, Collectively Exhaustive splitting
    strategy. This means that each session in the input data is either put into train or
    test set as a whole. So the input sessions maintain all their items, as opposed to
    holdout strategy.

    Attributes:
        test_frac: A float representing the fraction of datapoints the test set has.
        num_folds: An integer representing the number of folds the k-fold split has.
        filter_non_trained_test_items: A boolean dictating whether the test items that
            do not appear as a training item should be filtered out.
    """

    @abstractmethod
    def __init__(
        self,
        test_frac: float = 0.2,
        num_folds: int = 0,
        filter_non_trained_test_items: bool = False,
    ) -> None:
        """Inits an instance of MECESplit.

        Args:
            test_frac: A float representing the test fraction. Defaults to 0.2.
            num_folds: An integer representing the number of folds. Defaults to 0.
            filter_non_trained_test_items: A boolean indicating whether to filter out
                non-trained test items. Defaults to False.

        Raises:
            ValueError: If test_frac is not a float between 0 and 1 inclusive.
            ValueError: If num_folds is not a non-negative integer.
        """
        if not (0.0 <= test_frac <= 1.0):
            raise ValueError("test_frac must be a float between 0 and 1 inclusive.")
        if num_folds < 0:
            raise ValueError("num_folds must be a non-negative integer.")
        self.test_frac: float = test_frac
        self.num_folds: int = num_folds
        self.filter_non_trained_test_items: bool = filter_non_trained_test_items

    @abstractmethod
    def _split_train_test_ids(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Splits the session ids found in the given DataFrame into train and test sets.

        Args:
            df: A pandas DataFrame containing the session data.

        Returns:
            A tuple of numpy arrays where the first element contains the train session
            ids and the second element contains the test session ids.
        """
        pass

    @abstractmethod
    def _split_k_fold_ids(
        self, df: pd.DataFrame
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Splits the session ids found in the given DataFrame into folds of train and
        validation sets. The folds are returned in a list of tuples where the first
        element of a tuple represent the train set session ids and the second element
        represents the validation set session ids.

        Args:
            df: A pandas DataFrame containing the session data.

        Returns:
            A list of tuples of numpy arrays containing session ids.
        """
        pass

    def split_train_test(
        self, input_data: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Splits the input data into train and test sets using a strategy determined by
        the "strategy" attribute.

        Args:
            input_data: A pandas DataFrame containing the input session data.

        Returns:
            A tuple containing the train and test sets respectively.
        """
        train_session_ids, test_session_ids = self._split_train_test_ids(input_data)
        return MECESplit._get_train_test_sessions_with_ids(
            input_data,
            train_session_ids,
            test_session_ids,
            self.filter_non_trained_test_items,
        )

    def split_k_fold(
        self, input_data: pd.DataFrame
    ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """Splits the given input data into folds of train and validation sets using a
        strategy determined by the "strategy" attribute.

        Args:
            input_data: A pandas DataFrame containing the input session data.

        Returns:
            A list of tuples where each tuple contains the train and validation set
            respectively.
        """
        fold_ids: list[tuple[np.ndarray, np.ndarray]] = self._split_k_fold_ids(
            input_data
        )
        folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
        for train_ids, val_ids in fold_ids:
            folds.append(
                MECESplit._get_train_test_sessions_with_ids(
                    input_data, train_ids, val_ids, self.filter_non_trained_test_items
                )
            )
        return folds

    @staticmethod
    def _get_train_test_sessions_with_ids(
        df: pd.DataFrame,
        first_session_ids: np.ndarray,
        second_session_ids: np.ndarray,
        filter_non_trained_test_items: bool,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Splits the given dataframe into two based on the given session ids.

        Args:
            df: A pandas DataFrame to split into two.
            first_session_ids: A numpy array containing the session ids of the first
                split.
            second_session_ids: A numpy array containing the session ids of the second
                split.
            filter_non_trained_test_items: A boolean indicating whether to filter out
                non-trained test items.

        Returns:
            A tuple containing two DataFrames resulting from the split.
        """
        first_df: pd.DataFrame = df[df["SessionId"].isin(first_session_ids)]
        second_df: pd.DataFrame = df[df["SessionId"].isin(second_session_ids)]
        if filter_non_trained_test_items:
            second_df = MECESplit._filter_out_non_trained_items(first_df, second_df)

        first_df = first_df.reset_index(drop=True).sort_values(["SessionId", "Time"])
        second_df = second_df.reset_index(drop=True).sort_values(["SessionId", "Time"])
        return first_df, second_df

    @staticmethod
    def _filter_out_non_trained_items(
        train_split: pd.DataFrame, test_split: pd.DataFrame
    ) -> pd.DataFrame:
        """Filters out, from the test set, the items that does not appear in the train
        set.

        Args:
            train_split: A pandas DataFrame containing the train set.
            test_split: A pandas DataFrame containing the test set.

        Returns:
            A pandas DataFrame representing the filtered test set.
        """
        test_item_ids_unique: np.ndarray = test_split["ItemId"].unique()
        train_item_ids_unique: np.ndarray = train_split["ItemId"].unique()
        common_items_index: np.ndarray = np.isin(
            test_item_ids_unique, train_item_ids_unique
        )
        items_to_keep: np.ndarray = test_item_ids_unique[common_items_index]
        test_split = test_split.loc[test_split["ItemId"].isin(items_to_keep)]

        ts_length: pd.DataFrame = test_split.groupby("SessionId").size()
        # This operation might create unit sessions, which are taken out as well.
        return test_split[
            np.in1d(test_split.SessionId, ts_length[ts_length >= 2].index)
        ]

    def can_split_k_fold(self) -> bool:
        """Indicates whether this SplitStrategy instance can apply k-fold splitting.

        Returns:
            True if k-fold splitting is possible with this instance, False otherwise.
        """
        return self.num_folds > 0
