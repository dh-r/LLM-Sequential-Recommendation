"""This module contains a class representing a session-based dataset."""

import datetime
import logging
import math
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from main.data.abstract_dataset import Dataset, SplitStrategy


class SessionDataset(Dataset):
    """An abstract class representing a session-based dataset object. The underlying
    type of data structure is a pandas Dataframe of type
    ["SessionId": int, "ItemId": str, "Time": float, "Reward": float].
    """

    def __init__(
        self,
        filepath_or_bytes: Union[str, bytes],
        sample_size: int = None,
        sample_random_state: int = None,
        n_withheld: int = 1,
        evolving: bool = False,
    ) -> None:
        """Inits a SessionDataset object with the given configuration.

        Args:
            sample_size: An integer representing the sample size if the sessions must be
                sampled after loading. Defaults to None, in which case sampling is not
                done.
            sample_random_state: An integer representing the random state seed for
                sampling of sessions. Defaults to None, in which case the seed is
                selected randomly.
            n_withheld: An integer representing the number of last items to withhold
                from each session. Defaults to 1.
            evolving: A boolean indicating whether evolving session testing is applied,
                in which case multiple test cases are generated from a single session
                in a sliding window fashion. Defaults to False.

        Raises:
            ValueError: If sample_size not None and not a positive integer.
        """
        if sample_size is not None and sample_size < 1:
            raise ValueError("Sample size must be a positive integer.")
        super().__init__(filepath_or_bytes)
        self.sample_size: Optional[int] = sample_size
        self.sample_random_state: Optional[int] = sample_random_state
        self.n_withheld: int = n_withheld
        self.evolving: bool = evolving

    def load(self) -> None:
        """Loads the data using the filepath_or_bytes attribute."""
        if isinstance(self.filepath_or_bytes, bytes):
            file: BytesIO = BytesIO(self.filepath_or_bytes)
        else:
            file: str = self.filepath_or_bytes

        self.input_data = pd.read_csv(
            file,
            dtype={
                "SessionId": int,
                "ItemId": int,
                "Time": float,
                "Reward": float,
            },
        )

    def sample_input_data(self, sample_size: int, random_state: int = None) -> None:
        """Sets the input_data of the instance to a smaller sample of the original.

        Args:
            sample_size: An integer representing the sample size.
            random_state: An integer representing the random state seed. Defaults to
                None, in which case the seed is selected randomly.

        Returns:
            ValueError: If the given sample size exceeds the total number of sessions.
        """
        if sample_size > (n_sessions := self.get_unique_sample_count()):
            raise ValueError(
                f"The sample size cannot exceed the unique number of sessions."
                f" There are {n_sessions} sessions in the input data."
            )

        rng: np.random.Generator = np.random.default_rng(random_state)
        sampled_session_ids: np.ndarray = rng.choice(
            self.input_data["SessionId"].unique(), sample_size, replace=False
        )
        sample_mask: pd.Series = self.input_data["SessionId"].isin(sampled_session_ids)
        sampled_df: pd.DataFrame = self.input_data.loc[sample_mask].reset_index(
            drop=True
        )
        self.input_data = sampled_df

    def get_unique_item_count(self) -> int:
        """Returns the unique number of items in the input data.

        Returns:
            An integer representing the number of unique items.

        Raises:
            Exception: If input data has not been set.
        """
        if self.input_data is None:
            raise Exception("The dataset has not been loaded.")

        return self.input_data["ItemId"].nunique()

    def get_unique_sample_count(self) -> int:
        """Returns the unique number of sessions in the input data.

        Returns:
            An integer representing the number of unique sessions.

        Raises:
            Exception: If input data has not been set.
        """
        if self.input_data is None:
            raise Exception("The dataset has not been loaded.")

        return self.input_data["SessionId"].nunique()

    def get_item_counts(self) -> dict[int, int]:
        """Get the number of occurrences per item in the input data.

        Returns:
            An dict with the count per item id. For example get_items_counts[10]
            represents the count for item 10.

        Raises:
            Exception: If input data has not been set.
        """

        if self.input_data is None:
            raise Exception("The dataset has not been loaded.")

        return self.input_data["ItemId"].value_counts().to_dict()

    def get_sample_counts(self) -> dict[int, int]:
        """Get the number of interactions per sample in the input data.

        Returns:
            A dict with the count per sample id. For example get_sample_counts[10]
            represents the number of interactions for sample 10.

        Raises:
            Exception: If input data has not been set.
        """

        if self.input_data is None:
            raise Exception("The dataset has not been loaded.")

        return self.input_data["SessionId"].value_counts().to_dict()

    def get_num_interactions(self) -> int:
        """Get the total number of interactions in the input data.

        Returns:
            int: The total number of interactions in the input data.
        """
        return len(self.input_data)

    def get_average_session_length(self) -> float:
        """Get the average session length.

        Returns:
            float: The average session length.
        """
        return float(self.get_num_interactions()) / self.get_unique_sample_count()

    def load_and_split(
        self,
        split_strategy: SplitStrategy = None,
    ) -> None:
        """Loads and splits dataset into train and test set and prepares the test set
        for evaluation. If the given SplitStrategy allows, the method also generates
        k-folds on the train set and prepares the folds for evaluation.

        Args:
            split_strategy: An instance of a concrete SplitStrategy to use as the split
                logic. Defaults to None, in which case TemporalSplit with 0.2 test_frac
                and 2 folds is used.

        Raises:
            ValueError: If evolving session testing is asked but split strategy is not
                a MECESplit.
        """
        if self.evolving and not isinstance(split_strategy, MECESplit):
            raise ValueError(
                "You can only use a MECESplit with evolving session configuration on."
            )

        self.load()

        if self.sample_size is not None:
            self.sample_input_data(self.sample_size, self.sample_random_state)

        # The default strategy is TemporalSplit.
        if split_strategy is None:
            split_strategy = TemporalSplit()

        self.split_input_data(split_strategy)
        self.prepare_test_for_eval(self.n_withheld, self.evolving)

        if split_strategy.can_split_k_fold():
            self.split_train_k_fold(split_strategy)
            self.prepare_k_fold_for_eval(self.n_withheld, self.evolving)

    def prepare_test_for_eval(
        self, n_withheld: int = 1, evolving: bool = False
    ) -> None:
        """Initializes the test_data_eval attribute by calling the methods
        prepare_to_predict and extract_ground_truths on the test data.

        Args:
            n_withheld: An integer representing the number of last items to withhold
                from each session. Defaults to 1.
            evolving: A boolean indicating whether evolving session testing is applied,
                in which case multiple test cases are generated from a single session
                in a sliding window fashion.

        Raises:
            Exception: If the test data has not been initialized yet.
        """
        if not self.has_test_data():
            raise Exception("The test data has not been initialized.")

        self.test_data_eval = (
            self._prepare_to_predict(self.test_data, n_withheld, evolving),
            self._extract_ground_truths(self.test_data, n_withheld, evolving),
        )

    def prepare_k_fold_for_eval(
        self, n_withheld: int = 1, evolving: bool = False
    ) -> None:
        """Initializes the k_fold_eval attribute by calling the methods
        prepare_to_predict and extract_ground_truths on the validation folds of the
        instance.

        Args:
            n_withheld: An integer representing the number of last items to withhold
                from each session. Defaults to 1.
            evolving: A boolean indicating whether evolving session testing is applied,
                in which case multiple test cases are generated from a single session
                in a sliding window fashion.

        Raises:
            Exception: If the validation folds have not been initialized yet.
        """
        if not self.has_k_fold():
            raise Exception("The validation folds have not been initialized.")

        new_eval_list: list[tuple[pd.DataFrame, Any, dict[int, np.ndarray]]] = []
        for fold in self.get_k_fold():
            eval_tuple: tuple[pd.DataFrame, Any, dict[int, np.ndarray]] = (
                fold[0],
                self._prepare_to_predict(fold[1], n_withheld, evolving),
                self._extract_ground_truths(fold[1], n_withheld, evolving),
            )
            new_eval_list.append(eval_tuple)

        self.k_fold_eval = new_eval_list

    def _prepare_to_predict(
        self, data: pd.DataFrame, n_withheld: int = 1, evolving: bool = False
    ) -> dict[int, np.ndarray]:
        """Returns the sessions provided in the given DataFrame while excluding the last
        n_withheld items of each session.

        Args:
            data: A pandas DataFrame containing the session data.
            n_withheld: An integer representing the number of last items to withhold
                from each session. Defaults to 1.
            evolving: A boolean indicating whether evolving session testing is applied,
                in which case multiple test cases are generated from a single session
                in a sliding window fashion.

        Returns:
            A dictionary where keys are session ids and the values are lists of item ids
            representing sessions.

        Raises:
            ValueError: If the parameter n_withheld is not a positive integer.
        """
        if n_withheld < 1:
            raise ValueError("The parameter n_withheld must be a positive integer.")

        if evolving:
            max_id: int = data["SessionId"].max()
        predict_sessions: dict[int, np.ndarray] = {}
        for _, session_df in data.groupby("SessionId"):
            session_id: int = session_df["SessionId"].iloc[0]
            prompt_items: list[int] = session_df["ItemId"].tolist()[:-n_withheld]
            # We do not include a session if there are no items to make predictions for.
            if len(prompt_items) == 0:
                continue

            if evolving:
                for i in range(len(prompt_items)):
                    # Generating unique ids per test case.
                    case_id: int = i * 2 * max_id + session_id + 1
                    predict_sessions[case_id] = np.array(prompt_items[: i + 1])
            else:
                predict_sessions[session_id] = np.array(prompt_items)
        return predict_sessions

    def _extract_ground_truths(
        self, data: pd.DataFrame, n_withheld: int = 1, evolving: bool = False
    ) -> dict[int, np.ndarray]:
        """Returns the last n_withheld items of each session in the given DataFrame.

        Args:
            data: A pandas Dataframe containing the session data.
            n_withheld: An integer representing the number of items to withhold from
                each session. Defaults to 1.
            evolving: A boolean indicating whether evolving session testing is applied,
                in which case multiple test cases are generated from a single session
                in a sliding window fashion.

        Returns:
            A dictionary where keys are session ids and the values are lists of item ids
            representing sessions.

        Raises:
            ValueError: If the parameter n_withheld is not a positive integer.
        """
        if n_withheld < 1:
            raise ValueError("The parameter n_withheld must be a positive integer.")

        max_id: int = data["SessionId"].max()
        ground_truths: dict[int, np.ndarray] = {}
        for _, session_df in data.groupby("SessionId"):
            session_id: int = session_df["SessionId"].iloc[0]
            session_items: np.ndarray = session_df["ItemId"].to_numpy().astype(int)
            # We do not include a session if there are no items to make predictions for.
            if len(session_items) == n_withheld:
                continue

            if evolving:
                for i in range(len(session_items) - n_withheld):
                    # Generating unique ids per test case.
                    case_id: int = i * 2 * max_id + session_id + 1
                    ground_truths[case_id] = np.array(
                        session_items[i + 1 : i + 1 + n_withheld]
                    )
            else:
                ground_truths[session_id] = session_items[-n_withheld:]
        return ground_truths


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


class TemporalSplit(MECESplit):
    """A class representing a temporal split of sessions.

    Attributes:
        test_cutoff: A datetime object or a float representing an epoch time, indicating
            the cutoff point between the train and test set. If provided, it takes
            precedence over the test_frac during the train-test split.
        fold_cutoffs: A list of either datetime objects or floats representing epoch
            times that indicate the cutoff points of consecutive folds. If provided,
            these cutoff points take precedence over the num_folds given during k-fold
            splitting.
        use_test_cutoff: A boolean indicating whether the train-test split should be
            based on an explicit cutoff point.
        use_fold_cutoffs: A boolean indicating whether the folds should be split based
            on explicit cutoff points.
    """

    def __init__(
        self,
        test_frac: float = 0.2,
        num_folds: int = 0,
        filter_non_trained_test_items: bool = False,
        test_cutoff: Union[datetime.datetime, float] = None,
        fold_cutoffs: Union[list[datetime.datetime], list[float]] = None,
    ) -> None:
        """Inits a TemporalSplit instance with the given arguments.

        Args:
            test_frac: A float representing the test fraction. Defaults to 0.2.
            num_folds: An integer representing the number of folds. Defaults to 0.
            filter_non_trained_test_items: A boolean indicating whether to filter out
                non-trained test items. Defaults to False.
            test_cutoff: A datetime object or a float representing an epoch time,
                indicating the cutoff point between the train and test set. Defaults to
                None.
            fold_cutoffs: A list of either datetime objects or floats representing epoch
                times that indicate the cutoff points of consecutive folds. Defaults to
                None.

        Raises:
            ValueError: If test_frac is not None and not a float between 0 and 1
                inclusive.
            ValueError: If num_folds is not a positive integer.
            ValueError: If test_cutoff is not None and not in the allowed format.
            ValueError: If fold_cutoffs is not None and not in the allowed format.
        """
        super().__init__(test_frac, num_folds, filter_non_trained_test_items)
        self.use_test_cutoff: bool = False
        self.use_fold_cutoffs: bool = False

        if test_cutoff is not None:
            if isinstance(test_cutoff, datetime.datetime):
                self.test_cutoff: float = test_cutoff.timestamp()
            elif isinstance(test_cutoff, float):
                self.test_cutoff: float = test_cutoff
            else:
                raise ValueError(
                    "The cutoff points must be either datetime objects or floats."
                )
            self.use_test_cutoff = True

        if fold_cutoffs is not None:
            if all(isinstance(i, datetime.datetime) for i in fold_cutoffs):
                self.fold_cutoffs: list[float] = [i.timestamp() for i in fold_cutoffs]
            elif all(isinstance(i, float) for i in fold_cutoffs):
                self.fold_cutoffs: list[float] = fold_cutoffs
            else:
                raise ValueError(
                    "The cutoff points must be either datetime objects or floats."
                )
            self.fold_cutoffs = sorted(self.fold_cutoffs)
            self.use_fold_cutoffs = True
            self.num_folds = len(fold_cutoffs)

    def _split_train_test_ids(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Splits the session ids found in the given DataFrame to train and test ids.

        Args:
            df: A pandas DataFrame containing the session data.

        Returns:
            A tuple of numpy arrays that represent the session ids of the train and test
            set respectively.
        """
        if self.use_test_cutoff:
            return TemporalSplit._split_train_test_ids_cutoff(df, self.test_cutoff)
        else:
            return TemporalSplit._split_train_test_ids_fraction(df, self.test_frac)

    @staticmethod
    def _split_train_test_ids_cutoff(
        df: pd.DataFrame, test_cutoff: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Splits the session ids found in the given DataFrame by putting the sessions
        that took place earlier than the cutoff point in the train set, and the ones
        that took at or after the cutoff point at the test set.

        Args:
            df: A pandas DataFrame containing the session data.
            test_cutoff: A float representing the cutoff point epoch time.

        Returns:
            A tuple of numpy arrays containing the train ids and test ids respectively.

        Raises:
            ValueError: If the given cutoff point results in an empty train set or test
                set.
        """
        sorted_session_times: pd.DataFrame = TemporalSplit._get_sorted_session_times(df)
        n_sessions: int = sorted_session_times["SessionId"].nunique()
        n_test_sessions: int = len(
            sorted_session_times[sorted_session_times["Time"] >= test_cutoff]
        )
        n_train_sessions: int = n_sessions - n_test_sessions

        if n_train_sessions == 0:
            raise ValueError("The given cutoff point results in empty train set.")
        if n_test_sessions == 0:
            raise ValueError("The given cutoff point results in empty test set.")

        train_ids, test_ids = np.array_split(
            sorted_session_times["SessionId"].to_numpy(), [n_train_sessions]
        )
        return train_ids, test_ids

    @staticmethod
    def _split_train_test_ids_fraction(
        df: pd.DataFrame, test_frac: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Splits the session ids in the given DataFrame by recency where the most
        recent sessions are placed in the test set. The fraction of test sessions is
        determined by the test_frac attribute of the object.

        Args:
            df: A pandas DataFrame containing the session data.

        Returns:
            A tuple of numpy arrays where the first element contains the train session
            ids and the second element contains the test session ids.
        """
        sorted_session_times: pd.DataFrame = TemporalSplit._get_sorted_session_times(df)
        n_sessions: int = sorted_session_times["SessionId"].nunique()
        n_test_sessions: int = math.ceil(test_frac * n_sessions)
        n_train_sessions: int = n_sessions - n_test_sessions

        train_ids, test_ids = np.array_split(
            sorted_session_times["SessionId"].to_numpy(), [n_train_sessions]
        )
        return train_ids, test_ids

    def _split_k_fold_ids(
        self, df: pd.DataFrame
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Splits the session ids found in the given DataFrame to k-folds of train and
        validation sets.

        Args:
            df: A pandas DataFrame containing the session data.

        Returns:
            A list of tuples that contain the session ids of train and validation sets
            of the folds.
        """
        if self.use_fold_cutoffs:
            return TemporalSplit._split_k_fold_ids_cutoff(df, self.fold_cutoffs)
        else:
            return TemporalSplit._split_k_fold_ids_chain(df, self.num_folds)

    @staticmethod
    def _split_k_fold_ids_cutoff(
        df: pd.DataFrame, fold_cutoffs: list[float]
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Splits the sessions in the given DataFrame to folds using the cutoff points
        provided.

        Args:
            df: A pandas DataFrame containing the session data.
            fold_cutoffs: A list of floats containing epoch time cutoff points

        Returns:
            A list of tuples that contain the train and validation session ids of the
            folds.

        Raises:
            ValueError: If the given cutoff points results in empty folds. This can
                happen if there are no sessions found between the cutoff points.
        """
        sorted_session_times: pd.DataFrame = TemporalSplit._get_sorted_session_times(df)
        interval_indices: list[int] = [
            len(sorted_session_times[sorted_session_times["Time"] < fold_cutoffs[0]])
        ]
        for i, cutoff in enumerate(fold_cutoffs[1:], 1):
            n_interval: int = len(
                sorted_session_times[
                    (sorted_session_times["Time"] >= fold_cutoffs[i - 1])
                    & (sorted_session_times["Time"] < cutoff)
                ]
            )
            interval_indices.append(n_interval + interval_indices[i - 1])

        interval_sizes: np.ndarray = np.diff(interval_indices)
        if 0 in interval_sizes:
            raise ValueError("The given fold cutoff points result in empty folds.")

        splits: list[np.ndarray] = np.array_split(
            sorted_session_times["SessionId"].to_numpy(), interval_indices
        )
        return TemporalSplit._convert_split_indices_to_folds(splits)

    @staticmethod
    def _split_k_fold_ids_chain(
        df: pd.DataFrame, num_folds: int
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Splits the session ids in the given DataFrame into folds of train and
        validation ids. The folds are constructed in a forward chaining fashion where
        the starting number of sessions in the folds is the total number of sessions
        divided by the num_folds attribute plus 1. The training size of the folds
        increase by this very amount every fold, and the validation size is kept the
        same, except the last one which can be a bit smaller in case the division above
        has a remainder.

        The logic of the forward chaining is the following. The session ids are
        contained in bins, i.e. splits. The sessions in a single bin are not sorted, but
        the bins themselves are sorted. Thus, all the sessions in a given bin took place
        before the sessions that are contained in the following bins. The folds are
        formed by iteratively stacking these bins while going forward in time, hence the
        term forward chaining. For example, if we have 4 bins labelled 1 through 4, the
        folds would look like the following:

        fold 1 : train [1] validation [2]
        fold 2 : train [1, 2] validation [3]
        fold 3 : train [1, 2, 3] validation [4]

        Args:
            df: A pandas DataFrame containing the session data.

        Returns:
            A list of tuples that contain the train and validation session ids of the
            folds.

        Raises:
            ValueError: If the given num_folds results in empty folds. This only happens
                if num_folds is larger than the number of sessions.
        """
        sorted_session_times: pd.DataFrame = TemporalSplit._get_sorted_session_times(df)
        splits: list[np.ndarray] = np.array_split(
            sorted_session_times["SessionId"].to_numpy(), num_folds + 1
        )
        for split in splits:
            if split.size == 0:
                raise ValueError(
                    f"The given num_folds attribute of {num_folds} result in empty"
                    f" validation folds. Make sure that (num_folds + 1) is smaller than"
                    f" or equal to the number of training sessions of"
                    f" {sorted_session_times['SessionId'].nunique()}"
                )
        return TemporalSplit._convert_split_indices_to_folds(splits)

    @staticmethod
    def _convert_split_indices_to_folds(
        splits: list[np.ndarray],
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Constructs folds out in a forward chaining fashion using the given list that
        contains split session ids.

        Args:
            splits: A list of numpy array where each array contains a subset of session
                ids.

        Returns:
            A list of tuples that contain the train and validation session ids of the
            folds.
        """
        folds: list[tuple[np.ndarray, np.ndarray]] = [(splits[0], splits[1])]
        for i in range(1, len(splits) - 1):
            fold_train_ids: np.ndarray = np.concatenate(
                (folds[i - 1][0], folds[i - 1][1])
            )
            fold_val_ids: np.ndarray = splits[i + 1]
            folds.append((fold_train_ids, fold_val_ids))
        return folds

    @staticmethod
    def _get_sorted_session_times(df: pd.DataFrame) -> pd.DataFrame:
        """Returns a DataFrame containing the session ids and session times. The
        sessions are sorted by the session times, where the session time is the latest
        click time in a session.

        Args:
            df: A pandas DataFrame containing the session data.

        Returns:
            A pandas DataFrame containing the sorted session ids and times.
        """
        session_max_times: pd.Series = df.groupby("SessionId")["Time"].max()
        return session_max_times.sort_values().reset_index()


class HoldoutSplit(SplitStrategy):
    """A class representing the Holdout splitting strategy. This strategy puts each
    input session to both train and test sets. The last item(s) of the sessions are
    withheld from each session in the train set, and these items are used as ground
    truths for testing purposes. Same logic applies to validation, where last item(s)
    of the train set are used for validation. Note that for validation, only a single
    fold is available due to the nature of the splitting.

    Be wary of the fact that withholding fewer than n_test_items during the preparation
    for testing, splitting sessions to prompts and ground truths that is, would result
    in information leakage.

    Attributes:
        n_test_items: An integer representing the number of last items of each input
            session to use as test items.
        n_val_items: An integer representing the number of last items of each training
            session to use as validation items.
    """

    def __init__(
        self,
        n_test_items: int = 1,
        n_val_items: int = 1,
    ):
        """Inits a HoldoutSplit instance.

        Args:
            n_test_items: An integer representing the number of test items per session.
            n_val_items: An integer representing the number of validation items per
                session.
        """
        if n_test_items < 0:
            raise ValueError(
                "The parameter n_test_items must be a non-negative integer."
            )
        if n_val_items < 0:
            raise ValueError(
                "The parameter n_val_items must be a non-negative integer."
            )

        self.n_test_items: int = n_test_items
        self.n_val_items: int = n_val_items

    def split_train_test(
        self, input_data: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Splits the given input data into train and test sets.

        Args:
            input_data: The data to split.

        Returns:
            A tuple containing the train and test sets respectively.
        """
        train_df, test_df = self._holdout_last_items(
            input_data,
            self.n_test_items,
        )
        return train_df, test_df

    def split_k_fold(
        self, input_data: pd.DataFrame
    ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """Splits the given input data into a single fold of train and validation sets.

        Args:
            input_data: The data to split folds for.

        Returns:
            A list of tuples where each tuple contains the train and validation set
            respectively.
        """
        logging.info("This strategy can only offer a single fold.")
        train_df, val_df = HoldoutSplit._holdout_last_items(
            input_data,
            self.n_val_items,
        )
        return [(train_df, val_df)]

    def can_split_k_fold(self) -> bool:
        """Indicates whether this SplitStrategy instance can apply k-fold splitting.

        Returns:
            True if k-fold splitting is possible, False otherwise.
        """
        return True

    @staticmethod
    def _holdout_last_items(
        input_data: pd.DataFrame,
        n_heldout: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Holdouts a certain number of last items from each session in the given
        DataFrame.

        Args:
            input_data: A DataFrame containing the sessions.
            n_heldout: An integer indicating the number of last items to hold out.

        Returns:
            A DataFrame with last items withheld from each session.
        """
        train_df: pd.DataFrame = input_data.drop(
            input_data.groupby("SessionId").tail(n_heldout).index, axis=0
        ).reset_index(drop=True)

        # Remove sessions that cannot yield prompts due to not enough items.
        train_session_ids: np.ndarray = train_df["SessionId"].unique()
        test_df: pd.DataFrame = (
            input_data[input_data["SessionId"].isin(train_session_ids)]
            .copy()
            .reset_index(drop=True)
        )

        return train_df, test_df
