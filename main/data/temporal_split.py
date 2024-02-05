import math
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd

from main.data.mece_split import MECESplit
from main.data.random_split import RandomSplit


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
        fold_strategy: The folding strategy to use.
                Cross - Cross validation, in this case we do not consider the temporal
                    orderings of the sessions and simply create a cross-validation
                    set of folds.
                Chain - Forward chaining strategy, most easily explained with the
                    example that if you have three folds, then the training data
                    is temporally split into four bins, and the first fold will contain
                    bins 1 (train) and 2 (test), the second contains bins 1, 2 (train)
                    and 3 (test), and the third contains bins 1, 2, 3 (train) and
                    4 (test). Note that the first fold will therefore contain much fewer
                    data than the last fold.
                Cutoff - Use the passed `fold_cutoffs` parameter to split the folds.
                    Every fold corresponds to one timestamp from the passed list, where
                    all training sessions occurred before the timestamp, and all test
                    sessions occurred after the timestamp.
    """

    def __init__(
        self,
        test_frac: float = 0.2,
        num_folds: int = 0,
        filter_non_trained_test_items: bool = False,
        test_cutoff: datetime | float = None,
        fold_strategy: Literal["cross", "chain", "cutoff"] = "chain",
        fold_cutoffs: list[datetime] | list[float] = None,
        random_state: int = None,
    ) -> None:
        """Inits a TemporalSplit instance with the given arguments.

        Args:
            test_frac (optional, float): A float representing the test fraction.
                Defaults to 0.2.
            num_folds (optional, int): An integer representing the number of folds.
                Defaults to 0.
            filter_non_trained_test_items (optional, bool): A boolean indicating whether
                to filter out non-trained test items. Defaults to False.
            test_cutoff (optional, Union[datetime, float]): A datetime object or a float
                representing an epoch time, indicating the cutoff point between the
                train and test set. Defaults to None.
            fold_strategy (optional, Literal): The folding strategy to use.
                Cross - Cross validation, in this case we do not consider the temporal
                    orderings of the sessions and simply create a cross-validation
                    set of folds.
                Chain - Forward chaining strategy, most easily explained with the
                    example that if you have three folds, then the training data
                    is temporally split into four bins, and the first fold will contain
                    bins 1 (train) and 2 (test), the second contains bins 1, 2 (train)
                    and 3 (test), and the third contains bins 1, 2, 3 (train) and
                    4 (test). Note that the first fold will therefore contain much fewer
                    data than the last fold.
                Cutoff - Use the passed `fold_cutoffs` parameter to split the folds.
                    Every fold corresponds to one timestamp from the passed list, where
                    all training sessions occurred before the timestamp, and all test
                    sessions occurred after the timestamp.
                Defaults to "chain".
            fold_cutoffs (optional, Union[list[datetime], list[float]]): A list of
                either datetime objects or floats representing epoch times that indicate
                the cutoff points of consecutive folds. Defaults to None.
            random_state: An integer representing the random seed to use for the splits.
                This random state is only used when the folding strategy is set to
                "cross", in which case we use the functionality of RandomSplit.
                Defaults to None, in which case the seed is selected randomly.

        Raises:
            ValueError: If test_frac is not None and not a float between 0 and 1
                inclusive.
            ValueError: If num_folds is not a positive integer.
            ValueError: If test_cutoff is not None and not in the allowed format.
            ValueError: If fold_strategy is "cutoffs", but fold_cutoffs is None
                or not in the allowed format.
        """
        super().__init__(test_frac, num_folds, filter_non_trained_test_items)
        self.use_test_cutoff: bool = False
        self.fold_strategy = fold_strategy

        if test_cutoff is not None:
            if isinstance(test_cutoff, datetime):
                self.test_cutoff: float = test_cutoff.timestamp()
            elif isinstance(test_cutoff, float):
                self.test_cutoff: float = test_cutoff
            else:
                raise ValueError(
                    "The cutoff points must be either datetime objects or floats."
                )
            self.use_test_cutoff = True

        if self.fold_strategy == "cutoff":
            if fold_cutoffs is None:
                raise ValueError(
                    "Folding strategy set to cutoff, but fold_cutoffs" " is None."
                )
            if all(isinstance(i, datetime) for i in fold_cutoffs):
                self.fold_cutoffs: list[float] = [i.timestamp() for i in fold_cutoffs]
            elif all(isinstance(i, float) for i in fold_cutoffs):
                self.fold_cutoffs: list[float] = fold_cutoffs
            else:
                raise ValueError(
                    "The cutoff points must be either datetime objects or floats."
                )
            self.fold_cutoffs = sorted(self.fold_cutoffs)
            self.num_folds = len(fold_cutoffs)

        elif self.fold_strategy == "cross":
            self.cross_fold_splitter = RandomSplit(
                test_frac, num_folds, filter_non_trained_test_items, random_state
            )

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
        if self.fold_strategy == "cutoff":
            return TemporalSplit._split_k_fold_ids_cutoff(df, self.fold_cutoffs)
        elif self.fold_strategy == "chain":
            return TemporalSplit._split_k_fold_ids_chain(df, self.num_folds)
        elif self.fold_strategy == "cross":
            return self.cross_fold_splitter._split_k_fold_ids(df)
        else:
            raise ValueError(f"Unknown fold strategy. Got {self.fold_strategy}")

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
