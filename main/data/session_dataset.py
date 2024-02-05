"""This module contains a class representing a session-based dataset."""

import logging
from io import BytesIO
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from main.data.abstract_dataset import Dataset, SplitStrategy
from main.data.mece_split import MECESplit
from main.data.temporal_split import TemporalSplit
from main.exceptions import InvalidStateError


class SessionDataset(Dataset):
    """An abstract class representing a session-based dataset object. The underlying
    type of data structure is a pandas Dataframe of type
    ["SessionId": int, "ItemId": str, "Time": datetime str, "Reward": float].
    """

    def __init__(
        self,
        filepath_or_bytes: Union[str, bytes],
        sample_size: int = None,
        sample_random_state: int = None,
        n_withheld: int = 1,
        evolving: bool = False,
    ) -> None:
        """Inits a SessionDataset instance with the given configuration.

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

        self.item_data: Optional[pd.DataFrame] = None

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
                "Reward": float,
            },
        )
        self.input_data["Time"] = pd.to_datetime(
            self.input_data["Time"], format="%Y-%m-%d %H:%M:%S.%f"
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
            InvalidStateError: If input data has not been set.
        """
        if self.input_data is None:
            raise InvalidStateError("The dataset has not been loaded.")

        return self.input_data["ItemId"].nunique()

    def get_unique_sample_count(self) -> int:
        """Returns the unique number of sessions in the input data.

        Returns:
            An integer representing the number of unique sessions.

        Raises:
            InvalidStateError: If input data has not been set.
        """
        if self.input_data is None:
            raise InvalidStateError("The dataset has not been loaded.")

        return self.input_data["SessionId"].nunique()

    def get_item_counts(self) -> dict[int, int]:
        """Get the number of occurrences per item in the input data.

        Returns:
            An dict with the count per item id. For example get_items_counts[10]
            represents the count for item 10.

        Raises:
            InvalidStateError: If input data has not been set.
        """

        if self.input_data is None:
            raise InvalidStateError("The dataset has not been loaded.")

        return self.input_data["ItemId"].value_counts().to_dict()

    def get_sample_counts(self) -> dict[int, int]:
        """Get the number of interactions per sample in the input data.

        Returns:
            A dict with the count per sample id. For example get_sample_counts[10]
            represents the number of interactions for sample 10.

        Raises:
            InvalidStateError: If input data has not been set.
        """

        if self.input_data is None:
            raise InvalidStateError("The dataset has not been loaded.")

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
            InvalidStateError: If the test data has not been initialized yet.
        """
        if not self.has_test_data():
            raise InvalidStateError("The test data has not been initialized.")

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
            InvalidStateError: If the validation folds have not been initialized yet.
        """
        if not self.has_k_fold():
            raise InvalidStateError("The validation folds have not been initialized.")

        new_eval_list: list[tuple[pd.DataFrame, Any, dict[int, np.ndarray]]] = []
        for fold in self.get_k_fold():
            eval_tuple: tuple[pd.DataFrame, Any, dict[int, np.ndarray]] = (
                fold[0],
                self._prepare_to_predict(fold[1], n_withheld, evolving),
                self._extract_ground_truths(fold[1], n_withheld, evolving),
            )
            new_eval_list.append(eval_tuple)

        self.k_fold_eval = new_eval_list

    def set_item_data(self, item_data: pd.DataFrame) -> None:
        """Sets the item_data attribute of the instance to the given DataFrame. The
        DataFrame should contain an "ItemId" column.

        Args:
            item_data: A pandas DataFrame containing item data.
        """
        self.item_data = item_data

    def get_item_data(self) -> Optional[pd.DataFrame]:
        """Returns the item_data DataFrame.

        Returns:
            A pandas DataFrame containing the item_data of the dataset.

        Raises:
            ValueError: If the item_data is not set.
        """
        if self.item_data is None:
            # raise ValueError("item_data attribute is not set.")
            logging.info(
                "item_data is being tried to access, but it is not set. Returning None."
            )
            return None
        else:
            return self.item_data

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
