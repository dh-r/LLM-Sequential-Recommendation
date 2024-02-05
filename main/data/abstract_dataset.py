"""This module contains the abstract Dataset class and the abstract SplitStrategy
class."""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from typing import Any, Optional, TypeVar, Union

import numpy as np

from pandas.core.indexes.base import Index


class NumericIndex(Index):
    pass


class IntegerIndex(NumericIndex):
    pass


class Int64Index(IntegerIndex):
    pass


class UInt64Index(IntegerIndex):
    pass


class Float64Index(NumericIndex):
    pass


from main.exceptions import InvalidStateError

# A type variable denoting the data structure to be used in a Dataset implementation.
DatasetT = TypeVar("DatasetT")

# Used for pickling methods. It means any subclass of Dataset type.
ConcreteDataset = TypeVar("ConcreteDataset", bound="Dataset")


class Dataset(ABC):
    """An abstract class representing a dataset object.

    Data-specific dataset implementations must inherit from this class.

    Attributes:
        filepath_or_bytes: Either a string file_path or a bytes object.
        input_data: The complete dataset of the Dataset instance.
        train_data: The training dataset resulting from the train-test split of the
            input_data.
        test_data: The test dataset resulting from the train-test split of the
            input_data.
        test_data_eval: A tuple containing as first element the test data in a format
            acceptable by the predict method of a corresponding Model class, and a
            dictionary containing the ground truth for each test case as the second
            element.
        k_fold: A list of tuples where each tuple consists of a train-valid split of
            the train_data and represents a fold of the training dataset.
        k_fold_eval: A list containing, for each validation fold, a tuple 3 elements.
            The first is the training data of the fold, the second is the validation
            data of the fold prepared for prediction, and the third is the ground truths
            extracted from the validation data.
    """

    @abstractmethod
    def __init__(self, filepath_or_bytes: Union[str, bytes]) -> None:
        """Inits a Dataset instance with the given filepath.

        Args:
            filepath_or_bytes: Either a string file_path or a bytes object.
        """
        self.filepath_or_bytes: Union[str, bytes] = filepath_or_bytes
        self.input_data: Optional[DatasetT] = None
        self.train_data: Optional[DatasetT] = None
        self.test_data: Optional[DatasetT] = None
        self.test_data_eval: Optional[tuple[Any, dict[int, np.ndarray]]] = None
        self.k_fold: Optional[list[tuple[DatasetT, DatasetT]]] = None
        self.k_fold_eval: Optional[
            list[tuple[DatasetT, Any, dict[int, np.ndarray]]]
        ] = None

    @abstractmethod
    def load(self) -> None:
        """Loads the dataset from a file using the filepath attribute."""
        pass

    @abstractmethod
    def _prepare_to_predict(self, data: DatasetT) -> Any:
        """Converts the data given as input to a format that is acceptable by the
        concrete Model implementations that work with this Dataset object. One can pass
        the test data or a fold of validation data to this method, and the returned
        value can be passed to the predict method of a corresponding Model object.

        Args:
            data: A data object matching the generic type of the Dataset instance.

        Returns:
            An object containing the data argument in a format that the corresponding
            Model object's predict method expects.
        """
        pass

    @abstractmethod
    def _extract_ground_truths(self, data: DatasetT) -> dict[int, np.ndarray]:
        """Extracts the ground truths from each test case to be used during evaluation.

        Args:
            data: A data object matching the generic type of the Dataset instance.

        Returns:
            A dictionary where keys are identifiers to test cases and the values are
            lists of integers representing the ground truth items.
        """
        pass

    @abstractmethod
    def get_unique_item_count(self) -> int:
        """Computes the unique item count across the whole dataset.

        Returns:
            The count of items.
        """
        pass

    @abstractmethod
    def get_unique_sample_count(self) -> int:
        """Computes the unique amount of samples across the whole dataset.
        A sample could be, for example, a unique user or a unique session.

        Returns:
            The count of samples.
        """
        pass

    @abstractmethod
    def get_item_counts(self) -> dict[int, int]:
        """Get the number of occurrences per item in the full dataset.

        Returns:
            A dict with the count per item id. For example get_items_counts[10]
            represents the count for item 10.
        """
        pass

    @abstractmethod
    def get_sample_counts(self) -> dict[int, int]:
        """Get the number of interactions per sample in the full dataset.

        Returns:
            A dict with the count per sample id. For example get_sample_counts[10]
            represents the number of interactions for sample 10.
        """
        pass

    def load_and_split(self, split_strategy: SplitStrategy) -> None:
        """Loads and splits dataset into train and test set and prepares the test set
        for evaluation. If the given SplitStrategy allows, the method also generates
        k folds on the train set and prepares the folds for evaluation.

        Args:
            split_strategy: An instance of a concrete SplitStrategy to use as the split
                logic.
        """
        self.load()
        self.split_input_data(split_strategy)
        self.prepare_test_for_eval()

        if split_strategy.can_split_k_fold():
            self.split_train_k_fold(split_strategy)
            self.prepare_k_fold_for_eval()

    def split_input_data(self, split_strategy: SplitStrategy) -> None:
        """Splits the input data into train and test sets using the given splitting
        strategy.

        Args:
            split_strategy: An instance of a concrete SplitStrategy to use as the split
                logic.
        """
        self.train_data, self.test_data = split_strategy.split_train_test(
            self.input_data
        )

    def split_train_k_fold(self, split_strategy: SplitStrategy) -> None:
        """Splits the train data into k folds of train and validation sets and stores
        them in the k_fold attribute as a list of tuples.

        Args:
            split_strategy: An instance of a concrete SplitStrategy to use as the split
                logic.

        Raises:
            ValueError: If the provided SplitStrategy cannot do k-fold splitting.
            InvalidStateError: If the train data does not exist yet.
        """
        if not split_strategy.can_split_k_fold():
            raise ValueError("This instance cannot apply k-fold splitting.")
        if not self.has_train_data():
            raise InvalidStateError("The train data is not created yet.")

        self.k_fold = split_strategy.split_k_fold(self.train_data)

    def prepare_test_for_eval(self) -> None:
        """Initializes the test_data_eval attribute by calling the methods
        prepare_to_predict and extract_ground_truths on the test data.

        Raises:
            InvalidStateError: If the test data has not been initialized yet.
        """
        if not self.has_test_data():
            raise InvalidStateError("The test data has not been initialized.")

        self.test_data_eval = (
            self._prepare_to_predict(self.test_data),
            self._extract_ground_truths(self.test_data),
        )

    def prepare_k_fold_for_eval(self) -> None:
        """Initializes the k_fold_eval attribute by calling the methods
        prepare_to_predict and extract_ground_truths on the validation folds of the
        instance.

        Raises:
            InvalidStateError: If the validation folds have not been initialized yet.
        """
        if not self.has_k_fold():
            raise InvalidStateError("The validation folds have not been initialized.")

        new_eval_list: list[tuple[DatasetT, Any, dict[int, np.ndarray]]] = []
        for fold in self.get_k_fold():
            eval_tuple: tuple[DatasetT, Any, dict[int, np.ndarray]] = (
                fold[0],
                self._prepare_to_predict(fold[1]),
                self._extract_ground_truths(fold[1]),
            )
            new_eval_list.append(eval_tuple)

        self.k_fold_eval = new_eval_list

    def get_input_data(self) -> DatasetT:
        """Returns the input dataset.

        Returns:
            The input dataset.

        Raises:
            InvalidStateError: If the input data has not been set.
        """
        if not self.has_input_data():
            raise InvalidStateError("The input data has not been set.")
        return self.input_data

    def set_input_data(self, input_data: DatasetT) -> None:
        """Sets the input dataset of this instance with the given dataset.

        Args:
            input_data: The new input dataset.
        """
        self.input_data = input_data

    def has_input_data(self) -> bool:
        """Checks if the input dataset is initialized to a value.

        Returns:
            True if it is initialized, i.e. is not None. False otherwise.
        """
        return self.input_data is not None

    def get_train_data(self) -> DatasetT:
        """Returns the training dataset.

        Returns:
            The training dataset.

        Raises:
            InvalidStateError: If the training data has not been initialized.
        """
        if not self.has_train_data():
            raise InvalidStateError("Training data has not been initialized.")
        return self.train_data

    def set_train_data(self, train_data: DatasetT) -> None:
        """Sets the training dataset of this instance with the given dataset.

        Args:
            train_data: The new training dataset.
        """
        self.train_data = train_data

    def has_train_data(self) -> bool:
        """Checks if the training dataset is initialized to a value.

        Returns:
            True if it is initialized, i.e. is not None. False otherwise.
        """
        return self.train_data is not None

    def get_test_data(self) -> DatasetT:
        """Returns the test dataset.

        Returns:
            The test dataset.

        Raises:
            InvalidStateError: If the test data has not been initialized.
        """
        if not self.has_test_data():
            raise InvalidStateError("The test data has not been initialized.")
        return self.test_data

    def set_test_data(self, test_data: DatasetT) -> None:
        """Sets the test dataset of this instance with the given dataset.

        Args:
            test_data: The new test dataset.
        """
        self.test_data = test_data

    def has_test_data(self) -> bool:
        """Checks if the test dataset is initialized to a value.

        Returns:
            True if it is initialized, i.e. is not None. False otherwise.
        """
        return self.test_data is not None

    def get_test_data_eval(self) -> tuple[Any, dict[int, np.ndarray]]:
        """Returns the test dataset evaluation tuple.

        Returns:
            The test dataset evaluation tuple.

        Raises:
            InvalidStateError: If the test data evaluation tuple has not been
                initialized.
        """
        if not self.has_test_data_eval():
            raise InvalidStateError(
                "Test data evaluation tuple has not been initialized."
            )
        return self.test_data_eval

    def get_test_prompts(self) -> Any:
        """Returns the test dataset evaluation prompts.

        Returns:
            Any: the test set evaluation prompts.
        """
        return self.get_test_data_eval()[0]

    def get_test_ground_truths(self) -> dict[int, np.ndarray]:
        """Returns the test dataset evaluation ground truths.

        Returns:
            Dict[int, np.ndarray]: the test set evaluation ground truths.
        """
        return self.get_test_data_eval()[1]

    def has_test_data_eval(self) -> bool:
        """Checks if the test data evaluation tuple is initialized to a value.

        Returns:
            True if it is initialized, i.e. is not None. False otherwise.
        """
        return self.test_data_eval is not None

    def get_k_fold(self) -> list[tuple[DatasetT, DatasetT]]:
        """Returns the k_fold list.

        Returns:
            A list containing k folds.

        Raises:
            InvalidStateError: If the k folds evaluation tuple list is not initialized.
        """
        if not self.has_k_fold():
            raise InvalidStateError("k fold list has not been initialized")
        return self.k_fold

    def set_k_fold(self, k_fold: list[tuple[DatasetT, DatasetT]]) -> None:
        """Sets the k_fold list of the instance to the given k_fold list.

        Args:
            k_fold: The new list of k folds of the training dataset.
        """
        self.k_fold = k_fold

    def has_k_fold(self) -> bool:
        """Checks if the k_fold list contains anything.

        Returns:
            True if it contains folds. False otherwise.
        """
        return (self.k_fold is not None) and (len(self.k_fold) != 0)

    def get_k_fold_eval(self) -> list[tuple[DatasetT, Any, dict[int, np.ndarray]]]:
        """Returns the k_fold evaluation tuple list.

        Returns:
            The k_fold evaluation tuple list.

        Raises:
            InvalidStateError: If k_fold_eval list is empty.
        """
        if not self.has_k_fold_eval():
            raise InvalidStateError(
                "k-folds evaluation tuple list has not been initialized."
            )
        return self.k_fold_eval

    def has_k_fold_eval(self) -> bool:
        """Checks if the k_fold_eval list contains anything.

        Returns:
            True if it is non-empty. False otherwise.
        """
        return (self.k_fold_eval is not None) and (len(self.k_fold_eval) != 0)

    def to_pickle(self, filepath: Optional[str] = None) -> Optional[bytes]:
        """Convert the dataset to a pickle and optionally store it as a file.
        The method adds a ".pickle" extension to the path if the path does not already
        have it.

        Args:
            filepath: A string representing the path of the file to store the pickle at.
                If not specified, it will return the dataset as bytes.
        """

        # If no path is set, we return just the bytes.
        if filepath is None:
            return pickle.dumps(self)

        path_with_extension: str = (
            filepath if filepath.endswith(".pickle") else filepath + ".pickle"
        )
        with open(path_with_extension, mode="wb") as write_file:
            pickle.dump(self, write_file)

    @staticmethod
    def from_pickle(filepath_or_bytes: Union[str, bytes]) -> ConcreteDataset:
        """Loads a dataset, stored as a pickle, from a file or bytes.

        When a path is given, the method adds a ".pickle" extension to the path if the
        path does not already have it to ensure path compatibility between loading and
        storing Model objects.

        Args:
            filepath_or_bytes: Either a string file_path or a bytes object.

        Returns:
            A Dataset object loaded from the pickle file or bytes.
        """
        if not (
            isinstance(filepath_or_bytes, str) or isinstance(filepath_or_bytes, bytes)
        ):
            raise TypeError(
                "Expected either a string for a filepath or a bytes object."
            )

        if isinstance(filepath_or_bytes, bytes):
            return pickle.loads(filepath_or_bytes)

        path_with_extension: str = (
            filepath_or_bytes
            if filepath_or_bytes.endswith(".pickle")
            else filepath_or_bytes + ".pickle"
        )
        with open(path_with_extension, mode="rb") as read_file:
            return pickle.load(read_file)


class SplitStrategy(ABC):
    """An abstract class representing a splitting strategy."""

    NO_SPLIT: int = 0  # Synthetic sugar for a 100-0 train-test split.

    @abstractmethod
    def split_train_test(self, input_data: DatasetT) -> tuple[DatasetT, DatasetT]:
        """Splits the given input data into train and test sets.

        Args:
            input_data: The data to split.

        Returns:
            A tuple containing the train and test sets respectively.
        """
        pass

    @abstractmethod
    def split_k_fold(self, input_data: DatasetT) -> list[tuple[DatasetT, DatasetT]]:
        """Splits the given input data into k folds of train and validation sets.

        Args:
            input_data: The data to split into folds.

        Returns:
            A list of tuples where each tuple contains the train and validation set
            respectively.
        """
        pass

    @abstractmethod
    def can_split_k_fold(self) -> bool:
        """Indicates whether this SplitStrategy instance can apply k-fold splitting.

        Returns:
            True if k-fold splitting is possible, False otherwise.
        """
        pass
