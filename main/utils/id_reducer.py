import logging
from typing import Dict, Union, Any, Optional

import numpy as np
import pandas as pd


class DefaultDict(dict):
    """A dictionary subclass that defines the behaviour on missing values."""

    missing_found = False

    def __missing__(self, key: Any) -> int:
        """Defines default behaviour when a key is missing from the dict.

        Args:
            key (Any): The missing key

        Returns:
            int: -1
        """
        if not (DefaultDict.missing_found):
            DefaultDict.missing_found = True
            logging.warning(
                "Encountered unknown ID(s) while reducing IDs. Returned -1 as value "
                "instead."
            )
        return -1


class IDReducer:
    """A class to reduce IDs in the target column into a consecutive integer range,
    namely from 0 to the number of unique IDs exclusive."""

    def __init__(self, data: pd.DataFrame, target_column: str = "ItemId") -> None:
        """Initializes the IDReducer by extracting the IDs from the target column and
        constructing the lookup tables.

        Args:
            data (pd.DataFrame): The dataframe for which the target_column IDs need to
                be reduced.
            target_column (str, optional): The target column name. Defaults to "ItemId".
        """
        self.target_column = target_column
        self.ids = data[[target_column]].drop_duplicates(ignore_index=True)
        self.id_lookup = DefaultDict(
            self.ids.to_dict()[target_column]
        )  # Format {index: original_id}
        self.id_reverse_lookup = DefaultDict(
            {v: k for k, v in self.id_lookup.items()}
        )  # New format {original_id: index}

    def to_reduced(
        self, data: Union[pd.DataFrame, Dict[int, np.ndarray]]
    ) -> Optional[Union[pd.DataFrame, Dict[int, np.ndarray]]]:
        """Converts the IDs in data to the reduced IDs.

        This method chooses the right method for the conversion based on the type of
        the data argument.

        Args:
            data (Union[pd.DataFrame, Dict[int, np.ndarray]]): The input data
                containing unreduced IDs.

        Returns:
            Optional[Union[pd.DataFrame, Dict[int, np.ndarray]]]: The original data but
                with the IDs reduced to the consecutive integer range, namely from 0 to
                the number of unique IDs exclusive.
                If data does not match any type for which we have a reducing method,
                return None.
        """
        if isinstance(data, pd.DataFrame):
            return self.__to_reduced_from_df(data)
        elif isinstance(data, Dict):
            return self.__to_reduced_from_dict(data)

        return None

    def to_original(
        self, data: Union[pd.DataFrame, Dict[int, np.ndarray]]
    ) -> Optional[Union[pd.DataFrame, Dict[int, np.ndarray]]]:
        """Converts data containing reduced IDs back to data containing the original IDs.

        Args:
            data (Union[pd.DataFrame, Dict[int, np.ndarray]]): The input data containing
                reduced IDs.

        Returns:
            Optional[Union[pd.DataFrame, Dict[int, np.ndarray]]]: The input data but
                with the IDs converted back to their original value
                If data does not match any type for which we have a reducing method,
                return None.
        """

        if isinstance(data, pd.DataFrame):
            return self.__to_original_from_df(data)
        elif isinstance(data, Dict):
            return self.__to_original_from_dict(data)

        return None

    def get_to_original_array(self) -> np.ndarray:
        """Returns the mapping from reduced to original in a numpy array format, where
        the reduced id serves as an index to the array, and the value contained in that
        index corresponds to the original id.

        Returns:
            A numpy array containing the mapping.
        """
        return self.ids.to_numpy(dtype=int).ravel()

    def __to_reduced_from_df(self, data: pd.DataFrame) -> pd.DataFrame:
        """Converts the IDs in the target column to the reduced IDs.

        Args:
            data (pd.DataFrame): The dataframe from which the target column IDs need to
                be reduced.

        Returns:
            pd.DataFrame: The dataframe where the target column IDs have been reduced.
        """
        data = data.copy()
        data.loc[:, self.target_column] = data[self.target_column].map(
            self.id_reverse_lookup
        )
        return data

    def __to_reduced_from_dict(
        self, data: Dict[int, np.ndarray]
    ) -> Dict[int, np.ndarray]:
        """Converts the IDs, assumed to be in the np.ndarray, to the reduced IDs.

        Args:
            data (Dict[int, np.ndarray]): The input dictionary from which the IDs in the
                np.ndarray need to be reduced.

        Returns:
            Dict[int, np.ndarray]: The input dictionary where the IDs in the np.ndarray
                have been reduced.
        """
        data = {
            k: np.array([self.id_reverse_lookup[item] for item in v])
            for k, v in data.items()
        }
        return data

    def __to_original_from_df(self, data: pd.DataFrame) -> pd.DataFrame:
        """Converts reduced IDs in the target column back to their original IDs.

        Args:
            data (pd.DataFrame): The dataframe from which the target column IDs have
                been reduced.

        Returns:
            pd.DataFrame: The dataframe where the target column IDs have been converted
                back to their original ID.
        """
        data = data.copy()
        data.loc[:, self.target_column] = data[self.target_column].map(self.id_lookup)
        return data

    def __to_original_from_dict(
        self, data: Dict[int, np.ndarray]
    ) -> Dict[int, np.ndarray]:
        """Converts reduced IDs, assumed to be in the np.ndarray, back to their original
        IDs.

        Args:
            data (Dict[int, np.ndarray]): The input dictionary from which the IDs in the
                np.ndarray have been reduced.

        Returns:
            Dict[int, np.ndarray]: The input dictionary where the IDs in the np.ndarray
                have been converted back to their original IDs.
        """
        data = {
            k: np.array([self.id_lookup[item] for item in v]) for k, v in data.items()
        }
        return data
