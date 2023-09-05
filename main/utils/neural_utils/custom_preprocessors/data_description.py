import numpy as np
import pandas as pd

from typing import TypedDict


class DataDescription(TypedDict):
    num_users: int
    num_items: int
    item_distribution: pd.Series
    session_length_description: pd.Series


def get_data_description(data: pd.DataFrame) -> DataDescription:
    data_description: DataDescription = {
        "num_sessions": data["SessionId"].nunique(),
        "num_items": data["ItemId"].nunique(),
        "item_distribution": data["ItemId"].value_counts(normalize=True, sort=False),
        "session_length_description": data.groupby("SessionId")["ItemId"]
        .count()
        .describe(np.arange(0, 1.05, 0.05)),
    }
    return data_description
