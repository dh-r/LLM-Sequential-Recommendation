import pandas as pd
import numpy as np
import tensorflow as tf

from typing import Union


class TensorFactory:
    """TensorFactory is a class to encapsulate all functionality needed to convert
    sessions into the form expected by the session models. It truncates sessions that are
    too long, and pads sessions that are too short so that they all have uniform
    length. In addition, it removes any occurence of -1 from the original session,
    because these are assumed to correspond to unknown items."""

    PADDING_TARGET: int = -1
    RNG = np.random.default_rng()

    @staticmethod
    def to_sequence_tensor(sessions: Union[pd.DataFrame, dict], sequence_length: int):
        if isinstance(sessions, pd.DataFrame):
            sessions: pd.DataFrame = sessions[["SessionId", "ItemId"]]
            sessions: list[np.ndarray] = (
                sessions.groupby("SessionId")["ItemId"].apply(np.array).tolist()
            )
        elif isinstance(sessions, dict):
            sessions: list[np.ndarray] = list(sessions.values())
        return TensorFactory.__process_sessions(sessions, sequence_length)

    @staticmethod
    def __process_sessions(sessions: list, sequence_length: int) -> tf.Tensor:
        """Processes a list of non-uniform length sessions into uniform length sequences
        using truncating and padding. Also removes -1 from the original sessions,
        because these are assumed to be unknown items.

        Args:
            sessions (list): The list of sessions.
            sequence_length (int): The length that the sequences should become.

        Returns:
            tf.Tensor: A sequence tensor in the form expected by the BERTModel.
        """
        if sequence_length <= 0:
            raise ValueError("Sequence length must be positive.")

        desired_shape = (len(sessions), sequence_length)
        sequences = np.zeros(desired_shape, dtype=np.int32)

        for i, session in enumerate(sessions):
            session = np.delete(session, np.where(session == -1), axis=0)

            if len(session) > sequence_length:
                # Get last sequence_length items
                session = session[-sequence_length:]
            elif len(session) < sequence_length:
                # Pad sequence with PADDING_TARGET
                num_pad: int = sequence_length - len(session)
                session = np.pad(
                    session,
                    (num_pad, 0),
                    mode="constant",
                    constant_values=TensorFactory.PADDING_TARGET,
                )
            sequences[i, :] = session

        return tf.convert_to_tensor(sequences)
