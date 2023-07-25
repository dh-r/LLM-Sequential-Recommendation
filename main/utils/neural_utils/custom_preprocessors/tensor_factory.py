import pandas as pd
import numpy as np
import tensorflow as tf

from typing import Dict, Union


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

    @classmethod
    def slice_random(
        cls, sessions: list[np.ndarray], sequence_length: int, min_session_length: int
    ) -> list[np.ndarray]:
        """Gets random slices from the sessions.

        For example, a session [1, 2, 3, 4, 5] could be sliced into
        [2, 3, 4], [1, 2], or [3, 4, 5], but never into a session larger than
        sequence_length.

        Args:
            sessions (list[np.ndarray]): _description_
            sequence_length (int): _description_

        Returns:
            list[np.ndarray]: _description_
        """

        # Get random lengths
        max_session_length = sequence_length

        # Generate a probability distribution from the min_session_length to max_session_length
        unnormalized_min_prob = 1
        unnormalized_max_prob = 2
        unnormalized_probabilities = np.linspace(
            start=unnormalized_min_prob,
            stop=unnormalized_max_prob,
            num=max_session_length - min_session_length + 1,
        )
        normalized_probabilities = unnormalized_probabilities / np.sum(
            unnormalized_probabilities
        )

        # Generate a list of new session lengths for each session.
        new_session_lengths = np.random.choice(
            a=np.arange(min_session_length, max_session_length + 1),
            size=len(sessions),
            replace=True,
            p=normalized_probabilities,
        )

        # For each session, get a random offset, and add to new_sessions.
        new_sessions = []
        for session, new_length in zip(sessions, new_session_lengths):
            max_offset = max(len(session) - new_length + 1, 1)
            new_offset = cls.RNG.integers(0, max_offset)
            new_sessions.append(session[new_offset : new_offset + new_length])

        return new_sessions

    def slice_full(sessions: list[np.ndarray], sequence_length: int):
        new_sessions = []
        min_length = sequence_length
        for session in sessions:
            if len(session) > min_length + 1:
                for length in range(6, max(len(session) + 1, sequence_length + 1)):
                    max_offset = max(len(session) - length + 1, 1)
                    for offset in range(max_offset):
                        new_sessions.append(session[offset : offset + length])
            else:
                new_sessions.append(session)
        return new_sessions

    def random_subset(sessions: list[np.ndarray], sequence_length: int):
        new_sessions = []
        for session in sessions:
            new_session_indices = np.sort(
                np.random.choice(
                    np.arange(0, len(session)),
                    size=min(sequence_length, len(session)),
                    replace=False,
                )
            )
            new_sessions.append(session[new_session_indices])

        return new_sessions
