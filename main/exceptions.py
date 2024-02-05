"""This module contains custom exceptions used in the library."""


class InvalidStateError(Exception):
    """Should be thrown when the internal state of an object does not allow an operation."""

    pass
