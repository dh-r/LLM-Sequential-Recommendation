import itertools


def split_dict(x: dict, chunks: int) -> list[dict]:
    """Split a dictionary into multiple chunks.

    Args:
        x (dict): The dictionary to split.
        chunks (int): The number of chunks to split the dictionary into.

    Returns:
        List[Dict]: A list of dictionaries containing the original key-value pairs,
        split evenly across the specified number of chunks.

    """
    i = itertools.cycle(range(chunks))
    split = [dict() for _ in range(chunks)]
    for k, v in x.items():
        split[next(i)][k] = v
    return split
