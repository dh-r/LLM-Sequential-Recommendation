import pickle
import pandas as pd
import os

import openai
from openai.embeddings_utils import get_embeddings as openai_get_embeddings
from openai.embeddings_utils import get_embedding as openai_get_embedding

EMBEDDING_ENGINE = "text-embedding-ada-002"
COMPLETION_ENGINE = "text-ada-001"

# Get path to current folder.
cur_folder = __file__.removesuffix(os.path.basename(__file__))

# cur_folder includes the trailing /
with open(f"{cur_folder}key.txt", "r") as key_file:
    key = key_file.read().strip()

openai.api_key = key

# establish a cache of embeddings to avoid recomputing
# cache is a dict of tuples (text, engine) -> embedding, saved as a pickle file
# cur_folder includes the trailing /
embedding_cache_path_to_load = f"{cur_folder}openai_embeddings_cache.pkl"
embedding_cache_path_to_save = f"{cur_folder}openai_embeddings_cache.pkl"

# load the cache if it exists, and save a copy to disk

try:
    embedding_cache = pd.read_pickle(embedding_cache_path_to_load)
except FileNotFoundError:
    if "y" not in input(
        "Have you ensured that you downloaded the embedding cache from GCP?"
    ):
        raise Exception("No embedding cache found.")
    embedding_cache = {}
with open(embedding_cache_path_to_save, "wb") as embedding_cache_file:
    pickle.dump(embedding_cache, embedding_cache_file)


def in_cache(string: str):
    return (string, EMBEDDING_ENGINE) in embedding_cache.keys()


def prompt_openai(prompt: str, **kwargs) -> str:
    return openai.Completion.create(
        model=COMPLETION_ENGINE,
        prompt=prompt,
        **kwargs,
    )


# define a function to retrieve embeddings from the cache if present, and otherwise request via the API
def embedding_from_string(string: str) -> list:
    """Return embedding of given string, using a cache to avoid recomputing."""
    if (string, EMBEDDING_ENGINE) not in embedding_cache.keys():
        embedding_cache[(string, EMBEDDING_ENGINE)] = openai_get_embedding(
            string, EMBEDDING_ENGINE
        )
        with open(embedding_cache_path_to_save, "wb") as embedding_cache_file:
            pickle.dump(embedding_cache, embedding_cache_file)
    return embedding_cache[(string, EMBEDDING_ENGINE)]


def set_embeddings_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """Set the embeddings by using the name column in the DataFrame.

    If there is a name in the DataFrame that is not cached, we retrieve all the
    embeddings from openAI, because it is too slow to get the embeddings on a granular
    level.

    TODO: take the unknown embeddings and only get those from openAI, instead
    of all embeddings.

    Args:
        df (pd.DataFrame): The input dataframe, containing at least the name column.

    Returns:
        pd.DataFrame: The input dataframe with an additional embedding column. We do
            not actually have to return this, since the input dataframe is modified
            anyway.
    """
    names = list(df["name"])
    if all([(name, EMBEDDING_ENGINE) in embedding_cache for name in names]):
        embeddings = [embedding_cache[(name, EMBEDDING_ENGINE)] for name in names]
    else:
        embeddings = openai_get_embeddings(names, EMBEDDING_ENGINE)

        for name, embedding in zip(names, embeddings):
            embedding_cache[(name, EMBEDDING_ENGINE)] = embedding

        with open(embedding_cache_path_to_save, "wb") as embedding_cache_file:
            pickle.dump(embedding_cache, embedding_cache_file)

    df["ada_embedding"] = embeddings

    return df
