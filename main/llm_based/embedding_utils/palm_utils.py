import pickle
import pandas as pd
import os
from vertexai.language_models import TextEmbeddingModel

EMBEDDING_ENGINE = "textembedding-gecko@002"
EMBEDDING_MODEL = TextEmbeddingModel.from_pretrained(EMBEDDING_ENGINE)

# Get path to current folder.
cur_folder = __file__.removesuffix(os.path.basename(__file__))

# establish a cache of embeddings to avoid recomputing
# cache is a dict of tuples (text, engine) -> embedding, saved as a pickle file
# cur_folder includes the trailing /
embedding_cache_path_to_load = f"{cur_folder}palm_embeddings_cache.pkl"
embedding_cache_path_to_save = f"{cur_folder}palm_embeddings_cache.pkl"

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


def embedding_from_string(string: str) -> list:
    """Return embedding of given string, using a cache to avoid recomputing."""
    if (string, EMBEDDING_ENGINE) not in embedding_cache.keys():
        embedding_cache[(string, EMBEDDING_ENGINE)] = embed_texts([string])[0]
        with open(embedding_cache_path_to_save, "wb") as embedding_cache_file:
            pickle.dump(embedding_cache, embedding_cache_file)
    return embedding_cache[(string, EMBEDDING_ENGINE)]


def set_embeddings_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """Set the embeddings by using the name column in the DataFrame.

    If there is a name in the DataFrame that is not cached, we retrieve all the
    embeddings, because it is too slow to get the embeddings on a granular level.

    TODO: take the unknown embeddings and only get those instead of all embeddings

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
        embeddings = embed_texts(names)

        for name, embedding in zip(names, embeddings):
            embedding_cache[(name, EMBEDDING_ENGINE)] = embedding

        with open(embedding_cache_path_to_save, "wb") as embedding_cache_file:
            pickle.dump(embedding_cache, embedding_cache_file)

    df["gecko_embedding"] = embeddings

    return df


def embed_texts(text_list) -> list:
    batch_size = 250
    cur_step = batch_size
    cur_batch = text_list[0:cur_step]
    n_batch_processed = 0
    all_embeddings = []
    while len(cur_batch) > 0:
        batch_embeddings = [
            emb.values for emb in EMBEDDING_MODEL.get_embeddings(cur_batch)
        ]
        all_embeddings += batch_embeddings

        n_batch_processed += 1
        cur_batch = text_list[cur_step : cur_step + batch_size]
        cur_step += batch_size

        print(f"Proccessed batch {n_batch_processed}.")
    return all_embeddings
