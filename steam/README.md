# Steam

## Source
The files needed for this dataset can be found here: https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data. The files are `steam_games.json` and `steam_reviews.json`.

## Preprocessing
We use `create_sessions.ipynb` to parse and preprocess the raw data files to obtain the `sessions.csv` and `item_metadata.csv` files. The former contains the item-session interaction data, and the latter contains metadata such as name, category, etc. We use this information to obtain item embeddings. More details on this step can be found in the paper at section 7.1.1.

We then use `attach_embeddings.ipynb` to attach the embeddings to the `dataset.pickle` object.
