# Beauty 

## Source
The dataset was found [here](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html)  in "per-category files". The necessary files are the Beauty [reviews](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty.json.gz) and the Beauty [metadata](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Beauty.json.gz). Note this is the older version of the Amazon Beauty dataset. This is intentional, because this way it matches the dataset with most research papers (BERT4Rec, SASRec) using Beauty.

## Processing
The files `reviews_Beauty.json` and `meta_Beauty.json` are initially processed by `create_sessions.ipynb`, which as the name suggests creates a `sessions.csv`. The table contains the columns `index`, `SessionId`, `ItemId`, `Time`, `Reward`. The `Time` column contains the Unix timestamp of the interaction. We do not use the last column `Reward` in our experiments or evaluation. We consider reviews by the same reviewer as an interaction in the session belonging to that reviewer. We do some small processing on the item names; we remove some special characters.

We then use `format_timestamps.ipynb` to format the timestamps such that they contain timestamp strings rather than epoch time in seconds. This step produces `formatted_sessions.csv`.

The `formatted_sessions.csv` is converted to our standard Dataset (see `main/data/session_dataset.py`) object in `split_sessions.ipynb`. This notebook is responsible for splitting the train and test sessions (including the folds used for hyperparameter search) and processing the sessions to the desired format of our recommendation models. The final product of this directory is the `dataset.pickle`, which is a persisted ([pickled](https://docs.python.org/3/library/pickle.html)) Dataset object. We share this pickle to ensure reproducibility of our results. Note that the process of creating the `dataset.pickle` is deterministic, but the the mappings (asin -> ItemId) and (reviewerId -> SessionId) may differ across runs.
