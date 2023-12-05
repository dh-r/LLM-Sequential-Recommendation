# LLM-sequential-recommendations
Implementation and reproducible experiments behind the research paper "Leveraging Large Language Models for Sequential Recommendation" published in RecSys'23 Late-Breaking Results track. Please cite as follows.

```
@inproceedings{Harte2023leveraging,
author = {Harte, Jesse and Zorgdrager, Wouter and Louridas, Panos and Katsifodimos, Asterios and Jannach, Dietmar and Fragkoulis, Marios},
title = {Leveraging Large Language Models for Sequential Recommendation}, 
year = {2023},
isbn = {979-8-4007-0241-9/23/09},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3604915.3610639},
doi = {10.1145/3604915.3610639},
booktitle = {Proceedings of the 17th ACM Conference on Recommender Systems},
numpages = {7},
location = {Singapore, Singapore},
series = {RecSys '23}
}
```


## Installation

In order to develop code in this repository, you'll need `poetry`. In case you don't have `poetry` installed, follow the steps in the steps in the [official documentation](https://python-poetry.org/docs/#installation).

### Activate a shell in the environment

This will activate a shell within the current `poetry` environment. If the environment does not exist, it will be automatically created.

```bash
poetry shell
```

Alternatively, you can prefix your commands with 

```
poetry run 
```

which will have the same effect as executing commands within the `poetry` environment. 

### Install requirements and local package

This command will also install the `llm-sequential-recommendations` package in editable mode. This means that any changes made in the code will be automatically reflected, as long as they reside in directories that existed during the installation. If at any point you get errors that do not make sense, feel free to run the command again.

```bash
poetry install
```


# Repository organization 

## Overview
We organized the repository by distinguishing between dataset-related code (e.g. `beauty`), implementation code (inside `main`), notebooks for analysis and visualization (inside `notebooks`), and results (in `results`). We explain our dataset-related code in a separate `README.md` in the corresponding directory. 

## Data-related code 
For now we have included all code we used to process Beauty in the `beauty` directory with a separate `README.md`. Regarding the Delivery Hero dataset, we are discussing with our organization to make it available.

## Main directory
The implementation code in `main` first of all contains `data` and `eval`. The former contains the implementation of our `SessionDataset`, which is a convenient object to group together all the code and data related to a single dataset, including the train and test set, but also the train and test set of the validation folds. The latter contains the implementation of each of the metrics and `evaluation.py`, which converts the recommendations and ground-truths into the format expected by the metrics, evaluates the recommendation using the metrics, and subsequently provides a view of the evaluation on all metrics. 

Furthermore, `main` contains the implementations of each recommendation model in their respective directory inside the `main` directory. The exception here are the `LLMSeqSim` and `LLMSeqPrompt` that are grouped together under `openai`, since both of these models require the utilities in `openai_utils`. Both models require the `product_embeddings_openai.csv.gzip` file created by `create_embeddings.ipynb`. This notebook in turn uses uses `openai_utils.py`, which requires the OpenAI API key to be set in `key.txt`. We added the embeddings created by `create_embeddings.ipynb` for the Beauty dataset in the `beauty` directory. Note that `LLMSeqSim` is implemented in `main/openai/similarity_model`, and `LLMSeqPrompt` is implemented in `main/openai/prompt_model`. 

All models except `LLMSeqSim` and `LLMSeqPrompt` implement the interface of `abstract_model`. The `train` method accepts a `pd.DataFrame` containing `SessionId` and `ItemId` as columns. The predict method accepts a dictionary mapping `SessionId` to a 1-dimensional `np.ndarray` of items, sorted according to time. The predict method returns an other dictionary mapping `SessionId` to an other 1-dimensional `np.ndarray` of recommended items, sorted descendingly on confidence (so the most confidence is on position 0). For more implementation details on the models, refer to `online_material.md`.

## Running experiments 
Running experiments using the models and a given dataset is done in `run_experiments.ipynb`. You can set `DATASET_FILENAME` to the path of the dataset pickle. `EXPERIMENTS_FOLDER` is the directory to which the recommendations and configurations are persisted, and from which the evaluation retrieves all recommendation pickles to evaluate. We hardcoded the optimal configurations returned by our hypersearch. In addition, we persisted the weights of the neural models with the top-performing configurations to ensure reproducibility. 
