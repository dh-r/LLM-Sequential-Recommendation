import json
import numpy as np
import pandas as pd

from zipfile import ZipFile
from main.eval.metrics import MetricDependency
from main.eval.evaluation import Evaluation, EvaluationReport

STARTING_FOLDER = "../.."


def get_best_params(client, run_id):
    """Download best model params of specified parent run. If it's a child run, download its params."""
    artifacts = client.list_artifacts(run_id)
    model_params = {}
    # Parent run
    if "best_params.json" in [artifact.path for artifact in artifacts]:
        local_path = client.download_artifacts(run_id, "best_params.json")
        with open(local_path, "r") as f:
            model_params = json.load(f)
        return model_params
    for artifact in artifacts:
        # Child run
        if ".zip" in artifact.path:
            local_path = client.download_artifacts(run_id, artifact.path)
            with ZipFile(local_path, "r") as myzip:
                with myzip.open("config.pickle") as myfile:
                    model_params = pd.read_pickle(myfile)
            # This is passed as a separate argument to the BERTWithEmbeddings constructor
            if "product_embeddings_location" in model_params.keys():
                del model_params["product_embeddings_location"]
    if not model_params:
        raise Exception(f"Could not retrieve best params from run {run_id}.")
    return model_params


def train_models(model_runs, working_dir, model_class, mlflow_client):
    """Train a list of models on the folds of a dataset of the working directory."""
    trained_models = {}
    for model_name, run_id in model_runs.items():
        print(f"Training model {model_name} on folds' training data.")
        embedding_model = model_name.split("_")[1]
        model_dataset = pd.read_pickle(
            f"{working_dir}/{embedding_model}_dataset.pickle"
        )
        k_fold_eval = model_dataset.get_k_fold_eval()
        dataset = working_dir.split("/")[1]

        # Configure model parameters
        model_params = get_best_params(mlflow_client, run_id)

        # Get product embeddings file location
        product_embeddings_location = (
            f"{STARTING_FOLDER}/{dataset}/product_embeddings_{embedding_model}.csv.gzip"
        )

        # Initialize model
        if "BERTWEMB" in model_name:
            model = model_class(
                product_embeddings_location=product_embeddings_location, **model_params
            )
        else:
            model = model_class(**model_params)

        print(f"Number of folds: {len(k_fold_eval)}")
        k = 0

        # Iterate over folds
        for k_fold in k_fold_eval:
            print(f"Fold {k}")

            # Get training data of fold
            train_data_fold = k_fold[0]
            # Train model on training data of fold.
            if "BERTWEMB" in model_name:
                model.train(train_data_fold)
            else:
                # Item data is a dataframe with four columns: ItemId, embedding, class, category_size.
                # Category size indicates the number of unique values in category columns.
                model.train(train_data_fold, model_dataset.item_data)

            model_fold_name = f"{model_name}_fold_{k}"

            trained_models[model_fold_name] = model

            k += 1

    return trained_models


def remove_duplicates(recs: dict[int, np.ndarray]):
    """Remove duplicate recommendations."""
    new_recs = {}
    for s_id, s_recs in recs.items():
        sample_predictions_seen = set()
        sample_predictions = np.array(
            [
                sample_prediction
                for sample_prediction in s_recs
                if not (
                    sample_prediction in sample_predictions_seen
                    or sample_predictions_seen.add(sample_prediction)
                )
            ]
        )
        new_recs[s_id] = sample_predictions

    return new_recs


def truncate_to_top_k(recs: dict[int, np.ndarray], top_k):
    """Truncate recommendations to top-k."""
    return {k: v[:top_k] for k, v in recs.items()}


def evaluate_results(
    top_k_eval, preds, ground_truths, num_items, item_count, model_name, cores
):
    """Evaluate predictions with ground truths."""
    results = {}
    eval_reports = []
    print("Evaluating...")
    for top_k in top_k_eval:
        results.update(
            Evaluation.eval(
                preds,
                ground_truths,
                top_k=top_k,
                dependencies={
                    MetricDependency.NUM_ITEMS: num_items,
                    MetricDependency.ITEM_COUNT: item_count,
                },
                cores=cores,
                model_name=model_name,
            ).results
        )
    eval_reports.append(EvaluationReport(model_name, top_k=-1, results=results))
    return eval_reports


def format_results(eval_reports):
    """Format evaluation results."""
    results = Evaluation.from_results(eval_reports).results_as_table().data
    results = results.reset_index()

    data = results.copy()
    data["Model_1"] = data["index"].apply(lambda x: x.split("-")[0])
    data["Model_2"] = data["index"].apply(lambda x: x.split("-")[1])
    data["Comb Method"] = data["index"].apply(lambda x: x.split("-")[-2])
    data["Param"] = pd.to_numeric(
        data["index"].apply(lambda x: x.split("-")[-1]), errors="ignore"
    )
    data = data.drop("index", axis="columns") if "index" in data.columns else data

    metric_names = [
        "NDCG",
        "HitRate",
        "MRR",
        "Catalog coverage",
        "Serendipity",
        "Novelty",
        "Diversity",
    ]

    all_metrics = []
    for metric_name in metric_names:
        all_metrics.append(f"{metric_name}@10")

    for metric_name in metric_names:
        all_metrics.append(f"{metric_name}@20")

    all_metrics = [
        metric_name for metric_name in all_metrics if metric_name in data.columns
    ]

    data = data[["Model_1", "Model_2", "Comb Method", "Param", *all_metrics]]

    data = data.sort_values(by="NDCG@10", ascending=False)

    return data


def get_item_lookups(item_data):
    """Creates dictionaries for mapping dataset-specific item ids to/from global item ids."""
    item_dataset_to_global = {
        row["ItemId"]: index for index, row in item_data.iterrows()
    }
    item_global_to_dataset = {
        index: row["ItemId"] for index, row in item_data.iterrows()
    }

    return item_dataset_to_global, item_global_to_dataset


def map_to_global(items, item_dataset_to_global):
    """Maps a list of dataset-specific item ids to global item ids."""
    global_items = [
        item_dataset_to_global[item]
        for item in items
        if item in item_dataset_to_global.keys()
    ]
    return global_items


def map_from_global(items, item_global_to_dataset):
    """Maps a list of global item ids to dataset-specific item ids."""
    dataset_items = [
        item_global_to_dataset[item]
        for item in items
        if item in item_global_to_dataset.keys()
    ]
    return dataset_items
