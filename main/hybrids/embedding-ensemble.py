import mlflow
import numpy as np
import os
import pandas as pd
import time

from main.hybrids.properties import *
from main.hybrids.utils import *
from main.llm_based.similarity_model.llm_seq_sim import LLMSeqSim

# Configs
DATASETS = ["beauty", "steam"]
STARTING_FOLDER = "../.."
RESULTS_FOLDER = {
    "beauty": f"{STARTING_FOLDER}/results/beauty/main/hybrids",
    "steam:": f"{STARTING_FOLDER}/results/steam/main/hybrids",
}
WORKING_DIRS = [f"{STARTING_FOLDER}/{DATASET}" for DATASET in DATASETS]

TOP_K = 20
TOP_K_EVAL = [10, 20]
CUTOFF_RANK = [0, 1, 2, 3, 4, 5, 10, 20]
CUTOFF_CONF = np.linspace(0.8, 0.99, 10)
CORES = 60

LLMSEQSIM_RUNS = {
    "beauty": {
        "LSS_openai_original_size": "f1d983a1e51d42f8b2594c9f0727d533",
        "LSS_openai_dim_reduction": "b13fae9702f64f9e93ad8f32333db978",
        "LSS_palm_original_size": "2610c8839eb04bbca3cf4eb5d09f8404",
        "LSS_palm_dim_reduction": "d5e00058b7ec4d76b25d146ca9276ef2",
    },
    "steam": {
        "LSS_openai_original_size": "c278a0fe3fdb4521be3ec2ba4d09ea42",  # llmseqsim_openai-original_sized-hypersearch-3-days_35
        "LSS_openai_dim_reduction": "0e7754e33b3c46d3813681e655160874",
        "LSS_palm_original_size": "78b237813ea0485cbabc1fbd86ab8536",  # llmseqsim_palm-original_sized-hypersearch-3-days_48
        "LSS_palm_dim_reduction": "d1486afee3bc4243863c26ff4b864258",
    },
}

assert "MLFLOW_TRACKING_USERNAME" in os.environ
assert "MLFLOW_TRACKING_PASSWORD" in os.environ

mlflow.set_tracking_uri("https://mlflow-dev.dhr-services.com")

MLFLOW_CLIENT = mlflow.tracking.MlflowClient()


for working_dir in WORKING_DIRS:
    start_time = time.time()
    print(f"Working directory: {working_dir}")

    dataset = working_dir.split("/")[1]
    # Get model names and runs
    llmseqsim_runs = LLMSEQSIM_RUNS[dataset]
    # Get list of model names
    model_names = list(llmseqsim_runs.keys())
    # Get all pairs of embedding models
    embedding_pairs = [
        (a, b) for idx, a in enumerate(model_names) for b in model_names[idx + 1 :]
    ]

    eval_reports = []

    trained_models = train_models(llmseqsim_runs, working_dir, LLMSeqSim, MLFLOW_CLIENT)

    for property in ["rank", "conf"]:
        print(f"Property: {property}")

        best_hybrid_model_1 = None
        best_hybrid_model_2 = None
        best_hybrid_performance = -1
        best_hybrid_property_param = None

        for pair in embedding_pairs:
            print("Evaluating hybrid...")
            print(f"Model 1: {pair[0]}")
            print(f"Model 2: {pair[1]}")

            eval_results = []

            embedding_model_1 = pair[0].split("_")[1]

            dataset_1 = pd.read_pickle(
                f"{working_dir}/{embedding_model_1}_dataset.pickle"
            )

            k_fold_eval = dataset_1.get_k_fold_eval()
            num_items = dataset_1.get_unique_item_count()
            item_count = dataset_1.get_item_counts()

            print(f"Number of folds: {len(k_fold_eval)}")
            k = 0

            # Iterate over folds
            for k_fold in k_fold_eval:
                print(f"Fold {k}")

                # Get validation and ground truth data of fold
                val_data_fold = k_fold[1]
                ground_truths_fold = k_fold[2]

                model_1_name = f"{pair[0]}_fold_{k}"
                model_2_name = f"{pair[1]}_fold_{k}"

                model_1 = trained_models[model_1_name]
                model_2 = trained_models[model_2_name]

                model_2_val_recs = model_2.predict(
                    val_data_fold, top_k=TOP_K, return_scores=False
                )

                if property == "rank":
                    model_1_val_recs = model_1.predict(
                        val_data_fold, top_k=TOP_K, return_scores=False
                    )

                    eval_results.extend(
                        rank_property(
                            CUTOFF_RANK,
                            model_1_name,
                            model_2_name,
                            model_1_val_recs,
                            model_2_val_recs,
                            val_data_fold,
                            ground_truths_fold,
                            num_items,
                            item_count,
                            TOP_K,
                            TOP_K_EVAL,
                            CORES,
                        )
                    )
                else:
                    model_1_val_recs, model_1_val_scores = model_1.predict(
                        val_data_fold, top_k=TOP_K, return_scores=True
                    )

                    eval_results.extend(
                        conf_property(
                            CUTOFF_CONF,
                            model_1_name,
                            model_2_name,
                            model_1_val_recs,
                            model_2_val_recs,
                            model_1_val_scores,
                            val_data_fold,
                            ground_truths_fold,
                            num_items,
                            item_count,
                            TOP_K,
                            TOP_K_EVAL,
                            CORES,
                        )
                    )

                k += 1

            report = format_results(eval_results)
            report.to_csv(
                f"{RESULTS_FOLDER[dataset]}/embedding_ensemble_results_{property}_{pair[0]}_{pair[1]}.csv",
                index=False,
            )
            # Compute average NDCG across folds
            report = report.groupby("Param")["NDCG@20"].mean().reset_index()
            # Find property param with best average NDCG
            best_pair_performance = report["NDCG@20"].max()
            best_pair_property_param = report[
                report["NDCG@20"] == best_pair_performance
            ]["Param"].values[0]

            print(f"Best pair performance: {best_pair_performance}")
            print(f"Best pair {property} param: {best_pair_property_param}")

            if best_pair_performance > best_hybrid_performance:
                best_hybrid_performance = best_pair_performance
                best_hybrid_property_param = best_pair_property_param
                best_hybrid_model_1 = pair[0]
                best_hybrid_model_2 = pair[1]

        print(f"Best hybrid performance: {best_hybrid_performance}")
        print(f"Best hybrid {property} param: {best_hybrid_property_param}")
        print(f"Best hybrid embedding model 1: {best_hybrid_model_1}")
        print(f"Best hybrid embedding model 2: {best_hybrid_model_2}")

        print(f"Training best hybrid on entire training set...")
        print(f"Model 1: {best_hybrid_model_1}")
        print(f"Model 2: {best_hybrid_model_2}")

        best_hybrid_embedding_model_1 = best_hybrid_model_1.split("_")[1]
        best_hybrid_embedding_model_2 = best_hybrid_model_2.split("_")[1]

        dataset_1 = pd.read_pickle(
            f"{working_dir}/{best_hybrid_embedding_model_1}_dataset.pickle"
        )
        dataset_2 = pd.read_pickle(
            f"{working_dir}/{best_hybrid_embedding_model_2}_dataset.pickle"
        )

        test_prompts = dataset_1.get_test_prompts()
        ground_truths = dataset_1.get_test_ground_truths()
        num_items = dataset_1.get_unique_item_count()
        item_count = dataset_1.get_item_counts()

        # Configure model parameters
        model_1_params = get_best_params(
            MLFLOW_CLIENT, llmseqsim_runs[best_hybrid_model_1]
        )
        model_2_params = get_best_params(
            MLFLOW_CLIENT, llmseqsim_runs[best_hybrid_model_2]
        )

        # Initialize models
        model_1 = LLMSeqSim(**model_1_params)
        model_2 = LLMSeqSim(**model_2_params)

        model_1.train(dataset_1.get_train_data(), dataset_1.item_data)
        model_2.train(dataset_2.get_train_data(), dataset_2.item_data)

        print(f"Predicting with best hybrid on all test prompts...")

        model_2_recs = model_2.predict(test_prompts, top_k=TOP_K, return_scores=False)

        model_1_name = best_hybrid_model_1
        model_2_name = best_hybrid_model_2

        if property == "rank":
            model_1_recs = model_1.predict(
                test_prompts, top_k=TOP_K, return_scores=False
            )

            eval_reports.extend(
                rank_property(
                    [best_hybrid_property_param],
                    model_1_name,
                    model_2_name,
                    model_1_recs,
                    model_2_recs,
                    test_prompts,
                    ground_truths,
                    num_items,
                    item_count,
                    TOP_K,
                    TOP_K_EVAL,
                    CORES,
                )
            )
        else:
            model_1_recs, model_1_scores = model_1.predict(
                test_prompts, top_k=TOP_K, return_scores=True
            )

            eval_reports.extend(
                conf_property(
                    [best_hybrid_property_param],
                    model_1_name,
                    model_2_name,
                    model_1_recs,
                    model_2_recs,
                    model_1_scores,
                    test_prompts,
                    ground_truths,
                    num_items,
                    item_count,
                    TOP_K,
                    TOP_K_EVAL,
                    CORES,
                )
            )

    format_results(eval_reports).to_csv(
        f"{RESULTS_FOLDER[dataset]}/embedding_ensemble_results.csv", index=False
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Finished in {elapsed_time} seconds.")
