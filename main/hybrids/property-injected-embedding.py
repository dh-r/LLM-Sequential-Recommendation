import mlflow
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
CUTOFF_QUANTILES = [0.25, 0.5, 0.75]
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

    eval_reports = []

    for property in ["pop", "divers", "pop_divers"]:
        print(f"Property: {property}")

        best_hybrid_model = None
        best_hybrid_performance = -1
        best_hybrid_property_param = None

        for model_name, run_id in llmseqsim_runs.items():
            print("Evaluating hybrid...")
            print(f"Model: {model_name}")

            eval_results = []

            embedding_model = model_name.split("_")[1]
            model_dataset = pd.read_pickle(
                f"{working_dir}/{embedding_model}_dataset.pickle"
            )
            # k_fold_eval: A list containing, for each validation fold, a tuple 3 elements.
            #             The first is the training data of the fold, the second is the validation
            #             data of the fold prepared for prediction, and the third is the ground truths
            #             extracted from the validation data.
            k_fold_eval = model_dataset.get_k_fold_eval()
            num_items = model_dataset.get_unique_item_count()
            item_count = model_dataset.get_item_counts()
            # num_interactions = model_dataset.get_num_interactions()
            num_sessions = model_dataset.get_unique_sample_count()
            input_data = model_dataset.get_input_data()
            item_data = model_dataset.get_item_data()

            # Configure model parameters
            model_params = get_best_params(MLFLOW_CLIENT, run_id)

            # Initialize model
            model = LLMSeqSim(**model_params)

            print(f"Number of folds: {len(k_fold_eval)}")
            k = 0

            # Iterate over folds
            for k_fold in k_fold_eval:
                print(f"Fold {k}")

                # Get training, validation, and ground truth data of fold
                train_data_fold = k_fold[0]
                val_data_fold = k_fold[1]
                ground_truths_fold = k_fold[2]
                # Train models on training data of fold.
                # Item data is a dataframe with four columns: ItemId, embedding, class, category_size.
                # Category size indicates the number of unique values in category columns.
                model.train(train_data_fold, item_data)

                # Get more predictions than top-k because low-quality ones will be removed with property.
                model_val_recs = model.predict(
                    val_data_fold, top_k=TOP_K * 5, return_scores=False
                )

                model_fold_name = f"{model_name}_fold_{k}"

                if property == "pop":
                    eval_results.extend(
                        pop_property_single_model(
                            CUTOFF_QUANTILES,
                            model_fold_name,
                            model_val_recs,
                            ground_truths_fold,
                            num_items,
                            item_count,
                            num_sessions,
                            input_data,
                            TOP_K,
                            TOP_K_EVAL,
                            CORES,
                        )
                    )
                elif property == "divers":
                    eval_results.extend(
                        divers_property(
                            CUTOFF_QUANTILES,
                            model_fold_name,
                            model_val_recs,
                            ground_truths_fold,
                            num_items,
                            item_count,
                            item_data,
                            TOP_K,
                            TOP_K_EVAL,
                            CORES,
                        )
                    )
                else:
                    eval_results.extend(
                        pop_divers_property(
                            CUTOFF_QUANTILES,
                            CUTOFF_QUANTILES,
                            model_fold_name,
                            model_val_recs,
                            ground_truths_fold,
                            num_items,
                            item_count,
                            num_sessions,
                            input_data,
                            item_data,
                            TOP_K,
                            TOP_K_EVAL,
                            CORES,
                        )
                    )

                k += 1

            report = format_results(eval_results)
            report.to_csv(
                f"{RESULTS_FOLDER[dataset]}/property-injected_embedding_results_{property}_{model_name}.csv",
                index=False,
            )
            # Compute average NDCG across folds
            report = report.groupby("Param")["NDCG@20"].mean().reset_index()
            # Find property param with best average NDCG
            best_model_performance = report["NDCG@20"].max()
            best_model_property_param = report[
                report["NDCG@20"] == best_model_performance
            ]["Param"].values[0]

            print(f"Best model performance: {best_model_performance}")
            print(f"Best model {property} param: {best_model_property_param}")

            if best_model_performance > best_hybrid_performance:
                best_hybrid_performance = best_model_performance
                best_hybrid_property_param = best_model_property_param
                best_hybrid_model = model_name

        print(f"Best hybrid performance: {best_hybrid_performance}")
        print(f"Best hybrid {property} param: {best_hybrid_property_param}")
        print(f"Best hybrid embedding model: {best_hybrid_model}")

        print(f"Training best hybrid on entire training set...")
        print(f"Model: {best_hybrid_model}")

        best_hybrid_embedding_model = best_hybrid_model.split("_")[1]
        model_dataset = pd.read_pickle(
            f"{working_dir}/{best_hybrid_embedding_model}_dataset.pickle"
        )

        test_prompts = model_dataset.get_test_prompts()
        ground_truths = model_dataset.get_test_ground_truths()
        num_items = model_dataset.get_unique_item_count()
        item_count = model_dataset.get_item_counts()
        # num_interactions = dataset.get_num_interactions()
        num_sessions = model_dataset.get_unique_sample_count()
        input_data = model_dataset.get_input_data()
        item_data = model_dataset.get_item_data()

        # Configure model parameters
        model_params = get_best_params(MLFLOW_CLIENT, llmseqsim_runs[best_hybrid_model])

        # Initialize model
        model = LLMSeqSim(**model_params)

        model.train(model_dataset.get_train_data(), item_data)

        print(f"Predicting with best hybrid on all test prompts...")

        # Get more predictions than top-k because low-quality ones will be removed with property.
        model_recs = model.predict(test_prompts, top_k=TOP_K * 5, return_scores=False)

        if property == "pop":
            eval_reports.extend(
                pop_property_single_model(
                    [best_hybrid_property_param],
                    best_hybrid_model,
                    model_recs,
                    ground_truths,
                    num_items,
                    item_count,
                    num_sessions,
                    input_data,
                    TOP_K,
                    TOP_K_EVAL,
                    CORES,
                )
            )
        elif property == "divers":
            eval_reports.extend(
                divers_property(
                    [best_hybrid_property_param],
                    best_hybrid_model,
                    model_recs,
                    ground_truths,
                    num_items,
                    item_count,
                    item_data,
                    TOP_K,
                    TOP_K_EVAL,
                    CORES,
                )
            )
        else:
            best_hybrid_pop_param = float(best_hybrid_property_param.split("_")[0])
            best_hybrid_divers_param = float(best_hybrid_property_param.split("_")[1])

            eval_reports.extend(
                pop_divers_property(
                    [best_hybrid_pop_param],
                    [best_hybrid_divers_param],
                    best_hybrid_model,
                    model_recs,
                    ground_truths,
                    num_items,
                    item_count,
                    num_sessions,
                    input_data,
                    item_data,
                    TOP_K,
                    TOP_K_EVAL,
                    CORES,
                )
            )

    format_results(eval_reports).to_csv(
        f"{RESULTS_FOLDER[dataset]}/property-injected_embedding_results.csv",
        index=False,
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Finished in {elapsed_time} seconds.")
