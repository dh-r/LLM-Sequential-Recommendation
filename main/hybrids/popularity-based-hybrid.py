import mlflow
import os
import pandas as pd
import time

from main.hybrids.properties import *
from main.hybrids.utils import *
from main.llm_based.similarity_model.llm_seq_sim import LLMSeqSim
from main.sknn.sknn import SessionBasedCF  # SKNN with embeddings
from main.transformer.bert.bert_with_embeddings import BERTWithEmbeddings

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
CUTOFF_POP = [0.25, 0.5, 0.75]
CORES = 12

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

BERT_W_EMB_RUNS = {
    "beauty": {
        "BERTWEMB_openai": "f58077ba0cc24553b92b37bcc92309fc",  # bert_w_emb_wo_clipnorm-openai-final_hs-3-days_gpu
        "BERTWEMB_palm": "24f36e2fbb4d4583bd815e13e9cb3876",  # bert_w_emb_w_clipnorm-palm-final_hs-3-days_gpu_110
    },
    "steam": {
        "BERTWEMB_openai": "2a943c2fdf334ddca49c74250946acca",  # bert_w_emb_wo_clipnorm-openai-final_hs-3-days_gpu
        "BERTWEMB_palm": "cd694c43a10749da9dfeea30ca2b5eb7",  # bert_w_emb_wo_clipnorm-palm-final_hs-3-days_gpu
    },
}

SKNN_W_EMB_RUNS = {
    "beauty": {
        "SKNN_openai_original_size": "191ffbc3dd6c4cedb391d49c43e80384",
        "SKNN_openai_dim_reduction": "241b9c14539f45b6a8e62473cb6b5195",
        "SKNN_palm_original_size": "0db253b3c1bc437ba19a126860655224",
        "SKNN_palm_dim_reduction": "7f013567147e45e69ac38647462ad37b",  # sknn_palm-dim_reduction-hypersearch-3-days_71
    },
    "steam": {
        "SKNN_openai_original_size": "32d2c9dc3d2d45609e59328c6ab09cbb",
        "SKNN_openai_dim_reduction": "5a9b24d27f994252a153ea85cb9f1f07",
        "SKNN_palm_original_size": "abf21ec06ec2402383172e78c939620a",
        "SKNN_palm_dim_reduction": "3800e62aec6f4ea9a4b37774c5a2b75b",
    },
}

assert "MLFLOW_TRACKING_USERNAME" in os.environ
assert "MLFLOW_TRACKING_PASSWORD" in os.environ

mlflow.set_tracking_uri("https://mlflow-dev.dhr-services.com")

MLFLOW_CLIENT = mlflow.tracking.MlflowClient()


property = "pop"

for working_dir in WORKING_DIRS:
    start_time = time.time()
    print(f"Working directory: {working_dir}")

    dataset = working_dir.split("/")[1]
    # Get model names and runs
    llmseqsim_runs = LLMSEQSIM_RUNS[dataset]
    bert_runs = BERT_W_EMB_RUNS[dataset]
    sknn_runs = SKNN_W_EMB_RUNS[dataset]

    trained_llmseqsim_models = train_models(
        llmseqsim_runs, working_dir, LLMSeqSim, MLFLOW_CLIENT
    )

    eval_reports = []

    for neural_model in ["bert", "sknn"]:
        if neural_model == "bert":
            model_runs = bert_runs
            model_class = BERTWithEmbeddings
        else:
            model_runs = sknn_runs
            model_class = SessionBasedCF

        trained_neural_models = train_models(
            model_runs, working_dir, model_class, MLFLOW_CLIENT
        )

        best_hybrid_model_1 = None
        best_hybrid_model_2 = None
        best_hybrid_performance = -1
        best_hybrid_property_param = None

        for model_1_name, run_id_1 in llmseqsim_runs.items():
            embedding_model_1 = model_1_name.split("_")[1]
            dataset_1 = pd.read_pickle(
                f"{working_dir}/{embedding_model_1}_dataset.pickle"
            )
            k_fold_eval = dataset_1.get_k_fold_eval()
            num_items = dataset_1.get_unique_item_count()
            item_count = dataset_1.get_item_counts()
            # num_interactions = model_dataset.get_num_interactions()
            num_sessions = dataset_1.get_unique_sample_count()
            input_data = dataset_1.get_input_data()

            for model_2_name, run_id_2 in model_runs.items():
                print("Evaluating hybrid...")
                print(f"Model 1: {model_1_name}")
                print(f"Model 2: {model_2_name}")
                print(f"Number of folds: {len(k_fold_eval)}")

                eval_results = []

                k = 0

                # Iterate over folds
                for k_fold in k_fold_eval:
                    print(f"Fold {k}")

                    # Get validation and ground truth data of fold
                    val_data_fold = k_fold[1]
                    ground_truths_fold = k_fold[2]

                    model_1_fold_name = f"{model_1_name}_fold_{k}"
                    model_2_fold_name = f"{model_2_name}_fold_{k}"

                    model_1 = trained_llmseqsim_models[model_1_fold_name]
                    model_2 = trained_neural_models[model_2_fold_name]

                    model_1_val_recs = model_1.predict(
                        val_data_fold, top_k=TOP_K, return_scores=False
                    )
                    model_2_val_recs = model_2.predict(val_data_fold, top_k=TOP_K)

                    eval_results.extend(
                        pop_property(
                            CUTOFF_POP,
                            model_1_fold_name,
                            model_2_fold_name,
                            model_1_val_recs,
                            model_2_val_recs,
                            val_data_fold,
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
                    k += 1

                report = format_results(eval_results)
                report.to_csv(
                    f"{RESULTS_FOLDER[dataset]}/popularity-based_hybrid_results_{property}_{model_1_name}_{model_2_name}.csv",
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
                    best_hybrid_model_1 = model_1_name
                    best_hybrid_model_2 = model_2_name

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
        # num_interactions = model_dataset.get_num_interactions()
        num_sessions = dataset_1.get_unique_sample_count()
        input_data = dataset_1.get_input_data()

        # Configure model parameters
        model_1_params = get_best_params(
            MLFLOW_CLIENT, llmseqsim_runs[best_hybrid_model_1]
        )
        model_2_params = get_best_params(MLFLOW_CLIENT, model_runs[best_hybrid_model_2])

        # Initialize and train models
        model_1 = LLMSeqSim(**model_1_params)
        model_1.train(dataset_1.get_train_data(), dataset_1.item_data)

        if neural_model == "bert":
            # Get product embeddings file location
            product_embeddings_location = f"{STARTING_FOLDER}/{dataset}/product_embeddings_{best_hybrid_embedding_model_2}.csv.gzip"
            model_2 = BERTWithEmbeddings(
                product_embeddings_location=product_embeddings_location,
                **model_2_params,
            )
            model_2.train(dataset_2.get_train_data())
        else:
            model_2 = SessionBasedCF(**model_2_params)
            model_2.train(dataset_2.get_train_data(), dataset_2.item_data)

        print(f"Predicting with best hybrid on all test prompts...")

        model_1_recs = model_1.predict(test_prompts, top_k=TOP_K, return_scores=False)
        model_2_recs = model_2.predict(test_prompts, top_k=TOP_K)

        eval_reports.extend(
            pop_property(
                [best_hybrid_property_param],
                best_hybrid_model_1,
                best_hybrid_model_2,
                model_1_recs,
                model_2_recs,
                test_prompts,
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

    format_results(eval_reports).to_csv(
        f"{RESULTS_FOLDER[dataset]}/popularity-based_hybrid_results.csv", index=False
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Finished in {elapsed_time} seconds.")
