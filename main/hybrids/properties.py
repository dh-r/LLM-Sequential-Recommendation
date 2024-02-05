import numpy as np
import pandas as pd

from main.hybrids.utils import *
from main.reranking.rerank_pairwise_similarity import (
    PairwiseSimilarityFilterReranker,
)
from sklearn.metrics.pairwise import cosine_similarity


def rank_property(
    cutoff_ranks,
    model_1,
    model_2,
    model_1_recs,
    model_2_recs,
    test_prompts,
    ground_truths,
    num_items,
    item_count,
    top_k,
    top_k_eval,
    cores,
):
    """Combine recommendations of two models based on the rank property."""
    comb_method = "Rank"
    eval_reports = []

    for cutoff_rank in cutoff_ranks:
        print(f"Cutoff rank: {cutoff_rank}")
        preds = {
            session_id: np.concatenate(
                [model_1_recs[session_id][:cutoff_rank], model_2_recs[session_id]]
            )
            for session_id in test_prompts.keys()
        }
        preds = remove_duplicates(preds)
        preds = truncate_to_top_k(preds, top_k)

        model_name = f"{model_1}-{model_2}-{comb_method}-{cutoff_rank}"
        eval_reports.extend(
            evaluate_results(
                top_k_eval,
                preds,
                ground_truths,
                num_items,
                item_count,
                model_name,
                cores,
            )
        )
    return eval_reports


def conf_property(
    cutoff_confs,
    model_1,
    model_2,
    model_1_recs,
    model_2_recs,
    model_1_scores,
    test_prompts,
    ground_truths,
    num_items,
    item_count,
    top_k,
    top_k_eval,
    cores,
):
    """Combine recommendations of two models based on the confidence property."""
    comb_method = "Conf"
    eval_reports = []

    cutoff_ranks = []

    def find_cutoff_rank(s_id, cutoff_conf):
        s_scores = model_1_scores[s_id]
        for i in range(len(s_scores)):
            if s_scores[i] < cutoff_conf:
                cutoff_ranks.append(i)
                return i

        cutoff_ranks.append(len(s_scores))
        return len(s_scores)

    for cutoff_conf in cutoff_confs:
        cutoff_ranks = []

        preds = {}
        for session_id in test_prompts.keys():
            preds[session_id] = np.concatenate(
                [
                    model_1_recs[session_id][
                        : find_cutoff_rank(session_id, cutoff_conf)
                    ],
                    model_2_recs[session_id],
                ]
            )
        preds = remove_duplicates(preds)
        preds = truncate_to_top_k(preds, top_k)

        print(
            f"Description of cutoff ranks with conf {cutoff_conf}:\n {pd.DataFrame(np.array(cutoff_ranks)).describe()}"
        )

        model_name = f"{model_1}-{model_2}-{comb_method}-{cutoff_conf}"
        eval_reports.extend(
            evaluate_results(
                top_k_eval,
                preds,
                ground_truths,
                num_items,
                item_count,
                model_name,
                cores,
            )
        )
    return eval_reports


def pop_property(
    cutoff_pops,
    model_1,
    model_2,
    model_1_recs,
    model_2_recs,
    test_prompts,
    ground_truths,
    num_items,
    item_count,
    num_sessions,
    input_data,
    top_k,
    top_k_eval,
    cores,
):
    """Combine recommendations of two models based on the popularity property."""
    comb_method = "Pop"
    eval_reports = []

    popularities = {}
    # Compute item popularities
    for itemid in input_data["ItemId"].unique():
        # Based on frequency in sessions
        popularities[itemid] = (
            input_data[input_data["ItemId"] == itemid]["SessionId"].nunique()
            / num_sessions
        )
        # Based on frequency in interactions
        # popularities[itemid] = input_data[input_data["ItemId"] == itemid]["SessionId"].count() / num_interactions

    popularities_df = pd.DataFrame.from_dict(
        popularities, orient="index", columns=["Popularity"]
    )

    # Define popularity thresholds based on quantiles
    pop_quantiles = {}
    for quantile in cutoff_pops:
        pop_quantiles[quantile] = popularities_df["Popularity"].quantile(quantile)

    # Identify embedding and neural models
    if "LSS" in model_1:
        embedding_model_recs = model_1_recs
        neural_model_recs = model_2_recs
        print(f"Embedding model: {model_1}")
        print(f"Neural model: {model_2}")
    elif "LSS" in model_2:
        embedding_model_recs = model_2_recs
        neural_model_recs = model_1_recs
        print(f"Embedding model: {model_2}")
        print(f"Neural model: {model_1}")
    else:
        raise Exception(
            f"Could not identify the embedding and neural models in [{model_1}, {model_2}]"
        )

    # Iterate over popularity thresholds
    for quantile, cutoff_pop in pop_quantiles.items():
        print(f"Cutoff quantile: {quantile}")
        print(f"Cutoff popularity: {cutoff_pop}")
        preds = {}
        # Compute predictions for each test prompt:
        # If last item of test prompt is popular, use neural model for prediction; else use embedding model.
        for session_id, session in test_prompts.items():
            session_last_itemid = session[-1]
            if popularities[session_last_itemid] >= cutoff_pop:
                preds[session_id] = neural_model_recs[session_id]
            else:
                preds[session_id] = embedding_model_recs[session_id]

        preds = remove_duplicates(preds)
        preds = truncate_to_top_k(preds, top_k)

        model_name = f"{model_1}-{model_2}-{comb_method}-{quantile}"
        eval_reports.extend(
            evaluate_results(
                top_k_eval,
                preds,
                ground_truths,
                num_items,
                item_count,
                model_name,
                cores,
            )
        )
    return eval_reports


def pop_property_single_model(
    cutoff_pops,
    model,
    model_recs,
    ground_truths,
    num_items,
    item_count,
    num_sessions,
    input_data,
    top_k,
    top_k_eval,
    cores,
):
    """Filter a model's recommendations based on the popularity property."""
    comb_method = "Pop"
    eval_reports = []

    popularities = {}
    # Compute item popularities
    for itemid in input_data["ItemId"].unique():
        # Based on frequency in sessions
        popularities[itemid] = (
            input_data[input_data["ItemId"] == itemid]["SessionId"].nunique()
            / num_sessions
        )
        # Based on frequency in interactions
        # popularities[itemid] = input_data[input_data["ItemId"] == itemid]["SessionId"].count() / num_interactions

    popularities_df = pd.DataFrame.from_dict(
        popularities, orient="index", columns=["Popularity"]
    )

    # Define popularity thresholds based on quantiles
    pop_quantiles = {}
    for quantile in cutoff_pops:
        pop_quantiles[quantile] = popularities_df["Popularity"].quantile(quantile)

    # Iterate over popularity thresholds
    for quantile, cutoff_pop in pop_quantiles.items():
        print(f"Cutoff quantile: {quantile}")
        print(f"Cutoff popularity: {cutoff_pop}")

        preds = remove_duplicates(model_recs)

        for session_id, session_preds in preds.items():
            itemids_low_quality = []

            for itemid in session_preds:
                # If item is not popular add to list of low-quality recommended items
                if popularities[itemid] < cutoff_pop:
                    itemids_low_quality.append(itemid)

            for itemid in itemids_low_quality:
                # Remove low-quality items as long as there are enough recommendations
                # Note: Here we may want to keep a dictionary of quality scores and remove items based on their quality ranking
                if len(preds[session_id]) > top_k:
                    preds[session_id] = np.delete(
                        preds[session_id], np.where(preds[session_id] == itemid)
                    )

        preds = truncate_to_top_k(preds, top_k)

        model_name = f"{model}-None-{comb_method}-{quantile}"
        eval_reports.extend(
            evaluate_results(
                top_k_eval,
                preds,
                ground_truths,
                num_items,
                item_count,
                model_name,
                cores,
            )
        )
    return eval_reports


def divers_property(
    cutoff_sims,
    model,
    model_recs,
    ground_truths,
    num_items,
    item_count,
    item_data,
    top_k,
    top_k_eval,
    cores,
):
    """Filter a model's recommendations based on the diversity property: exclude recommendations
    whose similarity score to at least one previously predicted item exceeds a specified threshold.
    """
    comb_method = "Divers"
    eval_reports = []

    # Get item embeddings
    item_embeddings = np.stack(item_data["embedding"].values)

    # Compute item embedding distances
    similarity_matrix = cosine_similarity(X=item_embeddings, Y=None, dense_output=False)

    # Define similarity thresholds based on quantiles;
    # exclude the diagonal from the quantile computation because it's always 1 and adds noise
    similarity_matrix_no_diag = similarity_matrix[
        ~np.eye(similarity_matrix.shape[0], dtype=bool)
    ].reshape(similarity_matrix.shape[0], -1)
    quantiles = np.quantile(similarity_matrix_no_diag, q=cutoff_sims)

    sim_quantiles = {}
    for i in range(len(cutoff_sims)):
        sim_quantiles[cutoff_sims[i]] = quantiles[i]

    preds = remove_duplicates(model_recs)

    # Get mappings of dataset-specific item ids to/from global item ids
    item_dataset_to_global, item_global_to_dataset = get_item_lookups(item_data)

    # Iterate over similarity thresholds
    for quantile, cutoff_sim in sim_quantiles.items():
        print(f"Cutoff quantile: {quantile}")
        print(f"Cutoff similarity: {cutoff_sim}")

        reranker = PairwiseSimilarityFilterReranker(
            similarity_matrix, similarity_threshold=cutoff_sim, cores=cores
        )

        reranked_preds = {}
        for session_id, session_preds in preds.items():
            # Normalize predicted item ids
            session_preds_global = map_to_global(session_preds, item_dataset_to_global)

            # Rerank predictions using the diversity reranker and limit to top-k items
            reranked_session_preds, _ = reranker._rerank_single(
                session_id, session_preds_global, top_k
            )

            # Denormalize reranked item ids
            reranked_preds[session_id] = map_from_global(
                reranked_session_preds, item_global_to_dataset
            )

        model_name = f"{model}-None-{comb_method}-{quantile}"
        eval_reports.extend(
            evaluate_results(
                top_k_eval,
                reranked_preds,
                ground_truths,
                num_items,
                item_count,
                model_name,
                cores,
            )
        )
    return eval_reports


def pop_divers_property(
    cutoff_pops,
    cutoff_sims,
    model,
    model_recs,
    ground_truths,
    num_items,
    item_count,
    num_sessions,
    input_data,
    item_data,
    top_k,
    top_k_eval,
    cores,
):
    """Filter a model's recommendations based on the popularity and diversity properties:
    first apply a popularity threshold to pick popular recommendations, and then apply a
    diversity threshold to diversify them."""
    comb_method = "Pop_Divers"
    eval_reports = []

    popularities = {}
    # Compute item popularities
    for itemid in input_data["ItemId"].unique():
        # Based on frequency in sessions
        popularities[itemid] = (
            input_data[input_data["ItemId"] == itemid]["SessionId"].nunique()
            / num_sessions
        )
        # Based on frequency in interactions
        # popularities[itemid] = input_data[input_data["ItemId"] == itemid]["SessionId"].count() / num_interactions

    popularities_df = pd.DataFrame.from_dict(
        popularities, orient="index", columns=["Popularity"]
    )

    # Define popularity thresholds based on quantiles
    pop_quantiles = {}
    for quantile in cutoff_pops:
        pop_quantiles[quantile] = popularities_df["Popularity"].quantile(quantile)

    # Get item embeddings
    item_embeddings = np.stack(item_data["embedding"].values)

    # Compute item embedding distances
    similarity_matrix = cosine_similarity(X=item_embeddings, Y=None, dense_output=False)

    # Define similarity thresholds based on quantiles;
    # exclude the diagonal from the quantile computation because it's always 1 and adds noise
    similarity_matrix_no_diag = similarity_matrix[
        ~np.eye(similarity_matrix.shape[0], dtype=bool)
    ].reshape(similarity_matrix.shape[0], -1)
    quantiles = np.quantile(similarity_matrix_no_diag, q=cutoff_sims)

    sim_quantiles = {}
    for i in range(len(cutoff_sims)):
        sim_quantiles[cutoff_sims[i]] = quantiles[i]

    # Get mappings of dataset-specific item ids to/from global item ids
    item_dataset_to_global, item_global_to_dataset = get_item_lookups(item_data)

    # Iterate over popularity thresholds
    for quantile_pop, cutoff_pop in pop_quantiles.items():
        print(f"Cutoff pop quantile: {quantile_pop}")
        print(f"Cutoff popularity: {cutoff_pop}")

        preds = remove_duplicates(model_recs)

        for session_id, session_preds in preds.items():
            itemids_low_quality = []

            for itemid in session_preds:
                # If item is not popular add to list of low-quality recommended items
                if popularities[itemid] < cutoff_pop:
                    itemids_low_quality.append(itemid)

            for itemid in itemids_low_quality:
                # Remove low-quality items as long as there are enough recommendations
                # Note: Here we may want to keep a dictionary of quality scores and remove items based on their quality ranking
                if len(preds[session_id]) > top_k:
                    preds[session_id] = np.delete(
                        preds[session_id], np.where(preds[session_id] == itemid)
                    )

        # Iterate over similarity thresholds
        for quantile_sim, cutoff_sim in sim_quantiles.items():
            print(f"Cutoff sim quantile: {quantile_sim}")
            print(f"Cutoff similarity: {cutoff_sim}")

            reranker = PairwiseSimilarityFilterReranker(
                similarity_matrix, similarity_threshold=cutoff_sim, cores=cores
            )

            reranked_preds = {}
            for session_id, session_preds in preds.items():
                # Normalize predicted item ids
                session_preds_global = map_to_global(
                    session_preds, item_dataset_to_global
                )

                # Rerank predictions using the diversity reranker and limit to top-k items
                reranked_session_preds, _ = reranker._rerank_single(
                    session_id, session_preds_global, top_k
                )

                # Denormalize reranked item ids
                reranked_preds[session_id] = map_from_global(
                    reranked_session_preds, item_global_to_dataset
                )

            model_name = f"{model}-None-{comb_method}-{quantile_pop}_{quantile_sim}"
            eval_reports.extend(
                evaluate_results(
                    top_k_eval,
                    reranked_preds,
                    ground_truths,
                    num_items,
                    item_count,
                    model_name,
                    cores,
                )
            )
    return eval_reports
