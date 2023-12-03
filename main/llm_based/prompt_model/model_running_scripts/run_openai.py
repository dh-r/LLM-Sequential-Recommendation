import argparse
import json
import pickle
import random
import time
from collections import Counter

import openai

from main.data.session_dataset import *
from main.llm_based.embedding_utils import openai_utils
from main.llm_based.prompt_model.create_prompt import (
    create_prompt_completion_from_session,
)
from main.utils.top_k_computer import TopKComputer
from paths import ROOT_DIR

parser = argparse.ArgumentParser(description="Predict using a finetuned model.")
parser.add_argument("--working-dir", dest="working_dir")
parser.add_argument(
    "--model-name",
    dest="model_name",
)
parser.add_argument(
    "--embeddings-name",
    dest="embeddings_name",
)
parser.add_argument(
    "--top-k",
    dest="top_k",
    type=int,
)
parser.add_argument(
    "--temperature",
    type=float,
)
parser.add_argument(
    "--top-p",
    dest="top_p",
    type=float,
)
args = parser.parse_args()

WORKING_DIR = f"{ROOT_DIR}/{args.working_dir}"
MODEL_NAME = args.model_name
EMBEDDINGS_NAME = args.embeddings_name
TOP_K = args.top_k
TEMPERATURE = args.temperature
TOP_P = args.top_p

print(f"Configuration:\n{args}")

total_model_name = f"{MODEL_NAME}_temp_{TEMPERATURE}_top_p_{TOP_P}"

product_embeddings = pd.read_csv(
    f"{WORKING_DIR}/{EMBEDDINGS_NAME}.csv.gz", compression="gzip"
)
product_id_to_name = (
    product_embeddings[["global_product_id", "name"]]
    .set_index("global_product_id")
    .to_dict()["name"]
)
product_name_to_id = (
    product_embeddings[["global_product_id", "name"]]
    .set_index("name")
    .to_dict()["global_product_id"]
)
product_index_to_embedding = (
    product_embeddings[["global_product_id", "ada_embedding"]]
    .set_index("global_product_id")
    .to_dict()["ada_embedding"]
)
product_index_to_embedding = {
    k: np.array(json.loads(v)) for k, v in product_index_to_embedding.items()
}
product_index_to_embedding = np.array(list(product_index_to_embedding.values()))
product_index_to_id = list(product_id_to_name.keys())

dataset = SessionDataset.from_pickle(f"{WORKING_DIR}/dataset.pickle")
test = dataset.get_test_prompts()
test_keys = list(test.keys())

print(f"Will be computing for {len(test_keys)} sessions")
# Loop through sessions to get:
#   1. The value counts of recommended items.
#   2. The recommended item names that need to be re-embedded to get existing products
#       from the catalog.
session_id_to_prompts = {}
for i, session_id in enumerate(test_keys):
    test_session = test[session_id]

    test_session_used_in_prompt = test_session.copy()

    if i % 1 == 0:
        print(f"Now at {i}", end="\r")

    # Create prompt.
    prompt, _ = create_prompt_completion_from_session(
        test_session_used_in_prompt, product_id_to_name, 0
    )

    if (
        len(prompt) > 5000
    ):  # Ada has a maximum token length of 2049, so around 6-9k chars.
        # We need to truncate the prompt.
        # For now, we just take the last 20 items.
        # It's too slow to process on a case-by-case basis.
        print(
            f"Have to truncate with len(prompt): {len(prompt)} and len(session): {len(test_session_used_in_prompt)}"
        )
        test_session_used_in_prompt = test_session_used_in_prompt[-20:]
        prompt, _ = create_prompt_completion_from_session(
            test_session_used_in_prompt, product_id_to_name, 0
        )

    session_id_to_prompts[session_id] = prompt

recs_filename = f"{total_model_name}_recs.pickle"

# Maps session_id to -> dict(item_name -> number of occurrences in recommendation slate.)
session_id_to_value_counts = {}

try:
    with open(f"{WORKING_DIR}/{recs_filename}", "rb") as f:
        session_id_to_value_counts = pickle.loads(f.read())
except:
    # Create batches of the prompts.
    step_size = 20
    cur_step = 0

    session_id_and_prompts = list(session_id_to_prompts.items())

    # Create embedding for each item.
    # These embeddings are automatically saved in the cache, so that they
    # can be used in the following code immediately.
    while cur_step < len(session_id_to_prompts):
        print(f"Currently at {cur_step} of {len(session_id_to_prompts)}")

        cur_session_id_and_prompts = session_id_and_prompts[
            cur_step : cur_step + step_size
        ]

        cur_prompts = [
            session_id_and_prompt[1]
            for session_id_and_prompt in cur_session_id_and_prompts
        ]
        cur_sessions = [
            session_id_and_prompt[0]
            for session_id_and_prompt in cur_session_id_and_prompts
        ]

        # Call the API to complete the prompts.
        def call_openai():
            return openai.Completion.create(
                model=MODEL_NAME,
                prompt=cur_prompts,
                max_tokens=50,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                n=TOP_K,
                stop="###",
            )["choices"]

        try:
            choices = call_openai()
        except Exception as e:
            print(
                f"Failed call to openAI with exception {e}, trying again in 20 seconds.."
            )
            time.sleep(40)

            choices = call_openai()

        # Get the recommended item names.
        predicted_item_names = [choice["text"].strip() for choice in choices]

        # Reshape batched predicted item names.
        predicted_item_names = np.reshape(predicted_item_names, (-1, TOP_K))

        for session_id, session_predicted_item_names in zip(
            cur_sessions, predicted_item_names
        ):
            # Get the value counts for each recommended item name,
            # since the API might return duplicate items.
            value_counts = dict(Counter(session_predicted_item_names))

            session_id_to_value_counts[session_id] = value_counts

        cur_step += step_size

    with open(f"{WORKING_DIR}/{recs_filename}", "wb") as f:
        pickle.dump(session_id_to_value_counts, f)

to_embed = set()
num_unknown_recommendations = 0
num_known_recommendations = 0
num_duplicate_recommendations = 0
num_total_recommendations = 0
for session_id, value_counts in session_id_to_value_counts.items():
    for item_name in value_counts.keys():
        # If an item is not in the catalog, we need to embed it.
        if not (openai_utils.in_cache(item_name)):
            to_embed.add(item_name)
        num_total_recommendations += value_counts[item_name]
        if item_name not in product_name_to_id:
            num_unknown_recommendations += value_counts[item_name]
        else:
            num_known_recommendations += value_counts[item_name]

        if value_counts[item_name] > 1:
            num_duplicate_recommendations += value_counts[item_name]

to_embed = list(to_embed)
print(f"Must compute {len(to_embed)} new embeddings")
to_embed = pd.DataFrame(to_embed, columns=["name"])
to_embed = to_embed[to_embed["name"] != ""]

# Create batches of the items that we need to embed.
step_size = 1000
batch_product_lookup = to_embed.iloc[0:step_size]
cur_step = step_size
processed_batches = []
num_batches_processed = 0

# Create embedding for each item.
# These embeddings are automatically saved in the cache, so that they
# can be used in the following code immediately.
while not (batch_product_lookup.empty):
    print(f"Currently at {cur_step} of {len(to_embed)}")
    openai_utils.set_embeddings_from_df(batch_product_lookup)
    processed_batches.append(batch_product_lookup)

    batch_product_lookup = to_embed.iloc[cur_step : cur_step + step_size]
    cur_step += step_size

# Now that we have embeddings for everything we need, we can finalize the
# recommendations.
recommendations = {}
bug_item_list = []
num_sessions_done = 0
for session_id, value_counts in session_id_to_value_counts.items():
    session_item_names = [product_id_to_name[item] for item in test[session_id]]
    session_recommendations = []

    duplicate_replacements = []
    for item_name, count in value_counts.items():
        # If an item occurs more than once, we need its embedding to find
        # neighbouring items.
        # If an item is not in the catalog, we get a similar item that is in the catalog.
        if count > 1 or item_name not in product_name_to_id:
            # Assert that the item is in the cache, otherwise we would
            # retrieve these embeddings from openAI again, which is slow and expensive.
            if not openai_utils.in_cache(item_name):
                # This always happens when item_name is an empty string, so we just
                # create a zero embedding.
                item_embedding = np.zeros((1, 1024 + 512))
            else:
                # Get item similarity using embedding
                item_embedding = openai_utils.embedding_from_string(item_name)
                if isinstance(item_embedding, str):
                    item_embedding = json.loads(item_embedding)

                item_embedding = np.array([item_embedding], dtype=np.float64)
            predictions = (product_index_to_embedding @ item_embedding.T).T

            # Get neighbouring item(s), and extend the recommendations for this
            # session with the neighbouring item(s).
            top_k_item_ids_indices = TopKComputer.compute_top_k(
                predictions, top_k=count + TOP_K
            )[0]
            top_k_item_ids = [
                product_index_to_id[item_index] for item_index in top_k_item_ids_indices
            ]

            # Get names of the items that are not allowed to be added.
            already_recommended_names = [
                product_id_to_name[item]
                for item in session_recommendations + duplicate_replacements
            ]
            upcoming_recommendations = value_counts
            disallowed_items = (
                already_recommended_names
                + list(upcoming_recommendations.keys())
                + session_item_names
            )

            # Filter out disallowed items.
            top_k_item_ids = [
                item
                for item in top_k_item_ids
                if product_id_to_name[item] not in disallowed_items
            ]

            # We add the item itself if it exists.
            item_exists: bool = item_name in product_name_to_id
            if item_exists:
                item_id = product_name_to_id[item_name]
                session_recommendations.append(item_id)

            # Truncate.
            # If an item appeared `count` times, it needs `count - int(item_exists)` replacements.
            # If the item exists, we have added it already, so we only need count - 1 replacements.
            # If the item does not exist, we need count replacements.
            top_k_item_ids = top_k_item_ids[: count - int(item_exists)]

            duplicate_replacements.extend(top_k_item_ids)

        else:
            # Simply add the id to the list of recommendations
            item_id = product_name_to_id[item_name]
            session_recommendations.append(item_id)

    session_recommendations.extend(duplicate_replacements)

    num_sessions_done += 1
    if random.randint(0, 100) == 50:
        print(f"Num sessions done: {num_sessions_done}", end="\r")

    recommendations.update({session_id: session_recommendations})

predictions_pickle: bytes = pickle.dumps(recommendations)

filename = f"recs_openai_{total_model_name}"

with open(f"{WORKING_DIR}/{total_model_name}_statistics.txt", "w") as f:
    f.write(
        f"num_unknown = {num_unknown_recommendations}"
        f"\nnum_known = {num_known_recommendations}"
        f"\nnum_duplicate = {num_duplicate_recommendations}"
        f"\nnum_total = {num_total_recommendations}"
    )

with open(f"{WORKING_DIR}/{filename}.pickle", "wb") as file:
    file.write(predictions_pickle)
