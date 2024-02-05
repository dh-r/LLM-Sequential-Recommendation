import argparse

from main.llm_based.gpt.gpt_model import GPT

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cost_bound", type=int, default=50)
parser.add_argument("-n", "--name", type=str, default="")
parser.add_argument("dataset_name")
args = parser.parse_args()

dataset_name = args.dataset_name  # e.g., "beauty_rank_next_item"
training_file = f"../../data/{dataset_name}/gpt/train_cases.jsonl"
validation_file = f"../../data/{dataset_name}/gpt/validation_cases.jsonl"

if args.name != "":
    model_name = args.name
else:
    model_name = dataset_name

gpt_model = GPT("../../openai_api_key.txt")
gpt_model.prepare(
    training_file,
    cost_bound=args.cost_bound,
    unit_cost_1k_tokens=0.008,
    validation_dataset_path=validation_file,
)
gpt_model.finetune(model="gpt-3.5-turbo-1106", finetuned_model=model_name)
