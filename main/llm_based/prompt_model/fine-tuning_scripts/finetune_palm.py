from main.llm_based.palm.palm_model import Palm

dataset_name = ""
training_filepath = f"../../data/{dataset_name}/palm/train_cases.jsonl"
validation_filepath = f"../../data/{dataset_name}/palm/256_validation_cases.jsonl"

palm_model = Palm()
palm_model.prepare(
    training_filepath,
    validation_filepath,
)
palm_model.finetune(
    model="text-bison@001", tuned_model_name="", train_steps=200, validation_interval=10
)
