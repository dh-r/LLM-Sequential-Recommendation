The four folders `genitem`, `genlist`, `classify`, `rank` correspond to respectively section 4.1-4.4 in the paper. For each variant use the following steps:
1. Generate prompts for finetuning using `generate_finetune_prompts.ipynb`. This will generate 2 files: `train_cases.jsonl` and `validation_cases.jsonl`.
2. The training and validation files are input for the finetuning scripts. There is a script to fine-tune a GPT model or a PaLM model. Make sure to update OpenAI API keys or Google Credentials to make these scripts work. Consult the online_material.md for exact parameters.
3. Execute the `predict_gpt.ipnyb` or `predict_palm.ipynb` to run predictions on the fine-tuned GPT or PaLM model. These notebooks will result in 'recommendation pickles' which can be evaluated using the evaluation scripts. 


The notebooks have dependencies on the specified dataset (Steam or Beauty), embeddings and auxiliary lookup tables (e.g. product names).  