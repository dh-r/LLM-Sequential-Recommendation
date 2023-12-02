The following commands were used to fine-tune a model using the tutorial on the OpenAI 
docs https://platform.openai.com/docs/guides/fine-tuning/advanced-usage 
(commands are up-to-date as of 22 Jun 2023)

OpenAI requires its training and validation data to be in JSONL format. These files are created by `create_train_jsonl.ipynb`, which converts sessions into the format specified in the paper. We reserved 500 sessions to be the validation cases, which we use to evaluate whether a model has reasonably converged. For examples, refer to the `train_cases.jsonl` and `validation_cases.jsonl` files we placed in the `beauty/finetuning` directory. 

The command below creates a new model based on Ada (best price/performance ratio, according to OpenAI) and 
trains it for a given number of epochs.

`openai api fine_tunes.create -t train_cases.jsonl -v validation_cases.jsonl -m ada --suffix <YOUR_SUFFIX> --n_epochs <NUM_EPOCHS>`

Once finished fine-tuning, you can download the validation results with:

`openai api fine_tunes.results -i <YOUR_FINE_TUNE_JOB_ID> > results_<YOUR SUFFIX>.csv`

Add this result file to check_fine_tune_results.ipynb (in `beauty/finetuning`) to visualize the validation loss of the model. 

If it has not sufficiently converged, you can continue training the model again with the following command, and check the validation loss again. 
Make sure that you set the -m flag here to continue rather than start a new fine-tune.

`openai api fine_tunes.create -t train_cases.jsonl -v validation_cases.jsonl -m <FINE_TUNE_MODEL_NAME> --n_epochs <NUM_EPOCHS>`