from __future__ import annotations

import logging
import os
import time

import vertexai
from google.auth import default
from google.cloud import storage
from google.cloud.aiplatform_v1 import PipelineState
from vertexai.language_models import ChatModel, TextGenerationModel
from vertexai.language_models._language_models import TuningEvaluationSpec


class Palm:
    def __init__(self, credentials: str = None):
        credentials, _ = default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        super().__init__(auth=credentials)

        self.validation_file = None

    def name(self):
        return "palm2"

    def prepare(
        self,
        dataset_path: str,
        validation_dataset_path: str = None,
        project: str = "",
        bucket_name: str = "",
    ):
        storage_client: storage.Client = storage.Client(project=project)
        llm_bucket: storage.Bucket = storage_client.bucket(bucket_name=bucket_name)

        path_components: list[str] = dataset_path.split("/")
        finetune_file_location: str = "/".join(path_components[-3:])
        self.finetune_file = f"gs://{bucket_name}/{finetune_file_location}"
        logging.info(f"Finetune file upload location: {self.finetune_file}")

        if llm_bucket.blob(finetune_file_location).exists():
            logging.warning(
                f"File {dataset_path} has already been uploaded to the cloud."
            )
        else:
            _, data = super().prepare(dataset_path)
            blob = llm_bucket.blob(finetune_file_location)
            blob.upload_from_string(data, "text/jsonl", timeout=60)
            logging.info(f"Finetune file uploaded successfully.")

        if validation_dataset_path is not None:
            path_components: list[str] = validation_dataset_path.split("/")
            validation_file_location: str = "/".join(path_components[-3:])
            self.validation_file = f"gs://{bucket_name}/{validation_file_location}"
            logging.info(f"Validation file upload location: {self.validation_file}")

            if llm_bucket.blob(validation_file_location).exists():
                logging.warning(
                    f"File {validation_dataset_path} has already been uploaded."
                )
            else:
                _, data = super().prepare(validation_dataset_path)
                blob = llm_bucket.blob(validation_file_location)
                blob.upload_from_string(data, "text/jsonl", timeout=60)
                logging.info(f"Validation file uploaded successfully.")

    def finetune(
        self,
        model: str = "text-bison@001",
        project_id: str = "",
        tuning_location: str = "",
        tuned_model_location: str = "",
        tuned_model_name: str = "",
        train_steps: int = 50,
        validation_interval: int = 10,
    ):
        """Tune a new model, based on a prompt-response data. training_data can be
        either the GCS URI of a file formatted in JSONL format (for example
        f'gs://{bucket}/{filename}.jsonl), or a pandas DataFrame. Each training example
        should be JSONL record with two keys, for example:

        {
            "input_text": <input prompt>,
            "output_text": <associated output>
        },

        or the pandas DataFame should contain two columns: ['input_text', 'output_text']
        with rows for each training example.

        Args:
            model: A string indicating what foundational model to tune.
            project_id: GCP Project ID, used to initialize vertexai location.
            tuning_location: Tuning is supported in the following locations:
                'europe-west4', 'us-central1'
            tuned_model_location: Model deployment is only supported in the following
                locations: us-central1
            tuned_model_name: A string indicating the name of the finetuned model.
            train_steps: An integer indicating the number of epochs to use when tuning
                the model.
        """
        # vertexai.init(project=project_id, location=location, credentials=self.api_key)
        vertexai.init(project=project_id, credentials=self.auth)

        if model.startswith("text"):
            self.finetuned_model: TextGenerationModel = (
                TextGenerationModel.from_pretrained(model)
            )
        else:
            self.finetuned_model: ChatModel = ChatModel.from_pretrained(model)

        # Checking whether a job exists would be nice, but it entails a job
        # endpoint. Also, the check should rely on the finetune_file and
        # this is not easy to extract from a job.
        self.finetune_job = self.finetuned_model.tune_model(
            training_data=self.finetune_file,
            # Optional:
            model_display_name=tuned_model_name,
            train_steps=train_steps,
            tuning_job_location=tuning_location,
            tuned_model_location=tuned_model_location,
            tuning_evaluation_spec=TuningEvaluationSpec(
                evaluation_data=self.validation_file,
                evaluation_interval=validation_interval,
                enable_early_stopping=True,
                enable_checkpoint_selection=True,
            ),
        )

        logging.info(f"Created job {self.finetune_job}")

        self.job_monitor()

    def predict(self, prompt: str, max_tokens: int = 7, temperature: int = 0):
        return self.finetuned_model.predict(
            prompt, max_output_tokens=max_tokens, temperature=temperature
        )

    def delete(self, finetuned_model: str):
        pass

    def job_monitor(self):
        while self.get_finetune_job_state() in [
            PipelineState.PIPELINE_STATE_QUEUED,
            PipelineState.PIPELINE_STATE_PENDING,
            PipelineState.PIPELINE_STATE_RUNNING,
        ]:
            logging.info(f"Finetuning job state: {self.get_finetune_job_state().name}.")
            time.sleep(10)

        if self.get_finetune_job_state() != PipelineState.PIPELINE_STATE_SUCCEEDED:
            raise Exception(
                f"Finetuning not succeeded. State: {self.get_finetune_job_state().name}"
            )
        else:
            logging.info(f"Finetuning completed successfully: {self.finetune_job}")

    def get_finetune_job_state(self):
        return self.finetune_job._job.state

    @staticmethod
    def find_finetuned_models(model: str):
        base_model: ChatModel = ChatModel.from_pretrained(model)
        return base_model.list_tuned_model_names()

    @staticmethod
    def get_finetuned_model(model: str, finetuned_model: str) -> ChatModel:
        palm = Palm()
        base_model: ChatModel = ChatModel.from_pretrained(model)
        palm.finetuned_model = base_model.get_tuned_model(finetuned_model)
        return palm
