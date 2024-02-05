import logging
import time

import openai
import tiktoken


class GPT:
    def __init__(self, api_key_file: str):
        with open(api_key_file) as f:
            api_key = f.readline().rstrip("\n")
        super().__init__(auth=api_key)

        self.client = openai.OpenAI(api_key=api_key)
        self.validation_file = None

    def name(self):
        return "gpt3.5"

    def prepare(
        self,
        dataset_path: str,
        cost_bound: int,
        unit_cost_1k_tokens: float,
        dataset_id: str = None,
        validation_dataset_path: str = None,
        validation_dataset_id: str = None,
    ):
        if self.file_exists(dataset_id, dataset_path) and self.file_exists(
            validation_dataset_id, validation_dataset_path
        ):
            logging.warning(
                f"File {dataset_path} and {validation_dataset_path} has already been"
                " uploaded to OpenAI."
            )
            return

        dataset_path, data = super().prepare(dataset_path)

        GPT.compute_tokens_cost(dataset_path, data, cost_bound, unit_cost_1k_tokens)

        # Use file object of prepared data
        self.finetune_file: openai.File = self.client.files.create(
            file=open(dataset_path, "rb"), purpose="fine-tune"
        )
        self.file_monitor("training")

        self.validation_file: openai.File = self.client.files.create(
            file=open(validation_dataset_path, "rb"), purpose="fine-tune"
        )
        self.file_monitor("validation")

    def finetune(self, model: str, finetuned_model: str):
        if self.job_exists():
            return

        self.finetune_job: openai.FineTuningJob = self.client.fine_tuning.jobs.create(
            training_file=self.finetune_file.id,
            model=model,
            suffix=finetuned_model,
            validation_file=self.validation_file.id,
        )
        logging.info(f"Created job {self.finetune_job}")

        self.job_monitor()
        self.finetuned_model: str = self.finetune_job.fine_tuned_model

    def predict(self, messages: list[dict], max_tokens: int = 7, temperature: int = 0):
        completion = self.client.chat.completions.create(
            model=self.finetuned_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return completion["choices"][0]["message"]["content"]

    def delete(self, finetuned_model: str):
        """Requires Owner privileges"""
        self.client.models.delete(finetuned_model)

    def file_exists(self, dataset_id, dataset_path) -> bool:
        if dataset_id is not None:
            files = self.client.files.list()
            for file in files["data"]:
                if file.id == dataset_id and file.status in ["uploaded", "processed"]:
                    self.finetune_file = file
                    logging.warning(
                        f"File {dataset_path} with id {dataset_id} has already been"
                        f" uploaded: {file}"
                    )
                    return True

        return False

    def file_monitor(self, file_to_monitor: str) -> None:
        if file_to_monitor == "training":
            file = self.finetune_file
        elif file_to_monitor == "validation":
            file = self.validation_file
        else:
            raise ValueError(
                f"The given file_to_monitor of {file_to_monitor} is not recognized."
            )

        while file.status == "pending":
            logging.info(f"File is being uploaded: {file}.")
            time.sleep(10)
            file = self.client.files.retrieve(file.id)
        if file.status not in ["uploaded", "processed"]:
            raise Exception(
                f"File upload did not succeed. File is at state {file.status_details}"
            )
        else:
            if file_to_monitor == "training":
                self.finetune_file = file
            else:
                self.validation_file = file
            logging.info(f"File upload completed successfully. Ready for fine-tuning.")

    def job_exists(self) -> bool:
        jobs = self.client.fine_tuning.jobs.list()
        for job in jobs.data:
            if job.training_file == self.finetune_file.id and job.status in [
                "validating_files",
                "queued",
                "running",
                "succeeded",
            ]:
                logging.warning(
                    f"Finetuning job for file {self.finetune_file} is in progress or"
                    " already done."
                )
                logging.warning(f"{job}")
                return True

        return False

    def job_monitor(self):
        while self.finetune_job.status in ["validating_files", "queued", "running"]:
            logging.info(f"Finetuning job state: {self.finetune_job.status}.")
            time.sleep(10)
            self.finetune_job = self.client.fine_tuning.jobs.retrieve(
                self.finetune_job.id
            )

        if self.finetune_job.status != "succeeded":
            raise Exception(f"Finetuning not succeeded: {self.finetune_job}")
        else:
            logging.info(f"Finetuning completed successfully: {self.finetune_job}")

    def find_finetuned_models(self, pattern: str):
        matched_models = []
        models = self.client.models.list()["data"]
        for model in models:
            if pattern in model["id"]:
                matched_models.append(model["id"])

        return matched_models

    def predict_for_model(
        self,
        finetune_model: str,
        messages: list,
        max_tokens: int = 7,
        temperature: int = 0,
    ):
        completion = self.client.chat.completions.create(
            model=finetune_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return completion["choices"][0]["message"]["content"]

    @staticmethod
    def compute_tokens_cost(
        dataset_path: str,
        data: str,
        cost_bound: int,
        unit_cost_1k_tokens: float,
    ):
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        n_chars = len(data)
        n_tokens = len(encoding.encode(data))

        cost_est = (n_tokens / 1000) * unit_cost_1k_tokens
        if cost_est > cost_bound:
            raise Exception(
                f"Cost estimate ({cost_est}) for finetuning is higher than cost bound"
                f" ({cost_bound})"
            )
        else:
            logging.info(
                f"Finetuning for dataset {dataset_path} "
                f"with {n_chars} characters "
                f" (tokens={n_tokens}) is expected to cost {cost_est}."
            )
