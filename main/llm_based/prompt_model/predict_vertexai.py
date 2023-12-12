import time

from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value

api_endpoint: str = "us-central1-aiplatform.googleapis.com"
client_options = {"api_endpoint": api_endpoint}
client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)


def predict_with_vertexai_model(
    endpoint_id: str,
    project: str,
    location: str,
    prompts: list[str],
    temperature: float,
    top_p: float,
    top_k: int = 40,
    stop_sequence: str = "###",
) -> list[list[str]]:
    """Returns predictions for the given prompts using the given model and parameters.
    For more information on the parameters of the model, refer to
    https://developers.generativeai.google/api/rest/generativelanguage/models/generateText.

    Args:
        endpoint_id: A string indicating the endpoint id of the model.
        project: A string indicating the GCP project that hosts the model.
        location: A string indicating the location of the model in GCP.
        prompts: A list of strings containing the prompts for each prediction.
        temperature: A float indicating the temperature parameter.
        top_p: A float indicating the top_p parameter.
        top_k: An in indicating the top_k parameter.
        stop_sequence: A string indicating the stop sequence.

    Returns:
        A list that contains a list of predictions for each prompt.
    """
    instances = [{"prompt": p} for p in prompts]
    instances = [
        json_format.ParseDict(instance_dict, Value()) for instance_dict in instances
    ]
    parameters_dict = {
        "temperature": temperature,
        # Temperature controls the degree of randomness in token selection.
        "maxOutputTokens": 1024,
        # Token limit determines the maximum amount of text output.
        "topP": top_p,
        # Tokens are selected from most probable to least until the sum of their probabilities equals the top_p value.
        "topK": top_k,
        # A top_k of 1 means the selected token is the most probable among all tokens.
        "stopSequence": stop_sequence,
        "candidateCount": 5,
    }
    parameters = json_format.ParseDict(parameters_dict, Value())
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    all_predictions = []
    for i in instances:
        while True:
            try:
                response = client.predict(
                    endpoint=endpoint, instances=[i], parameters=parameters
                )
                predictions = [
                    remove_stop_sequence(dict(p)["content"], stop_sequence).strip()
                    for p in response.predictions
                ]
                break
            except Exception as e:
                print(f"Got an error {e}, trying again in 10 seconds...")
                time.sleep(10)
        all_predictions.append(predictions)
    return all_predictions


def remove_stop_sequence(pred: str, stop_seq: str) -> str:
    if pred.endswith(stop_seq):
        return pred.removesuffix(stop_seq)
    else:
        return pred
