from tensorflow import keras
from keras import activations

from main.utils.neural_utils.custom_activations.gelu import gelu


__STR_TO_ACTIVATION = {
    "gelu": gelu,
}


def to_activation(act) -> tuple[str, callable]:
    if callable(act):
        return act
    elif isinstance(act, str):
        return __STR_TO_ACTIVATION[act] if act in __STR_TO_ACTIVATION else act
    else:
        raise ValueError(f"Unknown activation type. Got {type(act)}")
