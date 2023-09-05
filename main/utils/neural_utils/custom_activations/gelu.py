import tensorflow as tf


def gelu(x: tf.Tensor) -> tf.Tensor:
    """Gaussian Error Linear Unit.

    Taken from the original BERT4Rec implementation at
    https://github.com/FeiSun/BERT4Rec/blob/615eaf2004abecda487a38d5b0c72f3dcfcae5b3/modeling.py

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
        input_tensor: float Tensor to perform activation.
    Returns:
        `input_tensor` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.math.erf(x / tf.sqrt(2.0)))
    return x * cdf
