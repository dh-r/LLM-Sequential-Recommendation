import numpy as np
import tensorflow as tf
from tensorflow import keras

from keras import backend as K
from keras import layers
from keras.regularizers import L2
from keras.losses import MeanSquaredError, SparseCategoricalCrossentropy

from typing import Tuple
import math

from main.utils.top_k_computer import TopKComputer
from main.data.side_information import SideInformation


class SideEncoder(keras.Model):
    def __init__(
        self,
        side_information: SideInformation,
        encoder_dimension: int,
        l2: float = 0,
        optimizer_kwargs: dict = {},
        is_verbose: bool = False,
    ) -> None:
        """A plug-and-play encoder of a general feature vector, supporting
        categorical, real and integer features.

        Setup.
        The goal of the side-encoder is to embed a general feature matrix to
        be used in other models. It assumes that this matrix is structured in such a
        way that the each column represents a single feature and that each row
        represents the object (user / item / session). In addition, the non-categorical
        features should be placed before the categorical features. The non-categorical
        features are assumed to be real or integer valued.

        Network architecture.
        We first embed the categorical features individually. These embeddings are
        concatenated together with the non-categorical features. For example, the
        feature vector [cont1, cont2, cat_1, cat_2] is transformed into
        [cont1, cont2, cat1_emb_1, .. , cat1_emb_n, cat2_emb_1, .. ,cat2_emb_n] where n
        is the cat_embedding size. This approach is inspired by the
        [Google Wide & Deep paper](https://arxiv.org/pdf/1606.07792.pdf).

        We then process this vector through hidden layers of
        the encoder, where the last hidden layer has an output dimension equal to
        encoder_dimension. This output is the encoding that can be used by other models.

        To facilitate pre-training the model, we attach a decoder network after the
        encoder, which is simply tasked with predicting the original input from the
        encodings. The last layer in the decoder must have a dimension equal to
        num_non_cat + sum(cat_sizes), so that we have an output for each
        non-categorical feature, and an output for each category in the categorical
        features. We use the MSE loss on the non-categorical features, whereas
        we use categorical cross-entropy on the categorical features. This way,
        the network pre-learns to efficiently encode a feature vector without
        needing any feedback from historical interactions like usual recommender models.

        Args:
            side_information (SideInformation): The side-information to embed.
            encoder_layer_dimensions (list[int]): The dimensions of the hidden layers
                in the encoder network.
            encoder_dimension (int): The dimension of the encoding.
            decoder_layer_dimensions (list[int]): The dimensions of the hidden layers
                in the decoder network.
            cat_embedding_size (int, optional): The embedding size used
                to embed the categorical features before these are fed to the encoding
                network. Defaults to 8.
            l2 (float, optional): The L2 regularization on the network. Defaults to 0.
            optimizer_kwargs (dict, optional): Optional keywords to initialize the Adam
                optimizer with.
            is_verbose (bool, optional): Debug and progress information.
                Defaults to False.
        """
        super().__init__()

        # Save model parameters.
        self.side_information = side_information
        self.num_non_cat = side_information["num_non_categorical_features"]
        self.num_cat = side_information["num_categorical_features"]
        self.cat_sizes = side_information["category_sizes"]

        self.is_verbose = is_verbose

        # Create category embedders.
        self.cat_embedders = []
        total_embedding_size = 0
        for cat_size in self.cat_sizes:
            cat_embedding_size = math.ceil(math.log(cat_size, 2))
            total_embedding_size += cat_embedding_size
            self.cat_embedders.append(
                layers.Embedding(
                    cat_size, cat_embedding_size, embeddings_regularizer=L2(l2)
                )
            )

        # Compute architecture.
        input_size_after_embeddings = self.num_non_cat + total_embedding_size
        (
            encoder_layer_dimensions,
            decoder_layer_dimensions,
        ) = SideEncoder.get_hidden_architecture(
            input_size_after_embeddings, encoder_dimension
        )

        # Create encoder.
        self.encoder = keras.Sequential()

        # Create hidden layers of encoder.
        for dimension in encoder_layer_dimensions:
            self.encoder.add(
                layers.Dense(
                    dimension, kernel_regularizer=L2(l2), bias_regularizer=L2(l2)
                )
            )

        # Create output layer of encoder.
        self.encoder.add(
            layers.Dense(
                encoder_dimension, kernel_regularizer=L2(l2), bias_regularizer=L2(l2)
            )
        )

        # Create decoder.
        self.decoder = keras.Sequential()

        # Create hidden layers of decoder.
        for dimension in decoder_layer_dimensions:
            self.decoder.add(
                layers.Dense(
                    dimension, kernel_regularizer=L2(l2), bias_regularizer=L2(l2)
                )
            )

        # Create output layer of decoder.
        final_layer_size = self.num_non_cat + sum(self.cat_sizes)
        self.decoder.add(
            layers.Dense(
                final_layer_size, kernel_regularizer=L2(l2), bias_regularizer=L2(l2)
            )
        )

        # Define losses.
        # See the side_encoder_loss method for more documentation on these losses.
        self.mse_loss = MeanSquaredError()
        self.cat_loss_funcs = []
        for _ in self.cat_sizes:
            self.cat_loss_funcs.append(SparseCategoricalCrossentropy(from_logits=True))

        # Compile with Adam and sigmoid binary crossentropy loss.
        adam = keras.optimizers.Adam(**optimizer_kwargs)
        self.compile(optimizer=adam, loss=SideEncoder.side_encoder_loss(self))

    def call(
        self, input: tf.Tensor, training: bool = None, decode: bool = True
    ) -> tf.Tensor:
        """The call method defines how the input tensor is propagated through the model
        to form the output tensor.

        Args:
            inputs (tf.Tensor): An input tensor. The columns of the tensor should
                correspond to the features, in an ordering where the
                non-categorical features come first, followed by integers representing
                the category for each of the categorical features. These categories
                should come in the exact same ordering as cat_sizes during the
                initialization of SideEncoder.
            training (bool): Whether the call is made for training or prediction.
            decode (bool): Whether to feed the encoding to the decoding network to get
                an estimate reconstruction of the original input.

        Returns:
            tf.Tensor: The encoding of the features, or the decodings if decode is
                True.
        """
        encoding = self.get_encodings_tensor(input)

        if decode:
            # Decode.
            return self.decoder(encoding)
        else:
            return encoding

    def pretrain(self, num_epochs: int = 3, batch_size: int = 512):
        """Pretrain the network using an auto-encoder approach.

        Args:
            num_epochs (int): The number of epochs to train for. Defaults to 3.
            batch_size (int): The batch size to be used during training.
        """
        features = self.side_information["features"]
        features_tensor = tf.convert_to_tensor(features, dtype=tf.float32)
        self.fit(
            features_tensor,
            features_tensor,
            batch_size=batch_size,
            epochs=num_epochs,
            verbose=2 if self.is_verbose else 0,
        )

    def get_encodings(self, features: np.ndarray) -> np.ndarray:
        """Get the encodings for the features.

        Args:
            features (np.ndarray): The features. The columns should be
                ordered as specified in the call method of SideEncoder.

        Returns:
            np.ndarray: The encodings for each row in the features input.
        """
        features_tensor = tf.convert_to_tensor(features, dtype=tf.float32)
        return self.get_encodings_tensor(features_tensor).numpy()

    def get_decodings(self, features: np.ndarray) -> np.ndarray:
        """Get the decodings predicted by the auto-encoder.

        Note that we must do some post-processing after prediction, since the network
        predicts a logit value for each category for every categorical feature.
        This method simply takes the highest logit value and returns that as the
        predicted category for each feature.

        Args:
            features (np.ndarray): The features. The columns should be
                ordered as specified in the call method of SideEncoder.

        Returns:
            np.ndarray: A reconstruction of the features.
        """
        features_tensor = tf.convert_to_tensor(features, dtype=tf.float32)
        pred = self.predict(features_tensor)

        # Split the array into non-categorical and categorical.
        # Note that this approach is different when using TensorFlow like in the loss,
        # because numpy expects indices to split on, instead of sizes like TensorFlow.
        sizes_pred = [self.num_non_cat, *self.cat_sizes]
        indices_pred = [sum(sizes_pred[: (i + 1)]) for i in range(len(sizes_pred) - 1)]
        pred_split = np.split(pred, indices_pred, axis=1)

        # Get the non-categorical predictions.
        non_cat_result = pred_split[0]

        # Get max item for each category
        cat_results = []
        for pred_cat in pred_split[1:]:
            top_cat = TopKComputer.compute_top_k(pred_cat, 1)
            cat_results.append(top_cat)

        # Concatenate and return results
        return np.concatenate([non_cat_result, *cat_results], axis=1)

    def get_encodings_tensor(self, input: tf.Tensor) -> tf.Tensor:
        """Get the encodings of the auto-encoder given a Tensor input.

        This method was created to support both forward propagation and manual
        calls to get_encodings.

        Args:
            input (tf.Tensor): The features in Tensor form. The columns should be
                ordered as specified in the call method of SideEncoder.

        Returns:
            np.ndarray: The encodings for each row in the features input.
        """
        non_cat: tf.Tensor = input[:, : self.num_non_cat]
        cat_variables: list[tf.Tensor] = [
            input[:, (self.num_non_cat + i)] for i in range(self.num_cat)
        ]

        # Embed categories.
        cat_embeddings = []
        for cat, cat_embedder in zip(cat_variables, self.cat_embedders):
            cur_cat_embedding = cat_embedder(cat)
            cat_embeddings.append(cur_cat_embedding)

        # Combine non cat real/integer variables with cat embeddings
        to_encode = tf.concat([non_cat, *cat_embeddings], axis=1)

        # Encode.
        encoding = self.encoder(to_encode)

        return encoding

    @staticmethod
    def side_encoder_loss(self):
        """This method returns the loss function for the side encoder.

        This particular construction of returning an inner method allows us to use
        member variables inside the loss function, which are necessary for splitting
        the tensor into the predictions for each feature.
        """

        def side_encoder_loss_inner(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
            """The encoder loss.

            The loss first splits the tensors into the non-categorical and
            categorical features. In the true tensor, the categorical features are
            sparsely represented (so an integer denoting the category for each feature).
            In contrast, in the predicted tensor, the categorical features are densely
            represented (so a logit value for each category in each categorical feature,
            i.e. [0.1, 0.1, 0.7, 0.1] for a categorical feature of size 4).

            For the non-categorical features, the side encoder uses MSE loss, since
            predicting the values for these non-categorical features is a regression
            task. For the categorical features, the side encoder uses categorical
            cross-entropy.

            Args:
                y_true (tf.Tensor): The true features.
                y_pred (tf.Tensor): The predicted features.

            Returns:
                tf.Tensor: The loss.
            """
            # Split into non category and category outputs.
            sizes_true = [self.num_non_cat, *[1 for _ in range(len(self.cat_sizes))]]
            sizes_pred = [self.num_non_cat, *self.cat_sizes]

            # true_split format:
            # [[non-cat1, non-cat2, ..], [cat1 (sparse)], [cat2 (sparse)], ..]
            true_split = tf.split(y_true, sizes_true, axis=1)

            # pred_split format:
            # [[non-cat1, non-cat2, ..], [cat1 (dense)], [cat2 (dense)], [cat3 (dense)]]
            pred_split = tf.split(y_pred, sizes_pred, axis=1)

            # Calculate loss on non-categorical features.
            non_cat_loss = self.mse_loss(true_split[0], pred_split[0])

            # TODO: Weigh loss per feature so that contribution to the loss
            # is rougly equal.
            total_loss = non_cat_loss

            # Calculate loss on categorical features.
            for true_cat, pred_cat, loss_func in zip(
                true_split[1:], pred_split[1:], self.cat_loss_funcs
            ):
                total_loss += loss_func(true_cat, pred_cat)

            return total_loss

        return side_encoder_loss_inner

    @staticmethod
    def get_hidden_architecture(
        input_size: int, encoder_dimension: int
    ) -> Tuple[list[int], list[int]]:
        """Compute the sizes of the hidden layers of the encoder and decoder network.

        Note that if encoder_dimension * 2 > input_size, then we do not
        return hidden layers at all, because we then assume that just the encoding
        layer (which is between the hidden layers) should suffice anyway.

        For the decoder network, we basically double the dimension at each layer
        starting from the encoding_dimension * 2, up until but excluding the input size.
        For the encoder, we basically take the reverse architecture of the decoder.

        Args:
            input_size (int): The input size after having calculated the embeddings
                for the categorical features.
            encoder_dimension (int): The dimension the encodings should have.

        Returns:
            Tuple(list[int], list[int]): The sizes of the hidden layers of the encoder
                and decoder network respectively.
        """
        hidden_layer_sizes = []
        cur_dim = encoder_dimension * 2
        while cur_dim <= input_size:
            hidden_layer_sizes.append(cur_dim)
            cur_dim = cur_dim * 2

        return list(reversed(hidden_layer_sizes)), hidden_layer_sizes
