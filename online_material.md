# Implementation details 

## Neural models 
Our implementations of BERT4Rec, SASRec and GRU4Rec are subclasses of our `NeuralModel` in `main/neural_model.py`, which contains all functionality for training and predicting with a Keras model. Here, we take 10% of the train sessions (with a maximum of 500 sessions) and place them in an "inner-validation" set, and use the inner-validation set for early stopping. More specifically, after each epoch, we evaluate the model on NDCG@20 on the inner-validation set. If the model has not improved the inner-validation NDCG@20 for 2 epochs, we stop training and restore the model weights to the epoch with the best inner-validation NDCG@20. Using the inner-validation NDCG@20 prevents our models from overfitting on the data. Note that 2 epochs may seem like a short early-stopping patience, but our models generally converge within 5-15 epochs, even with lower learning rates. Therefore, even with higher values for the early-stopping patience, the model weights will often be restored to the same epoch, so higher values for the early-stopping patience would result in more superfluous training. 

Note that all our neural models have the hyperparameter __N__, which is used to truncate long sessions to the last __N__ items. This is to ensure that sessions fit in the same tensor. We choose N = 20 for both the Beauty and the Anonymous dataset. 

Lastly, we train all models with the Keras AdamW optimizer. 

### BERT4Rec
We implemented BERT4Rec largely following the specifications in the [BERT4REC paper](https://dl.acm.org/doi/abs/10.1145/3357384.3357895) [1] with the transformer architecture described in [2]. A minor difference with our implementation is the initialization method: in our implementation we use the keras default GlorotUniform, except for the bias layer. The latter is initialized with zeros. The [accompanying implementation](https://github.com/FeiSun/BERT4Rec/blob/615eaf2004abecda487a38d5b0c72f3dcfcae5b3/modeling.py#L371) of the paper uses the truncated normal distribution, but we found that this initializer generally resulted in slightly lower performance than the initialization method described above. 

### SASRec

SASRec mainly adheres to the specifications in the original [SASRec paper](https://arxiv.org/abs/1808.09781) [3]. Note that SASRec in [3] proposes a different transformer layout than the one used in BERT4Rec and Attention is all you need [2]. SASRec in [3] additionally uses a dropout layer on the embeddings. Also, they originally only use one head instead of multi-head attention, and have some experiments demonstrating that using multiple heads did not lead to performance increases with their implementation. We accomodate the aforementioned differences with BERT4Rec in our SASRec implementation and allow for a variable number of heads. Moreover, we use the projection head from BERT4Rec instead of the prediction layer described in [3] because we found it generally improved performance. Lastly, our SASRec implementation is trained with categorical cross-entropy loss, again similar to BERT4Rec, without the negative sample described in [3]. 

### GRU4Rec 
Our GRU4Rec implementation slightly deviates from the original [GRU4Rec paper](https://arxiv.org/pdf/1511.06939.pdf) [4]. Instead of the session-parallel mini-batches, we train GRU4Rec using whole sessions in a batch. This way, the data generator that we need conveniently coincides with the one for SASRec. By processing sessions as a whole, we do not have to reset the internal state during a batch which makes the implementation overall much simpler. One side-effect of this is that GRU4Rec becomes unable to process sessions longer than its parameter __N__, but we generally found that truncating sessions to a sensible length improves performance, likely due to the removal of noise from the long-tail of the sessions. Furthermore, we simply train GRU4Rec with categorical cross-entropy loss, and we did not encounter the instability reported in [4] caused by this loss function. Lastly, we re-use the projection head from BERT4Rec instead of the dense prediction layer described in [3]. This improved model performance, as re-using the item embeddings in the projection head basically halves the model's size and thus alleviates overfitting. 

## SKNN 
We implemented SKNN according to the specifications in [5]. Our implementation parameterizes and adds several options. First of all, both dot-product and cosine similarity are supported. We parameterize the sampling strategy and support both the random sampling of sessions, and prioritizing recent sessions (if the timestamp is available). We added the parameter `idf_weighting`, which if set, weighs items proportional to their IDF-score when computing the session similarity. The `SKNN` class supports all variants of SKNN presented in [5], including V-SKNN (using the `decay` parameter), SF-SKNN (with `sequential_filter` set to `true`), S-SKNN (with `sequential_weighting` set to `true`) and regular SKNN. For V-SKNN, we use harmonic decay. 

# Data 
We preprocessed Beauty using the 5-core methodology, meaning we iteratively remove sessions and items until all sessions and items have at least five interactions. Note that in our test prompts there is a small amount of sessions that have less than five interactions. This is a result of our `filter_non_trained_test_items` flag in `beauty/split_sessions`, which removes all items from the prompts that were not included in the training data. In some rare cases, all five (or more) interactions of an item only appear in the test set, causing them to be filtered from the test prompts. 

We do not apply any pre-processing to our anonymous QCommerce sessions dataset to better simulate a real-world setting, except that we removed sessions with only one interaction from the test set. 

For both datasets, we employ a temporal splitting strategy to separate the data into a train and test set. So, all test sessions succeed train sessions in time. We believe that the temporal splitting strategy best simulates a real-world setting in comparison to other splitting strategies in the literature (evolving [1] or simple random sampling). 

# Hypersearch 
We hypersearch (hyperparameter-search) the neural models LLM2BERT, BERT4Rec, SASRec, GRU4Rec, and the SKNN variants with a Tree-Parzen-Estimator (TPE) sampler [7]. The optimization objective is the average validation NDCG@20 across the three folds. We start with 20 randomly-sampled configurations to avoid premature biases and then continue by evaluating the suggestions by the TPE sampler. We stop the hypersearch if the optimization objective has not been improved for 100 trials. We pre-empt the hypersearch if it has not stopped after 72 hours. 

The folds themselves are created by splitting the training data into validation-training and validation-testing sets. We again do this in a temporal fashion by splitting the train sessions into four bins, where the first bin contains the first 25% training sessions etc. The first fold then uses the first bin as the validation-training data, and uses the second bin as the validation-testing data. The second fold then uses the first and second bin as the validation-training data, and the third as the validation-testing data. The third fold uses the first three bins as the validation-training data, and the fourth as the validation-testing data. By using less data in the first few folds (in comparison to cross-validation), we can more quickly evaluate a configuration and preserve the temporal split in our hypersearch process as well. 

We validate our approach by finding that the average validation NDCG@20 of a model configuration is strongly correlated with the final NDCG@20 of a model configuration trained on the whole training data, evaluated on the hidden test set. We also found that the validation NDCG@20 on individual folds are correlated with the final NDCG@20, which suggests we can use pruning. Therefore, after each fold we prune the model configurations in the bottom 20% of validation NDCG@20 to speed-up our hypersearch. 

## LLMSeqPrompt
For the `LLMSeqPrompt` model, we hypersearch the prediction parameters. It would have been too costly and time-consuming to hypersearch the training parameters (e.g. learning rate, batch size, prompt_loss_weight), and hence we resorted to simply training the model with the default training parameters. For prediction we considered `temperature` and `top_p` to be influential parameters, and hence experimented with these values. These variables basically control the diversity of the completion, and can hence be used to control the amount of duplicates and hallucinations in the predictions by the fine-tuned model. We hypothesized that decreasing the amount of duplicates or hallucinations would improve performance, and thus we hypersearch the `temperature` and `top_p` parameters. Since [the OpenAI API reference](https://platform.openai.com/docs/api-reference/completions/create) generally recommends altering `temperature` or `top_p` but not both, we fix either `temperature` or `top_p` to 1 while hypersearching the other. 

## LLMSeqSim
For the `LLMSeqSim` model, we experimented with several combination strategies and similarity measures. For the latter, we found that the exact similarity measure barely affected the recommendations. This is because the LLM embeddings all roughly have a norm equal to 1. Therefore, our session embedding (an average of embeddings) also have a norm roughly equal to 1.  As a result, cosine and dot product trivially produce the same similarity scores. Moreover, because of the equal norms, euclidean similarity is very much correlated with the dot-product similarities, so much so that it barely affected the relative ranking. Since there are very little differences between similarity measures, and dot-product is fastest to compute, we settled on using dot-product in `LLMSeqSim`, and in `LLMSeqPrompt` for resolving duplicates and hallucinations as well. For the combination strategies, we experiment with different functions to weigh items when taking the weighted average. We experimented with constant (equal weights, so normal average), linear, quadratic, cubic and last_only weighing functions. Note that the last_only weighing function simply corresponds to taking the embedding of the last item as the session embedding. 


## Hyperparameter ranges 
The searched hyperparameter ranges are listed below. An "*" (asterisk) denotes that the range is the same throughout the column. An "-" (hyphen) denotes that a hyperparameter is not used for the corresponding model. 

| **Model/hyperparameter range** | **learning_rate** | **weight_decay** | **clipnorm** | **fit_batch_size** | **emb_dim** | **drop_rate** | **L** | **h** | **mask_prob** | **hidden_dim** |
|--------------------------------|-------------------|------------------|--------------|--------------------|-------------|---------------|-------|-------|---------------|----------------|
| **BERT4Rec**      | 0.0001 - 0.01     | 0-0.1            | 1-100 or None       | 32-512             | 16-512      | 0-0.9         | 1-4   | 1-4   | 0.05-0.9      | -              |
| **LLM2BERT4Rec**      | *                 | *                | *            | *                  | *           | *             | *     | *     | *             | -              |
| **SASRec**                     | *                 | *                | *            | *                  | *           | *             | *     | *     | -             | -              |
| **GRU4Rec**                    | *                 | *                | *            | *                  | *           | *             | -     | -     | -             | 16-512         |

|  | **k**  | **sample_size** | **sampling**         | **similarity_measure** |
|--------------------------------|--------|-----------------|----------------------|------------------------|
| **SKNN**                       | 50-500 | 500 - 2000      | {"random", "recent"} | {"dot", "cosine"}      |
| **S-SKNN**                     | *      | *               | *                    | *                      | 
| **SF-SKNN**                    | *      | *               | *                    | *                      |
| **V-SKNN**                     | *      | *               | *                    | *                      |

|  | **top_p**  | **temperature** |
|--------------------------------|--------|-----------------|
| **LLMSeqPrompt**                       | 0.25 - 1 | 0.25 - 1 |


|  | **weighing function**  |
|--------------------------------| -- |
| **LLMSeqSim**                       | {"constant", "linear", "quadratic", "cubic", "last_only"} |


## Optimal hyperparameters Beauty

| **Model / optimal hyperparameter** | **learning_rate** | **weight_decay** | **clipnorm** | **fit_batch_size** | **emb_dim** | **drop_rate** | **L** | **h** | **mask_prob** | **hidden_dim** |
|--------------------------------|-------------------|------------------|--------------|--------------------|-------------|---------------|-------|-------|---------------|----------------|
| **LLM2BERT4Rec**      | 0.0001     | 0.002        | 5        | 256             | 208      | 0.0         | 3   | 3   | 0.5      | -              |
| **BERT4Rec**      | 0.0001     | 0.004            | 5        | 256             | 512      | 0.25         | 2   | 2   | 0.65      | -              |
| **SASRec**                     | 0.001                 | 0                | 100            | 128                 | 320          | 0.3           | 1    | 1   | -             | -              |
| **GRU4Rec**                    | 0.001                 | 0                | None        | 128                | 208          | 0             | -     | -     | -             | 320         |

|  | **k**  | **sample_size** | **sampling**         | **similarity_measure** |
|--------------------------------|--------|-----------------|----------------------|------------------------|
| **V-SKNN**                     |    70   | 1090               | random                    | dot                      |

|  | **top_p**  | **temperature** |
|--------------------------------|--------|-----------------|
| **LLMSeqPrompt**                       | 1 | 0.25 |


|  | **weighing function**  |
|--------------------------------| -- |
| **LLMSeqSim**                       | last_only |

## Optimal hyperparameters Anonymous

| **Model / optimal hyperparameter** | **learning_rate** | **weight_decay** | **clipnorm** | **fit_batch_size** | **emb_dim** | **drop_rate** | **L** | **h** | **mask_prob** | **hidden_dim** |
|--------------------------------|-------------------|------------------|--------------|--------------------|-------------|---------------|-------|-------|---------------|----------------|
| **LLM2BERT4Rec**      | 0.0001     | 0        | 5        | 256             | 224      | 0.1         | 4   | 4   | 0.25      | -              |
| **BERT4Rec**      | 0.0001     | 0.007            | 12        | 256             | 176      | 0.15         | 1   | 1   | 0.4      | -              |
| **SASRec**                     | 0.0005                 | 0.02                | 20            | 256                 | 176          | 0.2           | 1    | 1   | -             | -              |
| **GRU4Rec**                    | 0.0013                 | 0.09                | 29        | 128                | 480          | 0.25             | -     | -     | -             | 176         |

|  | **k**  | **sample_size** | **sampling**         | **similarity_measure** | 
|--------------------------------|--------|-----------------|----------------------|------------------------|
| **V-SKNN**                     |    420   | 1040               | recent                    | dot                      | 

|  | **top_p**  | **temperature** |
|--------------------------------|--------|-----------------|
| **LLMSeqPrompt**                       | 1 | 0.5 |


|  | **weighing function**  |
|--------------------------------| -- |
| **LLMSeqSim**                       | constant |

# Additional discussion 
To reiterate our discussion on the results in our paper: 

- The gains obtained by initializing BERT4Rec with semantically-rich embeddings are substantial. We confirmed that the driver of performance are the semantics, instead of any statistical properties of the LLM embeddings. We did this by permuting the embeddings across the item catalogue, and found that the performance degraded heavily to the point where LLM2BERT4Rec had worse performance than vanilla BERT4Rec. 

- The `LLMSeqSim` model performance varies across datasets. On Beauty it is highly competitive, whereas on our Anonymous dataset the model has poor performance. The substantially-higher number of infrequent items makes it harder for the `LLMSeqSim` to find the correct recommendation. Furthermore, we find that in the Beauty dataset, items in the same session are often semantically related, most often through the brand names. In our Anonymous dataset, this is not the case. In addition, we find that `LLMSeqSim` has a high Catalog Coverage, since it does not take item popularity into account. 

- The `SKNN` models are, though non-neural, very competitive in performance, especially on short sessions. See `notebooks/session_length_hitrate_*.ipynb` for SKNN's performance against session length. 

This section is intended to extend this discussion. 

## Observations on optimal hyperparameter values 
We excluded our observations on the optimal hyperparameter values due to space limitations. However, looking at the optimal hyperparameter values of each model, we can make several observations. Most significantly, we find that `LLM2BERT4Rec` is less prone to overfitting, evidenced by less regularization (lower optimal values for weight decay and drop rate) and deeper architectures (higher optimal values for the number of encoder layers). Though, this is not the only driver of performance, as we also found that using the exact same model configuration still provides a significant performance boost. These observations suggest that by initializing BERT4Rec with LLM Embeddings, we initialize the model weights close to a better minimum in the hyperparameter landscape than when we initialize with random weights. 

Furthermore, it seems that in both cases `V-SKNN` is the most performant of the SKNN variants, indicating that it is necessary to assign a lower-weight to items in the long-tail of the session, which implies that these long-tail items are more likely to be less-related to the ground-truth item.

For the `LLMSeqSim` model, it seems that in Beauty the last item is generally most related to the last item, and hence LAST_ONLY is the optimal weighing function for Beauty. In contrast, for the Anonymous dataset, it seems that the CONSTANT weighing function is optimal. Note that this simply correponds to taking the average of the embeddings of all items in the session. This may be explained by the fact that the Anonymous dataset contains shorter sessions, and so most items in the session are relevant to the ground-truth item. 

For the `LLMSeqPrompt`, we surprisingly found that lowering the `temperature` to 0.25 for Beauty or 0.5 for the Anonymous dataset produced the best results. Generally, we find that lowering the `temperature` or `top_p` decreases the amount of hallucinated items, and increases the amount of duplicates. The  `temperature` and `top_p` parameters therefore directly control this trade-off. With the optimal hyperparameter configuration on `Beauty`, only 20% of the recommendations were unique, 80% of recommendations were duplicate recommendations, and 26% of the recommendations were hallucinated (not directly corresponding to an item in the catalogue). Hence, it seems to be the case that this model favours precision over diversity, since all duplicated recommendations are replaced with closely-related products. Hence, we get recommendation slates very similar to `LLMSeqSim` (very much focused on a single type of item or brand), with the added benefit that `LLMSeqPrompt` is not restricted to produce recommendations that are semantically similar to the test prompt, and can instead focus on recommending complementary products learned from the session data. 

## Other string embedders
We experimented with other string embedders than the `text-embedding-ada-002` embedder from OpenAI. The fact that retrieving the LLM Embeddings from OpenAI is pay-per-use limits its usability, and so we experimented with open-source embedders. We found [GenSim, topic modelling for humans](https://radimrehurek.com/gensim/models/word2vec.html) to have a Python API for retrieving word embeddings. Since these embedders map individual words to embeddings, not whole strings, we take the average of the embeddings of each individual word in the product names, if it exists in the vocabulary. We subsequently used these embeddings in our `LLMSeqSim` model. Surprisingly, the model with these open-source embeddings resulted in very poor performance and barely beat the `MostPopular` baseline. For example, the best model (of the open-source embedders) was `LLMSeqSim` with the `glove-wiki-gigaword-300` vocabulary with a NDCG@20 of 0.012 and HitRate@20 of 0.03. We attribute the difference in performance between the OpenAI LLM Embeddings and the embeddings by the open-source embedders to the fact that `text-embedding-ada-002` is able to embed the string as a whole, and recognizes brand names much more often. The former means that OpenAI *probably* has a more sophisticated mechanism in place to combine the embeddings of the tokens in the product name instead of the simple averaging we attempted for the open-source embeddings. The latter, recognizing brand names, seems to be a vital factor in performance when looking at the recommendations on an individual session level. The sessions for which the `LLMSeqSim` with `text-embedding-ada-002` embeddings correctly predicted the last item often had a strong semantic connection between the last item in the prompt and the ground-truth item. Most-often this semantic connection was through the brand name, and thus the performance of  `LLMSeqSim` with `text-embedding-ada-002` embeddings is significantly better than the performance of  `LLMSeqSim` with open-source embeddings. 

## LLM2SASRec, LLM2GRU4Rec 
Besides LLM2BERT4Rec, we also implemented LLM2SASRec and LLM2GRU4Rec. For reference, we paste the accuracy results on both datasets here below. 

Beauty dataset

|                 | **NDCG@10** | **HitRate@10** | **NDCG@20** | **HitRate@20** |
|-----------------|-------------|----------------|-------------|----------------|
| **SASRec**      | 0.026       | 0.051          | 0.033       | 0.080          |
| **LLM2SASRec**  | 0.033       | 0.067          | 0.044       | 0.110          |
| **GRU4Rec**     | 0.026       | 0.050          | 0.034       | 0.081          |
| **LLM2GRU4Rec** | 0.028       | 0.056          | 0.037       | 0.089          |

Anonymous dataset

|                 | **NDCG@10** | **HitRate@10** | **NDCG@20** | **HitRate@20** |
|-----------------|-------------|----------------|-------------|----------------|
| **SASRec**      | 0.084       | 0.149          | 0.100       | 0.212          |
| **LLM2SASRec**  | 0.094       | 0.167          | 0.112       | 0.239          |
| **GRU4Rec**     | 0.085       | 0.153          | 0.101       | 0.218          |
| **LLM2GRU4Rec** | 0.084       | 0.152          | 0.101       | 0.218          |

Evidently, we found that SASRec generally also benefits from being initialized with the LLM embeddings. In fact, we observe the same patterns for LLM2SASRec as we did for LLM2BERT4Rec; The LLM2SASRec model needs less regularization (drop_rate = 0.15, weight_decay = 0.007 on the Anonymous dataset), and is most optimal in deeper architectures (L = h = 4 in the Anonymous dataset). In contrast, for the LLM2GRU4Rec model the gains depend on the dataset; it shows to benefit from the LLM embeddings on Beauty, but not on the Anonymous dataset. This may be explained by the fact that Beauty has much fewer interactions per item on average when compared to the Anonymous dataset, so that the Anonymous dataset contains enough information per item to avoid overfitting by GRU4Rec, which the LLM embeddings seem to prevent for our transformer models. 

# References 

[1] Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang. 2019. BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer. In Proceedings of the 28th ACM International Conference on Information and Knowledge Management (CIKM ’19). 1441–1450.

[2] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. 2017. Attention is All you Need. In Advances in Neural Information Processing Systems, Vol. 30. Curran Associates, Inc.

[3] Wang-Cheng Kang and Julian J. McAuley. 2018. Self-Attentive Sequential Recommendation. In Proceedings of the 18th IEEE International Conference on Data Mining (ICDM 2018). 197–206.

[4] Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, and Domonkos Tikk. 2016. Session-based Recommendations with Recurrent Neural Networks. In 4th International Conference on Learning Representations, ICLR.

[5] Malte Ludewig and Dietmar Jannach. 2018. Evaluation of Session-based Recommendation Algorithms. User-Modeling and User-Adapted Interaction 28, 4–5 (2018), 331–390.

[6] Vito Walter Anelli, Alejandro Bellogín, Tommaso Di Noia, Dietmar Jannach, and Claudio Pomo. 2022. Top-N Recommendation Algorithms: A Quest
for the State-of-the-Art. In Proceedings of the 30th ACM Conference on User Modeling, Adaptation and Personalization (Barcelona, Spain) (UMAP ’22).
Association for Computing Machinery, 121–131

[7] James Bergstra, Rémi Bardenet, Yoshua Bengio, and Balázs Kégl. 2011. Algorithms for Hyper-Parameter Optimization. In Advances in Neural Information Processing Systems, J. Shawe-Taylor, R. Zemel, P. Bartlett, F. Pereira, and K.Q. Weinberger (Eds.), Vol. 24. Curran Associates, Inc.