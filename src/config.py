import os

model_name = 'NRMS'
# Currently included model
assert model_name in ['NRMS', 'NAML', 'LSTUR', 'DKN']


class BaseConfig():
    """
    General configurations appiled to all models
    """
    num_batches = 8000  # Number of batches to train
    num_batches_batch_loss = 50  # Number of batchs to show loss
    # Number of batchs to check metrics on validation dataset
    num_batches_validate = 200
    num_batches_save_checkpoint = 100
    batch_size = 128
    learning_rate = 0.001
    validation_proportion = 0.1
    num_workers = 4  # Number of workers for data loading
    num_clicked_news_a_user = 50  # Number of sampled click history for each user
    # Whether try to load checkpoint
    load_checkpoint = os.environ[
        'LOAD_CHECKPOINT'] == '1' if 'LOAD_CHECKPOINT' in os.environ else True
    num_words_title = 20
    num_words_abstract = 50
    word_freq_threshold = 3
    entity_freq_threshold = 3
    entity_confidence_threshold = 0.5
    negative_sampling_ratio = 4
    dropout_probability = 0.2
    # Modify the following by the output of `src/dataprocess.py`
    num_words = 1 + 31313
    num_categories = 1 + 274
    num_users = 1 + 50000
    word_embedding_dim = 300
    category_embedding_dim = 100
    # Modify the following only if you use another dataset
    entity_embedding_dim = 100
    # For additive attention
    query_vector_dim = 200


class NRMSConfig(BaseConfig):
    # For multi-head self-attention
    num_attention_heads = 15


class NAMLConfig(BaseConfig):
    # For CNN
    num_filters = 300
    window_size = 3


class LSTURConfig(BaseConfig):
    # For CNN
    num_filters = 300
    window_size = 3
    # 'ini' or 'con'. See paper for more detail
    long_short_term_method = 'ini'
    # TODO 'con' unimplemented currently
    masking_probability = 0.5


class DKNConfig(BaseConfig):
    # For CNN
    num_filters = 50
    window_sizes = [2, 3, 4]
