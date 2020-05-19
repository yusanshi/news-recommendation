import os

# Currently included model: 'NRMS', 'NAML', 'LSTUR'
model_name = 'NRMS'


class BaseConfig():
    """
    General configurations appiled to all models
    """
    num_batches = 8000  # Number of batches to train
    num_batches_batch_loss = 50  # Number of batchs to show loss
    # Number of batchs to check loss and accuracy on validation dataset
    num_batches_val_loss_and_acc = 500
    num_batches_save_checkpoint = 100
    batch_size = 64
    learning_rate = 0.001
    train_validation_split = (0.9, 0.1)
    num_workers = 4  # Number of workers for data loading
    num_clicked_news_a_user = 50  # Number of sampled click history for each user
    # Whether try to load checkpoint
    load_checkpoint = os.environ[
        'LOAD_CHECKPOINT'] == '1' if 'LOAD_CHECKPOINT' in os.environ else True
    num_words_title = 20
    word_freq_threshold = 3
    negative_sampling_ratio = 4
    inference_radio = 0.1
    dropout_probability = 0.2
    # Modify the following by the output of `src/dataprocess.py`
    num_words = 1 + 15352
    word_embedding_dim = 300
    # For additive attention
    query_vector_dim = 200
    category_embedding_dim = 100
    num_words_abstract = 50


class NRMSConfig(BaseConfig):
    # For multi-head self-attention
    num_attention_heads = 15
    num_categories = 1 + 274


class NAMLConfig(BaseConfig):
    # For CNN
    num_filters = 400
    window_size = 3


class LSTURConfig(BaseConfig):
    # For CNN
    num_filters = 300
    window_size = 3
    # 'ini' or 'con'. See paper for more detail
    long_short_term_method = 'ini'
    masking_probability = 0.5
    # Modify the following by the output of `src/dataprocess.py`
    num_users = 1 + 50000
    num_categories = 1 + 274
