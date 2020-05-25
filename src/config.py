import os

model_name = os.environ['MODEL_NAME'] if 'MODEL_NAME' in os.environ else 'NRMS'
# Currently included model
assert model_name in ['NRMS', 'NAML', 'LSTUR', 'DKN', 'HiFiArk', 'TANR', 'FIM']


class BaseConfig():
    """
    General configurations appiled to all models
    """
    num_batches = 8000  # Number of batches to train
    num_batches_batch_loss = 50  # Number of batchs to show loss
    # Number of batchs to check metrics on validation dataset
    num_batches_validate = 400
    num_batches_save_checkpoint = 200
    batch_size = 128
    learning_rate = 0.001
    validation_proportion = 0.1
    num_workers = 0  # Number of workers for data loading
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
    num_users = 1 + 49108
    word_embedding_dim = 300
    category_embedding_dim = 100
    # Modify the following only if you use another dataset
    entity_embedding_dim = 100
    # For additive attention
    query_vector_dim = 200


class NRMSConfig(BaseConfig):
    dataset_attributes = {
        "news": ['title'],
        "record": []
    }
    # For multi-head self-attention
    num_attention_heads = 15


class NAMLConfig(BaseConfig):
    dataset_attributes = {
        "news": ['category', 'subcategory', 'title', 'abstract'],
        "record": []
    }
    # For CNN
    num_filters = 300
    window_size = 3


class LSTURConfig(BaseConfig):
    dataset_attributes = {
        "news": ['category', 'subcategory', 'title', 'abstract'],
        "record": ['user', 'clicked_news_length']
    }
    # For CNN
    num_filters = 300
    window_size = 3
    # 'ini' or 'con'. See paper for more detail
    long_short_term_method = 'ini'
    # TODO 'con' unimplemented currently
    masking_probability = 0.5


class DKNConfig(BaseConfig):
    dataset_attributes = {
        "news": ['title', 'title_entities'],
        "record": []
    }
    # For CNN
    num_filters = 50
    window_sizes = [2, 3, 4]


class HiFiArkConfig(BaseConfig):
    dataset_attributes = {
        "news": ['title'],
        "record": []
    }
    # For CNN
    num_filters = 300
    window_size = 3
    num_pooling_heads = 5
    regularizer_loss_weight = 4.0  # TODO


class TANRConfig(BaseConfig):
    dataset_attributes = {
        "news": ['category', 'title'],
        "record": []
    }
    # For CNN
    num_filters = 300
    window_size = 3
    topic_classification_loss_weight = 0.1  # TODO


class FIMConfig(BaseConfig):
    dataset_attributes = {
        # Currently only title is used
        "news": ['category', 'subcategory', 'title'],
        "record": []
    }
    news_rep = {
        "num_filters": 300,
        "window_size": 3,
        "dilations": [1, 2, 3]
    }
    cross_matching = {
        "layers": [
            {
                "num_filters": 32,
                "window_size": (3, 3, 3),
                "stride": (1, 1, 1)
            },
            {
                "num_filters": 16,
                "window_size": (3, 3, 3),
                "stride": (1, 1, 1)
            }
        ],
        "max_pooling": {
            "window_size": (3, 3, 3),
            "stride": (3, 3, 3)
        }
    }
