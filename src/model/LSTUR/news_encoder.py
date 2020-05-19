import torch
import torch.nn as nn
import torch.nn.functional as F
from model.general.additive_attention import AdditiveAttention

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class NewsEncoder(torch.nn.Module):
    def __init__(self, config, pretrained_word_embedding):
        super(NewsEncoder, self).__init__()
        self.config = config
        if pretrained_word_embedding is None:
            self.word_embedding = nn.Embedding(config.num_words,
                                               config.word_embedding_dim,
                                               padding_idx=0)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(
                pretrained_word_embedding, freeze=False, padding_idx=0)
        self.category_embedding = nn.Embedding(config.num_categories,
                                               config.num_filters,
                                               padding_idx=0)
        assert config.window_size >= 1 and config.window_size % 2 == 1
        self.title_CNN = nn.Conv2d(
            1,
            config.num_filters,
            (config.window_size, config.word_embedding_dim),
            padding=(int((config.window_size - 1) / 2), 0))
        self.abstract_CNN = nn.Conv2d(
            1,
            config.num_filters,
            (config.window_size, config.word_embedding_dim),
            padding=(int((config.window_size - 1) / 2), 0))
        self.title_attention = AdditiveAttention(config.query_vector_dim,
                                                 config.num_filters)
        self.abstract_attention = AdditiveAttention(config.query_vector_dim,
                                                    config.num_filters)

    def forward(self, news):
        """
        Args:
            news:
                {
                    "category": Tensor(batch_size),
                    "subcategory": Tensor(batch_size),
                    "title": [Tensor(batch_size) * num_words_title],
                    "abstract": [Tensor(batch_size) * num_words_abstract]
                }
        Returns:
            (shape) batch_size, num_filters * 4
        """
        # Part 1: calculate category_vector

        # batch_size, num_filters
        category_vector = self.category_embedding(news['category'].to(device))

        # Part 2: calculate subcategory_vector

        # batch_size, num_filters
        subcategory_vector = self.category_embedding(
            news['subcategory'].to(device))

        # Part 3: calculate weighted_title_vector

        # batch_size, num_words_title, word_embedding_dim
        title_vector = F.dropout(self.word_embedding(
            torch.stack(news['title'], dim=1).to(device)),
                                 p=self.config.dropout_probability,
                                 training=self.training)
        # batch_size, num_filters, num_words_title
        convoluted_title_vector = self.title_CNN(
            title_vector.unsqueeze(dim=1)).squeeze(dim=3)
        # batch_size, num_filters, num_words_title
        activated_title_vector = F.dropout(F.relu(convoluted_title_vector),
                                           p=self.config.dropout_probability,
                                           training=self.training)
        # batch_size, num_filters
        weighted_title_vector = self.title_attention(
            activated_title_vector.transpose(1, 2))

        # Part 4: calculate weighted_abstract_vector

        # batch_size, num_words_abstract, word_embedding_dim
        abstract_vector = F.dropout(self.word_embedding(
            torch.stack(news['abstract'], dim=1).to(device)),
                                    p=self.config.dropout_probability,
                                    training=self.training)
        # batch_size, num_filters, num_words_abstract
        convoluted_abstract_vector = self.abstract_CNN(
            abstract_vector.unsqueeze(dim=1)).squeeze(dim=3)
        # batch_size, num_filters, num_words_abstract
        activated_abstract_vector = F.dropout(
            F.relu(convoluted_abstract_vector),
            p=self.config.dropout_probability,
            training=self.training)
        # batch_size, num_filters
        weighted_abstract_vector = self.abstract_attention(
            activated_abstract_vector.transpose(1, 2))

        # batch_size, num_filters * 4
        news_vector = torch.cat([
            category_vector, subcategory_vector, weighted_title_vector,
            weighted_abstract_vector
        ],
                                dim=1)
        return news_vector
