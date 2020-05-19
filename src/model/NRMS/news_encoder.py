import torch
import torch.nn as nn
import torch.nn.functional as F
from model.general.multihead_self_attention import MultiHeadSelfAttention
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

        self.multihead_self_attention = MultiHeadSelfAttention(
            config.word_embedding_dim, config.num_attention_heads)
        self.additive_attention = AdditiveAttention(config.query_vector_dim,
                                                    config.word_embedding_dim)

    def forward(self, news):
        """
        Args:
            news: Tensor(batch_size) * num_words_title
        Returns:
            (shape) batch_size, word_embedding_dim
        """
        # batch_size, num_words_title, word_embedding_dim
        news_vector = F.dropout(self.word_embedding(
            torch.stack(news, dim=1).to(device)),
            p=self.config.dropout_probability,
            training=self.training)
        # batch_size, num_words_title, word_embedding_dim
        multihead_news_vector = self.multihead_self_attention(news_vector)
        # batch_size,  word_embedding_dim
        final_news_vector = self.additive_attention(multihead_news_vector)
        return final_news_vector
