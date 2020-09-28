import torch
import torch.nn as nn
import torch.nn.functional as F
from model.general.attention.multihead_self import MultiHeadSelfAttention
from model.general.attention.additive import AdditiveAttention

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TextEncoder(torch.nn.Module):
    def __init__(self, word_embedding, word_embedding_dim, num_attention_heads,
                 query_vector_dim, dropout_probability):
        super(TextEncoder, self).__init__()
        self.word_embedding = word_embedding
        self.dropout_probability = dropout_probability
        self.multihead_self_attention = MultiHeadSelfAttention(
            word_embedding_dim, num_attention_heads)
        self.additive_attention = AdditiveAttention(query_vector_dim,
                                                    word_embedding_dim)

    def forward(self, text):
        # batch_size, num_words_text, word_embedding_dim
        text_vector = F.dropout(self.word_embedding(text),
                                p=self.dropout_probability,
                                training=self.training)
        # batch_size, num_words_text, word_embedding_dim
        multihead_text_vector = self.multihead_self_attention(text_vector)
        multihead_text_vector = F.dropout(multihead_text_vector,
                                          p=self.dropout_probability,
                                          training=self.training)
        # batch_size, word_embedding_dim
        final_text_vector = self.additive_attention(multihead_text_vector)
        return final_text_vector


class ElementEncoder(torch.nn.Module):
    def __init__(self, embedding, linear_input_dim, linear_output_dim):
        super(ElementEncoder, self).__init__()
        self.embedding = embedding
        self.linear = nn.Linear(linear_input_dim, linear_output_dim)

    def forward(self, element):
        return F.relu(self.linear(self.embedding(element)))


class NewsEncoder(torch.nn.Module):
    def __init__(self, config, pretrained_word_embedding):
        super(NewsEncoder, self).__init__()
        self.config = config
        if pretrained_word_embedding is None:
            word_embedding = nn.Embedding(config.num_words,
                                          config.word_embedding_dim,
                                          padding_idx=0)
        else:
            word_embedding = nn.Embedding.from_pretrained(
                pretrained_word_embedding, freeze=False, padding_idx=0)
        assert len(config.dataset_attributes['news']) > 0
        text_encoders_candidates = ['title', 'abstract']
        self.text_encoders = nn.ModuleDict({
            name:
            TextEncoder(word_embedding, config.word_embedding_dim,
                        config.num_attention_heads, config.query_vector_dim,
                        config.dropout_probability)
            for name in (set(config.dataset_attributes['news'])
                         & set(text_encoders_candidates))
        })
        category_embedding = nn.Embedding(config.num_categories,
                                          config.category_embedding_dim,
                                          padding_idx=0)
        element_encoders_candidates = ['category', 'subcategory']
        self.element_encoders = nn.ModuleDict({
            name:
            ElementEncoder(category_embedding, config.category_embedding_dim,
                           config.word_embedding_dim)
            for name in (set(config.dataset_attributes['news'])
                         & set(element_encoders_candidates))
        })
        if len(config.dataset_attributes['news']) > 1:
            self.final_attention = AdditiveAttention(config.query_vector_dim,
                                                     config.word_embedding_dim)

    def forward(self, news):
        """
        Args:
            news:
                {
                    "category": batch_size,
                    "subcategory": batch_size,
                    "title": batch_size * num_words_title,
                    "abstract": batch_size * num_words_abstract,
                }
        Returns:
            (shape) batch_size, word_embedding_dim
        """
        text_vectors = [
            encoder(news[name].to(device))
            for name, encoder in self.text_encoders.items()
        ]
        element_vectors = [
            encoder(news[name].to(device))
            for name, encoder in self.element_encoders.items()
        ]

        all_vectors = text_vectors + element_vectors

        if len(all_vectors) == 1:
            final_news_vector = all_vectors[0]
        else:
            final_news_vector = self.final_attention(
                torch.stack(all_vectors, dim=1))
        return final_news_vector
