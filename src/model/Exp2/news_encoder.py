import torch
import torch.nn as nn
import torch.nn.functional as F
from model.general.attention.multihead_self import MultiHeadSelfAttention
from model.general.attention.additive import AdditiveAttention
from transformers import RobertaModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TextEncoder(torch.nn.Module):
    def __init__(self, word_embedding_dim, num_attention_heads,
                 query_vector_dim, dropout_probability, roberta_level,
                 fine_tune):
        super(TextEncoder, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        if self.training:
            self.roberta.train()
        if not fine_tune:
            for param in self.roberta.parameters():
                param.requires_grad = False
        self.reduce_dim = nn.Linear(768, word_embedding_dim)
        self.dropout_probability = dropout_probability
        self.roberta_level = roberta_level
        self.multihead_self_attention = MultiHeadSelfAttention(
            word_embedding_dim, num_attention_heads)
        self.additive_attention = AdditiveAttention(query_vector_dim,
                                                    word_embedding_dim)

    def forward(self, input_ids, attention_mask):
        embeddings = self.roberta(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  return_dict=True)
        if self.roberta_level == 'word':
            # batch_size, num_words_text, word_embedding_dim
            text_vector = F.dropout(self.reduce_dim(
                embeddings['last_hidden_state']),
                                    p=self.dropout_probability,
                                    training=self.training)
            # batch_size, num_words_text, word_embedding_dim
            multihead_text_vector = self.multihead_self_attention(text_vector)
            multihead_text_vector = F.dropout(multihead_text_vector,
                                              p=self.dropout_probability,
                                              training=self.training)
            # batch_size, word_embedding_dim
            return self.additive_attention(multihead_text_vector)
        elif self.roberta_level == 'sentence':
            # batch_size, word_embedding_dim
            return self.reduce_dim(embeddings['pooler_output'])


class ElementEncoder(torch.nn.Module):
    def __init__(self, embedding, linear_input_dim, linear_output_dim):
        super(ElementEncoder, self).__init__()
        self.embedding = embedding
        self.linear = nn.Linear(linear_input_dim, linear_output_dim)

    def forward(self, element):
        return F.relu(self.linear(self.embedding(element)))


class NewsEncoder(torch.nn.Module):
    def __init__(self, config):
        super(NewsEncoder, self).__init__()
        self.config = config
        assert len(config.dataset_attributes['news']) > 0
        self.text_encoders = nn.ModuleDict()
        for x in ['title', 'abstract']:
            if f'{x}_roberta' in config.dataset_attributes[
                    'news'] and f'{x}_mask_roberta' in config.dataset_attributes[
                        'news']:
                self.text_encoders[x] = TextEncoder(config.word_embedding_dim,
                                                    config.num_attention_heads,
                                                    config.query_vector_dim,
                                                    config.dropout_probability,
                                                    config.roberta_level,
                                                    config.fine_tune)

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
                    "category": Tensor(batch_size),
                    "subcategory": Tensor(batch_size),
                    "title_roberta": Tensor(batch_size) * num_words_title,
                    "title_mask_roberta: Tensor(batch_size) * num_words_title,
                }
        Returns:
            (shape) batch_size, word_embedding_dim
        """
        text_vectors = [
            encoder(
                torch.stack(news[f"{name}_roberta"], dim=1).to(device),
                torch.stack(news[f"{name}_mask_roberta"], dim=1).to(device))
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
