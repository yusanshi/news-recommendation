import torch
import torch.nn as nn
import torch.nn.functional as F
from model.general.additive_attention import AdditiveAttention

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class KCNN(torch.nn.Module):
    """
    Knowledge-aware CNN (KCNN) based on Kim CNN.
    Input a news sentence (e.g. its title), produce its embedding vector.
    """
    def __init__(self, config, pretrained_word_embedding,
                 pretrained_entity_embedding, pretrained_context_embedding):
        super(KCNN, self).__init__()
        self.config = config
        if pretrained_word_embedding is None:
            self.word_embedding = nn.Embedding(config.num_words,
                                               config.word_embedding_dim,
                                               padding_idx=0)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(
                pretrained_word_embedding, freeze=False, padding_idx=0)
        self.entity_embedding = pretrained_entity_embedding
        self.context_embedding = pretrained_context_embedding
        self.transform_matrix = nn.Parameter(
            torch.empty(self.config.word_embedding_dim,
                        self.config.entity_embedding_dim).uniform_(-0.1, 0.1))
        self.transform_bias = nn.Parameter(
            torch.empty(self.config.word_embedding_dim).uniform_(-0.1, 0.1))

        self.conv_filters = nn.ModuleDict({
            str(x): nn.Conv2d(3 if self.context_embedding is not None else 2,
                              self.config.num_filters,
                              (x, self.config.word_embedding_dim))
            for x in self.config.window_sizes
        })
        self.additive_attention = AdditiveAttention(
            self.config.query_vector_dim, self.config.num_filters)

    def forward(self, news):
        """
        Args:
          news:
            {
                "title": [Tensor(batch_size) * num_words_title],
                "title_entity":[Tensor(batch_size) * num_words_title]
            }

        Returns:
          final_vector: batch_size, len(window_sizes) * num_filters
        """
        # batch_size, num_words_title, word_embedding_dim
        word_vector = self.word_embedding(
            torch.stack(news["title"], dim=1).to(device))
        # batch_size, num_words_title, entity_embedding_dim
        entity_vector = F.embedding(torch.stack(news["title_entity"], dim=1),
                                    self.entity_embedding).to(device)
        if self.context_embedding is not None:
            # batch_size, num_words_title, entity_embedding_dim
            context_vector = F.embedding(
                torch.stack(news["title_entity"], dim=1),
                self.context_embedding).to(device)

        # The abbreviations are the same as those in paper
        b = self.config.batch_size
        n = self.config.num_words_title
        d = self.config.word_embedding_dim
        k = self.config.entity_embedding_dim

        # batch_size, num_words_title, word_embedding_dim
        transformed_entity_vector = torch.tanh(
            torch.add(
                torch.bmm(self.transform_matrix.expand(b * n, -1, -1),
                          entity_vector.view(b * n, k, 1)).view(b, n, d),
                self.transform_bias.expand(b, n, -1)))

        if self.context_embedding is not None:
            # batch_size, num_words_title, word_embedding_dim
            transformed_context_vector = torch.tanh(
                torch.add(
                    torch.bmm(self.transform_matrix.expand(b * n, -1, -1),
                              context_vector.view(b * n, k, 1)).view(b, n, d),
                    self.transform_bias.expand(b, n, -1)))

            # batch_size, 3, num_words_title, word_embedding_dim
            multi_channel_vector = torch.stack([
                word_vector, transformed_entity_vector,
                transformed_context_vector
            ],
                                               dim=1)
        else:
            # batch_size, 2, num_words_title, word_embedding_dim
            multi_channel_vector = torch.stack(
                [word_vector, transformed_entity_vector], dim=1)

        pooled_vectors = []
        for x in self.config.window_sizes:
            # batch_size, num_filters, num_words_title + 1 - x
            convoluted = self.conv_filters[str(x)](
                multi_channel_vector).squeeze(dim=3)
            # batch_size, num_filters, num_words_title + 1 - x
            activated = F.relu(convoluted)
            # batch_size, num_filters
            # Here we use a additive attention module
            # instead of pooling in the paper
            pooled = self.additive_attention(activated.transpose(1, 2))
            # pooled = activated.max(dim=-1)[0]
            # # or
            # # pooled = F.max_pool1d(activated, activated.size(2)).squeeze(dim=2)
            pooled_vectors.append(pooled)
        # batch_size, len(window_sizes) * num_filters
        final_vector = torch.cat(pooled_vectors, dim=1)
        return final_vector
