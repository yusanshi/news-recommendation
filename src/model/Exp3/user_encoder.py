import torch
import torch.nn as nn
import torch.nn.functional as F


class UserEncoder(torch.nn.Module):
    def __init__(self, config):
        super(UserEncoder, self).__init__()
        self.config = config
        self.linear = nn.Linear(2 * config.word_embedding_dim,
                                config.query_vector_dim)
        self.attention_query_vector = nn.Parameter(
            torch.empty(config.query_vector_dim).uniform_(-0.1, 0.1))

    def forward(self, clicked_news_vector, popularity_vector):
        """
        Args:
            clicked_news_vector: batch_size, num_clicked_news_a_user, word_embedding_dim
        Returns:
            (shape) batch_size, word_embedding_dim
        """
        # batch_size, candidate_size, query_vector_dim
        temp = torch.tanh(
            self.linear(
                torch.cat((clicked_news_vector, popularity_vector), dim=-1)))
        # batch_size, candidate_size
        candidate_weights = F.softmax(torch.matmul(
            temp, self.attention_query_vector),
                                      dim=1)
        # batch_size, word_embedding_dim
        final_user_vector = torch.bmm(candidate_weights.unsqueeze(dim=1),
                                      clicked_news_vector).squeeze(dim=1)

        return final_user_vector
