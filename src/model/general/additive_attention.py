import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(torch.nn.Module):
    """
    A general additive attention module
    """
    def __init__(self, query_vector_dim, candidate_vector_dim):
        super(AdditiveAttention, self).__init__()
        self.linear = nn.Linear(candidate_vector_dim, query_vector_dim)
        self.attention_query_vector = nn.Parameter(
            torch.empty(query_vector_dim).uniform_(-0.1, 0.1))

    def forward(self, candidate_vector):
        """
        Args:
            candidate_vector: batch_size, candidate_size, candidate_vector_dim
        Returns:
            (shape) batch_size, candidate_vector_dim
        """
        # batch_size, candidate_size, query_vector_dim
        temp = torch.tanh(self.linear(candidate_vector))
        # batch_size, candidate_size
        candidate_weights = F.softmax(torch.matmul(
            temp, self.attention_query_vector),
                                      dim=1)
        # batch_size, candidate_vector_dim
        target = torch.bmm(candidate_weights.unsqueeze(dim=1),
                           candidate_vector).squeeze(dim=1)
        return target
