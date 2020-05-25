import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(torch.nn.Module):
    """
    A general multi-head self attention module.
    Originally for NRMS.
    """

    def __init__(self, candidate_vector_dim, num_attention_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert candidate_vector_dim % num_attention_heads == 0
        self.candidate_vector_dim = candidate_vector_dim
        self.num_attention_heads = num_attention_heads
        self.single_head_dim = int(candidate_vector_dim / num_attention_heads)
        self.Qs = nn.ParameterList([
            nn.Parameter(
                torch.empty(candidate_vector_dim,
                            candidate_vector_dim).uniform_(-0.1, 0.1))
            for _ in range(num_attention_heads)
        ])
        self.Vs = nn.ParameterList([
            nn.Parameter(
                torch.empty(
                    candidate_vector_dim,
                    self.single_head_dim,
                ).uniform_(-0.1, 0.1)) for _ in range(num_attention_heads)
        ])

    def forward(self, candidate_vector):
        """
        Args:
            candidate_vector: batch_size, candidate_size, candidate_vector_dim
        Returns:
            (shape) batch_size, candidate_size, candidate_vector_dim
        """
        container = []
        for i in range(self.num_attention_heads):
            # batch_size, candidate_size, candidate_vector_dim
            temp = torch.matmul(candidate_vector, self.Qs[i])
            # batch_size, candidate_size, candidate_size
            temp2 = torch.bmm(temp, candidate_vector.transpose(1, 2))
            # batch_size, candidate_size, candidate_size
            weights = F.softmax(temp2, dim=2)
            # batch_size, candidate_size, candidate_vector_dim
            weighted = torch.bmm(weights, candidate_vector)
            # batch_size, candidate_size, single_head_dim
            container.append(torch.matmul(weighted, self.Vs[i]))
        # batch_size, candidate_size, candidate_vector_dim
        target = torch.cat(container, dim=2)
        return target
