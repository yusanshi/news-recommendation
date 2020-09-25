import torch
import torch.nn.functional as F


class SelfAttention(torch.nn.Module):
    """
    A general self attention module.
    Originally for Hi-Fi Ark.
    """
    def __init__(self):
        super(SelfAttention, self).__init__()

    def forward(self, candidate_vector):
        """
        Args:
            candidate_vector: batch_size, candidate_size, candidate_vector_dim
        Returns:
            (shape) batch_size, candidate_size, candidate_vector_dim
        """
        # batch_size, candidate_size, candidate_size
        weights = F.softmax(torch.bmm(candidate_vector,
                                      candidate_vector.transpose(1, 2)),
                            dim=2)
        # batch_size, candidate_size, candidate_vector_dim
        self_attended_vector = torch.bmm(weights, candidate_vector)
        return self_attended_vector
