import torch
import torch.nn.functional as F


class SimilarityAttention(torch.nn.Module):
    """
    A general attention module based on similarity w.r.t. another vector
    """
    def __init__(self):
        super(SimilarityAttention, self).__init__()

    def forward(self, wrt_vector, candidate_vector):
        """
        Args:
            wrt_vector: batch_size, candidate_vector_dim
            candidate_vector: batch_size, candidate_size, candidate_vector_dim
        Returns:
            (shape) batch_size, candidate_vector_dim
        """
        # batch_size, candidate_size
        candidate_weights = F.softmax(torch.bmm(
            candidate_vector, wrt_vector.unsqueeze(dim=2)).squeeze(dim=2),
                                      dim=1)
        # batch_size, candidate_vector_dim
        target = torch.bmm(candidate_weights.unsqueeze(dim=1),
                           candidate_vector).squeeze(dim=1)
        return target
