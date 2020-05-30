import torch


class DotProductClickPredictor(torch.nn.Module):
    def __init__(self):
        super(DotProductClickPredictor, self).__init__()

    def forward(self, candidate_news_vector, user_vector):
        """
        Args:
            candidate_news_vector: batch_size, X
            user_vector: batch_size, X
        Returns:
            (shape): batch_size
        """
        # batch_size
        probability = torch.bmm(
            user_vector.unsqueeze(dim=1),
            candidate_news_vector.unsqueeze(dim=2)).flatten()
        return probability
