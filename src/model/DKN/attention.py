import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(torch.nn.Module):
    """
    Attention Net.
    Input embedding vectors (produced by KCNN) of a candidate news and all of user's clicked news,
    produce final user embedding vectors with respect to the candidate news.
    """
    def __init__(self, config):
        super(Attention, self).__init__()
        self.config = config
        self.dnn = nn.Sequential(
            nn.Linear(
                len(self.config.window_sizes) * 2 * self.config.num_filters,
                16), nn.Linear(16, 1))

    def forward(self, candidate_news_vector, clicked_news_vector):
        """
        Args:
            candidate_news_vector: batch_size, len(window_sizes) * num_filters
            clicked_news_vector: batch_size, num_clicked_news_a_user, len(window_sizes) * num_filters
        Returns:
            user_vector: batch_size, len(window_sizes) * num_filters
        """
        # batch_size, num_clicked_news_a_user
        clicked_news_weights = F.softmax(self.dnn(
            torch.cat((candidate_news_vector.expand(
                self.config.num_clicked_news_a_user, -1, -1).transpose(
                    0, 1), clicked_news_vector),
                      dim=2)).squeeze(dim=2),
                                         dim=1)

        # batch_size, len(window_sizes) * num_filters
        user_vector = torch.bmm(clicked_news_weights.unsqueeze(dim=1),
                                clicked_news_vector).squeeze(dim=1)
        return user_vector
