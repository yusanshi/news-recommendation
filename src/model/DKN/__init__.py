import torch
import torch.nn as nn
from model.DKN.kcnn import KCNN
from model.DKN.attention import Attention


class DKN(torch.nn.Module):
    """
    Deep knowledge-aware network.
    Input a candidate news and a list of user clicked news, produce the click probability.
    """

    def __init__(self, config, pretrained_word_embedding,
                 pretrained_entity_embedding, pretrained_context_embedding):
        super(DKN, self).__init__()
        self.config = config
        self.kcnn = KCNN(config, pretrained_word_embedding,
                         pretrained_entity_embedding,
                         pretrained_context_embedding)
        self.attention = Attention(config)
        self.dnn = nn.Sequential(
            nn.Linear(
                len(self.config.window_sizes) * 2 * self.config.num_filters,
                16), nn.Linear(16, 1))

    def forward(self, candidate_news, clicked_news):
        """
        Args:
          candidate_news:
            {
                "title": [Tensor(batch_size) * num_words_title],
                "title_entities":[Tensor(batch_size) * num_words_title]
            }
          clicked_news:
            [
                {
                    "title": [Tensor(batch_size) * num_words_title],
                    "title_entities":[Tensor(batch_size) * num_words_title]
                } * num_clicked_news_a_user
            ]
        Returns:
          click_probability: batch_size
        """
        # batch_size, len(window_sizes) * num_filters
        candidate_news_vector = self.kcnn(candidate_news)
        # batch_size, num_clicked_news_a_user, len(window_sizes) * num_filters
        clicked_news_vector = torch.stack(
            [self.kcnn(x) for x in clicked_news], dim=1)
        # batch_size, len(window_sizes) * num_filters
        user_vector = self.attention(candidate_news_vector,
                                     clicked_news_vector)
        # Sigmoid is done with BCEWithLogitsLoss
        # batch_size
        click_probability = self.dnn(
            torch.cat((user_vector, candidate_news_vector),
                      dim=1)).squeeze(dim=1)
        return click_probability

    def get_news_vector(self, news):
        # batch_size, len(window_sizes) * num_filters
        return self.kcnn(news)

    def get_user_vector(self, clicked_news_vector):
        """
        clicked_news_vector: batch_size, num_clicked_news_a_user, len(window_sizes) * num_filters
        """
        # batch_size, num_clicked_news_a_user, len(window_sizes) * num_filters
        return clicked_news_vector

    def get_prediction(self, candidate_news_vector, clicked_news_vector):
        """
        candidate_news_vector: batch_size, len(window_sizes) * num_filters
        clicked_news_vector: batch_size, num_clicked_news_a_user, len(window_sizes) * num_filters
        """
        # batch_size, len(window_sizes) * num_filters
        user_vector = self.attention(candidate_news_vector,
                                     clicked_news_vector)
        # batch_size
        click_probability = self.dnn(
            torch.cat((user_vector, candidate_news_vector),
                      dim=1)).squeeze(dim=1)
        return click_probability
