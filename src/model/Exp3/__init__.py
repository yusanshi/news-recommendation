import torch
import torch.nn as nn
from model.Exp3.news_encoder import NewsEncoder
from model.Exp3.user_encoder import UserEncoder
from model.general.click_predictor.dot_product import DotProductClickPredictor


class Exp3(torch.nn.Module):
    def __init__(self, config, pretrained_word_embedding=None):
        super(Exp3, self).__init__()
        self.config = config
        if pretrained_word_embedding is None:
            word_embedding = nn.Embedding(config.num_words,
                                          config.word_embedding_dim,
                                          padding_idx=0)
        else:
            word_embedding = nn.Embedding.from_pretrained(
                pretrained_word_embedding, freeze=False, padding_idx=0)
        self.news_encoder = NewsEncoder(config, word_embedding)
        self.user_encoder = UserEncoder(config)
        self.popularity_linear = nn.Linear(config.word_embedding_dim, 1)
        self.eta_linear = nn.Linear(config.word_embedding_dim, 1)
        self.click_predictor = DotProductClickPredictor()

    def forward(self, candidate_news, clicked_news):
        """
        Args:
            candidate_news:
                [
                    {
                        "title": batch_size * num_words_title
                    } * (1 + K)
                ]
            clicked_news:
                [
                    {
                        "title":batch_size * num_words_title
                    } * num_clicked_news_a_user
                ]
        Returns:
          click_probability: batch_size, 1 + K
        """
        # batch_size, 1 + K, word_embedding_dim
        candidate_news_vector = torch.stack(
            [self.news_encoder(x) for x in candidate_news], dim=1)
        # batch_size, num_clicked_news_a_user, word_embedding_dim
        clicked_news_vector = torch.stack(
            [self.news_encoder(x) for x in clicked_news], dim=1)
        clicked_news_popularity_vector = clicked_news_vector  # TODO
        # batch_size, word_embedding_dim
        user_vector = self.user_encoder(clicked_news_vector,
                                        clicked_news_popularity_vector)
        # batch_size, 1 + K
        candidate_news_popularity_score = self.popularity_linear(
            candidate_news_vector).squeeze(dim=-1)
        # batch_size
        eta = torch.sigmoid(self.eta_linear(user_vector).squeeze(dim=-1))
        # batch_size, 1 + K
        eta = eta.unsqueeze(dim=-1).expand_as(candidate_news_popularity_score)
        # batch_size, 1 + K
        click_probability = (1 - eta) * self.click_predictor(
            candidate_news_vector,
            user_vector) + eta * candidate_news_popularity_score

        return click_probability

    def get_news_vector(self, news):
        """
        Args:
            news:
                {
                    "title": batch_size * num_words_title
                },
        Returns:
            (shape) batch_size, word_embedding_dim
        """
        # batch_size, word_embedding_dim
        return self.news_encoder(news)

    def get_user_vector(self, clicked_news_vector,
                        clicked_news_popularity_vector):
        """
        Args:
            clicked_news_vector: batch_size, num_clicked_news_a_user, word_embedding_dim
        Returns:
            (shape) batch_size, word_embedding_dim
        """
        # batch_size, word_embedding_dim
        return self.user_encoder(clicked_news_vector,
                                 clicked_news_popularity_vector)

    def get_prediction(self, news_vector, user_vector):
        """
        Args:
            news_vector: candidate_size, word_embedding_dim
            user_vector: word_embedding_dim
        Returns:
            click_probability: candidate_size
        """
        # candidate_size
        return self.click_predictor(
            news_vector.unsqueeze(dim=0),
            user_vector.unsqueeze(dim=0)).squeeze(dim=0)
