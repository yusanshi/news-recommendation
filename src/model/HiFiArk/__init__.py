import torch
from model.HiFiArk.news_encoder import NewsEncoder
from model.HiFiArk.OMAP import OMAP
from model.general.click_predictor.DNN import DNNClickPredictor
from model.general.attention.self import SelfAttention
from model.general.attention.similarity import SimilarityAttention


class HiFiArk(torch.nn.Module):
    """
    Hi-Fi Ark network.
    Input 1 + K candidate news and a list of user clicked news, produce the click probability.
    """

    def __init__(self, config, pretrained_word_embedding=None, writer=None):
        super(HiFiArk, self).__init__()
        self.config = config
        self.news_encoder = NewsEncoder(config, pretrained_word_embedding)
        self.self_attention = SelfAttention()
        self.omap = OMAP(config)
        self.similarity_attention = SimilarityAttention()
        self.click_predictor = DNNClickPredictor(config.num_filters * 2)

    def forward(self, candidate_news, clicked_news):
        """
        Args:
            candidate_news:
                [
                    {
                        "title": Tensor(batch_size) * num_words_title
                    } * (1 + K)
                ]
            clicked_news:
                [
                    {
                        "title": Tensor(batch_size) * num_words_title
                    } * num_clicked_news_a_user
                ]
        Returns:
            click_probability: batch_size, 1 + K
            regularizer_loss: 0-dim tensor
        """
        # 1 + K, batch_size, num_filters
        candidate_news_vector = torch.stack(
            [self.news_encoder(x) for x in candidate_news])
        # batch_size, num_clicked_news_a_user, num_filters
        clicked_news_vector = torch.stack(
            [self.news_encoder(x) for x in clicked_news], dim=1)
        # batch_size, num_clicked_news_a_user, num_filters
        self_attended_clicked_news_vector = torch.add(
            self.self_attention(clicked_news_vector), clicked_news_vector)
        # batch_size, num_pooling_heads, num_filters
        user_archive_vector, regularizer_loss = self.omap(
            self_attended_clicked_news_vector)
        # 1 + K, batch_size, num_filters
        user_vector = torch.stack([self.similarity_attention(x,
                                                             user_archive_vector) for x in candidate_news_vector])
        # batch_size, 1 + K
        click_probability = torch.stack([self.click_predictor(x,
                                                              y) for x, y in zip(candidate_news_vector, user_vector)], dim=1)
        return click_probability, regularizer_loss

    def get_news_vector(self, news):
        """
        Args:
            news:
                {
                    "title": Tensor(batch_size) * num_words_title
                }
        Returns:
            (shape) batch_size, num_filters
        """
        # batch_size, num_filters
        return self.news_encoder(news)

    def get_user_vector(self, clicked_news_vector):
        """
        Args:
            clicked_news_vector: batch_size, num_clicked_news_a_user, num_filters
        Returns:
            (shape) batch_size, num_pooling_heads, num_filters
        """
        # batch_size, num_clicked_news_a_user, num_filters
        self_attended_clicked_news_vector = torch.add(
            self.self_attention(clicked_news_vector), clicked_news_vector)
        # batch_size, num_pooling_heads, num_filters
        user_archive_vector, _ = self.omap(self_attended_clicked_news_vector)
        return user_archive_vector

    def get_prediction(self, candidate_news_vector, user_archive_vector):
        """
        Args:
            candidate_news_vector: num_filters
            user_archive_vector: num_pooling_heads, num_filters
        Returns:
            click_probability: 0-dim tensor
        """
        # 1, num_filters
        user_vector = self.similarity_attention(
            candidate_news_vector.unsqueeze(dim=0),
            user_archive_vector.unsqueeze(dim=0))
        # 0-dim tensor
        click_probability = self.click_predictor(
            candidate_news_vector.unsqueeze(dim=0), user_vector).squeeze(dim=0)
        return click_probability
