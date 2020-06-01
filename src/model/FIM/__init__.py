import torch
from model.FIM.news_encoder import NewsEncoder
from model.FIM.aggregator import Aggregator


class FIM(torch.nn.Module):
    """
    FIM network.
    Input 1 + K candidate news and a list of user clicked news, produce the click probability.
    """

    def __init__(self, config, pretrained_word_embedding=None, writer=None):
        super(FIM, self).__init__()
        self.news_encoder = NewsEncoder(config, pretrained_word_embedding)
        self.aggregator = Aggregator(config)

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
            click_probability: batch_size
        """
        # 1 + K, batch_size, 1 + len(dilations), num_words_title, num_filters (HDC)
        candidate_news_vector = torch.stack(
            [self.news_encoder(x) for x in candidate_news])
        # batch_size, num_clicked_news_a_user, 1 + len(dilations), num_words_title, num_filters (HDC)
        clicked_news_vector = torch.stack(
            [self.news_encoder(x) for x in clicked_news], dim=1)
        # batch_size, 1 + K
        click_probability = torch.stack(
            [self.aggregator(x, clicked_news_vector) for x in candidate_news_vector], dim=1)
        return click_probability

    def get_news_vector(self, news):
        """
        Args:
            news:
                {
                    "title": Tensor(batch_size) * num_words_title
                }
        Returns:
            (shape) batch_size, 1 + len(dilations), num_words_title, num_filters (HDC)
        """
        return self.news_encoder(news)

    def get_user_vector(self, clicked_news_vector):
        """
        Args:
            clicked_news_vector: batch_size, num_clicked_news_a_user, 1 + len(dilations), num_words_title, num_filters (HDC)
        Returns:
            (shape) batch_size, num_clicked_news_a_user, 1 + len(dilations), num_words_title, num_filters (HDC)
        """
        return clicked_news_vector

    def get_prediction(self, candidate_news_vector, clicked_news_vector):
        """
        Args:
            candidate_news_vector: 1 + len(dilations), num_words_title, num_filters (HDC)
            clicked_news_vector: num_clicked_news_a_user, 1 + len(dilations), num_words_title, num_filters (HDC)
        Returns:
            click_probability: 0-dim tensor
        """
        click_probability = self.aggregator(
            candidate_news_vector.unsqueeze(dim=0),
            clicked_news_vector.unsqueeze(dim=0)).squeeze(dim=0)
        return click_probability
