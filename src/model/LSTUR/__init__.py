import torch
import torch.nn as nn
import torch.nn.functional as F
from model.LSTUR.news_encoder import NewsEncoder
from model.LSTUR.user_encoder import UserEncoder
from model.general.click_predictor.dot_product import DotProductClickPredictor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LSTUR(torch.nn.Module):
    """
    LSTUR network.
    Input 1 + K candidate news and a list of user clicked news, produce the click probability.
    """

    def __init__(self, config, pretrained_word_embedding=None, writer=None):
        """
        # ini
        user embedding: num_filters * 3
        news encoder: num_filters * 3
        GRU:
        input: num_filters * 3
        hidden: num_filters * 3

        # con
        user embedding: num_filter * 1.5
        news encoder: num_filters * 3
        GRU:
        input: num_fitlers * 3
        hidden: num_filter * 1.5
        """
        super(LSTUR, self).__init__()
        self.config = config
        self.news_encoder = NewsEncoder(config, pretrained_word_embedding)
        self.user_encoder = UserEncoder(config)
        self.click_predictor = DotProductClickPredictor()
        assert int(config.num_filters * 1.5) == config.num_filters * 1.5
        self.user_embedding = nn.Embedding(config.num_users,
                                           config.num_filters *
                                           3 if config.long_short_term_method == 'ini' else int(
                                               config.num_filters * 1.5),
                                           padding_idx=0)

    def forward(self, user, clicked_news_length, candidate_news, clicked_news):
        """
        Args:
            user: batch_size,
            clicked_news_length: batch_size,
            candidate_news:
                [
                    {
                        "category": Tensor(batch_size),
                        "subcategory": Tensor(batch_size),
                        "title": Tensor(batch_size) * num_words_title
                    } * (1 + K)
                ]
            clicked_news:
                [
                    {
                        "category": Tensor(batch_size),
                        "subcategory": Tensor(batch_size),
                        "title": Tensor(batch_size) * num_words_title
                    } * num_clicked_news_a_user
                ]
        Returns:
            click_probability: batch_size
        """
        # 1 + K, batch_size, num_filters * 3
        candidate_news_vector = torch.stack(
            [self.news_encoder(x) for x in candidate_news])
        # ini: batch_size, num_filters * 3
        # con: batch_size, num_filters * 1.5
        # TODO what if not drop
        user = F.dropout2d(self.user_embedding(user.to(device)).unsqueeze(dim=0),
                           p=self.config.masking_probability,
                           training=self.training).squeeze(dim=0)
        # batch_size, num_clicked_news_a_user, num_filters * 3
        clicked_news_vector = torch.stack(
            [self.news_encoder(x) for x in clicked_news], dim=1)
        # batch_size, num_filters * 3
        user_vector = self.user_encoder(user, clicked_news_length,
                                        clicked_news_vector)
        # batch_size, 1 + K
        click_probability = torch.stack([self.click_predictor(x,
                                                              user_vector) for x in candidate_news_vector], dim=1)
        return click_probability

    def get_news_vector(self, news):
        # batch_size, num_filters * 3
        return self.news_encoder(news)

    def get_user_vector(self, user, clicked_news_length, clicked_news_vector):
        """
        Args:
            user: batch_size
            clicked_news_length: batch_size
            clicked_news_vector: batch_size, num_clicked_news_a_user, num_filters * 3
        Returns:
            (shape) batch_size, num_filters * 3
        """
        # ini: batch_size, num_filters * 3
        # con: batch_size, num_filters * 1.5
        user = self.user_embedding(user.to(device))
        # batch_size, num_filters * 3
        return self.user_encoder(user, clicked_news_length,
                                 clicked_news_vector)

    def get_prediction(self, news_vector, user_vector):
        """
        Args:
            news_vector: num_filters * 3
            user_vector: num_filters * 3
        Returns:
            click_probability: 0-dim tensor
        """
        # 0-dim tensor
        click_probability = self.click_predictor(
            news_vector.unsqueeze(dim=0),
            user_vector.unsqueeze(dim=0)).squeeze(dim=0)
        return click_probability
