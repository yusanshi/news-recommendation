import torch
from model.NAML.news_encoder import NewsEncoder
from model.NAML.user_encoder import UserEncoder
from model.general.click_predictor.dot_product import DotProductClickPredictor


class NAML(torch.nn.Module):
    """
    NAML network.
    Input 1 + K candidate news and a list of user clicked news, produce the click probability.
    """

    def __init__(self, config, pretrained_word_embedding=None, writer=None):
        super(NAML, self).__init__()
        self.config = config
        self.news_encoder = NewsEncoder(config, pretrained_word_embedding, writer)
        self.user_encoder = UserEncoder(config)
        self.click_predictor = DotProductClickPredictor()

    def forward(self, candidate_news, clicked_news):
        """
        Args:
            candidate_news:
                [
                    {
                        "category": Tensor(batch_size),
                        "subcategory": Tensor(batch_size),
                        "title": Tensor(batch_size) * num_words_title,
                        "abstract": Tensor(batch_size) * num_words_abstract
                    } * (1 + K)
                ]
            clicked_news:
                [
                    {
                        "category": Tensor(batch_size),
                        "subcategory": Tensor(batch_size),
                        "title": Tensor(batch_size) * num_words_title,
                        "abstract": Tensor(batch_size) * num_words_abstract
                    } * num_clicked_news_a_user
                ]
        Returns:
            click_probability: batch_size
        """
        # 1 + K, batch_size, num_filters
        candidate_news_vector = torch.stack(
            [self.news_encoder(x) for x in candidate_news])
        # batch_size, num_clicked_news_a_user, num_filters
        clicked_news_vector = torch.stack(
            [self.news_encoder(x) for x in clicked_news], dim=1)
        # batch_size, num_filters
        user_vector = self.user_encoder(clicked_news_vector)
        # batch_size, 1 + K
        click_probability = torch.stack([self.click_predictor(x,
                                                              user_vector) for x in candidate_news_vector], dim=1)
        return click_probability

    def get_news_vector(self, news):
        """
        Args:
            news:
                {
                    "category": Tensor(batch_size),
                    "subcategory": Tensor(batch_size),
                    "title": Tensor(batch_size) * num_words_title,
                    "abstract": Tensor(batch_size) * num_words_abstract
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
            (shape) batch_size, num_filters
        """
        # batch_size, num_filters
        return self.user_encoder(clicked_news_vector)

    def get_prediction(self, news_vector, user_vector):
        """
        Args:
            news_vector: num_filters
            user_vector: num_filters
        Returns:
            click_probability: 0-dim tensor
        """
        # 0-dim tensor
        click_probability = self.click_predictor(
            news_vector.unsqueeze(dim=0),
            user_vector.unsqueeze(dim=0)).squeeze(dim=0)
        return click_probability
