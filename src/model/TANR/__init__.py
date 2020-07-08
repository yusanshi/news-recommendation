import torch
import torch.nn as nn
from model.TANR.news_encoder import NewsEncoder
from model.TANR.user_encoder import UserEncoder
from model.general.click_predictor.dot_product import DotProductClickPredictor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TANR(torch.nn.Module):
    """
    TANR network.
    Input 1 + K candidate news and a list of user clicked news, produce the click probability.
    """
    def __init__(self, config, pretrained_word_embedding=None, writer=None):
        super(TANR, self).__init__()
        self.config = config
        self.news_encoder = NewsEncoder(config, pretrained_word_embedding)
        self.user_encoder = UserEncoder(config)
        self.click_predictor = DotProductClickPredictor()
        self.topic_predictor = nn.Linear(config.num_filters,
                                         config.num_categories)

    def forward(self, candidate_news, clicked_news):
        """
        Args:
            candidate_news:
                [
                    {
                        "category": Tensor(batch_size),
                        "title": Tensor(batch_size) * num_words_title
                    } * (1 + K)
                ]
            clicked_news:
                [
                    {
                        "category": Tensor(batch_size),
                        "title": Tensor(batch_size) * num_words_title
                    } * num_clicked_news_a_user
                ]
        Returns:
            click_probability: batch_size, 1 + K
            topic_classification_loss: 0-dim tensor
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
        click_probability = torch.stack([
            self.click_predictor(x, user_vector) for x in candidate_news_vector
        ],
                                        dim=1)

        # batch_size * (1 + K + num_clicked_news_a_user), num_categories
        y_pred = self.topic_predictor(
            torch.cat(
                (candidate_news_vector.transpose(0, 1), clicked_news_vector),
                dim=1).view(-1, self.config.num_filters))
        # batch_size * (1 + K + num_clicked_news_a_user)
        y = torch.stack([x['category'] for x in candidate_news + clicked_news],
                        dim=1).flatten().to(device)
        class_weight = torch.ones(self.config.num_categories).to(device)
        class_weight[0] = 0
        criterion = nn.CrossEntropyLoss(weight=class_weight)
        topic_classification_loss = criterion(y_pred, y)

        return click_probability, topic_classification_loss

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
