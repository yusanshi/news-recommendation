import torch
from model.DKN.KCNN import KCNN
from model.DKN.attention import Attention
from model.general.click_predictor.DNN import DNNClickPredictor


class DKN(torch.nn.Module):
    """
    Deep knowledge-aware network.
    Input 1 + K candidate news and a list of user clicked news, produce the click probability.
    """
    def __init__(self,
                 config,
                 pretrained_word_embedding=None,
                 pretrained_entity_embedding=None,
                 pretrained_context_embedding=None,
                 writer=None):
        super(DKN, self).__init__()
        self.config = config
        self.kcnn = KCNN(config, pretrained_word_embedding,
                         pretrained_entity_embedding,
                         pretrained_context_embedding)
        self.attention = Attention(config)
        self.click_predictor = DNNClickPredictor(
            len(self.config.window_sizes) * 2 * self.config.num_filters)

    def forward(self, candidate_news, clicked_news):
        """
        Args:
            candidate_news:
                [
                    {
                        "title": Tensor(batch_size) * num_words_title,
                        "title_entities": Tensor(batch_size) * num_words_title
                    } * (1 + K)
                ]
            clicked_news:
                [
                    {
                        "title": Tensor(batch_size) * num_words_title,
                        "title_entities": Tensor(batch_size) * num_words_title
                    } * num_clicked_news_a_user
                ]
        Returns:
            click_probability: batch_size
        """
        # 1 + K, batch_size, len(window_sizes) * num_filters
        candidate_news_vector = torch.stack(
            [self.kcnn(x) for x in candidate_news])
        # batch_size, num_clicked_news_a_user, len(window_sizes) * num_filters
        clicked_news_vector = torch.stack([self.kcnn(x) for x in clicked_news],
                                          dim=1)
        # 1 + K, batch_size, len(window_sizes) * num_filters
        user_vector = torch.stack([
            self.attention(x, clicked_news_vector)
            for x in candidate_news_vector
        ])
        # batch_size, 1 + K
        click_probability = torch.stack([
            self.click_predictor(x, y)
            for (x, y) in zip(candidate_news_vector, user_vector)
        ],
                                        dim=1)
        return click_probability

    def get_news_vector(self, news):
        """
        Args:
            news:
                {
                    "title": Tensor(batch_size) * num_words_title,
                    "title_entities": Tensor(batch_size) * num_words_title
                }
        Returns:
            (shape) batch_size, len(window_sizes) * num_filters
        """
        # batch_size, len(window_sizes) * num_filters
        return self.kcnn(news)

    def get_user_vector(self, clicked_news_vector):
        """
        Args:
            clicked_news_vector: batch_size, num_clicked_news_a_user, len(window_sizes) * num_filters
        Returns:
            (shape) batch_size, num_clicked_news_a_user, len(window_sizes) * num_filters
        """
        # batch_size, num_clicked_news_a_user, len(window_sizes) * num_filters
        return clicked_news_vector

    def get_prediction(self, candidate_news_vector, clicked_news_vector):
        """
        Args:
            candidate_news_vector: len(window_sizes) * num_filters
            clicked_news_vector: num_clicked_news_a_user, len(window_sizes) * num_filters
        Returns:
            click_probability: 0-dim tensor
        """
        # 1, len(window_sizes) * num_filters
        user_vector = self.attention(candidate_news_vector.unsqueeze(dim=0),
                                     clicked_news_vector.unsqueeze(dim=0))
        # 0-dim tensor
        click_probability = self.click_predictor(
            candidate_news_vector.unsqueeze(dim=0), user_vector).squeeze(dim=0)
        return click_probability
