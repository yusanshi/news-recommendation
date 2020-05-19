import torch
import torch.nn as nn


class UserEncoder(torch.nn.Module):
    def __init__(self, config):
        super(UserEncoder, self).__init__()
        self.config = config
        self.gru = nn.GRU(config.num_filters * 4, config.num_filters * 4)

    def forward(self, user, clicked_news_vector):
        """
        Args:
            user: batch_size, num_filters * 4
            clicked_news_vector: batch_size, num_clicked_news_a_user, num_filters * 4
        Returns:
            (shape) batch_size, num_filters * 4
        """
        # 1, batch_size, num_filters * 4
        if self.config.long_short_term_method == 'ini':
            _, last_hidden = self.gru(clicked_news_vector.transpose(0, 1),
                                      user.unsqueeze(dim=0))
        else:
            _, last_hidden = self.gru(clicked_news_vector.transpose(0, 1))
            # TODO
        return last_hidden.squeeze(dim=0)
