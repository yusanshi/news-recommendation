import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class OMAP(torch.nn.Module):
    def __init__(self, config):
        super(OMAP, self).__init__()
        self.config = config
        self.W = nn.Parameter(
            torch.empty(config.num_filters,
                        config.num_pooling_heads).uniform_(-0.1, 0.1))

    def forward(self, self_attended_clicked_news_vector):
        """
        Args:
            self_attended_clicked_news_vector: batch_size, num_clicked_news_a_user, num_filters
        Returns:
            user_archive_vector: batch_size, num_pooling_heads, num_filters
            regularizer_loss: 0-dim tensor
        """
        # TODO pooling head installment

        # In case that drop_last=False, when real_batch_size != config.batch_size
        real_batch_size = self_attended_clicked_news_vector.size(0)
        # batch_size, num_pooling_heads, num_clicked_news_a_user
        weights = F.softmax(torch.bmm(self_attended_clicked_news_vector, self.W.expand(real_batch_size, -1,
                                                                                       -1)).transpose(1, 2),
                            dim=2)
        # batch_size, num_pooling_heads, num_filters
        user_archive_vector = torch.bmm(weights,
                                        self_attended_clicked_news_vector)

        if self.training:
            # num_pooling_heads, num_pooling_heads
            left = torch.mm(self.W.transpose(0, 1), self.W)
            # num_pooling_heads, num_pooling_heads
            right = (torch.ones(self.config.num_pooling_heads,
                                self.config.num_pooling_heads) - torch.eye(self.config.num_pooling_heads)).to(device)
            regularizer_loss = torch.mul(left, right).norm(p='fro')
        else:
            regularizer_loss = None
        return user_archive_vector, regularizer_loss
