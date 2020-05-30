import torch
import torch.nn as nn
import torch.nn.functional as F


class OMAP(torch.nn.Module):
    def __init__(self, config):
        super(OMAP, self).__init__()
        self.config = config

    def forward(self, self_attended_clicked_news_vector, clicked_news_vector):
        """
        Args:
            self_attended_clicked_news_vector: batch_size, num_clicked_news_a_user, num_filters
            clicked_news_vector: batch_size, num_clicked_news_a_user, num_filters
        Returns:
            user_archive_vector: batch_size, num_pooling_heads, num_filters
            regularizer_loss: 0-dim tensor
        """
        # TODO
        # num_filters, num_pooling_heads
        W = torch.empty(self.config.num_filters,
                        self.config.num_pooling_heads).uniform_(-0.1, 0.1)

        # In case that drop_last=False, when real_batch_size != config.batch_size
        real_batch_size = self_attended_clicked_news_vector.size(0)
        # batch_size, num_pooling_heads, num_clicked_news_a_user
        weights = F.softmax(torch.bmm(self_attended_clicked_news_vector,
                                      W.expand(real_batch_size, -1,
                                               -1)).transpose(1, 2),
                            dim=2)
        # batch_size, num_pooling_heads, num_filters
        user_archive_vector = torch.bmm(weights,
                                        self_attended_clicked_news_vector)

        # The following does the same thing, simpler but not so efficient
        # archives = []
        # for pooling_head in W.transpose(0, 1):
        #     # batch_size, num_clicked_news_a_user
        #     candidate_weights = F.softmax(torch.matmul(
        #         self_attended_clicked_news_vector, pooling_head), dim=1)
        #     # batch_size, num_filters
        #     archives.append(torch.bmm(candidate_weights.unsqueeze(
        #         dim=1), self_attended_clicked_news_vector).squeeze(dim=1))
        # user_archive_vector = torch.stack(archives, dim=1)

        if self.training:
            regularizer_loss = torch.mul(torch.mm(W.transpose(0, 1), W), torch.ones(self.config.num_pooling_heads,
                                                                                    self.config.num_pooling_heads) - torch.eye(self.config.num_pooling_heads)).norm(p='fro')
        else:
            regularizer_loss = None
        return user_archive_vector, regularizer_loss
