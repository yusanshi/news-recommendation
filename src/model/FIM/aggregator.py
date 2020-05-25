import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Aggregator(torch.nn.Module):
    def __init__(self, config):
        super(Aggregator, self).__init__()
        self.config = config
        in_channels = 1 + len(config.news_rep['dilations'])
        layers = []
        for layer in self.config.cross_matching['layers']:
            layers.append(
                nn.Conv3d(in_channels,
                          layer['num_filters'],
                          layer['window_size'],
                          stride=layer['stride']))
            in_channels = layer['num_filters']

        self.conv_filters = nn.ModuleList(layers)
        self.max_pooling = nn.MaxPool3d(
            config.cross_matching['max_pooling']['window_size'],
            stride=config.cross_matching['max_pooling']['stride'])
        self.dnn = nn.Sequential(
            # TODO magic number 6000 should be expressed with configuration parameters
            nn.Linear(6000, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, candidate_news_vector, clicked_news_vector):
        """
        Args:
            candidate_news_vector: batch_size, 1 + len(dilations), num_words_title, num_filters (HDC)
            clicked_news_vector: batch_size, num_clicked_news_a_user, 1 + len(dilations), num_words_title, num_filters (HDC)
        Returns:
            (shape) batch_size
        """
        # batch_size, 1 + len(dilations), num_clicked_news_a_user, num_words_title, num_words_title
        matching_matrices = torch.matmul(
            clicked_news_vector,
            candidate_news_vector.expand(
                self.config.num_clicked_news_a_user, -1, -1, -1,
                -1).transpose(0, 1).transpose(-1, -2)).div(
                    self.config.news_rep['num_filters']).transpose(1, 2)
        for layer in self.conv_filters:
            matching_matrices = layer(matching_matrices)

        pooled = self.max_pooling(matching_matrices)
        # batch_size, X
        integrated_matching_vector = pooled.view(pooled.size(0), -1)
        # batch_size
        click_probability = self.dnn(integrated_matching_vector).squeeze(dim=1)
        return click_probability
