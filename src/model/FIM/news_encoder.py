import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class NewsEncoder(torch.nn.Module):
    def __init__(self, config, pretrained_word_embedding):
        super(NewsEncoder, self).__init__()
        self.config = config
        if pretrained_word_embedding is None:
            self.word_embedding = nn.Embedding(config.num_words,
                                               config.word_embedding_dim,
                                               padding_idx=0)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(
                pretrained_word_embedding, freeze=False, padding_idx=0)
        # TODO currently we assume num_filters in HDC
        # is equal to word embedding dim for simplicity
        assert config.word_embedding_dim == config.news_rep['num_filters']
        assert config.news_rep[
            'window_size'] >= 1 and config.news_rep['window_size'] % 2 == 1
        self.conv_filters = nn.ModuleDict({
            str(x): nn.Conv2d(
                1,
                config.news_rep['num_filters'],
                (config.news_rep['window_size'],
                 config.news_rep['num_filters']),
                padding=(int(x * (config.news_rep['window_size'] - 1) / 2), 0),
                dilation=(x, 1))
            for x in self.config.news_rep['dilations']
        })
        self.layer_norm = nn.LayerNorm(
            (config.num_words_title, config.news_rep['num_filters']))

    def forward(self, news):
        """
        Args:
            news:
                {
                    "title": Tensor(batch_size) * num_words_title
                }
        Returns:
            (shape) batch_size, 1 + len(dilations), num_words_title, num_filters (HDC)
        """
        # batch_size, num_words_title, word_embedding_dim (num_filters (HDC))
        title_vector = self.word_embedding(
            torch.stack(news['title'], dim=1).to(device))

        stacked_layers = [title_vector]
        last = title_vector
        for x in self.config.news_rep['dilations']:
            # batch_size, num_words_title, num_filters (HDC)
            convoluted = self.conv_filters[str(x)](
                last.unsqueeze(dim=1)).squeeze(dim=3).transpose(-1, -2)
            # batch_size, num_words_title, num_filters (HDC)
            last = self.layer_norm(F.relu(convoluted))
            # TODO layer normalization
            stacked_layers.append(last)
        # batch_size, 1 + len(dilations), num_words_title, num_filters (HDC)
        multi_grained = torch.stack(stacked_layers, dim=1)
        return multi_grained
