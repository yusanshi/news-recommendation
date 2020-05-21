from torch.utils.data import Dataset
import pandas as pd
from ast import literal_eval
from config import model_name

if model_name == 'NRMS':
    from config import NRMSConfig as Config
elif model_name == 'NAML':
    from config import NAMLConfig as Config
elif model_name == 'LSTUR':
    from config import LSTURConfig as Config
elif model_name == 'DKN':
    from config import DKNConfig as Config
else:
    print("Model name not included!")
    exit()


class BaseDataset(Dataset):
    def __init__(self, behaviors_path, news_path, attributes):
        super(BaseDataset, self).__init__()
        self.attributes = attributes
        assert all(attribute in [
            'user', 'category', 'subcategory', 'title', 'abstract',
            'title_entities', 'abstract_entities'
        ] for attribute in attributes)
        self.behaviors_parsed = pd.read_table(behaviors_path)
        self.news_parsed = pd.read_table(
            news_path,
            index_col='id',
            converters={
                attribute: literal_eval
                for attribute in set(attributes) & set([
                    'title', 'abstract', 'title_entities', 'abstract_entities'
                ])
            })
        self.padding = {}
        if 'category' in attributes:
            self.padding['category'] = 0
        if 'subcategory' in attributes:
            self.padding['subcategory'] = 0
        if 'title' in attributes:
            self.padding['title'] = [0] * Config.num_words_title
        if 'abstract' in attributes:
            self.padding['abstract'] = [0] * Config.num_words_abstract
        if 'title_entities' in attributes:
            self.padding['title_entities'] = [0] * Config.num_words_title
        if 'abstract_entities' in attributes:
            self.padding['abstract_entities'] = [0] * Config.num_words_abstract

    def __len__(self):
        return len(self.behaviors_parsed)

    def __getitem__(self, idx):
        def news2dict(news, df):
            return {
                key: df.loc[news][key]
                for key in self.attributes if key != 'user'
            } if news in df.index else self.padding

        item = {}
        row = self.behaviors_parsed.iloc[idx]
        if 'user' in self.attributes:
            item['user'] = row.user
        item["clicked"] = row.clicked
        item["candidate_news"] = news2dict(row.candidate_news,
                                           self.news_parsed)
        item["clicked_news"] = [
            news2dict(x, self.news_parsed)
            for x in row.clicked_news.split()[:Config.num_clicked_news_a_user]
        ]
        repeated_times = Config.num_clicked_news_a_user - \
            len(item["clicked_news"])
        assert repeated_times >= 0
        item["clicked_news"].extend([self.padding] * repeated_times)

        return item


class NRMSDataset(BaseDataset):
    def __init__(self, behaviors_path, news_path):
        super(NRMSDataset, self).__init__(behaviors_path, news_path, ['title'])


class NAMLDataset(BaseDataset):
    def __init__(self, behaviors_path, news_path):
        super(NAMLDataset,
              self).__init__(behaviors_path, news_path,
                             ['category', 'subcategory', 'title', 'abstract'])


class LSTURDataset(BaseDataset):
    def __init__(self, behaviors_path, news_path):
        super(LSTURDataset, self).__init__(
            behaviors_path, news_path,
            ['user', 'category', 'subcategory', 'title', 'abstract'])


class DKNDataset(BaseDataset):
    def __init__(self, behaviors_path, news_path):
        super(DKNDataset, self).__init__(behaviors_path, news_path,
                                         ['title', 'title_entities'])
