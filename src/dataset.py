from torch.utils.data import Dataset
import pandas as pd
from ast import literal_eval
from config import model_name
import importlib

try:
    config = getattr(importlib.import_module('config'), f"{model_name}Config")
except AttributeError:
    print(f"{model_name} not included!")
    exit()


class BaseDataset(Dataset):
    def __init__(self, behaviors_path, news_path, attributes):
        super(BaseDataset, self).__init__()
        self.attributes = attributes
        assert all(attribute in [
            'category', 'subcategory', 'title', 'abstract', 'title_entities',
            'abstract_entities'
        ] for attribute in attributes['news'])
        assert all(attribute in ['user', 'clicked_news_length']
                   for attribute in attributes['record'])

        self.behaviors_parsed = pd.read_table(behaviors_path)
        self.news_parsed = pd.read_table(
            news_path,
            index_col='id',
            converters={
                attribute: literal_eval
                for attribute in set(attributes['news']) & set([
                    'title', 'abstract', 'title_entities', 'abstract_entities'
                ])
            })
        self.padding = {}
        if 'category' in attributes['news']:
            self.padding['category'] = 0
        if 'subcategory' in attributes['news']:
            self.padding['subcategory'] = 0
        if 'title' in attributes['news']:
            self.padding['title'] = [0] * config.num_words_title
        if 'abstract' in attributes['news']:
            self.padding['abstract'] = [0] * config.num_words_abstract
        if 'title_entities' in attributes['news']:
            self.padding['title_entities'] = [0] * config.num_words_title
        if 'abstract_entities' in attributes['news']:
            self.padding['abstract_entities'] = [0] * config.num_words_abstract

    def __len__(self):
        return len(self.behaviors_parsed)

    def __getitem__(self, idx):
        def news2dict(news, df):
            return {key: df.loc[news][key]
                    for key in self.attributes['news']
                    } if news in df.index else self.padding

        item = {}
        row = self.behaviors_parsed.iloc[idx]
        if 'user' in self.attributes['record']:
            item['user'] = row.user
        item["clicked"] = list(map(int, row.clicked.split()))
        item["candidate_news"] = [
            news2dict(x, self.news_parsed) for x in row.candidate_news.split()
        ]
        item["clicked_news"] = [
            news2dict(x, self.news_parsed)
            for x in row.clicked_news.split()[:config.num_clicked_news_a_user]
        ]
        if 'clicked_news_length' in self.attributes['record']:
            item['clicked_news_length'] = len(item["clicked_news"])
        repeated_times = config.num_clicked_news_a_user - \
            len(item["clicked_news"])
        assert repeated_times >= 0
        item["clicked_news"].extend([self.padding] * repeated_times)

        return item
