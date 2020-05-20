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
    """
    TODO
    """

    def __init__(self, behaviors_path, news_path, attributes):
        super(BaseDataset, self).__init__()
        self.attributes = attributes
        assert all(attribute in ['user', 'category', 'subcategory', 'title', 'abstract',
                                 'title_entities', 'abstract_entities'] for attribute in attributes)
        self.behaviors_parsed = pd.read_table(behaviors_path)
        self.news_parsed = pd.read_table(news_path,
                                         index_col='id',
                                         converters={
                                             attribute: literal_eval
                                             for attribute in set(attributes) & ['title', 'abstract', 'title_entities', 'abstract_entities']
                                         })

    def __len__(self):
        return len(self.behaviors_parsed)

    def __getitem__(self, idx):

        def news2dict(news, df):
            pass


class NRMSDataset(Dataset):
    def __init__(self, behaviors_path, news_path):
        super(NRMSDataset, self).__init__()
        self.behaviors_parsed = pd.read_table(behaviors_path)
        self.news_parsed = pd.read_table(news_path,
                                         index_col='id',
                                         converters={'title': literal_eval})

    def __len__(self):
        return len(self.behaviors_parsed)

    def __getitem__(self, idx):

        def news2dict(news, df):
            return {
                "title": df.loc[news].title
            } if news in df.index else {
                "title": [0] * Config.num_words_title
            }

        item = {}
        row = self.behaviors_parsed.iloc[idx]
        item["user"] = row.user
        item["clicked"] = row.clicked
        item["candidate_news"] = news2dict(
            row.candidate_news, self.news_parsed)
        item["clicked_news"] = [
            news2dict(x, self.news_parsed)
            for x in row.clicked_news.split()[:Config.num_clicked_news_a_user]
        ]

        padding = {"title": [0] * Config.num_words_title}

        repeated_times = Config.num_clicked_news_a_user - \
            len(item["clicked_news"])
        assert repeated_times >= 0
        item["clicked_news"].extend([padding] * repeated_times)
        return item


class NAMLDataset(Dataset):
    def __init__(self, behaviors_path, news_path):
        super(NAMLDataset, self).__init__()
        self.behaviors_parsed = pd.read_table(behaviors_path)
        self.news_parsed = pd.read_table(news_path,
                                         index_col='id',
                                         converters={
                                             'title': literal_eval,
                                             'abstract': literal_eval
                                         })

    def __len__(self):
        return len(self.behaviors_parsed)

    def __getitem__(self, idx):
        """
        example:
            {
                clicked: 0
                candidate_news:
                    {
                        "category": 0,
                        "subcategory": 0,
                        "title": [0] * num_words_title,
                        "abstract": [0] * num_words_abstract
                    }
                clicked_news:
                    [
                        {
                            "category": 0,
                            "subcategory": 0,
                            "title": [0] * num_words_title,
                            "abstract": [0] * num_words_abstract
                        } * num_clicked_news_a_user
                    ]
            }
        """
        def news2dict(news, df):
            return {
                "category": df.loc[news].category,
                "subcategory": df.loc[news].subcategory,
                "title": df.loc[news].title,
                "abstract": df.loc[news].abstract
            } if news in df.index else {
                "category": 0,
                "subcategory": 0,
                "title": [0] * Config.num_words_title,
                "abstract": [0] * Config.num_words_abstract
            }

        item = {}
        row = self.behaviors_parsed.iloc[idx]
        item["clicked"] = row.clicked
        item["candidate_news"] = news2dict(row.candidate_news,
                                           self.news_parsed)
        item["clicked_news"] = [
            news2dict(x, self.news_parsed)
            for x in row.clicked_news.split()[:Config.num_clicked_news_a_user]
        ]
        padding = {
            "category": 0,
            "subcategory": 0,
            "title": [0] * Config.num_words_title,
            "abstract": [0] * Config.num_words_abstract
        }
        repeated_times = Config.num_clicked_news_a_user - \
            len(item["clicked_news"])
        assert repeated_times >= 0
        item["clicked_news"].extend([padding] * repeated_times)

        return item


class LSTURDataset(Dataset):
    def __init__(self, behaviors_path, news_path):
        super(LSTURDataset, self).__init__()
        self.behaviors_parsed = pd.read_table(behaviors_path)
        self.news_parsed = pd.read_table(news_path,
                                         index_col='id',
                                         converters={
                                             'title': literal_eval,
                                             'abstract': literal_eval
                                         })

    def __len__(self):
        return len(self.behaviors_parsed)

    def __getitem__(self, idx):
        """
        example:
            {
                user: 1,
                clicked: 0,
                candidate_news:
                    {
                        "category": 0,
                        "subcategory": 0,
                        "title": [0] * num_words_title,
                        "abstract": [0] * num_words_abstract
                    },
                clicked_news:
                    [
                        {
                            "category": 0,
                            "subcategory": 0,
                            "title": [0] * num_words_title,
                            "abstract": [0] * num_words_abstract
                        } * num_clicked_news_a_user
                    ]
            }
        """
        def news2dict(news, df):
            return {
                "category": df.loc[news].category,
                "subcategory": df.loc[news].subcategory,
                "title": df.loc[news].title,
                "abstract": df.loc[news].abstract
            } if news in df.index else {
                "category": 0,
                "subcategory": 0,
                "title": [0] * Config.num_words_title,
                "abstract": [0] * Config.num_words_abstract
            }

        item = {}
        row = self.behaviors_parsed.iloc[idx]
        item["user"] = row.user
        item["clicked"] = row.clicked
        item["candidate_news"] = news2dict(row.candidate_news,
                                           self.news_parsed)
        item["clicked_news"] = [
            news2dict(x, self.news_parsed)
            for x in row.clicked_news.split()[:Config.num_clicked_news_a_user]
        ]
        padding = {
            "category": 0,
            "subcategory": 0,
            "title": [0] * Config.num_words_title,
            "abstract": [0] * Config.num_words_abstract
        }
        repeated_times = Config.num_clicked_news_a_user - \
            len(item["clicked_news"])
        assert repeated_times >= 0
        item["clicked_news"].extend([padding] * repeated_times)

        return item


class DKNDataset(Dataset):
    def __init__(self, behaviors_path, news_path):
        super(DKNDataset, self).__init__()
        self.behaviors_parsed = pd.read_table(behaviors_path)
        self.news_parsed = pd.read_table(news_path,
                                         index_col='id',
                                         converters={
                                             'title': literal_eval,
                                             'title_entities': literal_eval
                                         })

    def __len__(self):
        return len(self.behaviors_parsed)

    def __getitem__(self, idx):
        """
        example:
            {
                clicked: 0
                candidate_news:
                    {
                        "title": [0] * num_words_title,
                        "title_entities": [0] * num_words_title
                    }
                clicked_news:
                    [
                        {
                            "title": [0] * num_words_title,
                            "title_entities": [0] * num_words_title
                        } * num_clicked_news_a_user
                    ]
            }
        """
        def news2dict(news, df):
            return {
                "title": df.loc[news].title,
                "title_entities": df.loc[news].title_entities
            } if news in df.index else {
                "title": [0] * Config.num_words_title,
                "title_entities": [0] * Config.num_words_title
            }

        item = {}
        row = self.behaviors_parsed.iloc[idx]
        item["clicked"] = row.clicked
        item["candidate_news"] = news2dict(row.candidate_news,
                                           self.news_parsed)
        item["clicked_news"] = [
            news2dict(x, self.news_parsed)
            for x in row.clicked_news.split()[:Config.num_clicked_news_a_user]
        ]
        padding = {
            "title": [0] * Config.num_words_title,
            "title_entities": [0] * Config.num_words_title
        }
        repeated_times = Config.num_clicked_news_a_user - \
            len(item["clicked_news"])
        assert repeated_times >= 0
        item["clicked_news"].extend([padding] * repeated_times)

        return item
