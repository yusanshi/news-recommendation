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


class NRMSDataset(Dataset):
    def __init__(self, behaviors_path, news_path):
        super(NRMSDataset, self).__init__()
        self.behaviors = pd.read_table(behaviors_path)
        self.news_parsed = pd.read_table(news_path,
                                         index_col='id',
                                         converters={'title': literal_eval})

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        """
        example:
            {
                clicked: 0
                candidate_news:
                    [0] * num_words_title,
                clicked_news:
                    [[0] * num_words_title]] * num_clicked_news_a_user
            }
        """
        def id2title(news, df):
            return df.loc[news].title if news in df.index else [
                0
            ] * Config.num_words_title

        item = {}
        row = self.behaviors.iloc[idx]
        item["user"] = row.user
        item["clicked"] = row.clicked
        item["candidate_news"] = id2title(row.candidate_news, self.news_parsed)
        item["clicked_news"] = [
            id2title(x, self.news_parsed)
            for x in row.clicked_news.split()[:Config.num_clicked_news_a_user]
        ]

        padding = [0] * Config.num_words_title

        repeated_times = Config.num_clicked_news_a_user - \
            len(item["clicked_news"])
        assert repeated_times >= 0
        item["clicked_news"].extend([padding] * repeated_times)
        return item


class NAMLDataset(Dataset):
    def __init__(self, behaviors_path, news_path):
        super(NAMLDataset, self).__init__()
        self.behaviors = pd.read_table(behaviors_path)
        self.news_parsed = pd.read_table(news_path,
                                         index_col='id',
                                         converters={
                                             'title': literal_eval,
                                             'abstract': literal_eval
                                         })

    def __len__(self):
        return len(self.behaviors)

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
        row = self.behaviors.iloc[idx]
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
        self.behaviors = pd.read_table(behaviors_path)
        self.news_parsed = pd.read_table(news_path,
                                         index_col='id',
                                         converters={
                                             'title': literal_eval,
                                             'abstract': literal_eval
                                         })

    def __len__(self):
        return len(self.behaviors)

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
        row = self.behaviors.iloc[idx]
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
    pass