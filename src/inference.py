import torch
from config import model_name
from tqdm import tqdm
from train import latest_checkpoint
from torch.utils.data import Dataset, DataLoader
import json
import os
import pandas as pd
from ast import literal_eval


class NewsDataset(Dataset):
    def __init__(self, news_path):
        super(NewsDataset, self).__init__()
        self.news_parsed = pd.read_table(news_path,
                                         converters={
                                             'title': literal_eval,
                                             'abstract': literal_eval
                                         })

    def __len__(self):
        return len(self.news_parsed)

    def __getitem__(self, idx):
        row = self.news_parsed.iloc[idx]
        item = {
            "id": row.id,
            "category": row.category,
            "subcategory": row.subcategory,
            "title": row.title,
            "abstract": row.abstract,
        }
        return item


class UserDataset(Dataset):
    def __init__(self, behaviors_path, user2int_path):
        super(UserDataset, self).__init__()
        self.behaviors = pd.read_table(behaviors_path,
                                       header=None,
                                       usecols=[0, 2],
                                       names=['user', 'clicked_news'])
        self.behaviors.fillna(' ', inplace=True)
        self.behaviors.drop_duplicates(inplace=True)
        user2int = dict(pd.read_table(user2int_path).values.tolist())
        for row in self.behaviors.itertuples():
            self.behaviors.at[row.Index, 'user'] = user2int[
                row.user] if row.user in user2int else 0

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        row = self.behaviors.iloc[idx]
        item = {
            "user":
            row.user,
            "clicked_news_string":
            row.clicked_news,
            "clicked_news":
            row.clicked_news.split()[:Config.num_clicked_news_a_user]
        }
        repeated_times = Config.num_clicked_news_a_user - len(
            item["clicked_news"])
        assert repeated_times >= 0
        item["clicked_news"].extend(['PADDED_NEWS'] * repeated_times)

        return item


class BehaviorsDataset(Dataset):
    def __init__(self, behaviors_path):
        super(BehaviorsDataset, self).__init__()
        self.behaviors = pd.read_table(
            behaviors_path,
            header=None,
            usecols=range(4),
            names=['user', 'time', 'clicked_news', 'impressions'])
        self.behaviors.fillna(' ', inplace=True)
        self.behaviors.impressions = self.behaviors.impressions.str.split()

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        row = self.behaviors.iloc[idx]
        item = {
            "user": row.user,
            "time": row.time,
            "clicked_news_string": row.clicked_news,
            "impressions": row.impressions
        }
        return item


@torch.no_grad()
def inference():
    model = Model(Config).to(device)
    checkpoint_path = latest_checkpoint(
        os.path.join('./checkpoint', model_name))
    if checkpoint_path is None:
        print('No checkpoint file found!')
        exit()
    print(f"Load saved parameters in {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    news_dataset = NewsDataset('./data/test/news_parsed.tsv')
    news_dataloader = DataLoader(news_dataset,
                                 batch_size=Config.batch_size,
                                 shuffle=False,
                                 num_workers=Config.num_workers,
                                 drop_last=False)

    news2vector = {}

    with tqdm(total=len(news_dataloader),
              desc="Calculating vectors for news") as pbar:
        for minibatch in news_dataloader:
            news_ids = minibatch["id"]
            if any(id not in news2vector for id in news_ids):
                news_vector = model.get_news_vector(minibatch)
                for id, vector in zip(news_ids, news_vector):
                    if id not in news2vector:
                        news2vector[id] = vector
            pbar.update(1)

    news2vector['PADDED_NEWS'] = torch.zeros(
        list(news2vector.values())[0].size())

    user_dataset = UserDataset('data/test/behaviors.tsv',
                               'data/train/user2int.tsv')
    user_dataloader = DataLoader(user_dataset,
                                 batch_size=Config.batch_size,
                                 shuffle=False,
                                 num_workers=Config.num_workers,
                                 drop_last=False)

    user2vector = {}
    with tqdm(total=len(user_dataloader),
              desc="Calculating vectors for users") as pbar:
        for minibatch in user_dataloader:
            user_strings = minibatch["clicked_news_string"]
            if any(user_string not in user2vector
                   for user_string in user_strings):
                clicked_news_vector = torch.stack([
                    torch.stack([news2vector[x].to(device) for x in news_list],
                                dim=0)
                    for news_list in minibatch["clicked_news"]
                ],
                    dim=0).transpose(0, 1)
                if model_name == 'LSTUR':
                    user_vector = model.get_user_vector(
                        minibatch['user'], clicked_news_vector)
                else:
                    user_vector = model.get_user_vector(clicked_news_vector)
                for user, vector in zip(user_strings, user_vector):
                    if user not in user2vector:
                        user2vector[user] = vector
            pbar.update(1)

    behaviors_dataset = BehaviorsDataset('data/test/behaviors.tsv')
    behaviors_dataloader = DataLoader(behaviors_dataset,
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=Config.num_workers)

    # For writing inference results
    submission_answer_file = open('./data/test/answer.json', 'w')
    with tqdm(total=len(behaviors_dataloader),
              desc="Calculating probabilities") as pbar:
        for minibatch in behaviors_dataloader:
            user_inference = {
                "uid": minibatch['user'][0][1:],
                "time": minibatch['time'][0],
                "impression": {
                    news[0].split('-')[0][1:]: model.get_prediction(
                        news2vector[news[0].split('-')[0]], user2vector[
                            minibatch['clicked_news_string'][0]]).item()
                    for news in minibatch['impressions']
                }
            }

            submission_answer_file.write(json.dumps(user_inference) + '\n')
            pbar.update(1)

    submission_answer_file.close()


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)
    if model_name == 'NRMS':
        from model.NRMS import NRMS as Model
        from dataset import NRMSDataset as Dataset
        from config import NRMSConfig as Config
    elif model_name == 'NAML':
        from model.NAML import NAML as Model
        from dataset import NAMLDataset as Dataset
        from config import NAMLConfig as Config
    elif model_name == 'LSTUR':
        from model.LSTUR import LSTUR as Model
        from dataset import LSTURDataset as Dataset
        from config import LSTURConfig as Config
    elif model_name == 'DKN':
        from model.DKN import DKN as Model
        from dataset import DKNDataset as Dataset
        from config import DKNConfig as Config
    else:
        print("Model name not included!")
        exit()
    print(f'Inferencing model {model_name}')
    inference()
