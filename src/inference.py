import torch
from config import model_name
from tqdm import tqdm
from train import latest_checkpoint
from torch.utils.data import DataLoader
import json
import copy
import os


@torch.no_grad()
def inference():
    dataset = Dataset('data/test/behaviors_parsed.tsv',
                      'data/test/news_parsed.tsv')
    print(f"Load inference dataset with size {len(dataset)}.")
    dataloader = iter(
        DataLoader(dataset,
                   batch_size=Config.batch_size,
                   shuffle=False,
                   num_workers=Config.num_workers,
                   drop_last=False))

    model = Model(Config).to(device)
    checkpoint_path = latest_checkpoint(
        os.path.join('./checkpoint', Config.model))
    print(f"Load saved parameters in {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    y_pred = []
    y = []
    assert Config.inference_radio > 0 and Config.inference_radio <= 1
    total = int(len(dataloader) * Config.inference_radio)
    with tqdm(total=total, desc="Inferering") as pbar:
        for _ in range(total):
            minibatch = next(dataloader)
            y_pred.extend(
                model(minibatch["candidate_news"],
                      minibatch["clicked_news"]).tolist())
            y.extend(minibatch["clicked"].float().tolist())
            pbar.update(1)

    y_pred = iter(y_pred)
    y = iter(y)

    # For locating and order validating
    truth_file = open('./data/test/truth.json', 'r')
    # For writing inference results
    submission_answer_file = open('./data/test/answer.json', 'w')
    try:
        for line in truth_file.readlines():
            user_truth = json.loads(line)
            user_inference = copy.deepcopy(user_truth)
            for k in user_truth['impression'].keys():
                assert next(y) == user_truth['impression'][k]
                user_inference['impression'][k] = next(y_pred)
            submission_answer_file.write(json.dumps(user_inference) + '\n')
    except StopIteration:
        print(
            'Warning: Behaviors not fully inferenced. You can still run evaluate.py, but the evaluation result would be inaccurate.'
        )

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
    else:
        print("Model name not included!")
        exit()
    print(f'Training model {model_name}')
    inference()
