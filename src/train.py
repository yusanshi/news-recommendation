from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import BaseDataset
import torch
import torch.nn as nn
import time
import numpy as np
from config import model_name
from tqdm import tqdm
import os
from pathlib import Path
from evaluate import evaluate
import importlib

try:
    Model = getattr(importlib.import_module(f"model.{model_name}"), model_name)
    Config = getattr(importlib.import_module('config'), f"{model_name}Config")
except (AttributeError, ModuleNotFoundError):
    print(f"{model_name} not included!")
    exit()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def latest_checkpoint(directory):
    if not os.path.exists(directory):
        return None
    all_checkpoints = {
        int(x.split('.')[-2].split('-')[-1]): x
        for x in os.listdir(directory)
    }
    if not all_checkpoints:
        return None
    return os.path.join(directory,
                        all_checkpoints[max(all_checkpoints.keys())])


def train():
    writer = SummaryWriter(log_dir=f"./runs/{model_name}")

    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')

    try:
        pretrained_word_embedding = torch.from_numpy(
            np.load('./data/train/pretrained_word_embedding.npy')).float()
    except FileNotFoundError:
        pretrained_word_embedding = None

    if model_name == 'DKN':
        try:
            pretrained_entity_embedding = torch.from_numpy(
                np.load(
                    './data/train/pretrained_entity_embedding.npy')).float()
        except FileNotFoundError:
            pretrained_entity_embedding = None

        # TODO: currently context is not available
        try:
            pretrained_context_embedding = torch.from_numpy(
                np.load(
                    './data/train/pretrained_context_embedding.npy')).float()
        except FileNotFoundError:
            pretrained_context_embedding = None

        model = Model(Config, pretrained_word_embedding,
                      pretrained_entity_embedding,
                      pretrained_context_embedding).to(device)
    else:
        model = Model(Config, pretrained_word_embedding).to(device)

    print(model)

    dataset = BaseDataset('data/train/behaviors_parsed.tsv',
                          'data/train/news_parsed.tsv', Config.dataset_attributes)

    print(f"Load training dataset with size {len(dataset)}.")

    dataloader = iter(
        DataLoader(dataset,
                   batch_size=Config.batch_size,
                   shuffle=True,
                   num_workers=Config.num_workers,
                   drop_last=True))

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([Config.negative_sampling_ratio]).float().to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate)
    start_time = time.time()
    loss_full = []
    exhaustion_count = 0
    epoch = 0

    if Config.load_checkpoint:
        checkpoint_dir = os.path.join('./checkpoint', model_name)
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        checkpoint_path = latest_checkpoint(checkpoint_dir)
        if checkpoint_path is not None:
            print(f"Load saved parameters in {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            model.train()

    with tqdm(total=Config.num_batches, desc="Training") as pbar:
        for i in range(1, Config.num_batches + 1):
            try:
                minibatch = next(dataloader)
            except StopIteration:
                exhaustion_count += 1
                tqdm.write(
                    f"Training data exhausted for {exhaustion_count} times after {i} batches, reuse the dataset."
                )
                dataloader = iter(
                    DataLoader(dataset,
                               batch_size=Config.batch_size,
                               shuffle=True,
                               num_workers=Config.num_workers,
                               drop_last=True))
                minibatch = next(dataloader)

            epoch += 1
            if model_name == 'LSTUR':
                y_pred = model(minibatch["user"],
                               minibatch["clicked_news_length"],
                               minibatch["candidate_news"],
                               minibatch["clicked_news"])
            elif model_name == 'HiFiArk':
                y_pred, regularizer_loss = model(minibatch["candidate_news"],
                                                 minibatch["clicked_news"])
            elif model_name == 'TANR':
                y_pred, topic_classification_loss = model(minibatch["candidate_news"],
                                                          minibatch["clicked_news"])
            else:
                y_pred = model(minibatch["candidate_news"],
                               minibatch["clicked_news"])
            y = minibatch["clicked"].float().to(device)
            loss = criterion(y_pred, y)
            if model_name == 'HiFiArk':
                # if i % 10 == 0:
                #     print(loss.item(), '\t', regularizer_loss.item())
                loss += Config.regularizer_loss_weight * regularizer_loss
            elif model_name == 'TANR':
                # if i % 10 == 0:
                #     print(loss.item(), '\t', topic_classification_loss.item())
                loss += Config.topic_classification_loss_weight * topic_classification_loss
            loss_full.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('Train/Loss', loss.item(), i)

            if i % Config.num_batches_save_checkpoint == 0:
                torch.save(
                    {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch
                    }, f"./checkpoint/{model_name}/ckpt-{epoch}.pth")

            if i % Config.num_batches_batch_loss == 0:
                tqdm.write(
                    f"Time {time_since(start_time)}, batches {i}, current loss {loss.item():.6f}, average loss: {np.mean(loss_full):.6f}"
                )

            if i % Config.num_batches_validate == 0:
                val_auc, val_mrr, val_ndcg5, val_ncg10 = evaluate(
                    model, './data/val')
                writer.add_scalar('Validation/AUC', val_auc, i)
                writer.add_scalar('Validation/MRR', val_mrr, i)
                writer.add_scalar('Validation/nDCG@5', val_ndcg5, i)
                writer.add_scalar('Validation/nDCG@10', val_ncg10, i)
                tqdm.write(
                    f"Time {time_since(start_time)}, batches {i}, validation AUC: {val_auc:.6f}, validation MRR: {val_mrr:.6f}, validation nDCG@5: {val_ndcg5:.6f}, validation nDCG@10: {val_ncg10:.6f}, "
                )

            pbar.update(1)

    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch
        }, f"./checkpoint/{model_name}/ckpt-{epoch}.pth")


def time_since(since):
    """
    Format elapsed time string.
    """
    now = time.time()
    elapsed_time = now - since
    return time.strftime("%H:%M:%S", time.gmtime(elapsed_time))


if __name__ == '__main__':
    print('Using device:', device)
    print(f'Training model {model_name}')
    train()
