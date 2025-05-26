import random
import string
import torch
import wandb
import itertools
import argparse


import numpy as np
import pandas as pd

from torch.utils import data
from torch.utils.data import DataLoader
from tqdm import tqdm

from Code import Dataloader
from Code import Dataset
from Code import ModelBertV1
from Code import Train

from sklearn.model_selection import train_test_split

import Code.ModelBertV1 as ModelLib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import datetime


def load_model(path, map_location='cuda'):
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    config = checkpoint.get('config', None)
    # config['df_config'].address_limit = config['cnt_known_address']
    model = getattr(ModelBertV1, config['model'])(
                known_address_len=config['cnt_known_address'],
                **config['model_params']
            )

    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint.get('epoch', None)

    return model, config, epoch

def get_ts(data):
    return (data.loc[:, 'timestamp'].values.astype('datetime64[ns]') -  np.datetime64('1970-01-01T00:00:00')) \
            / np.timedelta64(1, 'ns')


class TimeEmbeddingRegressionDataset(data.Dataset):
    def __init__(self, df, model, sample_len, apply_log=False,
                 device='cpu', batch_size=128, cnt_batchs=-1,
                 target_type: str="timestamp", target_transform_params={}):
        self.model = model.eval().to(device)
        self.df = df.reset_index(drop=True)
        self.sample_len = sample_len
        self.apply_log = apply_log
        self.device = device
        self.batch_size = batch_size
        self.target_name = target_type
        self.target_transform_params = target_transform_params

        if cnt_batchs == -1:
            cnt_batchs = len(df) // (self.batch_size * self.sample_len) + 1
        self.cnt_batchs = cnt_batchs

        self.samples = self.build_samples(df)
        self.embeddings, self.targets = self.compute_all_embeddings()

    def build_samples(self, df):
        return list(range(
            min(len(df), self.batch_size * self.sample_len * self.cnt_batchs) // self.sample_len * self.sample_len))

    def __len__(self):
        return len(self.samples)

    def compute_target(self, idx: int) -> float:
        if self.target_name == "timestamp":
            return self.df.iloc[idx]["timestamp"]

        if self.target_name == "1h_volume":
            return self.df.iloc[idx]["1h_volume"]
        else:
            raise ValueError(f"Unsupported target: {self.target_name}")

    def get_target_transform(self, target):
        if self.target_name == "timestamp":
            mean = self.target_transform_params['mean']
            std = self.target_transform_params['std']

            return (target - mean) / std
        else:
            raise ValueError(f"Unsupported target: {self.target_name}")
        
    def revers_target_transfrom(self, pred):
        if self.target_name == "timestamp":
            mean = self.target_transform_params['mean']
            std = self.target_transform_params['std']
            return pred * std + mean
        else:
            raise ValueError(f"Unsupported target: {self.target_name}")

    def compute_all_embeddings(self):
        all_embs, all_tgts = [], []
        loader = DataLoader(self.samples[::self.sample_len], batch_size=self.batch_size, shuffle=True)

        for batch_indices in tqdm(itertools.islice(loader, self.cnt_batchs), desc="Embedding calculation", total=
                                  min(self.cnt_batchs, len(loader))):
            batch = [self._prepare_sample(int(idx)) for idx in batch_indices]

            smaples = self._collate_inputs([b[0] for b in batch])
            targets = torch.stack([b[1] for b in batch]).to(self.device)

            msk_ind, change_ind, save_ind, model_inputs = Train.batch_to_model(
                smaples,
                p_msk=0.0,
                p_change=0.0,
                p_save=1.0,
                device=self.device,
                change_strategy=Train.uniform_change_strategy
            )

            # for k, v in model_inputs.items():
            #     print(f"{k}: {v.shape}")
            #     print(v.device)

            with torch.no_grad():
                result = self.model(**model_inputs)['result']

            all_embs.append(result.cpu())
            all_tgts.append(targets.cpu())

        print(torch.cat(all_embs, 0).shape)
        return torch.cat(all_embs, 0).reshape(-1, all_embs[0].shape[-1]), torch.cat(all_tgts, 0).reshape(-1, 1)

    def _prepare_sample(self, idx):
        i = idx
        batch_df = self.df.iloc[i:i + self.sample_len]
        ts = get_ts(batch_df)
        time_features = Dataset.extract_time_features(ts)
        from_ind = batch_df['from'].values
        to_ind = batch_df['to'].values
        values = batch_df['value'].values

        volume_converter = lambda x: x
        if self.apply_log:
            volume_converter = lambda x: np.log(x)

        sample = {
            'numeric_features': torch.tensor(values, dtype=torch.float).T.unsqueeze(0).unsqueeze(-1),
            'from_address': torch.tensor(from_ind, dtype=torch.long).T.unsqueeze(0),
            'to_address': torch.tensor(to_ind, dtype=torch.long).T.unsqueeze(0),
            'time_features': torch.tensor(time_features, dtype=torch.float).T.unsqueeze(0),
            'value': torch.tensor(volume_converter(values)).unsqueeze(0).unsqueeze(-1)
        }

        target = torch.tensor([self.compute_target(i) for i in range(i, i + self.sample_len)], dtype=torch.float)
        return sample, target

    def _collate_inputs(self, input_list):
        batched = {}
        for key in input_list[0].keys():
            batched[key] = torch.cat([torch.tensor(item[key]) for item in input_list], dim=0)
        return batched

    def __getitem__(self, idx):
        return self.embeddings[idx], self.targets[idx]



class SimpleRegressor(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.ReLU(),
            nn.LazyLinear(1)
        )

    def forward(self, x):
        return self.net(x)


class TargetLogger:
    def __init__(self, target_type: str, run_name):
        self.target_type = target_type
        self.run_id = ''.join(random.choice(string.ascii_lowercase) for _ in range(5))
        self.run = wandb.init(name=f'{run_name} {self.run_id} {self.target_type}')

    def log(self, train_loss: float, val_loss: float, lr: float, epoch: int):
        log_data = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": lr,
            "epoch": epoch,
        }

        # if self.target_type == "timestamp":
        #     log_data["val_timedelta"] = datetime.timedelta(seconds=val_loss)

        wandb.log(log_data)

    def finish(self):
        self.run.finish()


def train_time_regressor(model, train_dataloader, val_dataloader, num_epochs=100,
                         lr=10, device='cuda', std=1, mean=0, normilize=True, logger: TargetLogger | None = None):
    std = float(std)
    mean = float(mean)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True,
        min_lr=1e-8,
        threshold=1,
    )

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        for emb_time, target in tqdm(train_dataloader, desc=f"Epoch {epoch + 1} - Train"):
            if normilize:
                emb_time = emb_time / emb_time.norm(2, 1, keepdim=True)
            emb_time = emb_time.to(device)
            target = (target.to(device).unsqueeze(-1).float() - mean) / std

            pred = model(emb_time)
            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for emb_time, target in tqdm(val_dataloader, desc=f"Epoch {epoch + 1} - Val"):
                if normilize:
                    emb_time = emb_time / emb_time.norm(2, 1, keepdim=True)
                emb_time = emb_time.to(device)
                target = (target.to(device).unsqueeze(-1).float() - mean) / std

                pred = model(emb_time)
                loss = criterion(pred, target)
                total_val_loss += loss.item()

        avg_val_loss = (total_val_loss / len(val_dataloader)) ** 0.5 * std
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch + 1}")
        print(f"\nLr: {optimizer.param_groups[0]['lr']}")
        avg_train_loss = (total_train_loss / len(train_dataloader)) ** 0.5 * std
        print(f"\nTrain Loss: {avg_train_loss:.4f}")
        print(f"\nVal Loss: {avg_val_loss:.4f}")
        print(f"\nTimedelta: {datetime.timedelta(seconds=avg_val_loss)}")

        lr = optimizer.param_groups[0]['lr']
        logger.log(avg_train_loss, avg_val_loss, lr, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--device", type=str)

    parser.add_argument("--target_type", type=str, default="timestamp")
    parser.add_argument("--precomputing_batch_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=2**14)
    parser.add_argument("--cnt_batchs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--model_epoch", type=int, default=1951)
    parser.add_argument("--study_epoch", type=int, default=100)
    parser.add_argument("--run_hash", type=str, default='qjmctedofg')

    args = parser.parse_args()

    precomputing_batch_size = args.precomputing_batch_size
    device = args.device
    target_type = args.target_type
    batch_size = args.batch_size
    lr = args.lr
    hidden_dim = args.hidden_dim
    model_epochs = args.model_epoch
    study_epoch = args.study_epoch
    run_name = args.run_name
    run_hash =  args.run_hash

    path = ('/home/motatasher/Documents/University/Diploma/'
                'Diploma/models/')
    import os

    for run in os.listdir(path):
        folder_path = os.path.join(path, run)
        if folder_path[-2:] != 'pt' and run.split('_')[-1] == run_hash:
            epochs = {int(x.replace('.', '_').split('_')[-2]): x for x in os.listdir(folder_path)}
            config_path = os.path.join(folder_path, epochs.get(model_epochs, max(epochs.keys())))

    print(config_path)
    model, config, epoch = load_model(config_path)

    df_raw = getattr(Dataloader, config['dataset_fabric'])(**config['df_config'], address_limit=config['cnt_known_address'])

    if target_type == '1h_volume':
        ts = df_raw["timestamp"].values
        values = df_raw["value"].values
        df_size = len(df_raw)
        prefix_sum = np.zeros(df_size + 1)
        prefix_sum[1:] = np.cumsum(values)
        future_volume = np.zeros(df_size)
        j = 0
        for i in tqdm(range(df_size)):
            while j < df_size and ts[j] <= ts[i] + 3600:
                j += 1
            future_volume[i] = prefix_sum[j] - prefix_sum[i + 1]
        df_raw["1h_volume"] = future_volume

    train_data, val_data = train_test_split(df_raw, test_size=1/4, shuffle=False)
    print(f"square std of all: {df_raw['value'].std() ** 2}\nTrain: {train_data['value'].std() ** 2}\nVal: {val_data['value'].std() ** 2}")

    time_model = SimpleRegressor(hidden_dim)

    train_df = TimeEmbeddingRegressionDataset(train_data, model, 100, apply_log=True, cnt_batchs=-1, device=device,
                                            target_type=target_type, batch_size=precomputing_batch_size)

    train_datalodader =  DataLoader(train_df, batch_size=batch_size, drop_last=True,
                                    num_workers=12, persistent_workers=True, pin_memory=True)

    val_df = TimeEmbeddingRegressionDataset(val_data, model, 100, apply_log=True, cnt_batchs=-1, device=device,
                                            target_type=target_type, batch_size=precomputing_batch_size)

    val_datalodader =  DataLoader(val_df, batch_size=batch_size, drop_last=True,
                                num_workers=12, persistent_workers=True, pin_memory=True)

    wandb_logger = TargetLogger(target_type, run_name)

    train_time_regressor(time_model, train_datalodader, val_datalodader, num_epochs=study_epoch,
                        lr=lr, device=device, std=train_df[:][1].std(), mean=train_df[:][1].mean(),
                        logger=wandb_logger)

    wandb_logger.finish()
