import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig, BertModel
import numpy as np

def extract_time_features(timestamps):
    return np.array([timestamps])

class TransactionDataset(Dataset):
    def __init__(self, data, known_address, sample_len, apply_log=False):
        self.data = data
        self.known_address = known_address
        self.sample_len = sample_len

        self.IND_FROM = self.data.columns.get_loc('from')
        self.IND_VALUE = self.data.columns.get_loc('value')
        self.IND_TO = self.data.columns.get_loc('to')
        self.timestamp = (self.data.loc[:, 'timestamp'].astype('datetime64[ns]') -  np.datetime64('1970-01-01T00:00:00')) \
            / np.timedelta64(1, 'ns')
        self.data = data.values
        self.apply_log = apply_log
        self.volume_converter = lambda x: x
        if self.apply_log:
            self.volume_converter = lambda x: np.log(x)


    def __len__(self):
        return len(self.data) // self.sample_len - 1

    def __getitem__(self, idx):
        start_idx = idx * self.sample_len + np.random.randint(0, self.sample_len)
        end_idx = start_idx + self.sample_len

        sample = self.data[start_idx: end_idx]
        ts = self.timestamp[start_idx: end_idx]
        time_features = extract_time_features(ts)
        from_address = sample[:, self.IND_FROM]
        to_address = sample[:, self.IND_TO]
        from_ind = np.full_like(from_address, len(self.known_address), dtype=int)
        to_ind = np.full_like(to_address, len(self.known_address), dtype=int)

        for ind, addr in enumerate(self.known_address):
            from_ind[from_address == addr] = ind
            to_ind[to_address == addr] = ind

        numeric_features = [sample[:, self.IND_VALUE]]

        numeric_features = torch.tensor(np.array([x.astype(float) for x in numeric_features]), dtype=torch.float)

        to_ind = torch.tensor(to_ind, dtype=torch.long)
        from_ind = torch.tensor(from_ind, dtype=torch.long)

        to_ind = to_ind.permute(*torch.arange(to_ind.ndim - 1, -1, -1))
        from_ind = to_ind.permute(*torch.arange(from_ind.ndim - 1, -1, -1))
        return {
            'numeric_features': numeric_features.T,
            'from_address': from_ind,
            'to_address': to_ind,
            'time_features': torch.tensor(time_features, dtype=torch.float).T,
            'value': torch.tensor(self.volume_converter(sample[:, self.IND_VALUE]).astype(float), dtype=torch.float)
        }
