import logging

import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

log = logging.getLogger(__name__)


class pdDataset(Dataset):
    def __init__(self, features, labels=None):
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx])

    def create_dataloader(self, shuffle, batch_size):
        return DataLoader(self, shuffle=shuffle, batch_size=batch_size)


class pdDatasetPrediction(Dataset):
    def __init__(self, features):
        self.features = features.astype(np.float32)
        if isinstance(features, pd.DataFrame):
            self.features = self.features.values.astype(np.float32)
        # print(self.features)
        log.info(f"Creating dataset with {len(self.features)} samples")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx])

    def create_dataloader(self, batch_size):
        return DataLoader(self, shuffle=False, batch_size=batch_size, pin_memory=False)


# Pytorch dataset class for xarray.DataArray
class xrDataset(Dataset):
    def __init__(self, features, labels=None):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx])

    def create_dataloader(self, shuffle, batch_size):
        return DataLoader(self, shuffle=shuffle, batch_size=batch_size)


class xrDatasetPrediction(Dataset):
    def __init__(self, features):
        self.features = features
        log.info(f"Creating dataset with {len(self.features)} samples")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx])

    def create_dataloader(self, batch_size, shuffle):
        return DataLoader(self, shuffle=shuffle, batch_size=batch_size, pin_memory=False)
