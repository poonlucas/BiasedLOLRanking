import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MatchDataset(Dataset):
    def __init__(self, data_csv):
        self.match_data = pd.read_csv(data_csv)

    def __len__(self):
        return (len(self.match_data) + 1) // 5

    def __getitem__(self, idx):
        index = idx * 5
        match = torch.tensor(self.match_data.iloc[index:index + 5, 1:32].values.reshape(5, 31)).float()
        label = torch.tensor(self.match_data.iloc[index:index + 5, 0].values.reshape(1, 5)).flatten().float()
        return match, label