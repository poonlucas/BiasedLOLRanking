import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MatchDataset(Dataset):
    def __init__(self, data_csv):
        self.match_data = pd.read_csv(data_csv)
        # apply one hot vector on team position (role)
        team_pos = self.match_data.iloc[:, 6]
        onehot_team_pos = torch.nn.functional.one_hot(torch.tensor(team_pos, dtype=torch.int64), 5)
        self.match_data = pd.concat([self.match_data.iloc[:, :5], pd.DataFrame(onehot_team_pos)], axis=1)

    def __len__(self):
        return (len(self.match_data) + 1) // 5

    def __getitem__(self, idx):
        index = idx * 5
        match = torch.tensor(self.match_data.iloc[index:index + 5, 1:10].values.reshape(5, 9)).float()
        label = torch.tensor(self.match_data.iloc[index:index + 5, 0].values.reshape(1, 5)).flatten().float()
        return match, label
