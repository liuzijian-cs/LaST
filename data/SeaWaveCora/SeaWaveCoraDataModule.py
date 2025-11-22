import os
import torch
import numpy as np
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from torch.utils.data import Dataset, DataLoader


class SeaWaveCoraDataModule(LightningDataModule):
    def __init__(self,
                 data_root: str = os.path.join(os.getcwd(), "data"),
                 batch_size: int = 16,
                 val_batch_size: int = 16,
                 num_workers: int = 4,
                 len_back: int = 7,
                 len_pred: int = 7,
                 ):
        super().__init__()
        self.data_path = os.path.join(data_root,"SeaWaveCora", 'data.npy')
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.len_back = len_back
        self.len_pred = len_pred

        self.train_dataset = self.valid_dataset = self.test_dataset = None

    def prepare_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"{self.data_path} not found!")

    def setup(self, stage=None):
        self.train_dataset = SeaWaveDataset(self.data_path, self.len_back, self.len_pred, True)
        self.test_dataset = SeaWaveDataset(self.data_path, self.len_back, self.len_pred, False)

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return self.test_dataloader()

    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset, batch_size=self.val_batch_size, shuffle=True,
                          num_workers=self.num_workers)


class SeaWaveDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 len_back: int = 7,
                 len_pred: int = 7,
                 is_train: bool = True,
                 ):
        super(SeaWaveDataset, self).__init__()
        self.dataset = np.load(data_path, allow_pickle=True).item()
        self.data = self.dataset['data']
        # self.time = self.dataset['time']
        # self.lat = self.dataset['lat']
        # self.lon = self.dataset['lon']

        self.data_mean = np.mean(self.data)
        self.data_std = np.std(self.data)

        self.len_back = len_back
        self.len_pred = len_pred

        self.start_index = 0 if is_train else 9861  # data index of 1989-2015
        self.end_index = 9861 if is_train else 11686  # data index of 2016-2020

    def __getitem__(self, idx):
        if idx + self.len_back + self.len_pred > len(self.data):
            raise IndexError("Index out of range.")
        # 获取输入和输出数据
        data_back = self.data[idx:idx + self.len_back]
        data_pred = self.data[idx + self.len_back:idx + self.len_back + self.len_pred]

        data_back = (data_back - self.data_mean) / self.data_std
        data_pred = (data_pred - self.data_mean) / self.data_std

        data_back = torch.tensor(data_back, dtype=torch.float32).unsqueeze(1)
        data_pred = torch.tensor(data_pred, dtype=torch.float32).unsqueeze(1)
        return data_back, data_pred

    def __len__(self):
        return self.end_index - self.start_index - self.len_back - self.len_pred + 1
