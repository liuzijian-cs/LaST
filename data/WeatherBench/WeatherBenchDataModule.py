import os
import torch
import xarray as xr
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import lightning as L
from utils import printf, Color as Co
from typing import List, Optional
from .WeatherBenchDataset import WeatherBenchDataset


class WeatherBenchDataModule(L.LightningDataModule):
    """
    This module heavily references OpenSTL: https://github.com/chengtan9907/OpenSTL
    and
    https://github.com/pangeo-data/WeatherBench
    https://github.com/google-research/weatherbench2
    """

    def __init__(self,
                 data: str = 't2m',
                 data_split: str = '5_625',
                 data_root: str = os.path.join(os.getcwd(), "data"),
                 batch_size: int = 32,
                 val_batch_size: int = 32,
                 num_workers: int = 4,
                 len_back: int = 12,
                 len_pred: int = 12,
                 ):
        super().__init__()
        assert data_split in ['5_625', '2_8125', '1_40625']
        self.data = data
        self.data_split = data_split
        self.data_dir = os.path.join(data_root, 'WeatherBench', data_split)
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        # self.len_back = len_back
        # self.len_pred = len_pred
        self.idx_in = list(range(-len_back + 1, 1))  # [-11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0]
        self.idx_out = list(range(1, len_pred + 1))  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

        # Config:
        # self.year_train = [str(year) for year in range(1979, 2016)]  # train: 1979-2015
        # self.year_valid = ['2016']  # Valid
        # self.year_test = ['2017', '2018']  # Test
        # self.year_train = [str(year) for year in range(1979, 2016)]  # train: 1979-2015
        self.year_train = ['2010', '2015']  # train: 1979-2015
        self.year_valid = ['2016']  # Valid
        self.year_test = ['2017', '2018']  # Test

        self.train_dataset = self.valid_dataset = self.test_dataset = None

        self.save_hyperparameters()

    def setup(self, stage=None):
        levels = ['1000'] if self.data == 'r' else ['50']
        self.train_dataset = WeatherBenchDataset(data_root=self.data_dir, data_name=self.data,
                                                 data_split=self.data_split, training_time=self.year_train,
                                                 idx_in=self.idx_in, idx_out=self.idx_out, step=1, levels=levels)
        self.valid_dataset = WeatherBenchDataset(data_root=self.data_dir, data_name=self.data,
                                                 data_split=self.data_split, training_time=self.year_valid,
                                                 idx_in=self.idx_in, idx_out=self.idx_out, step=1, levels=levels,
                                                 mean=self.train_dataset.mean, std=self.train_dataset.std)
        self.test_dataset = WeatherBenchDataset(data_root=self.data_dir, data_name=self.data,
                                                data_split=self.data_split, training_time=self.year_test,
                                                idx_in=self.idx_in, idx_out=self.idx_out, step=1, levels=levels,
                                                mean=self.train_dataset.mean, std=self.train_dataset.std)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        # Sync with openstl and other papers, this place is set to test_dataset
        return DataLoader(self.test_dataset, batch_size=self.val_batch_size, shuffle=False,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.val_batch_size, shuffle=False,
                          num_workers=self.num_workers)

    def predict_dataloader(self):
        return self.test_dataloader()
