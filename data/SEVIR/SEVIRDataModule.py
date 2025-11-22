"""
Code is adapted from https://github.com/gaozhihan/PreDiff
"""

"""
Code is adapted from https://github.com/amazon-science/earth-forecasting-transformer/blob/e60ff41c7ad806277edc2a14a7a9f45585997bd7/src/earthformer/datasets/sevir/sevir_torch_wrap.py
Add data augmentation.
Only return "VIL" data in `torch.Tensor` format instead of `Dict`
"""

import os
import numpy as np
import torch
import random
import lightning as L
import datetime
from .SEVIRDataLoader import SEVIRTorchDataset, SEVIRDataLoader

from torch.utils.data import Dataset as TorchDataset, DataLoader, random_split


class SEVIRDataModule(L.LightningDataModule):
    def __init__(self,
                 data: str = 'sevirlr',
                 data_root: str = os.path.join(os.getcwd(), "data"),
                 batch_size: int = 16,
                 val_batch_size: int = 16,
                 num_workers: int = 4,
                 len_back: int = 4,
                 len_pred: int = 4,
                 ):
        super().__init__()
        assert data in ['sevirlr', 'sevir']
        self.data = data
        self.data_root = data_root
        self.data_dir = os.path.join(data_root, 'SEVIR', data)

        self.batch_size = batch_size
        self.valid_batch_size = val_batch_size
        self.num_workers = num_workers

        self.len_back = len_back
        self.len_pred = len_pred

        self.train_dataset = self.valid_dataset = self.test_dataset = None

        self.catalog_path = os.path.join(self.data_dir, "CATALOG.csv")
        self.raw_data_dir = os.path.join(self.data_dir, "data")
        self.raw_seq_len = 49 if data == 'sevir' else 25
        self.interval_real_time = 5 if data == 'sevir' else 10
        self.img_height = 384 if data == 'sevir' else 128
        self.img_width = 384 if data == 'sevir' else 128

        self.train_valid_split_date = datetime.datetime(*[2019, 1, 1])
        self.valid_test_split_date = datetime.datetime(*[2019, 6, 1])

    def prepare_data(self):
        if os.path.exists(self.data_dir):
            assert os.path.exists(self.catalog_path), f"CATALOG.csv not found! Should be located at {self.catalog_path}"
            assert os.path.exists(self.raw_data_dir), f"SEVIR data not found! Should be located at {self.raw_data_dir}"
        else:
            raise FileNotFoundError(f"{self.data_dir} not found!")

    def setup(self, stage=None):
        public_conf = {
            "seq_len": self.len_back + self.len_pred,
            "raw_seq_len": self.raw_seq_len,
            "stride": self.len_back,
            "layout": "NTCHW",
            "sevir_catalog": self.catalog_path,
            "sevir_data_dir": self.raw_data_dir,
            "rescale_method": "01",  #
        }
        self.train_dataset = SEVIR(**public_conf, start_date=None, end_date=self.train_valid_split_date, shuffle=True)
        self.valid_dataset = SEVIR(**public_conf, start_date=self.train_valid_split_date,
                                   end_date=self.valid_test_split_date, shuffle=False)
        self.test_dataset = SEVIR(**public_conf, start_date=self.valid_test_split_date, end_date=None, shuffle=False)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return self.test_dataloader()

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


class SEVIR(TorchDataset):
    def __init__(self,
                 seq_len: int = 13,
                 raw_seq_len: int = 25,
                 stride: int = 6,
                 layout: str = "NTCHW",
                 sevir_catalog: str = os.path.join("data", "SEVIR", "sevirlr", "CATALOG.csv"),
                 sevir_data_dir: str = os.path.join("data", "SEVIR", "sevirlr", "data"),
                 start_date: datetime.datetime = None,
                 end_date: datetime.datetime = None,
                 shuffle: bool = False,
                 rescale_method: str = "01",
                 ):
        assert rescale_method in ["sevir", "01"]
        self.len_back = stride
        self.len_pred = seq_len - stride
        self.sevir_dataloader = SEVIRDataLoader(
            data_types=["vil"],
            seq_len=seq_len,
            raw_seq_len=raw_seq_len,
            sample_mode="sequent",
            stride=stride,
            batch_size=1,
            layout=layout,
            sevir_catalog=sevir_catalog,
            sevir_data_dir=sevir_data_dir,
            start_date=start_date,
            end_date=end_date,
            shuffle=shuffle,
            rescale_method=rescale_method,
            downsample_dict=None,  #
        )

    def __getitem__(self, idx):
        data_dict = self.sevir_dataloader._idx_sample(index=idx)
        data = data_dict["vil"].squeeze(0)  # [T,C,H,W]
        return data[:self.len_back, :, :], data[self.len_back:, :, :, :]

    def __len__(self):
        return self.sevir_dataloader.__len__()
