import numpy as np
import torch
import random
import lightning as L
import os
from torch.utils.data import DataLoader, Dataset


class TaxiBJDataModule(L.LightningDataModule):
    def __init__(self,
                 data_root: str = os.path.join(os.getcwd(), "data"),
                 batch_size: int = 32,
                 val_batch_size: int = 32,
                 num_workers: int = 4,
                 len_back: int = 4,
                 len_pred: int = 4,
                 ):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.valid_batch_size = val_batch_size
        self.num_workers = num_workers
        self.back_len = len_back
        self.pred_len = len_pred

        self.train_dataset = self.valid_dataset = self.test_dataset = None

    def prepare_data(self):
        if not os.path.exists(os.path.join(self.data_root, "TaxiBJ", "dataset.npz")):
            raise FileNotFoundError(
                "Dataset not found, please download TaxiBJ dataset to ./data_root/TaxiBJ/dataset.npz \n"
                "数据集未找到，请下载TaxiBJ数据集到 ./data_root/TaxiBJ/dataset.npz")

    def setup(self, stage=None):
        dataset = np.load(os.path.join(self.data_root, "TaxiBJ", "dataset.npz"))
        # X_train, Y_train [20461,4,2,32,32], X_test, Y_test [500,4,2,32,32]
        X_train, Y_train, X_test, Y_test = (
            dataset['X_train'], dataset['Y_train'], dataset['X_test'], dataset['Y_test']
        )
        assert X_train.shape[1] == self.back_len and Y_train.shape[1] == self.pred_len
        self.train_dataset = TaxibjDataset(X=X_train, Y=Y_train)
        self.test_dataset = TaxibjDataset(X=X_test, Y=Y_test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.valid_batch_size, num_workers=self.num_workers,
                          shuffle=False)

    def test_dataloader(self):
        return self.val_dataloader()

    def predict_dataloader(self):
        return self.test_dataloader()


class TaxibjDataset(Dataset):
    """Taxibj <https://arxiv.org/abs/1610.00081>`_ Dataset"""

    def __init__(self, X, Y, use_augment=False, data_name='taxibj'):
        super(TaxibjDataset, self).__init__()
        self.X = (X + 1) / 2  # channel is 2
        self.Y = (Y + 1) / 2
        self.use_augment = use_augment
        self.mean = 0
        self.std = 1
        self.data_name = data_name

    def _augment_seq(self, seqs):
        """Augmentations as a video sequence"""
        # Random Flip
        if random.randint(0, 1):
            seqs = torch.flip(seqs, dims=(3,))  # horizontal flip
        return seqs

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        data = torch.tensor(self.X[index, ::]).float()
        labels = torch.tensor(self.Y[index, ::]).float()
        if self.use_augment:
            len_data = data.shape[0]  # 4
            seqs = self._augment_seq(torch.cat([data, labels], dim=0))
            data = seqs[:len_data, ...]
            labels = seqs[len_data:, ...]
        return data, labels