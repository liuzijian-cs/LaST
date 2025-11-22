"""
The dataset is largely influenced by OpenSTL (https://github.com/chengtan9907/OpenSTL),
and their license is the Apache-2.0 License.
"""
import os
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L


class HumanDataModule(L.LightningDataModule):
    def __init__(self,
                 data_root: str = os.path.join(os.getcwd(), "data"),
                 batch_size: int = 16,
                 val_batch_size: int = 16,
                 num_workers: int = 4,
                 len_back: int = 10,
                 len_pred: int = 10,
                 ):
        super().__init__()
        self.data_dir = os.path.join(data_root, "Human")
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.len_back = len_back
        self.len_pred = len_pred

        self.step = 5

        self.train_dataset = self.valid_dataset = self.test_dataset = None

    def prepare_data(self):
        if os.path.exists(self.data_dir):
            assert os.path.exists(os.path.join(self.data_dir, "images", "train.txt")), "train.txt not found!"
            assert os.path.exists(os.path.join(self.data_dir, "images", "test.txt")), "test.txt not found!"
        else:
            raise FileNotFoundError(f"{self.data_dir} not found!")

    def setup(self, stage=None):
        public_conf = {
            "data_root": self.data_dir,
            "image_size": 256,
            "pre_seq_length": 4,
            "aft_seq_length": 4,
            "step": 5,
        }

        self.train_dataset = HumanDataset(**public_conf, list_path=os.path.join(self.data_dir, "images", "train.txt"))
        self.valid_dataset = HumanDataset(**public_conf, list_path=os.path.join(self.data_dir, "images", "test.txt"))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.val_batch_size, shuffle=False,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return self.val_dataloader()


class HumanDataset(Dataset):
    """ Human 3.6M Dataset
        <http://vision.imar.ro/human3.6m/pami-h36m.pdf>`_

    Args:
        data_root (str): Path to the dataset.
        list_path (str): Path to the txt list file.
        image_size (int: The target resolution of Human3.6M images.
        pre_seq_length (int): The input sequence length.
        aft_seq_length (int): The output sequence length for prediction.
        step (int): Sampling step in the time dimension (defaults to 5).
        use_augment (bool): Whether to use augmentations (defaults to False).
    """

    def __init__(self, data_root, list_path, image_size=256,
                 pre_seq_length=4, aft_seq_length=4, step=5, use_augment=False, data_name='human'):
        super(HumanDataset, self).__init__()
        self.data_root = data_root
        self.file_list = None
        self.image_size = image_size
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        self.seq_length = pre_seq_length + aft_seq_length
        self.step = step
        self.use_augment = use_augment
        self.input_shape = (self.seq_length, self.image_size, self.image_size, 3)
        with open(list_path, 'r') as f:
            self.file_list = f.readlines()
        self.mean = None
        self.std = None
        self.data_name = data_name

    def _augment_seq(self, imgs, h, w):
        """Augmentations for video"""
        ih, iw, _ = imgs[0].shape
        # Random Crop
        length = len(imgs)
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        for i in range(length):
            imgs[i] = imgs[i][x:x + h, y:y + w, :]
        # Random Rotation
        if random.randint(-2, 1):
            for i in range(length):
                imgs[i] = cv2.rotate(imgs[i], cv2.ROTATE_90_CLOCKWISE)
        elif random.randint(-2, 1):
            for i in range(length):
                imgs[i] = cv2.rotate(imgs[i], cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif random.randint(-2, 1):
            for i in range(length):
                imgs[i] = cv2.flip(imgs[i], flipCode=1)  # horizontal flip
        # to tensor
        for i in range(length):
            imgs[i] = torch.from_numpy(imgs[i].copy()).float()
        return imgs

    def _to_tensor(self, imgs):
        for i in range(len(imgs)):
            imgs[i] = torch.from_numpy(imgs[i].copy()).float()
        return imgs

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        item_list = self.file_list[idx].split(',')
        begin = int(item_list[1])
        end = begin + self.seq_length * self.step

        raw_input_shape = self.input_shape if not self.use_augment \
            else (self.seq_length, int(self.image_size / 0.975), int(self.image_size / 0.975), 3)
        img_seq = []
        i = 0
        for j in range(begin, end, self.step):
            # e.g., images/S11_Walking.60457274_001621.jpg
            base_str = '0' * (6 - len(str(j))) + str(j) + '.jpg'
            file_name = os.path.join(self.data_root, item_list[0] + base_str)
            image = cv2.imread(file_name)
            if image is None:
                raise FileNotFoundError(f"{file_name} not found!")
            if image.shape[0] != raw_input_shape[2]:
                image = cv2.resize(image, (raw_input_shape[1], raw_input_shape[2]), interpolation=cv2.INTER_CUBIC)
            img_seq.append(image)
            i += 1

        # augmentation
        if self.use_augment:
            img_seq = self._augment_seq(img_seq, h=self.image_size, w=self.image_size)
        else:
            img_seq = self._to_tensor(img_seq)

        # transform
        img_seq = torch.stack(img_seq, 0).permute(0, 3, 1, 2) / 255  # min-max to [0, 1]
        data = img_seq[:self.pre_seq_length, ...]
        labels = img_seq[self.aft_seq_length:, ...]

        return data, labels
