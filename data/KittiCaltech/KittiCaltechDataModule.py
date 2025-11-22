"""
The dataset is largely influenced by OpenSTL (https://github.com/chengtan9907/OpenSTL),
and their license is the Apache-2.0 License.
"""
import hickle as hkl
import os
import os.path as osp
import cv2
import random
import numpy as np
from skimage.transform import resize
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import lightning as L


class KittiCaltechDataModule(L.LightningDataModule):
    def __init__(self,
                 data_root: str = os.path.join(os.getcwd(), "data"),
                 batch_size: int = 16,
                 val_batch_size: int = 16,
                 num_workers: int = 4,
                 len_back: int = 10,
                 len_pred: int = 1,
                 ):
        super().__init__()
        self.data_dir = os.path.join(data_root, "KittiCaltech")
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.len_back = len_back
        self.len_pred = len_pred

        self.train_dataset = self.valid_dataset = self.test_dataset = None

    def prepare_data(self):
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"{self.data_dir} not found!")

    def setup(self, stage=None):
        input_param = {
            'paths': {'kitti': osp.join(self.data_dir, 'kitti_hkl'),
                      'caltech': osp.join(self.data_dir, 'caltech')},
            'seq_length': (self.len_back + self.len_pred),
            'input_data_type': 'float32',
            'input_shape': (128, 160),
        }
        input_handle = DataProcess(input_param)
        train_data, train_idx = input_handle.load_data('train')
        test_data, test_idx = input_handle.load_data('test')

        self.train_dataset = KittiCaltechDataset(train_data, train_idx, self.len_back, self.len_pred)
        self.test_dataset = KittiCaltechDataset(test_data, test_idx, self.len_back, self.len_pred)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return self.test_dataloader()

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.val_batch_size, shuffle=False,
                          num_workers=self.num_workers)


class KittiCaltechDataset(Dataset):
    """KittiCaltech <https://dl.acm.org/doi/10.1177/0278364913491297>`_ Dataset"""

    def __init__(self, datas, indices, pre_seq_length, aft_seq_length,
                 require_back=False, use_augment=False, data_name='kitticaltech'):
        super(KittiCaltechDataset, self).__init__()
        self.datas = datas.swapaxes(2, 3).swapaxes(1, 2)
        self.indices = indices
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        self.require_back = require_back
        self.use_augment = use_augment
        self.mean = 0
        self.std = 1
        self.data_name = data_name

    def _augment_seq(self, imgs, crop_scale=0.95):
        """Augmentations for video"""
        _, _, h, w = imgs.shape  # original shape, e.g., [12, 3, 128, 160]
        imgs = F.interpolate(imgs, scale_factor=1 / crop_scale, mode='bilinear')
        _, _, ih, iw = imgs.shape
        # Random Crop
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        imgs = imgs[:, :, x:x + h, y:y + w]
        # Random Flip
        if random.randint(0, 1):
            imgs = torch.flip(imgs, dims=(3,))  # horizontal flip
        return imgs

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        batch_ind = self.indices[i]
        begin = batch_ind
        end1 = begin + self.pre_seq_length
        end2 = end1 + self.aft_seq_length
        data = torch.tensor(self.datas[begin:end1, ::]).float()
        labels = torch.tensor(self.datas[end1:end2, ::]).float()
        if self.use_augment:
            imgs = self._augment_seq(torch.cat([data, labels], dim=0), crop_scale=0.95)
            data = imgs[:self.pre_seq_length, ...]
            labels = imgs[self.pre_seq_length:self.pre_seq_length + self.aft_seq_length, ...]
        return data, labels


def process_im(im, desired_sz):
    # cite the `process_im` code from PredNet, Thanks!
    # https://github.com/coxlab/prednet/blob/master/process_kitti.py
    target_ds = float(desired_sz[0]) / im.shape[0]
    im = resize(im, (desired_sz[0], int(np.round(target_ds * im.shape[1]))), preserve_range=True)
    d = int((im.shape[1] - desired_sz[1]) / 2)
    im = im[:, d:d + desired_sz[1]]
    return im


class DataProcess(object):

    def __init__(self, input_param):
        self.paths = input_param['paths']
        self.seq_len = input_param['seq_length']
        self.input_shape = input_param['input_shape']  # (128, 160)

    def load_data(self, mode='train'):
        """Loads the dataset.
        Args:
          paths: paths of train/test dataset.
          mode: Training or testing.
        Returns:
          A dataset and indices of the sequence.
        """
        if mode == 'train' or mode == 'val':
            kitti_root = self.paths['kitti']
            data = hkl.load(osp.join(kitti_root, 'X_' + mode + '.hkl'))
            data = data.astype('float') / 255.0
            fileidx = hkl.load(
                osp.join(kitti_root, 'sources_' + mode + '.hkl'))

            indices = []
            index = len(fileidx) - 1
            while index >= self.seq_len - 1:
                if fileidx[index] == fileidx[index - self.seq_len + 1]:
                    indices.append(index - self.seq_len + 1)
                    index -= self.seq_len - 1
                index -= 1

        elif mode == 'test':
            caltech_root = self.paths['caltech']
            # find the cache file
            caltech_cache = osp.join(caltech_root, 'data_cache.npy')
            if osp.exists(caltech_cache):
                data = np.load(caltech_cache).astype('float') / 255.0
                indices = np.load(osp.join(caltech_root, 'indices_cache.npy'))
            else:
                print(f'loading caltech from {caltech_root}, which requires some times...')
                data = []
                fileidx = []
                for seq_id in os.listdir(caltech_root):
                    if osp.isdir(osp.join(caltech_root, seq_id)) is False:
                        continue
                    for item in os.listdir(osp.join(caltech_root, seq_id)):
                        seq_file = osp.join(caltech_root, seq_id, item)
                        print(seq_file)
                        cap = cv2.VideoCapture(seq_file)
                        cnt_frames = 0
                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            cnt_frames += 1
                            if cnt_frames % 3 == 0:
                                frame = process_im(frame, self.input_shape) / 255.0
                                data.append(frame)
                                fileidx.append(seq_id + item)
                data = np.asarray(data)

                indices = []
                index = len(fileidx) - 1
                while index >= self.seq_len - 1:
                    if fileidx[index] == fileidx[index - self.seq_len + 1]:
                        indices.append(index - self.seq_len + 1)
                        index -= self.seq_len - 1
                    index -= 1

                # save the cache file
                data_cache = data * 255
                np.save(caltech_cache, data_cache.astype('uint8'))
                indices_cache = np.asarray(indices)
                np.save(osp.join(caltech_root, 'indices_cache.npy'), indices_cache.astype('int32'))

        return data, indices
