import random
import numpy as np
import os.path as osp
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import tqdm
import xarray as xr

data_map = {
    'z': 'geopotential',
    't': 'temperature',
    'tp': 'total_precipitation',
    't2m': '2m_temperature',
    'r': 'relative_humidity',
    's': 'specific_humidity',
    'u10': '10m_u_component_of_wind',
    'u': 'u_component_of_wind',
    'v10': '10m_v_component_of_wind',
    'v': 'v_component_of_wind',
    'tcc': 'total_cloud_cover',
    "lsm": "constants",
    "o": "constants",
    "l": "constants",
}

mv_data_map = {
    **dict.fromkeys(['mv', 'mv4'], ['r', 't', 'u', 'v']),
    'mv5': ['z', 'r', 't', 'u', 'v'],
    'uv10': ['u10', 'v10'],
    'mv12': ['lsm', 'o', 't2m', 'u10', 'v10', 'l', 'z', 'u', 'v', 't', 'r', 's']
}

data_keys_map = {
    'o': 'orography',
    'l': 'lat2d',
    's': 'q'
}


class WeatherBenchDataset(Dataset):
    """Wheather Bench Dataset <http://arxiv.org/abs/2002.00469>`_

    Args:
        data_root (str): Path to the dataset.
        data_name (str|list): Name(s) of the weather modality in Wheather Bench.
        training_time (list): The arrange of years for training.
        idx_in (list): The list of input indices.
        idx_out (list): The list of output indices to predict.
        step (int): Sampling step in the time dimension.
        levels (int|list|"all"): Level(s) to use.
        data_split (str): The resolution (degree) of Wheather Bench splits.
        use_augment (bool): Whether to use augmentations (defaults to False).
    """

    def __init__(self, data_root, data_name, training_time,
                 idx_in, idx_out, step=1, levels=['50'], data_split='5_625',
                 mean=None, std=None,
                 transform_data=None, transform_labels=None, use_augment=False):
        super().__init__()
        self.data_root = data_root
        self.data_split = data_split
        self.training_time = training_time
        self.idx_in = np.array(idx_in)
        self.idx_out = np.array(idx_out)
        self.step = step
        self.data = None
        self.mean = mean
        self.std = std
        self.transform_data = transform_data
        self.transform_labels = transform_labels
        self.use_augment = use_augment

        self.time = None
        self.time_size = self.training_time
        shape = int(32 * 5.625 / float(data_split.replace('_', '.')))
        self.shape = (shape, shape * 2)

        self.data, self.mean, self.std = [], [], []

        if levels == 'all':
            levels = ['50', '250', '500', '600', '700', '850', '925']
        levels = levels if isinstance(levels, list) else [levels]
        levels = [int(level) for level in levels]
        if isinstance(data_name, str) and data_name in mv_data_map:
            data_names = mv_data_map[data_name]
        else:
            data_names = data_name if isinstance(data_name, list) else [data_name]

        for name in tqdm.tqdm(data_names):
            data, mean, std = self._load_data_xarray(data_name=name, levels=levels)
            self.data.append(data)
            self.mean.append(mean)
            self.std.append(std)

        for i, data in enumerate(self.data):
            if data.shape[0] != self.time_size:
                self.data[i] = data.repeat(self.time_size, axis=0)

        self.data = np.concatenate(self.data, axis=1)
        self.mean = np.concatenate(self.mean, axis=1)
        self.std = np.concatenate(self.std, axis=1)

        self.valid_idx = np.array(
            range(-idx_in[0], self.data.shape[0]-idx_out[-1]-1))

    def _load_data_xarray(self, data_name, levels):
        """Loading full data with xarray"""
        try:
            dataset = xr.open_mfdataset(self.data_root+'/{}/{}*.nc'.format(
                data_map[data_name], data_map[data_name]), combine='by_coords')
        except (AttributeError, ValueError):
            assert False and 'Please install xarray and its dependency (e.g., netcdf4), ' \
                                'pip install xarray==0.19.0,' \
                                'pip install netcdf4 h5netcdf dask'
        except OSError:
            print("OSError: Invalid path {}/{}/*.nc".format(self.data_root, data_map[data_name]))
            assert False

        if 'time' not in dataset.indexes:
            dataset = dataset.expand_dims(dim={"time": 1}, axis=0)
        else:
            dataset = dataset.sel(time=slice(*self.training_time))
            dataset = dataset.isel(time=slice(None, -1, self.step))
            self.time_size = dataset.dims['time']

        if 'level' not in dataset.indexes:
            dataset = dataset.expand_dims(dim={"level": 1}, axis=1)
        else:
            dataset = dataset.sel(level=np.array(levels))

        if data_name in data_keys_map:
            data = dataset.get(data_keys_map[data_name]).values
        else:
            data = dataset.get(data_name).values

        mean = data.mean().reshape(1, 1, 1, 1)
        std = data.std().reshape(1, 1, 1, 1)
        data = (data - mean) / std

        return data, mean, std

    def _augment_seq(self, seqs, crop_scale=0.96):
        """Augmentations as a video sequence"""
        _, _, h, w = seqs.shape  # original shape, e.g., [4, 1, 128, 256]
        seqs = F.interpolate(seqs, scale_factor=1 / crop_scale, mode='bilinear')
        _, _, ih, iw = seqs.shape
        # Random Crop
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        seqs = seqs[:, :, x:x+h, y:y+w]
        # Random Flip
        if random.randint(0, 1):
            seqs = torch.flip(seqs, dims=(3, ))  # horizontal flip
        return seqs

    def __len__(self):
        return self.valid_idx.shape[0]

    def __getitem__(self, index):
        index = self.valid_idx[index]
        data = torch.tensor(self.data[index+self.idx_in])
        labels = torch.tensor(self.data[index+self.idx_out])
        if self.use_augment:
            len_data = self.idx_in.shape[0]
            seqs = self._augment_seq(torch.cat([data, labels], dim=0), crop_scale=0.96)
            data = seqs[:len_data, ...]
            labels = seqs[len_data:, ...]
        return data, labels