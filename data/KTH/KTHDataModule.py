import os
import cv2
import torch
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import lightning as L
from utils import printf, Color as Co


class KTHDataModule(L.LightningDataModule):
    def __init__(self,
                 data_root: str = os.path.join(os.getcwd(), "data"),
                 batch_size: int = 32,
                 val_batch_size: int = 32,
                 num_workers: int = 4,
                 len_back: int = 10,
                 len_pred: int = 10,
                 img_size: int = None,  # if img_size = 128 > [128,128] else default [120,160]
                 ):
        super().__init__()
        self.data_dir = os.path.join(data_root, "KTH")
        # self.data_dir = data_root
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.len_back = len_back
        self.len_pred = len_pred
        self.img_size = img_size

        # Config:
        self.category = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']
        self.train_id = [f"{i:02}" for i in range(1, 17)]  # person 1-16 for train(16)
        self.test_id = [f"{i:02}" for i in range(17, 26)]  # person 17-25 for test(9)

        self.train_dataset = self.valid_dataset = self.test_dataset = None

        self.save_hyperparameters()

    def data_integrity_check(self):
        # 数据完整性检查：检查目录 self.data_dir 是否存在 self.category 中的文件夹
        printf(s="Setup data", m=f"{Co.B}Checking data integrity >>> {Co.C}{self.data_dir}{Co.RE} ")
        missing_folders = []
        for category in self.category:
            folder_path = os.path.join(self.data_dir, category)
            if not os.path.isdir(folder_path):
                missing_folders.append(category)
        if not missing_folders:
            printf(s="Setup data", m=f"{Co.B}Checking data integrity [{Co.G}Successful{Co.B}]{Co.RE} ")
        else:
            printf(s="Setup data", err="Missing folder", m=f"{', '.join(missing_folders)}")
            raise FileNotFoundError("[data/KTH/KTHDataModule] Missing folder Error.")

    def create_data(self, stage: str = 'train'):
        assert stage in ['train', 'valid', 'test']
        printf(s="Setup data",
               m=f"{Co.B}Creating {Co.C}{stage}{Co.B} dataset(task:{Co.C}{self.len_back}{Co.B} > {Co.C}{self.len_pred}{Co.B}, image size:[{Co.C}{self.img_size}*{self.img_size}{Co.B}]){Co.RE} ")
        self.data_integrity_check()
        person_id = self.train_id if stage == 'train' else self.test_id

        current_category = -1  # 0-boxing, 1-hand clapping, 2-hand waving, 3-walking, 4-jogging, 5-running

        frames = []  # video frame
        indices = []  # The starting index of the sequence for each person
        index = 0

        with tqdm(total=len(self.category) * len(person_id) * 4, desc=f"{Co.P}[Processing videos ]{Co.RE} ",
                  unit="video") as pbar:
            for category in self.category:
                current_category += 1
                for pid in person_id:
                    for d in ['d1', 'd2', 'd3', 'd4']:
                        file = "person" + pid + "_" + category + "_" + d + "_uncomp.avi"
                        pbar.set_description(
                            f"{Co.P}[Processing videos ] : {Co.B}(total {len(frames)} frames) {Co.C}{category}{Co.B} - {Co.C}{file}{Co.RE}")
                        # The official handclapping will be missing this video file & the second video is damaged.
                        if file == "person13_handclapping_d3_uncomp.avi" or file == "person01_boxing_d4_uncomp.avi":
                            pbar.update(1)
                            continue
                        path = os.path.join(self.data_dir, category, file)
                        if not os.path.exists(path):
                            printf(s="Setup data", err="Missing files",
                                   m=f"{Co.B}Video file {Co.C}{path}{Co.B} is not exist.{Co.RE}")
                            raise FileNotFoundError
                        cap = cv2.VideoCapture(path)
                        if not cap.isOpened():
                            printf(s="Setup data", err=f"Unable to open the video file {Co.C}{path}{Co.RE}", )
                            raise FileNotFoundError("Unable to open the video file")

                        indices.append(index)
                        len_frame = 0
                        # Extracting video frames
                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            # Generate Frames
                            frame = frame[:, :, 0]  # Convert to grayscale by extracting the first channel
                            frame = cv2.resize(frame,
                                               (self.img_size, self.img_size)) if self.img_size is not None else frame
                            frames.append(frame)
                            len_frame = len_frame + 1
                        index = index + len_frame
                        pbar.update(1)
        frames = np.stack(frames, axis=0).astype(np.uint8)
        printf(s="Setup data",
               m=f"{Co.G}Created dateset successfully, {Co.C}{len(frames)} {Co.B}frames with {Co.C}{len(indices)} {Co.B}videos, {Co.Y}Saving file, please waiting... {Co.RE} ")
        saving_path = os.path.join(self.data_dir, stage)
        np.savez_compressed(saving_path, frames=frames, indices=indices)
        # Load: data = np.load('train.npz', allow_pickle=True)  frames = data['frames']  indices = data['indices']
        printf(s="Setup data",
               m=f"{Co.G}Created dateset successfully, {Co.C}{len(frames)} {Co.B}frames with {Co.C}{len(indices)} {Co.B}videos, file has been saved to {Co.C}{saving_path}{Co.RE} ")

    def prepare_data(self):
        train_data_path = os.path.join(self.data_dir, 'train.npz')
        valid_data_path = os.path.join(self.data_dir, 'valid.npz')
        if not os.path.exists(train_data_path):
            self.create_data(stage='train')
        if not os.path.exists(valid_data_path):
            self.create_data(stage='valid')

    def setup(self, stage=None):
        self.train_dataset = KTH(data_dir=self.data_dir, is_train=True, len_back=self.len_back, len_pred=self.len_pred)
        self.valid_dataset = KTH(data_dir=self.data_dir, is_train=False, len_back=self.len_back, len_pred=self.len_pred)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.val_batch_size, shuffle=False,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return self.val_dataloader()

    def predict_dataloader(self):
        return self.val_dataloader()



class KTH(Dataset):
    def __init__(self,
                 data_dir: str,
                 is_train: bool = True,
                 len_back: int = 10,
                 len_pred: int = 20,
                 ):
        self.data_dir = data_dir
        self.is_train = is_train
        self.len_back = len_back
        self.len_pred = len_pred

        data_file = os.path.join(data_dir, 'train.npz') if is_train else os.path.join(data_dir, 'valid.npz')
        try:
            data = np.load(data_file, allow_pickle=True)
        except FileNotFoundError:
            raise Exception(f"Data file not found: {data_file}")
        self.frames = data['frames']
        self.indices = self.generate_indices(data['indices'])

    def generate_indices(self, indices_frame_start):
        indices = []
        indices_frame_end = list(indices_frame_start[1:]) + [len(self.frames)]
        for start_idx, end_idx in zip(indices_frame_start, indices_frame_end):
            max_start_idx = end_idx - (self.len_back + self.len_pred) + 1
            for idx in range(start_idx, max_start_idx):
                indices.append(idx)
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        batch_indices = self.indices[idx]
        start_idx = batch_indices
        end1_idx = start_idx + self.len_back
        end2_idx = end1_idx + self.len_pred
        data_back = torch.tensor(self.frames[start_idx:end1_idx], dtype=torch.float32).unsqueeze(1) / 255.0
        data_pred = torch.tensor(self.frames[end1_idx:end2_idx], dtype=torch.float32).unsqueeze(1) / 255.0
        return data_back, data_pred


if __name__ == '__main__':
    test_data_module = KTHDataModule(
        data_root="",
        batch_size=4,
        val_batch_size=4,
        num_workers=0,
        len_back=10,
        len_pred=10,
        img_size=128,
    )
    test_data_module.prepare_data()
