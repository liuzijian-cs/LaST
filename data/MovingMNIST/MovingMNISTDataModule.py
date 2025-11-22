import numpy as np
import torch
import random
import lightning as L
import os
import gzip
import torchvision
from torch.utils.data import DataLoader, Dataset


class MovingMNISTDataModule(L.LightningDataModule):
    def __init__(self,
                 data_root: str = os.path.join(os.getcwd(), "data"),
                 batch_size: int = 32,
                 val_batch_size: int = 32,
                 num_workers: int = 4,
                 len_back: int = 10,
                 len_pred: int = 10,
                 img_size: int = 64,
                 num_objects: list = None,
                 dataset_length: int = None,
                 ):
        super().__init__()
        self.data_dir = os.path.join(data_root, "MovingMNIST")
        self.batch_size = batch_size
        self.valid_batch_size = val_batch_size
        self.num_workers = num_workers
        self.len_back = len_back
        self.len_pred = len_pred
        self.image_size = img_size
        self.num_objects = num_objects if num_objects is not None else [2]
        self.dataset_length = int(dataset_length if dataset_length is not None else int(1e4))
        self.train_dataset = self.valid_dataset = self.test_dataset = None

        self.save_hyperparameters()

    def prepare_data(self):
        # Download MNIST datasets
        torchvision.datasets.MNIST(self.data_dir, train=True, download=True)
        torchvision.datasets.MNIST(self.data_dir, train=False, download=True)
        torchvision.datasets.MovingMNIST(self.data_dir, download=True)

    def setup(self, stage=None):
        self.train_dataset = MovingMNIST(data_dir=self.data_dir, is_train=True,
                                         len_back=self.len_back, len_pred=self.len_pred, img_size=self.image_size,
                                         num_objects=self.num_objects, dataset_length=self.dataset_length)
        self.valid_dataset = MovingMNIST(data_dir=self.data_dir, is_train=False,
                                         len_back=self.len_back, len_pred=self.len_pred, img_size=self.image_size,
                                         num_objects=self.num_objects, dataset_length=self.dataset_length)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.valid_batch_size, shuffle=False,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.valid_batch_size, shuffle=False,
                          num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.valid_batch_size, shuffle=False,
                          num_workers=self.num_workers)


class MovingMNIST(Dataset):
    def __init__(self,
                 data_dir: str,
                 is_train: bool = True,
                 len_back: int = 10,
                 len_pred: int = 10,
                 img_size: int = 64,
                 num_objects: list = None,
                 dataset_length: int = None,
                 ):
        super(MovingMNIST, self).__init__()
        self.data_dir = data_dir
        self.is_train = is_train
        self.len_back = len_back
        self.len_pred = len_pred
        self.img_size = img_size
        self.num_objects = num_objects if num_objects is not None else [2]
        self.dataset_length = int(dataset_length if dataset_length is not None else int(1e4))
        self.mnist = None
        self.dataset = None
        self.digit_size_ = 28
        self.step_length_ = 0.1

        self.train_path = os.path.join(self.data_dir, 'MovingMNIST', 'mnist_train_seq.npy')  # useless
        self.test_path = os.path.join(self.data_dir, 'MovingMNIST', 'mnist_test_seq.npy')

        if self.is_train or self.num_objects[0] != 2:
            with gzip.open(os.path.join(self.data_dir, "MNIST", 'raw', "train-images-idx3-ubyte.gz"), 'rb') as f:
                self.mnist = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)  # [60000,28,28]
        else:
            self.dataset = np.load(self.test_path) if os.path.exists(self.test_path) else self._generate_dataset()
            self.dataset = self.dataset[:, :, np.newaxis, :, :]  # add channel dim [20,10000,1,64,64]

    def _generate_dataset(self, is_train: bool = True):
        with gzip.open(os.path.join(self.data_dir, 'MNIST', 'raw', "train-images-idx3-ubyte.gz"), 'rb') as f:
            self.mnist = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)  # (60000,28,28)
        sequences = []
        for _ in range(self.dataset_length):
            num_digits = random.choice(self.num_objects)  # Sample number of objects
            images = self._generate_moving_mnist(num_digits)  # [20,64,64,1]
            sequences.append(images)
        new_dataset = np.stack(sequences, axis=1).astype(np.uint8)
        np.save(self.train_path if is_train else self.test_path, new_dataset)  # save
        return new_dataset  # [20,10000,64,64]

    def _get_random_trajectory(self, back_len):
        ''' Generate a random sequence of a MNIST digit '''
        canvas_size = self.img_size - self.digit_size_
        x = random.random()
        y = random.random()
        theta = random.random() * 2 * np.pi
        v_y = np.sin(theta)
        v_x = np.cos(theta)
        start_y = np.zeros(back_len)
        start_x = np.zeros(back_len)
        for i in range(back_len):
            # Take a step along velocity.
            y += v_y * self.step_length_
            x += v_x * self.step_length_

            # Bounce off edges.
            if x <= 0:
                x = 0
                v_x = -v_x
            if x >= 1.0:
                x = 1.0
                v_x = -v_x
            if y <= 0:
                y = 0
                v_y = -v_y
            if y >= 1.0:
                y = 1.0
                v_y = -v_y
            start_y[i] = y
            start_x[i] = x
        # Scale to the size of the canvas.
        start_y = (canvas_size * start_y).astype(np.int32)
        start_x = (canvas_size * start_x).astype(np.int32)
        return start_y, start_x

    def _generate_moving_mnist(self, num_digits=2):
        '''
        Get random trajectories for the digits and generate a video.
        '''
        data = np.zeros((self.len_back + self.len_pred, self.img_size, self.img_size), dtype=np.float32)
        for n in range(num_digits):
            # Trajectory
            start_y, start_x = self._get_random_trajectory(self.len_back + self.len_pred)
            ind = random.randint(0, self.mnist.shape[0] - 1)
            digit_image = self.mnist[ind]
            for i in range(self.len_back + self.len_pred):
                top = start_y[i]
                left = start_x[i]
                bottom = top + self.digit_size_
                right = left + self.digit_size_
                # Draw digit
                data[i, top:bottom, left:right] = np.maximum(data[i, top:bottom, left:right], digit_image)
        return data

    def __getitem__(self, idx):
        # r, w = 1, self.image_size  # w = int(64 / r)
        # images = images.reshape((length, w, r, w, r)).transpose(0, 2, 4, 1, 3).reshape((length, r * r, w, w))
        images = self._generate_moving_mnist(random.choice(self.num_objects))[:, np.newaxis, :, :] if self.is_train \
            else self.dataset[:, idx, ...]
        return (torch.from_numpy(images[:self.len_back] / 255.0).contiguous().float(),
                torch.from_numpy(images[self.len_back:self.len_back + self.len_pred] / 255.0).contiguous().float())

    def __len__(self):
        return self.dataset_length


if __name__ == '__main__':
    # test
    import matplotlib.pyplot as plt


    def visualize_data(input_data, output_data):
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(input_data[0, 0, 0].numpy(), cmap='gray')
        ax[0].set_title("Sample Input")
        ax[1].imshow(output_data[0, 0, 0].numpy(), cmap='gray')
        ax[1].set_title("Sample Output")
        plt.show()


    # TEST LightningDataModule:
    data_module_lightning = MovingMNISTDataModule(
        data_dir=os.path.join(os.getcwd(), "data"),
        batch_size=1,
        num_workers=0,
        len_back=10,
        len_pred=10,
        image_size=64,
        num_objects=[2],
        dataset_length=10000
    )
    data_module_lightning.prepare_data()
    data_module_lightning.setup()
    train_loader = data_module_lightning.train_dataloader()
    test_loader = data_module_lightning.test_dataloader()
    # 显示前5个批次的数据
    for i, (input_data, output_data) in enumerate(train_loader):
        if i >= 5:
            break
        print(f"Train Batch {i + 1} - Input shape: {input_data.shape}, Output shape: {output_data.shape}")
        visualize_data(input_data, output_data)
    for i, (input_data, output_data) in enumerate(test_loader):
        if i >= 5:
            break
        print(f"Test Batch {i + 1} - Input shape: {input_data.shape}, Output shape: {output_data.shape}")
        visualize_data(input_data, output_data)

        # # test MovingMNIST
        # print(os.path.join(os.getcwd(), "data"))
        # dataset = MovingMNIST(data_dir=os.path.join(os.getcwd(), "data"), is_train=True, seq_len=10, pred_len=10,
        #                       image_size=64, num_objects=[2],
        #                       dataset_length=1e4)
        #
        # data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
        # # Fetch data and display
        # for i, (input_data, output_data) in enumerate(data_loader):
        #     if i >= 5:  # Display 5 batches
        #         break
        #     print(f"Batch {i + 1}")
        #     print("Input data shape:", input_data.shape)
        #     print("Output data shape:", input_data.shape)
        #     # Visualization
        #     fig, ax = plt.subplots(1, 2)
        #     ax[0].imshow(input_data[0, 0, 0].numpy(), cmap='gray')
        #     ax[0].set_title("Sample Input")
        #     ax[1].imshow(output_data[0, 0, 0].numpy(), cmap='gray')
        #     ax[1].set_title("Sample Output")
        #     plt.show()
