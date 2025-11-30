import os
import abc
import numpy as np
from typing import Optional, List


class BaseVisualizer(abc.ABC):
    """
    抽象基类，定义所有可视化器类必须实现的接口。
    """

    def __init__(self,
                 data_mean: List[float] = 0.0,
                 data_std: List[float] = 255.0,
                 save_dir: str = os.path.join(os.getcwd(), 'Visualization'),
                 ):
        """
        初始化 BaseVisualizer。

        Args:
            save_dir (str): 文件存储的目录路径。
            filename (str): 文件的基本名称（不包含扩展名）。
        """
        self.data_mean = data_mean
        self.data_std = data_std
        self.save_dir = save_dir

    @abc.abstractmethod
    def visualize(self,
                  true: np.ndarray,
                  pred: np.ndarray,
                  index: int,
                  state: str = 'val',
                  save_png: bool = True,
                  save_gif: bool = False,
                  heat_cmap: str = 'viridis',
                  ):
        """
        处理和转换输入的图像帧。

        Args:
            true (np.ndarray): 输入的真实图像帧，形状通常为 [C, H, W] 或 [H, W]。
            pred (np.ndarray): 输入的预测图像帧，形状通常为 [C, H, W] 或 [H, W]。
            index (int): 当前帧或图像的索引，用于区分保存的文件。
            state (str): 处于的阶段 ‘val’ or 'test'。
            save_png (bool): 是否保存为 PNG 文件。
            save_gif (bool): 是否保存为 GIF 文件。
        """
        pass

    def _filename(self, index: int, state: str = "", ext: str = "png") -> str:
        return os.path.join(self.save_dir, f"{index:03}_{state}.{ext}")
