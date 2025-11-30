import os
import torch
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from .BaseVisualizer import BaseVisualizer


class HeatVisualizer(BaseVisualizer):
    def __init__(self,
                 data_mean: List[float],
                 data_std: List[float],
                 save_dir: str = os.path.join(os.getcwd(), 'Visualization'),
                 ):
        super().__init__(data_mean=data_mean, data_std=data_std, save_dir=save_dir)

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
       可视化方法，将视频数据的 T 个时间步绘制成每一行分别对应真实值、预测值、和误差值。

       Args:
           true (np.ndarray): 真实图像，形状为 [T, C, H, W]。
           pred (np.ndarray): 预测图像，形状为 [T, C, H, W]。
           index (int): 当前绘制的索引。
           state (str): 当前阶段，例如 'val' 或 'test'。
           save_png (bool): 是否保存为 PNG 文件。
           save_gif (bool): 是否保存为 GIF 文件。
           heat_cmap: viridis...
       """
        # heat_cmap = "Oranges"
        # 确保输入的 true 和 pred 的形状为 [T, C, H, W]
        assert true.ndim == 3 and pred.ndim == 3, "Expected input shapes [T, H, W]."
        T = true.shape[0]  # 时间步数

        true = (true - np.min(true)) / (np.max(true) - np.min(true))  # 0-1归一化
        pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))  # 0-1归一化


        # 处理输入图像, 将 [T, C, H, W] 处理为 [T, H, W]
        # true = [self._process_frame(true[t]) for t in range(T)]
        # pred = [self._process_frame(pred[t]) for t in range(T)]
        err = [np.abs(true[t] - pred[t]) for t in range(T)]

        vm = np.max(true)
        vn = np.min(true)

        # 创建图表，T 张图像在同一行
        fig, axes = plt.subplots(3, T, figsize=(T * 3, 9))  # 每行 T 张图片，高度为 9

        # 绘制真实值
        for t in range(T):
            axes[0, t].imshow(true[t], cmap=heat_cmap, vmin=vn, vmax=vm)  # viridis RdYlBu_r
            axes[0, t].set_title(f'True t={t}')
            axes[0, t].axis('off')

        # 绘制预测值
        for t in range(T):
            axes[1, t].imshow(pred[t], cmap=heat_cmap, vmin=vn, vmax=vm)
            axes[1, t].set_title(f'Pred t={t}')
            axes[1, t].axis('off')

        # 绘制误差图
        for t in range(T):
            axes[2, t].imshow(err[t], cmap=heat_cmap, vmin=vn, vmax=vm)
            axes[2, t].set_title(f'Error t={t}')
            axes[2, t].axis('off')

        plt.tight_layout()

        # 保存为 PNG
        if save_png:
            plt.savefig(self._filename(index=index, state=state))

        # # 将误差图添加到 GIF 帧
        # if save_gif:
        #     # 将误差图转换为 RGB 图像
        #     err_rgb = plt.cm.hot(err_img / err_img.max())[:, :, :3] if err_img.max() != 0 else plt.cm.hot(err_img)[
        #                                                                                        :, :, :3]
        #     self.gif_frames.append((err_rgb * 255).astype(np.uint8))
        #     print(f"Added frame to GIF. Total frames: {len(self.gif_frames)}")

        plt.close(fig)

    # def _process_frame(self, frame: np.ndarray) -> np.ndarray:
    #     """
    #     处理输入帧，确保为 [H, W] 并归一化到 0-255。
    #
    #     Args:
    #         frame (np.ndarray): 输入的图像帧，形状为 [C, H, W]。
    #
    #     Returns:
    #         np.ndarray: 处理后的图像，类型为 uint8。
    #     """
    #     # 检查帧的通道数
    #     if frame.ndim == 3:  # [C, H, W]
    #         # 如果通道数匹配mean和std的长度，逐个通道处理
    #         C = frame.shape[0]
    #         if len(self.data_mean) != C or len(self.data_std) != C:
    #             raise ValueError(f"data_mean 和 data_std 的长度必须与通道数 C={C} 匹配。")
    #
    #         processed_frame = np.zeros_like(frame)
    #         for c in range(C):
    #             processed_frame[c] = frame[c] * self.data_std[c] + self.data_mean[c]
    #
    #         # 如果只有一个通道，去掉通道维度
    #         if C == 1:
    #             processed_frame = processed_frame.squeeze(0)
    #
    #     elif frame.ndim == 2:  # [H, W]
    #         # 单通道情况，直接应用第一个均值和标准差
    #         processed_frame = frame * self.data_std[0] + self.data_mean[0]
    #
    #     else:
    #         raise ValueError(f"Unexpected frame shape: {frame.shape}. Expected [C, H, W] or [H, W].")
    #
    #     # 将数据裁剪到 0-255 范围内，并转换为 uint8
    #     processed_frame = np.clip(processed_frame, 0, 255).astype(np.uint8)
    #
    #     return processed_frame
