import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from .BaseVisualizer import BaseVisualizer


class RGBVisualizer(BaseVisualizer):
    def __init__(self,
                 data_mean: List[float] = 0.0,
                 data_std: List[float] = 255.0,
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
                  heat_cmap: str = 'hot',
                  ):
        assert true.ndim == 4 and pred.ndim == 4, "Expected input shapes [T, C, H, W]."
        T = true.shape[0]

        true = np.clip(true, 0, 1).astype(np.float32)  # 强制确保数值在 [0, 1] 范围内
        pred = np.clip(pred, 0, 1).astype(np.float32)

        err = np.abs(true - pred)
        err = np.clip(err, 0, 1).astype(np.float32)

        # 创建一个图表，T 张图像分三行显示
        fig, axes = plt.subplots(3, T, figsize=(T * 3, 9))  # 三行，T列，每行是真实值、预测值和误差值

        if T == 1:
            axes[0].imshow(true[0, :, :, :].transpose(1, 2, 0)[..., ::-1])  # 转换为 HxWxC
            axes[0].set_title(f'True t={0}')
            axes[0].axis('off')
            axes[1].imshow(pred[0, :, :, :].transpose(1, 2, 0)[..., ::-1])  # 转换为 HxWxC
            axes[1].set_title(f'Pred t={0}')
            axes[1].axis('off')
            axes[2].imshow(err[0, :, :, :].transpose(1, 2, 0)[..., ::-1], cmap='hot', vmin=0, vmax=1)  # 转换为 HxWxC
            axes[2].set_title(f'Error t={0}')
            axes[2].axis('off')
        else:
            # 绘制真实值图像
            for t in range(T):
                axes[0, t].imshow(true[t, :, :, :].transpose(1, 2, 0)[..., ::-1])  # 转换为 HxWxC
                axes[0, t].set_title(f'True t={t}')
                axes[0, t].axis('off')

            # 绘制预测值图像
            for t in range(T):
                axes[1, t].imshow(pred[t, :, :, :].transpose(1, 2, 0)[..., ::-1])  # 转换为 HxWxC
                axes[1, t].set_title(f'Pred t={t}')
                axes[1, t].axis('off')

            # 绘制误差图
            for t in range(T):  # 转换为 HxWxC
                axes[2, t].imshow(err[t, :, :, :].transpose(1, 2, 0)[..., ::-1], cmap=heat_cmap, vmin=0, vmax=1)
                axes[2, t].set_title(f'Error t={t}')
                axes[2, t].axis('off')

        plt.tight_layout()

        # 保存为 PNG
        if save_png:
            plt.savefig(self._filename(index=index, state=state))

        # 保存为 GIF (如果需要)
        if save_gif:
            # 如果需要保存 GIF 可以继续扩展这部分代码
            pass

        plt.close(fig)
