import os
import torch
import lightning as L
from utils import Color as Co, printf
from typing import Any, override, Union, List
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT

from .MapVisualizer import MapVisualizer
from .GrayVisualizer import GrayVisualizer
from .HeatVisualizer import HeatVisualizer
from .RGBVisualizer import RGBVisualizer

supported_visualizer = {
    'gray': GrayVisualizer,
    'heat': HeatVisualizer,
    'rgb': RGBVisualizer,
    'map': MapVisualizer,

}

__all__ = ['Visualizer']


class Visualizer(Callback):
    def __init__(self,
                 style: str,
                 i_image: int = 0,
                 data_mean: float = 0.0,
                 data_std: float = 255.0,
                 n_val_img: Union[int, str] = 10,
                 n_test_img: Union[int, str] = 10,
                 save_dir: str = os.path.join(os.getcwd(), 'Visualization'),
                 save_png: bool = True,
                 save_gif: bool = False,
                 draw_per_channel: bool = False,
                 heat_cmap="viridis",
                 ):
        super().__init__()
        """
        初始化 Visualizer 回调函数
        
        style (str): 可视化风格，例如 'gray', 'heat', 'rgb', 'remote_sensing'。
        s_data_y (List[int]): 绘制图片的大小信息[T,C,H,W]，如[10,1,64,64]。
        i_image (int): 验证&测试阶段每个批次中要绘制的图片索引。
        n_val_img (int or 'all'): 验证阶段绘制的图片数量或 'all'。
        n_test_img (int or 'all'): 测试阶段绘制的图片数量或 'all'。
        save_dir: 图片输出路径。
        save_png (bool): 是否保存为 PNG 文件。
        save_gif (bool): 是否保存为 GIF 文件。
        """
        self.save_path = save_dir
        self.i_image = i_image
        self.n_val_img = n_val_img
        self.n_test_img = n_test_img
        self.draw_per_channel = draw_per_channel
        self.save_png = save_png
        self.save_gif = save_gif
        self.heat_cmap = heat_cmap

        os.makedirs(save_dir, exist_ok=True)
        visualizer = supported_visualizer[style]
        if visualizer is None:
            raise ValueError(
                f'Unknown visualization style: {style}, You can modify the utils/callbacks, the Visualization to add support Visualization methods')
        self.visualizer = visualizer(data_mean=data_mean, data_std=data_std, save_dir=self.save_dir)

        # 初始化计数器
        self.val_image_count = 0
        self.test_image_count = 0

    @property
    def save_dir(self) -> str:
        return os.path.join(self.save_path, 'Visualization')

    # @override
    # def on_sanity_check_start(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
    #     os.makedirs(self.save_dir, exist_ok=True)

    @override
    def on_validation_start(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        os.makedirs(self.save_dir, exist_ok=True)

    @override
    def on_test_start(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        os.makedirs(self.save_dir, exist_ok=True)

    @override
    def on_validation_epoch_start(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        self.val_image_count = 0

    @override
    def on_test_epoch_start(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        self.test_image_count = 0

    @override
    def on_test_batch_end(self,
                           trainer: "L.Trainer",
                           pl_module: "L.LightningModule",
                           outputs: STEP_OUTPUT,
                           batch: Any,
                           batch_idx: int,
                           dataloader_idx: int = 0,
                           ) -> None:
        return self.on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    @override
    def on_validation_batch_end(
            self,
            trainer: "L.Trainer",
            pl_module: "L.LightningModule",
            outputs: STEP_OUTPUT,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        if self.val_image_count >= self.n_val_img:  # 已达到绘制数量
            return

        seq_pred = outputs['seq_pred']
        seq_true = outputs['seq_true']

        if seq_pred is None or seq_true is None:
            raise ValueError("[Visualization callback] Outputs must contain 'seq_pred' and 'seq_true'.")

        batch_size = seq_pred.shape[0]
        if self.i_image > batch_size:
            printf(s="Visualization Callback", err="Index of image(i_image) must be greater than batch_size",
                   m=f"Index of image(i_image) will be set to 0.")
            self.i_image = 0

        if self.draw_per_channel:
            for channel in range(seq_pred.shape[2]):
                self.visualizer.visualize(
                    true=seq_true[self.i_image, :, channel, :, :].cpu().numpy(),  # [T,H,W]
                    pred=seq_pred[self.i_image, :, channel, :, :].cpu().numpy(),  # [T,H,W]
                    index=trainer.current_epoch,
                    state=f"val_{batch_idx}_{self.i_image}_channel{channel}",
                    save_png=self.save_png,
                    save_gif=self.save_gif,
                    heat_cmap=self.heat_cmap,
                )
        else:
            self.visualizer.visualize(
                true=seq_true[self.i_image].cpu().numpy(),  # [T, C, H, W]
                pred=seq_pred[self.i_image].cpu().numpy(),  # [T, C, H, W]
                index=trainer.current_epoch,
                state=f"val_{batch_idx}_{self.i_image}",
                save_png=True,
                save_gif=False,
            )
        self.val_image_count += 1
