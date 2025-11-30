import os
import torch
from typing import List
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from .BaseVisualizer import BaseVisualizer

class MapVisualizer(BaseVisualizer):
    def __init__(self,
                 data_mean: List[float],
                 data_std: List[float],
                 lat: np.ndarray,
                 lon: np.ndarray,
                 save_dir: str = os.path.join(os.getcwd(), 'Visualization'),
                 ):
        super().__init__(data_mean=data_mean, data_std=data_std, save_dir=save_dir)
        self.lat = lat
        self.lon = lon

    def visualize(self,
                  true: np.ndarray,
                  pred: np.ndarray,
                  index: int,
                  state: str = 'val',
                  save_png: bool = True,
                  save_gif: bool = False,
                  heat_cmap: str = 'viridis',
                  ):
        # 确保输入的 true 和 pred 的形状为 [T, C, H, W]
        assert true.ndim == 3 and pred.ndim == 3, "Expected input shapes [T, H, W]."
        T = true.shape[0]  # 时间步数

        # 计算误差
        err = np.abs(true - pred)

        # 设置地图投影
        projection = ccrs.PlateCarree()

        # 根据时间步数动态调整图像大小
        figsize = (5 * T, 15)  # 每个时间步5英寸，高度15英寸（三行）
        fig, axes = plt.subplots(3, T, figsize=figsize, subplot_kw={'projection': projection})

        for t in range(T):
            # 绘制真实值
            ax_true = axes[0, t]
            ax_true.set_extent([self.lon.min(), self.lon.max(), self.lat.min(), self.lat.max()], crs=projection)
            ax_true.coastlines(resolution='110m')
            ax_true.add_feature(cfeature.BORDERS, linestyle=':')
            ax_true.add_feature(cfeature.LAND, facecolor='lightgray')
            ax_true.add_feature(cfeature.OCEAN, facecolor='lightblue')
            im_true = ax_true.pcolormesh(self.lon, self.lat, true[t], transform=projection, cmap='viridis',
                                         shading='auto')
            if t == 0:
                ax_true.set_ylabel('True', fontsize=12)
            ax_true.set_xticks([])
            ax_true.set_yticks([])

            # 绘制预测值
            ax_pred = axes[1, t]
            ax_pred.set_extent([self.lon.min(), self.lon.max(), self.lat.min(), self.lat.max()], crs=projection)
            ax_pred.coastlines(resolution='110m')
            ax_pred.add_feature(cfeature.BORDERS, linestyle=':')
            ax_pred.add_feature(cfeature.LAND, facecolor='lightgray')
            ax_pred.add_feature(cfeature.OCEAN, facecolor='lightblue')
            im_pred = ax_pred.pcolormesh(self.lon, self.lat, pred[t], transform=projection, cmap='viridis',
                                         shading='auto')
            if t == 0:
                ax_pred.set_ylabel('Pred', fontsize=12)
            ax_pred.set_xticks([])
            ax_pred.set_yticks([])

            # 绘制误差图
            ax_err = axes[2, t]
            ax_err.set_extent([self.lon.min(), self.lon.max(), self.lat.min(), self.lat.max()], crs=projection)
            ax_err.coastlines(resolution='110m')
            ax_err.add_feature(cfeature.BORDERS, linestyle=':')
            ax_err.add_feature(cfeature.LAND, facecolor='lightgray')
            ax_err.add_feature(cfeature.OCEAN, facecolor='lightblue')
            im_err = ax_err.pcolormesh(self.lon, self.lat, err[t], transform=projection, cmap='coolwarm',
                                       shading='auto')
            if t == 0:
                ax_err.set_ylabel('Error', fontsize=12)
            ax_err.set_xticks([])
            ax_err.set_yticks([])

            # 添加颜色条
            # 真值和预测值共享一个颜色条
        cbar_ax_true = fig.add_axes([0.1, 0.95, 0.8, 0.02])
        fig.colorbar(im_true, cax=cbar_ax_true, orientation='horizontal', label='Temperature (K)')

        # 误差图单独一个颜色条
        cbar_ax_err = fig.add_axes([0.1, 0.45, 0.8, 0.02])
        fig.colorbar(im_err, cax=cbar_ax_err, orientation='horizontal', label='Error (K)')

        plt.tight_layout(rect=[0, 0, 1, 0.9])

        # 保存为 PNG
        if save_png:
            filename = self._filename(index=index, state=state)
            plt.savefig(filename, dpi=300)

        # 这里暂不实现保存为 GIF 的功能

        plt.close(fig)