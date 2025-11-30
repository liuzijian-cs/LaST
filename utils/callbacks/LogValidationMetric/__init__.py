import torch
import torchmetrics
import lightning as L
import numpy as np
from utils import Color as Co, printf
from typing import Any, override, Optional, List, Dict
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT

from .MeanSquaredErrorVP import MeanSquaredErrorVP
from .MeanAbsoluteErrorVP import MeanAbsoluteErrorVP
from .RootMeanSquaredErrorVP import RootMeanSquaredErrorVP
from .SEVIRSkillScore import SEVIRSkillScore

__all__ = ['LogValidationMetric']

supported_metrics = {
    'mae': MeanAbsoluteErrorVP(),
    'mse': MeanSquaredErrorVP(),
    'rmse': RootMeanSquaredErrorVP(),
    'psnr': torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0),  # 默认采用归一化后的数据进行计算
    'ssim': torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0),  # 默认采用归一化后的数据进行计算
    'mae_pixel': torchmetrics.MeanAbsoluteError(),
    'mse_pixel': torchmetrics.MeanSquaredError(),
    'sevir': SEVIRSkillScore(layout="NTCHW"),

}


class LogValidationMetric(Callback):
    def __init__(self,
                 metrics=None,
                 inverse_transform=False,
                 data_mean: Optional[List[float]] = None,
                 data_std: Optional[List[float]] = None):
        super().__init__()
        self.inverse_transform = inverse_transform
        self.data_mean = data_mean
        self.data_std = data_std

        self.metrics = torch.nn.ModuleDict({})
        self.__init_metrics(metrics)

    def __init_metrics(self, metrics):
        metrics = metrics if metrics is not None else ['mse']
        for metric in metrics:
            if metric in supported_metrics:
                self.metrics[metric] = supported_metrics[metric]
            else:
                printf(s='LogValidationMetric', err='ValueError', m=f'{Co.C}{metric}{Co.RE} is not supported.')
                raise ValueError(
                    f'[utils/callbacks/LogValidationMetric/__init__.py] Invalid metrics: {metric}, please reconfigure parser.metrics.')

    def _metric_update(self, seq_pred, seq_true):  # 弃用
        seq_pred = seq_pred.reshape(-1, *seq_pred.shape[2:])  # (B, T, C, H, W) -> (B*T, C, H, W)
        seq_true = seq_true.reshape(-1, *seq_true.shape[2:])  # (B, T, C, H, W) -> (B*T, C, H, W)

        for metric in self.metrics:
            self.metrics[metric].update(seq_pred.detach().float(), seq_true.detach().float())

    def sevir_log_score_epoch_end(self, score_dict: Dict, prefix, pl_module):

        for metrics in ['csi', 'pod', 'sucr', 'bias']:
            for thresh in [16, 74, 133, 160, 181, 219]:
                score_mean = np.mean(score_dict[thresh][metrics]).item()
                pl_module.log(f"{prefix}_{metrics}_{thresh}_epoch", score_mean,
                              prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            score_avg_mean = score_dict.get("avg", None)
            if score_avg_mean is not None:
                score_avg_mean = np.mean(score_avg_mean[metrics]).item()
                pl_module.log(f"{prefix}_{metrics}_avg_epoch", score_avg_mean,
                              prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

    def _metrics_compute(self, pl_module):
        result = {}
        for metric in self.metrics:
            if metric == "sevir":
                self.sevir_log_score_epoch_end(self.metrics[metric].compute(), "valid", pl_module)
            else:
                result[metric] = self.metrics[metric].compute()
        return result

    def _metrics_reset(self):
        for metric in self.metrics:
            self.metrics[metric].reset()

    @override
    def setup(self, trainer: "L.Trainer", pl_module: "L.LightningModule", stage: str = None) -> None:
        # 将指标模块移动到模型所在的设备
        self.metrics = self.metrics.to(pl_module.device)

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
        seq_pred = outputs['seq_pred'].reshape(-1, *outputs['seq_pred'].shape[2:])  # (B, T, C, H, W) -> (B*T, C, H, W)
        seq_true = outputs['seq_true'].reshape(-1, *outputs['seq_true'].shape[2:])  # (B, T, C, H, W) -> (B*T, C, H, W)

        if self.inverse_transform:
            # if seq_pred.shape[2] == 1:
            #     seq_pred = seq_pred * self.data_std[0] + self.data_mean[0]  # todo channel == 1 only
            #     seq_true = seq_true * self.data_std[0] + self.data_mean[0]
            # else:
            for c in range(seq_pred.shape[1]):
                seq_pred[:,  c, :, :] = seq_pred[:,  c, :, :] * self.data_std[c] + self.data_mean[c]
                seq_true[:,  c, :, :] = seq_true[:,  c, :, :] * self.data_std[c] + self.data_mean[c]

        if seq_pred is None or seq_true is None:
            raise ValueError("[LogValidationMetric callback] Outputs must contain 'seq_pred' and 'seq_true'.")

        for metric in self.metrics:
            self.metrics[metric].update(seq_pred.detach().float().contiguous(), seq_true.detach().float().contiguous())

    @override
    def on_validation_epoch_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        result = self._metrics_compute(pl_module)
        pl_module.log_dict(result, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self._metrics_reset()

    @override
    def on_test_batch_end(
        self,
        trainer: "L.Trainer",
        pl_module: "L.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        return self.on_validation_batch_end(trainer, pl_module, outputs,batch, batch_idx, dataloader_idx)

    @override
    def on_test_epoch_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        return self.on_validation_epoch_end(trainer, pl_module)


