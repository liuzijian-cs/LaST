"""
关于 PredFormer 实现的说明：

本项目中集成的 PredFormer 模块，系本人基于原论文（2024年10月版）思想进行的非官方复现（完成于2024年11月）。

值得说明的是，在复现过程中，我们基于 PredFormer-TS 架构进行了细致的调试。实验结果显示，该复现版本在特定测试条件下展现出了令人惊喜的性能，甚至在部分指标上略优于原论文汇报数据。出于实验一致性的考虑，本项目中汇报的所有数据均基于此复现版本生成。

我们对 PredFormer 原作者卓越的研究工作致以崇高的敬意。原作者已开源了基于 OpenSTL 的官方实现，建议各位研究同仁参考官方代码以获取最原汁原味的设计思路。

EN：

The PredFormer module integrated into this project is an unofficial reproduction based on the concepts presented in the original paper (October 2024 version), completed by the author in November 2024.

It is worth noting that during the reproduction process, we conducted meticulous tuning based on the PredFormer-TS architecture. Experimental results indicate that this reproduced version exhibits surprisingly strong performance under specific testing conditions, with some metrics slightly surpassing those reported in the original paper. To maintain experimental consistency, all data reported in this project are generated based on this reproduction.

We extend our highest respect to the original authors of PredFormer for their outstanding research. The authors have released an official implementation based on OpenSTL, and we strongly recommend that fellow researchers consult the official repository to grasp the most authentic design philosophy.
"""

import torch
import torch.nn as nn
import lightning as L
from typing import List

from .module import PatchEmbedding, PatchReconstruction, GatedAttentionUnit


class PredFormer(L.LightningModule):
    def __init__(self,
                 d_model: int = 128,
                 s_patch: int = 4,
                 n_heads: int = 8,
                 n_layer: int = 4,
                 r_forward: int = 4,
                 dropout: float = 0.0,
                 drop_path: float = 0.0,
                 data_back: List[int] = None,
                 data_pred: List[int] = None,
                 lr: float = 1e-3,
                 lr_scheduler='cosine',
                 attn_bias: bool = False,
                 ):
        super().__init__()

        # Parameters:
        assert lr_scheduler in ['cosine', 'onecycle']
        self.data_back = data_back if data_back is not None else [4, 2, 32, 32]
        self.data_pred = data_pred if data_pred is not None else [4, 2, 32, 32]
        self.lr = lr
        self.lr_scheduler = lr_scheduler

        # Lightning Framework:
        self.example_input_array = torch.rand(1, *self.data_back)
        self.save_hyperparameters()

        # Criterion:
        self.criterion = nn.MSELoss(reduction='mean')

        # Spatial Patch Embedding & Spatial Patch Reconstruction:
        self.patch_embedding = PatchEmbedding(d_model, s_patch, self.data_back)
        self.patch_reconstruction = PatchReconstruction(d_model, s_patch, self.data_back, self.data_pred)

        # Spatio-Temporal Local Perception Attention:
        drop_path_rate = [drop_path for _ in range(n_layer)]  # Default
        # drop_path_rate = [drop_path * (i / (n_layer - 1)) for i in range(n_layer)]

        self.attn = nn.ModuleList(
            nn.ModuleList([
                GatedAttentionUnit(d_model, n_heads, dropout, drop_path_rate[i], r_forward, attn_bias),
                GatedAttentionUnit(d_model, n_heads, dropout, drop_path_rate[i], r_forward, attn_bias),
            ]) for i in range(n_layer)
        )

    def forward(self, x: torch.Tensor, return_attn: bool = False) -> torch.Tensor:
        x = self.patch_embedding(x)
        B, D, T, H, W = x.shape
        N = H * W
        x = x.flatten(-2, -1).permute(0, 2, 3, 1)  # [B,T,N,D]

        for (t_attn, s_attn) in self.attn:
            x = t_attn(x.transpose(1, 2).flatten(0, 1)).reshape(B, N, T, D)
            x = s_attn(x.transpose(1, 2).flatten(0, 1)).reshape(B, T, N, D)
        x = x.permute(0, 3, 1, 2).reshape(B, D, T, H, W)

        x = self.patch_reconstruction(x)
        return x

    def shared_step(self, batch):
        seq_back, seq_true = batch
        seq_pred = self(seq_back)
        loss = self.criterion(seq_pred, seq_true)
        return loss, seq_pred, seq_true

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.shared_step(batch)
        self.log('train_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, seq_pred, seq_true = self.shared_step(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        return {  # Callback: LogValidationMetric need 'seq_pred' and 'seq_true'
            'loss': loss,
            'seq_pred': seq_pred,
            'seq_true': seq_true,
        }

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-2)  # AdamW Optimizer
        if self.lr_scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=self.trainer.estimated_stepping_batches,
                eta_min=5e-8
            )
        elif self.lr_scheduler == 'onecycle':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                max_lr=self.lr,
                epochs=self.trainer.max_epochs,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.3,
                div_factor=25,
                final_div_factor=10000,
            )
        else:
            scheduler = None
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }