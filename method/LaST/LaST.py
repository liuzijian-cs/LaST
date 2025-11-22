import torch
import torch.nn as nn
import lightning as L
from typing import List

from .module import SpatialPatchEmbedding, SpatialPatchReconstruction, SpatialLocalAwareAttentionBlock, \
    TemporalLocalAwareAttentionBlock


class LaST(L.LightningModule):
    def __init__(self,
                 d_model: int = 128,
                 s_patch: int = 4,
                 n_heads: int = 8,
                 n_layer: int = 4,
                 s_kernel: int = 1,
                 r_forward: int = 4,
                 dropout: float = 0.0,
                 drop_path: float = 0.0,
                 s_local: List[int] = None,
                 data_back: List[int] = None,
                 data_pred: List[int] = None,
                 lr: float = 1e-3,
                 lr_scheduler='cosine',
                 attn_bias: bool = False,
                 ):
        """
            d_model:   Hidden dimension.
            s_patch:   Spatial patch size; side length of square spatial patches.
            n_heads:   Number of attention heads; parallel heads in multi-head attention.
            n_layer:   Number of STLAA layers; stacked spatio-temporal local aware attention blocks.
            s_kernel:  Kernel size for SPE/SPR; 1 means linear projection, >1 uses convolution for embedding/reconstruction.
            r_forward: FFN expansion ratio; width multiplier inside feed-forward network.
            dropout:   Dropout ratio; probability used in attention and FFN.
            drop_path: Drop path ratio; stochastic depth probability per layer.
            s_local:   Local receptive field sizes; list [temporal_k, height_k, width_k].
            data_back: Back window shape; input history tensor shape [T_back, C, H, W].
            data_pred: Prediction window shape; target future tensor shape [T_pred, C, H, W].
            lr:        Learning rate; initial optimizer step size.
            lr_scheduler: Learning rate scheduler; 'cosine' or 'onecycle'.
            attn_bias: Whether attention linear projections use bias; if True adds bias to Q/K/V/O.
        """
        super().__init__()

        # Parameters:
        assert lr_scheduler in ['cosine', 'onecycle']
        self.s_local = s_local if s_local is not None else [3, 3, 3]
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
        self.patch_embedding = SpatialPatchEmbedding(d_model, s_patch, self.data_back, s_kernel)
        self.patch_reconstruction = SpatialPatchReconstruction(d_model, s_patch, self.data_back, self.data_pred, s_kernel)

        # Spatio-Temporal Local Perception Attention:
        drop_path_rate = [drop_path for _ in range(n_layer)]  # Default (same as PredFormer)
        # drop_path_rate = [drop_path * (i / (n_layer - 1)) for i in range(n_layer)]

        self.attn = nn.ModuleList(
            nn.ModuleList([
                TemporalLocalAwareAttentionBlock(d_model, n_heads, self.s_local[0], dropout, drop_path_rate[i],
                                                      r_forward, attn_bias),
                SpatialLocalAwareAttentionBlock(d_model, n_heads, self.s_local[1:], dropout, drop_path_rate[i],
                                                     r_forward, attn_bias)
            ]) for i in range(n_layer)
        )

    def forward(self, x: torch.Tensor, return_attn: bool = False) -> torch.Tensor:
        x = self.patch_embedding(x)

        for (t_attn, s_attn) in self.attn:
            x = t_attn(x)
            x = s_attn(x)

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