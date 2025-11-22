import torch
import torch.nn as nn
from typing import List
from .Common import SpatialPatchEmbedding, SpatialPatchReconstruction, DropPath, SpatialDepthWiseGatedLinearUnit, TemporalDepthWiseGatedLinearUnit
from .Attention import SpatialLocalAwareAttention, TemporalLocalAwareAttention


class SpatialLocalAwareAttentionBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 s_local: List[int],
                 dropout: float = 0.0,
                 drop_path: float = 0.0,
                 r_forward: int = 4,
                 attn_bias: bool = False,
                 ):
        super(SpatialLocalAwareAttentionBlock, self).__init__()
        self.attn = SpatialLocalAwareAttention(d_model, n_heads, s_local, dropout, attn_bias)
        self.cGlu = SpatialDepthWiseGatedLinearUnit(d_model, dropout, r_forward, attn_bias)
        self.drop_attn = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.drop_cGlu = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm_attn = nn.GroupNorm(1, d_model, eps=1e-6)
        self.norm_cGLU = nn.GroupNorm(1, d_model, eps=1e-6)

    def forward(self, x: torch.Tensor, return_attn: bool = False) -> torch.Tensor:
        B, D, T, H, W = x.shape
        x_ = x
        if return_attn:
            x, attn_map = self.attn(x, return_attn)
        else:
            x = self.attn(x)
        x = self.norm_attn(self.drop_attn(x) + x_)  # Add & Drop & Norm
        x = self.norm_cGLU(self.drop_cGlu(self.cGlu(x)) + x)
        return x if not return_attn else (x, attn_map)


class TemporalLocalAwareAttentionBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 s_local: int,
                 dropout: float = 0.0,
                 drop_path: float = 0.0,
                 r_forward: int = 4,
                 attn_bias: bool = False,
                 ):
        super(TemporalLocalAwareAttentionBlock, self).__init__()
        self.attn = TemporalLocalAwareAttention(d_model, n_heads, s_local, dropout, attn_bias)
        self.cGlu = TemporalDepthWiseGatedLinearUnit(d_model, dropout, r_forward, attn_bias)
        self.drop_attn = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.drop_cGlu = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm_attn = nn.GroupNorm(1, d_model, eps=1e-6)
        self.norm_cGLU = nn.GroupNorm(1, d_model, eps=1e-6)

    def forward(self, x: torch.Tensor, return_attn: bool = False) -> torch.Tensor:
        B, D, T, H, W = x.shape
        x_ = x
        if return_attn:
            x, attn_map = self.attn(x, return_attn)
        else:
            x = self.attn(x)
        x = self.norm_attn(self.drop_attn(x) + x_)  # Add & Drop & Norm
        x = self.norm_cGLU(self.drop_cGlu(self.cGlu(x)) + x)
        return x if not return_attn else (x, attn_map)
