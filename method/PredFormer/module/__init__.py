import torch
import torch.nn as nn
from typing import List
from .Common import PatchEmbedding, PatchReconstruction, DropPath, SwiGLU
from .Attention import MultiHeadAttention


class GatedAttentionUnit(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 dropout: float = 0.0,
                 drop_path: float = 0.0,
                 r_forward: int = 4,
                 attn_bias: bool = False,
                 ):
        super(GatedAttentionUnit, self).__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout, attn_bias)
        self.sglu = SwiGLU(d_model, dropout, r_forward, attn_bias)
        self.drop_attn = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.drop_sglu = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        x = self.drop_attn(self.attn(x)) + x
        x = self.drop_sglu(self.sglu(x)) + x
        return x
