import torch
import torch.nn as nn
from typing import List

# Module List:
# 1. SpatialLocalAwareAttention
# 2. TemporalLocalAwareAttention

class SpatialLocalAwareAttention(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 s_local: List[int],
                 dropout: float = 0.0,
                 attn_bias: bool = False,
                 ):
        super(SpatialLocalAwareAttention, self).__init__()

        # Parameters:
        assert d_model % n_heads == 0
        self.s_local = s_local if s_local is not None else [3, 3]
        self.len_local = self.s_local[0] * self.s_local[1]
        self.n_heads = n_heads
        self.d_heads = d_model // n_heads
        self.factor = self.d_heads ** -0.5
        self.eps = 1e-6

        # Norms & Drops:
        self.init_norm = nn.LayerNorm(normalized_shape=d_model, eps=self.eps)
        self.attn_drop = nn.Dropout(dropout)

        # Projections:
        self.q_proj = nn.Linear(d_model, d_model, bias=attn_bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=attn_bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=attn_bias)
        self.output_proj = nn.Linear(d_model, d_model, bias=attn_bias)
        self.k_proj_local = nn.Linear(d_model, d_model, bias=attn_bias)
        self.v_proj_local = nn.Linear(d_model, d_model, bias=attn_bias)
        self.q_bias = nn.Parameter(nn.init.normal_(torch.empty(1, self.n_heads, 1, self.d_heads), mean=0.0, std=0.02))

        self.unfold = nn.Unfold(kernel_size=tuple(self.s_local), padding=tuple([s // 2 for s in s_local]), stride=1)
        self.mask_local = None

    @torch.no_grad()
    def _mask_local(self, H: int, W: int) -> torch.Tensor:
        return self.unfold(torch.ones(1, 1, H, W)).squeeze(0).transpose(-1, -2) == 0

    def forward(self, x: torch.Tensor, return_attn: bool = False) -> torch.Tensor:
        B, D, T, H, W = x.shape
        x = self.init_norm(x.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, D))  # [BT,HW,D]
        self.mask_local = self._mask_local(H, W).to(x.device) if self.mask_local is None else self.mask_local

        # 1. Query:
        q = self.q_proj(x).reshape(B * T, H * W, self.n_heads, self.d_heads).transpose(1, 2) + self.q_bias  # [B,h,N,d]

        # 2. Spatial Global Self Attention:
        k = self.k_proj(x).reshape(B * T, H * W, self.n_heads, self.d_heads).transpose(1, 2)  # [B,h,N,d]
        v = self.v_proj(x).reshape(B * T, H * W, self.n_heads, self.d_heads).transpose(1, 2)  # [B,h,N,d]
        attn_global = torch.einsum('bhnd,bhgd->bhng', q, k) * self.factor

        # 3. Spatial Local Perception Self Attention:
        k_ = self.k_proj_local(x).reshape(B * T, H, W, D).permute(0, 3, 1, 2)  # [BT,D,H,W]
        v_ = self.v_proj_local(x).reshape(B * T, H, W, D).permute(0, 3, 1, 2)  # [BT,D,H,W]
        k_ = self.unfold(k_).reshape(B * T, self.n_heads, self.d_heads, self.len_local, H * W).permute(0, 1, 4, 2, 3)
        v_ = self.unfold(v_).reshape(B * T, self.n_heads, self.d_heads, self.len_local, H * W).permute(0, 1, 4, 2, 3)
        attn_local = torch.einsum('bhnd,bhndl->bhnl', q, k_).masked_fill(self.mask_local, float('-inf')) * self.factor

        # 4. Aggregating global and local information:
        attn_map = torch.cat([attn_global, attn_local], dim=-1).softmax(dim=-1)
        attn_global, attn_local = self.attn_drop(attn_map).split([attn_global.size(-1), attn_local.size(-1)], dim=-1)
        x_global = torch.einsum('bhng,bhgd->bhnd', attn_global, v)
        x_local = torch.einsum('bhnl,bhndl->bhnd', attn_local, v_)

        # 5. Output Projection:
        x = self.output_proj((x_global + x_local).transpose(1, 2).reshape(B * T, H, W, D))
        x = x.reshape(B, T, H, W, D).permute(0, 4, 1, 2, 3)  # [B,D,T,H,W]

        return x if not return_attn else (x, attn_map)


class TemporalLocalAwareAttention(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 s_local: int = 3,
                 dropout: float = 0.0,
                 attn_bias: bool = False,
                 ):
        super(TemporalLocalAwareAttention, self).__init__()

        # Parameters:
        assert d_model % n_heads == 0
        self.s_local = s_local
        self.len_local = s_local
        self.n_heads = n_heads
        self.d_heads = d_model // n_heads
        self.factor = self.d_heads ** -0.5

        self.eps = 1e-6

        # Norms & Drops:
        self.init_norm = nn.LayerNorm(normalized_shape=d_model, eps=self.eps)
        self.attn_drop = nn.Dropout(dropout)

        # Projections:
        self.q_proj = nn.Linear(d_model, d_model, bias=attn_bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=attn_bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=attn_bias)
        self.output_proj = nn.Linear(d_model, d_model, bias=attn_bias)
        self.k_proj_local = nn.Linear(d_model, d_model, bias=attn_bias)
        self.v_proj_local = nn.Linear(d_model, d_model, bias=attn_bias)
        self.q_bias = nn.Parameter(nn.init.normal_(torch.empty(1, self.n_heads, 1, self.d_heads), mean=0.0, std=0.02))
        self.mask_local = None

    def _unfold(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.pad(x, pad=(self.s_local // 2, self.s_local // 2))
        x = x.unfold(-1, self.s_local, 1)
        return x

    @torch.no_grad()
    def _mask_local(self, T: int) -> torch.Tensor:  # [T, len_local]
        return torch.Tensor(self._unfold(torch.ones(1, T)).squeeze(0) == 0)

    def forward(self, x: torch.Tensor, return_attn: bool = False) -> torch.Tensor:
        B, D, T, H, W = x.shape
        x = self.init_norm(x.permute(0, 3, 4, 2, 1).reshape(B * H * W, T, D))  # [BHW,T,D]
        self.mask_local = self._mask_local(T).to(x.device) if self.mask_local is None else self.mask_local

        # 1. Query:
        q = self.q_proj(x).reshape(B * H * W, T, self.n_heads, self.d_heads).transpose(1, 2) + self.q_bias  # [B,h,T,d]

        # 2. Temporal Global Self Attention:
        k = self.k_proj(x).reshape(B * H * W, T, self.n_heads, self.d_heads).transpose(1, 2)  # [B,h,T,d]
        v = self.v_proj(x).reshape(B * H * W, T, self.n_heads, self.d_heads).transpose(1, 2)  # [B,h,T,d]
        attn_global = torch.einsum('bhnd,bhgd->bhng', q, k) * self.factor

        # 3. Temporal Local Perception Self Attention:
        k_ = self.k_proj_local(x).transpose(-1, -2)  # [BHW,D,T]
        v_ = self.v_proj_local(x).transpose(-1, -2)  # [BHW,D,T]
        k_ = self._unfold(k_).reshape(B * H * W, self.n_heads, self.d_heads, T, self.len_local).transpose(-3, -2)
        v_ = self._unfold(v_).reshape(B * H * W, self.n_heads, self.d_heads, T, self.len_local).transpose(-3, -2)
        attn_local = torch.einsum('bhnd,bhndl->bhnl', q, k_).masked_fill(self.mask_local, float('-inf')) * self.factor

        # 4. Aggregating global and local information:
        attn_map = torch.cat([attn_global, attn_local], dim=-1).softmax(dim=-1)
        attn_global, attn_local = self.attn_drop(attn_map).split([attn_global.size(-1), attn_local.size(-1)], dim=-1)
        x_global = torch.einsum('bhng,bhgd->bhnd', attn_global, v)
        x_local = torch.einsum('bhnl,bhndl->bhnd', attn_local, v_)

        # 5. Output Projection:
        x = self.output_proj((x_global + x_local).transpose(1, 2).reshape(B * H * W, T, D))
        x = x.reshape(B, H, W, T, D).permute(0, 4, 3, 1, 2)  # [B,D,T,H,W]

        return x if not return_attn else (x, attn_map)
