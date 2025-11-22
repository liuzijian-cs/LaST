import math
import torch
import torch.nn as nn
from typing import List


# Module List:
# 1. BasicLinear
# 2. BasicConv2d
# 3. SpatiotemporalPosEncoding3D
# 4. SpatiotemporalPosEncoding2D
# 5. SpatialPatchEmbedding
# 6. SpatialPatchReconstruction
# 7. TemporalDepthWiseGatedLinearUnit
# 8. SpatialDepthWiseGatedLinearUnit
# 9. DropPath


class BasicLinear(nn.Module):
    def __init__(self,
                 d_in: int,
                 d_out: int,
                 bias: bool = True,
                 activation: bool = True,
                 ):
        super(BasicLinear, self).__init__()
        self.linear = nn.Linear(in_features=d_in, out_features=d_out, bias=bias)
        self.norm_activation = nn.Sequential(
            nn.LayerNorm(d_out, eps=1e-6),
            nn.SiLU()
        ) if activation else nn.Identity()

    def forward(self, x):
        B, D, T, H, W = x.shape
        x = self.linear(x.permute(0, 2, 3, 4, 1))  # Linear([B,T,H,W,D])
        x = self.norm_activation(x)  # [B,T,H,W,D]
        x = x.permute(0, 4, 1, 2, 3)  # [B,D,T,H,W]
        return x  # [B,D,T,H,W]


class BasicConv2d(nn.Module):
    def __init__(self,
                 d_in: int,
                 d_out: int,
                 s_kernel: int = 3,
                 bias: bool = True,
                 activation: bool = True,
                 ):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=d_in, out_channels=d_out, kernel_size=s_kernel, bias=bias, padding='same')
        self.norm_activation = nn.Sequential(
            nn.GroupNorm(1, d_out, eps=1e-6),
            nn.SiLU()
        ) if activation else nn.Identity()

    def forward(self, x: torch.Tensor):
        B, D, T, H, W = x.shape
        x = self.conv(x.transpose(1, 2).flatten(0, 1))  # BT,D,H,W
        x = self.norm_activation(x)
        x = x.reshape(B, T, *x.shape[1:]).transpose(1, 2)
        return x  # [B,D,T,H,W]


class SpatiotemporalPosEncoding3D(nn.Module):
    def __init__(self, d_model: int, T: int, H: int, W: int):
        super(SpatiotemporalPosEncoding3D, self).__init__()
        assert d_model % 4 == 0, "d_model must be divisible by 4 for separate T, H, W encoding."

        self.d_t = d_model // 4
        self.d_h = self.d_w = (d_model - self.d_t) // 2

        self.pe_t = self.generate_sinusoidal_encoding(T, self.d_t)
        self.pe_h = self.generate_sinusoidal_encoding(H, self.d_h)
        self.pe_w = self.generate_sinusoidal_encoding(W, self.d_w)
        self.register_buffer("pe", self.combine_encodings(self.pe_t, self.pe_h, self.pe_w))

    def generate_sinusoidal_encoding(self, length, d_model):
        position = torch.arange(length, dtype=torch.float).unsqueeze(1)  # Shape: [length, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )  # Shape: [d_model/2]
        encoding = torch.zeros((length, d_model))  # Shape: [length, d_model]
        encoding[:, 0::2] = torch.sin(position * div_term)  # Even dimensions
        encoding[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions
        return encoding

    def combine_encodings(self, pe_t, pe_h, pe_w):
        T, d_t = pe_t.shape
        H, d_h = pe_h.shape
        W, d_w = pe_w.shape

        pe_t = pe_t.unsqueeze(1).unsqueeze(2).expand(T, H, W, d_t)  # [T, H, W, d_t]
        pe_h = pe_h.unsqueeze(0).unsqueeze(2).expand(T, H, W, d_h)  # [T, H, W, d_h]
        pe_w = pe_w.unsqueeze(0).unsqueeze(1).expand(T, H, W, d_w)  # [T, H, W, d_w]

        pe = torch.cat([pe_t, pe_h, pe_w], dim=-1)  # [T, H, W, d_t + d_h + d_w]
        pe = pe.permute(3, 0, 1, 2).unsqueeze(0)  # [1, D, T, H, W]
        return pe

    def forward(self, x):
        return x + self.pe


class SpatiotemporalPosEncoding2D(nn.Module):
    def __init__(self, d_model: int, T: int, H: int, W: int):
        super(SpatiotemporalPosEncoding2D, self).__init__()
        assert d_model % 2 == 0, "d_model must be divisible by 4 for separate T, H, W encoding."

        self.d_hw = d_model // 2
        self.d_t = d_model // 2

        self.pe_t = self.generate_sinusoidal_encoding(T, self.d_t)
        self.pe_hw = self.generate_sinusoidal_encoding(H * W, self.d_hw)
        self.register_buffer("pe", self.combine_encodings(self.pe_t, self.pe_hw))

    def generate_sinusoidal_encoding(self, length, d_model):
        position = torch.arange(length, dtype=torch.float).unsqueeze(1)  # Shape: [length, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )  # Shape: [d_model/2]
        encoding = torch.zeros((length, d_model))  # Shape: [length, d_model]
        encoding[:, 0::2] = torch.sin(position * div_term)  # Even dimensions
        encoding[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions
        return encoding

    def combine_encodings(self, pe_t, pe_hw):
        T, d_t = pe_t.shape
        HW, d_hw = pe_hw.shape

        pe_t = pe_t.unsqueeze(1).expand(T, HW, d_t)  # [T, HW, d_t]
        pe_hw = pe_hw.unsqueeze(0).expand(T, HW, d_hw)  # [T, HW, d_hw]

        pe = torch.cat([pe_t, pe_hw], dim=-1)  # [T, HW, d_t + d_hw]
        pe = pe.permute(2, 0, 1).unsqueeze(0)  # [1, D, T, HW]
        return pe

    def forward(self, x):
        B, D, T, H, W = x.shape
        return x + self.pe.reshape(1, D, T, H, W)


class SpatialPatchEmbedding(nn.Module):
    def __init__(self,
                 d_model: int,
                 s_patch: int,
                 data_back: List[int],
                 s_kernel: int = 1,
                 ):
        super(SpatialPatchEmbedding, self).__init__()

        self.patch_split = nn.PixelUnshuffle(downscale_factor=s_patch)
        self.patch_embed = BasicLinear(d_in=data_back[1] * (s_patch ** 2), d_out=d_model, activation=True) if (
                s_kernel == 1) else BasicConv2d(d_in=data_back[1] * (s_patch ** 2), d_out=d_model, s_kernel=s_kernel)
        self.position_encode = SpatiotemporalPosEncoding3D(d_model=d_model, T=data_back[0], H=data_back[-2] // s_patch,
                                                           W=data_back[-1] // s_patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_split(x).transpose(1, 2)  # [B,C,T,H,W]
        x = self.patch_embed(x)  # [B,D,T,H,W]
        x = self.position_encode(x)  # [B,D,T,H,W]
        return x  # [B,D,T,H,W]


class SpatialPatchReconstruction(nn.Module):
    def __init__(self,
                 d_model: int,
                 s_patch: int,
                 data_back: List[int],
                 data_pred: List[int],
                 s_kernel: int = 1,
                 ):
        super(SpatialPatchReconstruction, self).__init__()
        self.len_back = data_back[0]
        self.len_pred = data_pred[0]

        # Time Translator: (Only used when `len_back` != `len_pred)
        self.translator = nn.Sequential(
            nn.Linear(in_features=d_model * self.len_back, out_features=d_model * self.len_pred),
            nn.SiLU(),
            nn.LayerNorm(d_model * self.len_pred),
        ) if self.len_pred != self.len_back else nn.Identity()

        self.o_proj = BasicLinear(d_in=d_model, d_out=data_pred[1] * (s_patch ** 2), activation=False) \
            if s_kernel == 1 else (
            BasicConv2d(d_in=d_model, d_out=data_pred[1] * (s_patch ** 2), s_kernel=s_kernel, activation=False))
        self.patch_invert = nn.PixelShuffle(upscale_factor=s_patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D, T, H, W = x.shape
        x = (self.translator(x.reshape(B, D * T, H * W).transpose(-1, -2))
             .transpose(-1, -2).reshape(B, D, self.len_pred, H, W))
        x = self.o_proj(x)
        x = self.patch_invert(x.transpose(1, 2))
        return x  # [B,T,C,H,W]


class TemporalDepthWiseGatedLinearUnit(nn.Module):
    def __init__(self,
                 d_model: int,
                 dropout: float,
                 r_forward: int = 4,
                 attn_bias: bool = False,
                 ):
        super(TemporalDepthWiseGatedLinearUnit, self).__init__()
        d_hidden = d_model * r_forward

        self.gate_proj = nn.Linear(d_model, d_hidden, bias=attn_bias)
        self.gate_silu = nn.Conv1d(d_model, d_hidden, 3, 1, 1, groups=d_model, bias=attn_bias)
        self.down_proj = nn.Linear(d_hidden, d_model, bias=attn_bias)
        self.silu = nn.SiLU()
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):
        B, D, T, H, W = x.shape
        x_gate = self.gate_proj(x.permute(0, 3, 4, 2, 1).reshape(B * H * W, T, D))
        x_silu = self.silu(self.gate_silu(x.permute(0, 3, 4, 1, 2).reshape(B * H * W, D, T))).transpose(-1, -2)
        x = self.down_proj(self.drop(x_gate * x_silu)).reshape(B, H, W, T, D).permute(0, 4, 3, 1, 2)
        return x


class SpatialDepthWiseGatedLinearUnit(nn.Module):
    def __init__(self,
                 d_model: int,
                 dropout: float,
                 r_forward: int = 4,
                 attn_bias: bool = False,
                 ):
        super(SpatialDepthWiseGatedLinearUnit, self).__init__()
        d_hidden = d_model * r_forward

        self.gate_proj = nn.Linear(d_model, d_hidden, bias=attn_bias)
        self.gate_silu = nn.Conv2d(d_model, d_hidden, 3, 1, 1, groups=d_model, bias=attn_bias)
        self.down_proj = nn.Linear(d_hidden, d_model, bias=attn_bias)
        self.silu = nn.SiLU()
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):
        B, D, T, H, W = x.shape
        x_gate = self.gate_proj(x.permute(0, 2, 3, 4, 1).reshape(B * T, H, W, D))
        x_silu = self.silu(self.gate_silu(x.permute(0, 2, 1, 3, 4).reshape(B * T, D, H, W))).permute(0, 2, 3, 1)
        x = self.down_proj(self.drop(x_gate * x_silu)).reshape(B, T, H, W, D).permute(0, 4, 1, 2, 3)
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob > 0.0 and self.training:
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
            x = x.div(keep_prob) * random_tensor
        return x
