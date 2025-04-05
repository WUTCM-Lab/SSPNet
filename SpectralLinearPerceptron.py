# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# Demystify Mamba in Vision: A Linear Attention Perspective
# Modified by Dongchen Han
# -----------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.fft
from einops import rearrange
from Visualization import cluster_and_visualize1

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, dropout=0, norm=nn.BatchNorm2d, act_func=nn.ReLU):
        super(ConvLayer, self).__init__()
        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
            dilation=(dilation, dilation),
            groups=groups,
            bias=bias,
        )
        self.norm = norm(num_features=out_channels) if norm else None
        self.act = act_func() if act_func else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x

class AbsolutePositionalWeighting(torch.nn.Module):
    def __init__(self, H, W, C):
        super().__init__()
        self.H = H
        self.W = W

        # 计算中心坐标 (基于0的索引)
        self.center_h = (H - 1) // 2
        self.center_w = (W - 1) // 2

        # 生成dx和dy的网格
        i_indices = torch.arange(H)
        j_indices = torch.arange(W)
        grid_i, grid_j = torch.meshgrid(i_indices, j_indices, indexing='ij')
        dx = (grid_i - self.center_h).long()  # (H, W)
        dy = (grid_j - self.center_w).long()

        # 计算最大偏移量
        self.dh = max(torch.abs(dx).max().item(), self.center_h)
        self.dw = max(torch.abs(dy).max().item(), self.center_w)

        # 可学习的权重参数表（使用sigmoid约束到(0,1)范围）
        self.pos_weights = torch.nn.Parameter(
            torch.randn(2 * self.dh + 1, 2 * self.dw + 1, C)
        )
        torch.nn.init.normal_(self.pos_weights, mean=0, std=0.1)
        # 预计算索引映射
        self.register_buffer('dx_indices', dx + self.dh)
        self.register_buffer('dy_indices', dy + self.dw)

    def forward(self, x):
        B, H, W, C = x.shape
        assert H == self.H and W == self.W, "Input size mismatch"

        # 获取位置权重并应用sigmoid激活
        position_weights = torch.sigmoid(  # 约束到(0,1)范围
            self.pos_weights[self.dx_indices, self.dy_indices, :]  # (H, W, C)
        ).unsqueeze(0)  # 扩展为(B, H, W, C)

        # 进行逐元素乘法加权
        x_weighted = x * position_weights
        return x_weighted

class LinearAttention(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, qkv_bias=True, **kwargs):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.elu = nn.ELU()
        self.lepe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        # self.rope = RoPE(shape=(input_resolution[0], input_resolution[1], dim))

        self.APE = AbsolutePositionalWeighting(input_resolution[0], input_resolution[1], dim)
    def forward(self, x1, x2, x3):
        """
        Args:
            x: input features with shape of (B, N, C)
        """
        b, n, c = x1.shape
        h = int(n ** 0.5)
        w = int(n ** 0.5)
        num_heads = self.num_heads
        head_dim = c // num_heads


        # q, k, v: b, n, c
        q, k, v = x1, x2, x3

        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0
        q_rope = self.APE(q.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k_rope = self.APE(k.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k_rope.transpose(-2, -1) * (n ** -0.5)) @ (v * (n ** -0.5))
        x = q_rope @ kv * z

        x = x.transpose(1, 2).reshape(b, n, c)
        v = v.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = x + self.lepe(v).permute(0, 2, 3, 1).reshape(b, n, c)

        return x


    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'

class AbsoluteLinearAttention(nn.Module):
    def __init__(self, in_features, input_resolution, num_heads, qkv_bias=True):
        super().__init__()
        self.input_resolution = input_resolution
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=3 // 2, groups=in_features)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=5, padding=5 // 2, groups=in_features)
        self.conv3 = nn.Conv2d(in_features, in_features, kernel_size=7, padding=7 // 2, groups=in_features)
        self.attn = LinearAttention(dim=in_features, input_resolution=input_resolution, num_heads=num_heads, qkv_bias=qkv_bias)

        self.projector = nn.Conv2d(in_features, in_features, kernel_size=1, )

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.reshape(B, H, W, C)
        x = rearrange(x, 'b h w c -> b c h w')
        conv1_x = self.conv1(x)
        conv2_x = self.conv2(x)
        conv3_x = self.conv3(x)

        conv1_x = rearrange(conv1_x, 'b c h w -> b (h w) c')
        conv2_x = rearrange(conv2_x, 'b c h w -> b (h w) c')
        conv3_x = rearrange(conv3_x, 'b c h w -> b (h w) c')

        x = self.attn(conv1_x, conv2_x, conv3_x)

        return x

class Spectral_Linear_Perceptron(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.cpe1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.in_proj = nn.Linear(dim, dim)
        self.act_proj = nn.Linear(dim, dim)
        self.dwc = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.act = nn.ELU()

        self.ALA = AbsoluteLinearAttention(in_features=dim, input_resolution=input_resolution, num_heads=num_heads,
                                                                  qkv_bias=qkv_bias)

        self.out_proj = nn.Linear(dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.cpe2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x + self.cpe1(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        shortcut = x

        x = self.norm1(x)
        x = self.SpectralPerceptionBlock(x)
        x = shortcut + self.drop_path(x)
        x1 = self.cpe2(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)

        x = x1 + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def SpectralPerceptionBlock(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        act_res = self.act(self.act_proj(x))
        x = self.in_proj(x).view(B, H, W, C)
        x = self.act(self.dwc(x.permute(0, 3, 1, 2))).permute(0, 2, 3, 1).view(B, L, C)

        identity = x
        x = self.ALA(x)
        x = x + identity
        x = self.out_proj(x * act_res)

        return x
    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"mlp_ratio={self.mlp_ratio}"










