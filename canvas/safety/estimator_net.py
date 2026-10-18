"""Estimator network.

Adapted from MiVOLO (https://github.com/WildChlamydia/MiVOLO, Apache-2.0,
Copyright 2023 Irina Tolstykh, Maxim Kuprashevich), which builds on timm
(https://github.com/huggingface/pytorch-image-models, Apache-2.0).
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from timm.layers import Mlp, make_divisible, trunc_normal_
from timm.layers.bottleneck_attn import PosEmbedRel
from timm.models.volo import VOLO

D1_LAYERS = (4, 4, 8, 2)
D1_EMBED_DIMS = (192, 384, 384, 384)
D1_NUM_HEADS = (6, 12, 12, 12)


def _stem_output(size: tuple[int, int], stem: nn.Sequential) -> tuple[int, int]:
    out = list(size)
    for module in stem:
        if isinstance(module, nn.Conv2d):
            out = [
                (out[i] + 2 * int(module.padding[i]) - module.dilation[i] * (module.kernel_size[i] - 1) - 1)
                // module.stride[i]
                + 1
                for i in range(2)
            ]
    return (out[0], out[1])


def _conv_output(size: tuple[int, int], conv: nn.Conv2d) -> tuple[int, int]:
    out = [
        (size[i] + 2 * int(conv.padding[i]) - conv.dilation[i] * (conv.kernel_size[i] - 1) - 1)
        // conv.stride[i]
        + 1
        for i in range(2)
    ]
    return (out[0], out[1])


class CrossBottleneckAttn(nn.Module):
    """CrossBottleneckAttn."""

    def __init__(self, dim: int, dim_out: int, feat_size: tuple[int, int], num_heads: int = 1) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.dim_head_qk = make_divisible(dim_out * 1.0, divisor=8) // num_heads
        self.dim_head_v = dim_out // num_heads
        self.dim_out_qk = num_heads * self.dim_head_qk
        self.dim_out_v = num_heads * self.dim_head_v
        self.scale = self.dim_head_qk ** -0.5
        self.qkv_f = nn.Conv2d(dim, self.dim_out_qk * 2 + self.dim_out_v, 1, bias=False)
        self.qkv_p = nn.Conv2d(dim, self.dim_out_qk * 2 + self.dim_out_v, 1, bias=False)
        self.pos_embed = PosEmbedRel(feat_size, dim_head=self.dim_head_qk, scale=self.scale)
        self.norm = nn.LayerNorm([self.dim_out_v * 2, *feat_size])
        self.mlp = Mlp(
            in_features=self.dim_out_v * 2,
            hidden_features=int(dim * 4),
            act_layer=nn.GELU,
            out_features=dim_out,
            drop=0,
            use_conv=True,
        )
        self.pool = nn.Identity()

    def _qkv(self, x: torch.Tensor, conv: nn.Conv2d) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = x.shape[0]
        x = conv(x)
        q, k, v = torch.split(x, [self.dim_out_qk, self.dim_out_qk, self.dim_out_v], dim=1)
        q = q.reshape(batch * self.num_heads, self.dim_head_qk, -1).transpose(-1, -2)
        k = k.reshape(batch * self.num_heads, self.dim_head_qk, -1)
        v = v.reshape(batch * self.num_heads, self.dim_head_v, -1).transpose(-1, -2)
        return q, k, v

    def _attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, batch: int, h: int, w: int) -> torch.Tensor:
        attn = (q @ k) * self.scale + self.pos_embed(q)
        attn = attn.softmax(dim=-1)
        return (attn @ v).transpose(-1, -2).reshape(batch, self.dim_out_v, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        batch, channels, h, w = x.shape
        half = channels // 2
        q_f, k_f, v_f = self._qkv(x[:, :half], self.qkv_f)
        q_p, k_p, v_p = self._qkv(x[:, half:], self.qkv_p)
        out_f = self._attend(q_f, k_p, v_p, batch, h, w)
        out_p = self._attend(q_p, k_f, v_f, batch, h, w)
        joined = self.norm(torch.cat((out_f, out_p), dim=1))
        return self.pool(self.mlp(joined))


def _stem(in_chans: int, hidden_dim: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_chans, hidden_dim, kernel_size=7, stride=stride, padding=3, bias=False),
        nn.BatchNorm2d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(hidden_dim),
        nn.ReLU(inplace=True),
    )


class PairEmbed(nn.Module):
    """PairEmbed."""

    def __init__(self, img_size: int, patch_size: int, hidden_dim: int, embed_dim: int, stem_stride: int = 2) -> None:
        super().__init__()
        self.conv = True
        self.conv1 = _stem(3, hidden_dim, stem_stride)
        self.conv2 = _stem(3, hidden_dim, stem_stride)
        step = patch_size // stem_stride
        self.proj1 = nn.Conv2d(hidden_dim, embed_dim, kernel_size=step, stride=step)
        self.proj2 = nn.Conv2d(hidden_dim, embed_dim, kernel_size=step, stride=step)
        feat = _conv_output(_stem_output((img_size, img_size), self.conv1), self.proj1)
        self.map = CrossBottleneckAttn(embed_dim, dim_out=embed_dim, num_heads=1, feat_size=feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        first = self.proj1(self.conv1(x[:, :3]))
        second = self.proj2(self.conv2(x[:, 3:]))
        return self.map(torch.cat([first, second], dim=1))


class EstimatorNet(VOLO):
    """EstimatorNet."""

    def __init__(self, img_size: int = 384, num_classes: int = 3, **kwargs: Any) -> None:
        super().__init__(
            layers=list(D1_LAYERS),
            img_size=img_size,
            in_chans=6,
            num_classes=num_classes,
            embed_dims=list(D1_EMBED_DIMS),
            num_heads=list(D1_NUM_HEADS),
            **kwargs,
        )
        self.patch_embed = PairEmbed(img_size, 8, 64, D1_EMBED_DIMS[0])  # type: ignore[assignment]
        trunc_normal_(self.pos_embed, std=0.02)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward features."""
        x = self.patch_embed(x).permute(0, 2, 3, 1)
        x = self.forward_tokens(x)
        if self.post_network is not None:
            x = self.forward_cls(x)
        return self.norm(x)

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        """Forward head."""
        out = self.head(x[:, 0])
        if self.aux_head is not None:
            out = out + 0.5 * self.aux_head(x[:, 1:]).max(1)[0]
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        return self.forward_head(self.forward_features(x))
