from __future__ import annotations

import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vision_mamba import MambaUnet as ViM_seg


class PixelProjector(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True),
        )

    def forward(self, x_channels_last: torch.Tensor) -> torch.Tensor:
        # VSSM decoder returns B x H x W x C. Projector consumes B x C x H x W.
        x = x_channels_last.permute(0, 3, 1, 2).contiguous()
        z = self.net(x)
        return F.normalize(z, p=2, dim=1)


class MACS3RoboBranch(nn.Module):
    """One Mamba-UNet branch with a pixel projection head.

    The branch uses the original Mamba-UNet VSSM encoder-decoder, but exposes
    the final decoder feature map for pixel-level contrastive self-supervision.
    """

    def __init__(self, config, img_size, num_classes: int, projection_dim: int = 128):
        super().__init__()
        self.segmentor = ViM_seg(config, img_size=img_size, num_classes=num_classes)
        embed_dim = int(config.MODEL.VSSM.EMBED_DIM)
        self.projector = PixelProjector(embed_dim, projection_dim)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        # Match the original MambaUnet behavior for grayscale inputs while
        # preserving RGB inputs used in surgical endoscopy.
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        features, skips = self.segmentor.mamba_unet.forward_features(x)
        decoder_features = self.segmentor.mamba_unet.forward_up_features(features, skips)
        logits = self.segmentor.mamba_unet.up_x4(decoder_features)
        if return_embedding:
            embedding = self.projector(decoder_features)
            return logits, embedding
        return logits

    def load_from(self, config) -> None:
        self.segmentor.load_from(config)


def maybe_disable_missing_pretrain(config) -> None:
    """Avoid failing when the yaml points to a pretrained checkpoint that is not present."""
    ckpt = getattr(config.MODEL, "PRETRAIN_CKPT", None)
    if ckpt and not os.path.exists(ckpt):
        config.defrost()
        config.MODEL.PRETRAIN_CKPT = None
        config.freeze()
