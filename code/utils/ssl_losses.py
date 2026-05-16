from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Multi-class soft Dice loss with optional ignore label."""

    def __init__(self, num_classes: int, ignore_index: Optional[int] = None, include_background: bool = True, eps: float = 1e-5):
        super().__init__()
        self.num_classes = int(num_classes)
        self.ignore_index = ignore_index
        self.include_background = include_background
        self.eps = eps

    def forward(self, probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == 4 and target.size(1) == 1:
            target = target[:, 0]
        target = target.long()
        valid_mask = torch.ones_like(target, dtype=torch.bool)
        if self.ignore_index is not None:
            valid_mask = target != int(self.ignore_index)
            target = target.clone()
            target[~valid_mask] = 0
        one_hot = F.one_hot(target.clamp_min(0), num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        valid_mask_f = valid_mask.unsqueeze(1).float()
        probs = probs * valid_mask_f
        one_hot = one_hot * valid_mask_f
        start_class = 0 if self.include_background else 1
        losses = []
        for c in range(start_class, self.num_classes):
            p = probs[:, c]
            y = one_hot[:, c]
            intersect = torch.sum(p * y)
            denom = torch.sum(p * p) + torch.sum(y * y)
            losses.append(1.0 - (2.0 * intersect + self.eps) / (denom + self.eps))
        if len(losses) == 0:
            return probs.sum() * 0.0
        return torch.stack(losses).mean()


class PixelContrastiveLoss(nn.Module):
    """Symmetric pixel-level InfoNCE over sampled decoder features.

    z_a and z_b are L2-normalised tensors shaped B x C x H x W. The same
    flattened pixel index across branches is the positive pair, all sampled
    positions in the current mini-batch are negatives.
    """

    def __init__(self, temperature: float = 0.1, max_samples: int = 2048):
        super().__init__()
        self.temperature = float(temperature)
        self.max_samples = int(max_samples)

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        if z_a.numel() == 0 or z_b.numel() == 0:
            return z_a.sum() * 0.0
        if z_a.shape != z_b.shape:
            raise ValueError(f"Contrastive inputs must have the same shape, got {z_a.shape} and {z_b.shape}")
        b, c, h, w = z_a.shape
        z_a = z_a.permute(0, 2, 3, 1).reshape(-1, c)
        z_b = z_b.permute(0, 2, 3, 1).reshape(-1, c)
        n = z_a.size(0)
        sample_n = min(self.max_samples, n)
        if sample_n < n:
            idx = torch.randperm(n, device=z_a.device)[:sample_n]
            z_a = z_a[idx]
            z_b = z_b[idx]
        logits = torch.matmul(z_a, z_b.t()) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_ab = F.cross_entropy(logits, labels)
        loss_ba = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_ab + loss_ba)


def supervised_ce_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ce_loss: nn.Module,
    dice_loss: DiceLoss,
) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    return 0.5 * (ce_loss(logits, target.long()) + dice_loss(probs, target))
