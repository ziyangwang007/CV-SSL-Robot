from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch

from config import get_config
from networks.mac_s3robo import MACS3RoboBranch, maybe_disable_missing_pretrain


def build_parser():
    parser = argparse.ArgumentParser(description="Compute rough parameter count and optional FLOPs for MambaUNet branch")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--patch_size", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--projection_dim", type=int, default=128)
    parser.add_argument("--cfg", type=str, default="code/configs/vmamba_tiny.yaml")
    parser.add_argument("--flops", action="store_true", help="Use fvcore FLOP counting when dependencies are available")
    # config args
    parser.add_argument("--opts", default=None, nargs="+")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--cache-mode", type=str, default="part", choices=["no", "full", "part"])
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--accumulation-steps", type=int, default=None)
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument("--amp-opt-level", type=str, default="O1", choices=["O0", "O1", "O2"])
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--throughput", action="store_true")
    return parser


def resolve(path: str) -> str:
    p = Path(path)
    if p.exists() or p.is_absolute():
        return str(p)
    for c in [Path.cwd() / path, Path(__file__).resolve().parents[1] / path]:
        if c.exists():
            return str(c)
    return str(p)


def main():
    args = build_parser().parse_args()
    args.cfg = resolve(args.cfg)
    config = get_config(args)
    config.defrost(); config.MODEL.PRETRAIN_CKPT = None; config.freeze(); maybe_disable_missing_pretrain(config)
    model = MACS3RoboBranch(config, args.patch_size, args.num_classes, args.projection_dim)
    params = sum(p.numel() for p in model.segmentor.parameters())
    trainable = sum(p.numel() for p in model.segmentor.parameters() if p.requires_grad)
    print(f"MambaUNet segmentor params: {params / 1e6:.4f} M")
    print(f"Trainable segmentor params: {trainable / 1e6:.4f} M")
    print(f"Projector params: {sum(p.numel() for p in model.projector.parameters()) / 1e6:.4f} M")
    if args.flops:
        try:
            flops = model.segmentor.mamba_unet.flops(shape=(3, args.patch_size[0], args.patch_size[1]))
            print(f"MambaUNet FLOPs: {flops / 1e9:.4f} G")
        except Exception as exc:
            print(f"FLOP counting failed: {exc}")


if __name__ == "__main__":
    main()
