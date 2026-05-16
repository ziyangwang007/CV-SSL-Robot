from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    try:
        from tensorboardX import SummaryWriter
    except Exception:  # pragma: no cover
        class SummaryWriter:  # minimal no-op fallback for environments without tensorboard
            def __init__(self, *args, **kwargs):
                pass
            def add_scalar(self, *args, **kwargs):
                pass
            def add_image(self, *args, **kwargs):
                pass
            def close(self):
                pass

from config import get_config
from dataloaders.robot_h5 import RoboticH5Dataset, RandomGenerator
from networks.mac_s3robo import MACS3RoboBranch, maybe_disable_missing_pretrain
from utils.robot_metrics import evaluate_loader
from utils.ssl_losses import DiceLoss, supervised_ce_dice_loss


def build_parser():
    parser = argparse.ArgumentParser(description="Fully supervised Mamba-UNet for surgical robot segmentation")
    parser.add_argument("--root_path", type=str, default="../data/robotic")
    parser.add_argument("--train_split", type=str, default=None)
    parser.add_argument("--val_split", type=str, default=None)
    parser.add_argument("--exp", type=str, default="EndoVis2017/MambaUNet_Fully")
    parser.add_argument("--model_dir", type=str, default="../model")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--binary_mask", action="store_true")
    parser.add_argument("--ignore_label", type=int, default=255)
    parser.add_argument("--image_key", type=str, default="image")
    parser.add_argument("--label_key", type=str, default="label")
    parser.add_argument("--max_iterations", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--base_lr", type=float, default=0.01)
    parser.add_argument("--patch_size", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--deterministic", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--save_interval", type=int, default=3000)
    parser.add_argument("--load_pretrained", action="store_true")
    parser.add_argument("--pretrained_ckpt", type=str, default=None)
    parser.add_argument("--projection_dim", type=int, default=128)

    parser.add_argument("--cfg", type=str, default="code/configs/vmamba_tiny.yaml")
    parser.add_argument("--opts", default=None, nargs="+")
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


def resolve_path(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    p = Path(path)
    if p.exists() or p.is_absolute():
        return str(p)
    for c in [Path.cwd() / path, Path(__file__).resolve().parents[1] / path]:
        if c.exists():
            return str(c)
    return str(p)


def setup(args):
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        cudnn.benchmark = True
        cudnn.deterministic = False
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def save_metrics(path: Path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)


def main():
    args = build_parser().parse_args()
    setup(args)
    args.cfg = resolve_path(args.cfg)
    args.train_split = resolve_path(args.train_split)
    args.val_split = resolve_path(args.val_split)
    snapshot_path = Path(args.model_dir) / args.exp
    snapshot_path.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=str(snapshot_path / "log.txt"), level=logging.INFO, format="[%(asctime)s] %(message)s")
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    config = get_config(args)
    config.defrost()
    if args.pretrained_ckpt is not None:
        config.MODEL.PRETRAIN_CKPT = args.pretrained_ckpt
    if not args.load_pretrained:
        config.MODEL.PRETRAIN_CKPT = None
    config.freeze(); maybe_disable_missing_pretrain(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MACS3RoboBranch(config, img_size=args.patch_size, num_classes=args.num_classes, projection_dim=args.projection_dim).to(device)
    if args.load_pretrained:
        model.load_from(config)

    train_set = RoboticH5Dataset(args.root_path, "train", RandomGenerator(args.patch_size, args.ignore_label), args.train_split, args.image_key, args.label_key, args.binary_mask, args.ignore_label)
    val_set = RoboticH5Dataset(args.root_path, "val", None, args.val_split, args.image_key, args.label_key, args.binary_mask, args.ignore_label)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1)
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4)
    ce_loss = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
    dice_loss = DiceLoss(args.num_classes, ignore_index=args.ignore_label, include_background=False)
    writer = SummaryWriter(str(snapshot_path / "tensorboard"))
    best = 0.0; iter_num = 0; rows = []
    max_epoch = args.max_iterations // max(1, len(train_loader)) + 1
    for _ in tqdm(range(max_epoch), ncols=90):
        for batch in train_loader:
            images = batch["image"].to(device); labels = batch["label"].to(device)
            logits = model(images)
            loss = supervised_ce_dice_loss(logits, labels, ce_loss, dice_loss)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            iter_num += 1
            lr_ = args.base_lr * (1.0 - iter_num / args.max_iterations) ** 0.9
            for group in optimizer.param_groups:
                group["lr"] = lr_
            writer.add_scalar("train/loss", loss.item(), iter_num); writer.add_scalar("train/lr", lr_, iter_num)
            if iter_num % args.eval_interval == 0:
                model.eval(); metrics = evaluate_loader(model, val_loader, args.num_classes, tuple(args.patch_size), device); model.train()
                for k, v in metrics.items(): writer.add_scalar(f"val/{k}", v, iter_num)
                rows.append({"iter": iter_num, **metrics}); save_metrics(snapshot_path / "validation_metrics.csv", rows)
                logging.info(f"iter {iter_num}: {metrics}")
                if metrics["dice"] > best:
                    best = metrics["dice"]; torch.save(model.state_dict(), snapshot_path / "mambaunet_best.pth")
            if iter_num % args.save_interval == 0:
                torch.save(model.state_dict(), snapshot_path / f"iter_{iter_num}.pth")
            if iter_num >= args.max_iterations:
                break
        if iter_num >= args.max_iterations:
            break
    torch.save(model.state_dict(), snapshot_path / "mambaunet_final.pth")
    writer.close()


if __name__ == "__main__":
    main()
