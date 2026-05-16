from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Optional

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
from dataloaders.robot_h5 import RoboticH5Dataset, RandomGenerator, TwoStreamBatchSampler, split_labeled_unlabeled
from networks.mac_s3robo import MACS3RoboBranch, maybe_disable_missing_pretrain
from utils import ramps
from utils.robot_metrics import evaluate_loader, evaluate_loader_pair
from utils.ssl_losses import DiceLoss, PixelContrastiveLoss, supervised_ce_dice_loss


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MAC-S3Robo semi-supervised surgical robot segmentation")
    parser.add_argument("--root_path", type=str, default="../data/robotic", help="Dataset root containing train/val/test H5 files")
    parser.add_argument("--train_split", type=str, default=None, help="Optional train split file")
    parser.add_argument("--val_split", type=str, default=None, help="Optional validation split file")
    parser.add_argument("--exp", type=str, default="EndoVis2017/MAC_S3Robo", help="Experiment name under --model_dir")
    parser.add_argument("--model_dir", type=str, default="../model", help="Directory for checkpoints and logs")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of segmentation classes including background")
    parser.add_argument("--binary_mask", action="store_true", help="Convert every non-zero mask value to foreground. Use for EndoVis 2017 binary instrument segmentation.")
    parser.add_argument("--ignore_label", type=int, default=255)
    parser.add_argument("--image_key", type=str, default="image")
    parser.add_argument("--label_key", type=str, default="label")

    parser.add_argument("--max_iterations", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--labeled_bs", type=int, default=12, help="Number of labelled samples per batch")
    parser.add_argument("--labeled_ratio", type=float, default=0.10, help="Fraction of train samples used as labelled data")
    parser.add_argument("--base_lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patch_size", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--deterministic", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--lambda_self", type=float, default=1.0, help="Maximum weight for pixel contrastive self-supervision")
    parser.add_argument("--lambda_semi", type=float, default=0.1, help="Maximum weight for cross pseudo-label supervision")
    parser.add_argument("--consistency_rampup", type=float, default=200.0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--contrastive_dim", type=int, default=128)
    parser.add_argument("--contrastive_pixels", type=int, default=2048)

    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--save_interval", type=int, default=3000)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--load_pretrained", action="store_true", help="Load VMamba pretrained checkpoint specified by --pretrained_ckpt or config yaml")
    parser.add_argument("--pretrained_ckpt", type=str, default=None, help="Optional VMamba pretrained checkpoint path")
    parser.add_argument("--branch2_noise_std", type=float, default=0.0, help="Optional tiny parameter perturbation for branch B after pretrained loading")

    # Arguments consumed by config.get_config from the Mamba-UNet codebase.
    parser.add_argument("--cfg", type=str, default="code/configs/vmamba_tiny.yaml", help="Path to VMamba config yaml")
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


def resolve_path(path: Optional[str], base: Optional[Path] = None) -> Optional[str]:
    if path is None:
        return None
    p = Path(path)
    if p.exists() or p.is_absolute():
        return str(p)
    candidates = []
    if base is not None:
        candidates.append(base / path)
    candidates.append(Path.cwd() / path)
    candidates.append(Path(__file__).resolve().parents[1] / path)
    for c in candidates:
        if c.exists():
            return str(c)
    return str(p)


def set_reproducibility(seed: int, deterministic: int) -> None:
    if deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        cudnn.benchmark = True
        cudnn.deterministic = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logger(snapshot_path: Path) -> None:
    snapshot_path.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(snapshot_path / "log.txt"),
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def get_current_ramp_weight(max_value: float, iter_num: int, rampup: float) -> float:
    return float(max_value) * float(ramps.sigmoid_rampup(iter_num, rampup))


def perturb_parameters(model: nn.Module, std: float) -> None:
    if std <= 0:
        return
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad and p.dtype.is_floating_point:
                p.add_(torch.randn_like(p) * std)


def save_metrics_csv(path: Path, rows) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def create_models(args, config, device):
    model_a = MACS3RoboBranch(config, img_size=args.patch_size, num_classes=args.num_classes, projection_dim=args.contrastive_dim).to(device)
    model_b = MACS3RoboBranch(config, img_size=args.patch_size, num_classes=args.num_classes, projection_dim=args.contrastive_dim).to(device)
    if args.load_pretrained:
        model_a.load_from(config)
        model_b.load_from(config)
        perturb_parameters(model_b, args.branch2_noise_std)
    return model_a, model_b


def train(args, snapshot_path: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.cfg = resolve_path(args.cfg, Path(__file__).resolve().parents[1])
    args.train_split = resolve_path(args.train_split, Path(__file__).resolve().parents[1]) if args.train_split else None
    args.val_split = resolve_path(args.val_split, Path(__file__).resolve().parents[1]) if args.val_split else None
    config = get_config(args)
    config.defrost()
    if args.pretrained_ckpt is not None:
        config.MODEL.PRETRAIN_CKPT = args.pretrained_ckpt
    if not args.load_pretrained:
        config.MODEL.PRETRAIN_CKPT = None
    config.freeze()
    maybe_disable_missing_pretrain(config)

    model_a, model_b = create_models(args, config, device)
    model_a.train(); model_b.train()

    train_set = RoboticH5Dataset(
        base_dir=args.root_path,
        split="train",
        split_file=args.train_split,
        transform=RandomGenerator(args.patch_size, ignore_label=args.ignore_label),
        image_key=args.image_key,
        label_key=args.label_key,
        binary_mask=args.binary_mask,
        ignore_label=args.ignore_label,
    )
    val_set = RoboticH5Dataset(
        base_dir=args.root_path,
        split="val",
        split_file=args.val_split,
        transform=None,
        image_key=args.image_key,
        label_key=args.label_key,
        binary_mask=args.binary_mask,
        ignore_label=args.ignore_label,
    )

    labeled_idxs, unlabeled_idxs = split_labeled_unlabeled(len(train_set), args.labeled_ratio, args.seed)
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - args.labeled_bs)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
        np.random.seed(args.seed + worker_id)

    train_loader = DataLoader(
        train_set,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1)

    optimizer_a = optim.SGD(model_a.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer_b = optim.SGD(model_b.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    ce_loss = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
    ce_pseudo = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(args.num_classes, ignore_index=args.ignore_label, include_background=False)
    contrastive_loss = PixelContrastiveLoss(temperature=args.temperature, max_samples=args.contrastive_pixels)

    writer = SummaryWriter(str(snapshot_path / "tensorboard"))
    logging.info(str(args))
    logging.info(f"Device: {device}")
    logging.info(f"Labelled samples: {len(labeled_idxs)} / {len(train_set)} ({args.labeled_ratio:.3f})")
    logging.info(f"Unlabelled samples: {len(unlabeled_idxs)}")
    logging.info(f"Iterations per epoch: {len(train_loader)}")

    with open(snapshot_path / "labeled_indices.txt", "w", encoding="utf-8") as f:
        for idx in labeled_idxs:
            f.write(f"{idx}\t{train_set.sample_list[idx]}\n")
    with open(snapshot_path / "unlabeled_indices.txt", "w", encoding="utf-8") as f:
        for idx in unlabeled_idxs:
            f.write(f"{idx}\t{train_set.sample_list[idx]}\n")

    iter_num = 0
    max_epoch = args.max_iterations // max(1, len(train_loader)) + 1
    best_a = 0.0
    best_b = 0.0
    best_ensemble = 0.0
    metric_rows = []

    for _ in tqdm(range(max_epoch), ncols=90):
        for sampled_batch in train_loader:
            images = sampled_batch["image"].to(device, non_blocking=True)
            labels = sampled_batch["label"].to(device, non_blocking=True)

            logits_a, z_a = model_a(images, return_embedding=True)
            logits_b, z_b = model_b(images, return_embedding=True)
            probs_a = torch.softmax(logits_a, dim=1)
            probs_b = torch.softmax(logits_b, dim=1)

            sup_a = supervised_ce_dice_loss(logits_a[:args.labeled_bs], labels[:args.labeled_bs], ce_loss, dice_loss)
            sup_b = supervised_ce_dice_loss(logits_b[:args.labeled_bs], labels[:args.labeled_bs], ce_loss, dice_loss)
            sup_loss = sup_a + sup_b

            if args.labeled_bs < images.size(0):
                logits_a_u = logits_a[args.labeled_bs:]
                logits_b_u = logits_b[args.labeled_bs:]
                pseudo_a = torch.argmax(probs_a[args.labeled_bs:].detach(), dim=1)
                pseudo_b = torch.argmax(probs_b[args.labeled_bs:].detach(), dim=1)
                semi_a = ce_pseudo(logits_a_u, pseudo_b)
                semi_b = ce_pseudo(logits_b_u, pseudo_a)
                semi_loss = semi_a + semi_b
                self_loss = contrastive_loss(z_a[args.labeled_bs:], z_b[args.labeled_bs:])
            else:
                semi_loss = logits_a.sum() * 0.0
                self_loss = logits_a.sum() * 0.0

            lambda_self = get_current_ramp_weight(args.lambda_self, iter_num, args.consistency_rampup)
            lambda_semi = get_current_ramp_weight(args.lambda_semi, iter_num, args.consistency_rampup)
            total_loss = sup_loss + lambda_self * self_loss + lambda_semi * semi_loss

            optimizer_a.zero_grad(set_to_none=True)
            optimizer_b.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer_a.step()
            optimizer_b.step()

            iter_num += 1
            lr_ = args.base_lr * (1.0 - iter_num / args.max_iterations) ** 0.9
            for group in optimizer_a.param_groups:
                group["lr"] = lr_
            for group in optimizer_b.param_groups:
                group["lr"] = lr_

            writer.add_scalar("train/lr", lr_, iter_num)
            writer.add_scalar("train/total_loss", total_loss.item(), iter_num)
            writer.add_scalar("train/sup_loss", sup_loss.item(), iter_num)
            writer.add_scalar("train/semi_loss", semi_loss.item(), iter_num)
            writer.add_scalar("train/self_loss", self_loss.item(), iter_num)
            writer.add_scalar("train/lambda_self", lambda_self, iter_num)
            writer.add_scalar("train/lambda_semi", lambda_semi, iter_num)

            if iter_num % args.log_interval == 0:
                logging.info(
                    "iter %d | loss %.5f | sup %.5f | semi %.5f (w %.4f) | self %.5f (w %.4f) | lr %.6f"
                    % (iter_num, total_loss.item(), sup_loss.item(), semi_loss.item(), lambda_semi, self_loss.item(), lambda_self, lr_)
                )

            if iter_num > 0 and iter_num % args.eval_interval == 0:
                model_a.eval(); model_b.eval()
                metrics_a = evaluate_loader(model_a, val_loader, args.num_classes, tuple(args.patch_size), device)
                metrics_b = evaluate_loader(model_b, val_loader, args.num_classes, tuple(args.patch_size), device)
                metrics_e = evaluate_loader_pair(model_a, model_b, val_loader, args.num_classes, tuple(args.patch_size), device)
                for prefix, metrics in [("val_a", metrics_a), ("val_b", metrics_b), ("val_ensemble", metrics_e)]:
                    for k, v in metrics.items():
                        writer.add_scalar(f"{prefix}/{k}", v, iter_num)
                metric_rows.append({"iter": iter_num, "branch": "a", **metrics_a})
                metric_rows.append({"iter": iter_num, "branch": "b", **metrics_b})
                metric_rows.append({"iter": iter_num, "branch": "ensemble", **metrics_e})
                save_metrics_csv(snapshot_path / "validation_metrics.csv", metric_rows)

                if metrics_a["dice"] > best_a:
                    best_a = metrics_a["dice"]
                    torch.save(model_a.state_dict(), snapshot_path / "mac_s3robo_branch_a_best.pth")
                if metrics_b["dice"] > best_b:
                    best_b = metrics_b["dice"]
                    torch.save(model_b.state_dict(), snapshot_path / "mac_s3robo_branch_b_best.pth")
                if metrics_e["dice"] > best_ensemble:
                    best_ensemble = metrics_e["dice"]
                    torch.save({"branch_a": model_a.state_dict(), "branch_b": model_b.state_dict()}, snapshot_path / "mac_s3robo_ensemble_best.pth")
                logging.info(
                    "VAL iter %d | A dice %.4f | B dice %.4f | Ensemble dice %.4f"
                    % (iter_num, metrics_a["dice"], metrics_b["dice"], metrics_e["dice"])
                )
                model_a.train(); model_b.train()

            if iter_num > 0 and iter_num % args.save_interval == 0:
                torch.save(model_a.state_dict(), snapshot_path / f"branch_a_iter_{iter_num}.pth")
                torch.save(model_b.state_dict(), snapshot_path / f"branch_b_iter_{iter_num}.pth")

            if iter_num >= args.max_iterations:
                break
        if iter_num >= args.max_iterations:
            break

    torch.save(model_a.state_dict(), snapshot_path / "mac_s3robo_branch_a_final.pth")
    torch.save(model_b.state_dict(), snapshot_path / "mac_s3robo_branch_b_final.pth")
    writer.close()
    logging.info(f"Training finished. Best A={best_a:.4f}, B={best_b:.4f}, Ensemble={best_ensemble:.4f}")


def main() -> None:
    args = build_parser().parse_args()
    set_reproducibility(args.seed, args.deterministic)
    ratio_tag = f"{int(round(args.labeled_ratio * 100)):02d}p"
    snapshot_path = Path(args.model_dir) / args.exp / ratio_tag
    setup_logger(snapshot_path)
    train(args, snapshot_path)


if __name__ == "__main__":
    main()
