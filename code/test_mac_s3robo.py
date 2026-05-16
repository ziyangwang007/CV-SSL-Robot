from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_config
from dataloaders.robot_h5 import RoboticH5Dataset
from networks.mac_s3robo import MACS3RoboBranch, maybe_disable_missing_pretrain
from utils.robot_metrics import mean_multiclass_metrics, predict_single, predict_single_pair


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate MAC-S3Robo checkpoints on EndoVis H5 splits")
    parser.add_argument("--root_path", type=str, default="../data/robotic")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--split_file", type=str, default=None)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--binary_mask", action="store_true")
    parser.add_argument("--ignore_label", type=int, default=255)
    parser.add_argument("--image_key", type=str, default="image")
    parser.add_argument("--label_key", type=str, default="label")
    parser.add_argument("--patch_size", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--projection_dim", type=int, default=128)
    parser.add_argument("--checkpoint", type=str, default=None, help="Ensemble checkpoint containing branch_a and branch_b state dicts")
    parser.add_argument("--checkpoint_a", type=str, default=None)
    parser.add_argument("--checkpoint_b", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="../model/MAC_S3Robo_predictions")
    parser.add_argument("--save_predictions", action="store_true")

    # Arguments consumed by config.get_config.
    parser.add_argument("--cfg", type=str, default="code/configs/vmamba_tiny.yaml")
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


def resolve_path(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    p = Path(path)
    if p.exists() or p.is_absolute():
        return str(p)
    candidates = [Path.cwd() / path, Path(__file__).resolve().parents[1] / path]
    for c in candidates:
        if c.exists():
            return str(c)
    return str(p)


def load_models(args, config, device):
    config.defrost()
    config.MODEL.PRETRAIN_CKPT = None
    config.freeze()
    maybe_disable_missing_pretrain(config)
    model_a = MACS3RoboBranch(config, img_size=args.patch_size, num_classes=args.num_classes, projection_dim=args.projection_dim).to(device)
    model_b = MACS3RoboBranch(config, img_size=args.patch_size, num_classes=args.num_classes, projection_dim=args.projection_dim).to(device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        if "branch_a" in ckpt and "branch_b" in ckpt:
            model_a.load_state_dict(ckpt["branch_a"])
            model_b.load_state_dict(ckpt["branch_b"])
            return model_a, model_b
        raise KeyError("--checkpoint must contain keys 'branch_a' and 'branch_b'.")
    if args.checkpoint_a:
        model_a.load_state_dict(torch.load(args.checkpoint_a, map_location=device))
    else:
        model_a = None
    if args.checkpoint_b:
        model_b.load_state_dict(torch.load(args.checkpoint_b, map_location=device))
    else:
        model_b = None
    return model_a, model_b


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8)).save(path)


def main():
    args = build_parser().parse_args()
    args.cfg = resolve_path(args.cfg)
    args.split_file = resolve_path(args.split_file)
    if args.checkpoint:
        args.checkpoint = resolve_path(args.checkpoint)
    if args.checkpoint_a:
        args.checkpoint_a = resolve_path(args.checkpoint_a)
    if args.checkpoint_b:
        args.checkpoint_b = resolve_path(args.checkpoint_b)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_config(args)
    model_a, model_b = load_models(args, config, device)
    if model_a is None and model_b is None:
        raise ValueError("Provide --checkpoint or at least one of --checkpoint_a / --checkpoint_b")

    dataset = RoboticH5Dataset(
        base_dir=args.root_path,
        split=args.split,
        split_file=args.split_file,
        transform=None,
        image_key=args.image_key,
        label_key=args.label_key,
        binary_mask=args.binary_mask,
        ignore_label=args.ignore_label,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for batch in tqdm(loader, ncols=90):
        image = batch["image"][0]
        label = batch["label"][0].cpu().numpy().astype(np.int64)
        name = batch.get("name", [f"case_{len(rows):04d}"])[0]
        if model_a is not None and model_b is not None:
            pred = predict_single_pair(image, model_a, model_b, tuple(args.patch_size), device)
            branch = "ensemble"
        elif model_a is not None:
            pred = predict_single(image, model_a, tuple(args.patch_size), device)
            branch = "a"
        else:
            pred = predict_single(image, model_b, tuple(args.patch_size), device)
            branch = "b"
        metrics = mean_multiclass_metrics(pred, label, args.num_classes).as_dict()
        rows.append({"case": name, "branch": branch, **metrics})
        if args.save_predictions:
            save_mask(pred, out_dir / "masks" / f"{name}.png")

    mean_row = {"case": "MEAN", "branch": rows[0]["branch"] if rows else "NA"}
    for key in ["dice", "acc", "precision", "sensitivity", "specificity"]:
        mean_row[key] = float(np.mean([r[key] for r in rows])) if rows else 0.0
    rows.append(mean_row)
    with open(out_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(mean_row)


if __name__ == "__main__":
    main()
