#!/usr/bin/env python3
"""Create labelled/unlabelled split files for semi-supervised EndoVis experiments.

Usage:
  python scripts/create_ssl_splits.py --train_file data/splits/endovis2017_train.txt --ratio 0.1 --out_dir data/splits --prefix endovis2017
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out_dir", default="data/splits")
    parser.add_argument("--prefix", default="endovis")
    args = parser.parse_args()

    with open(args.train_file, "r", encoding="utf-8") as f:
        samples = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    if not (0 < args.ratio < 1):
        raise ValueError("--ratio must be in (0, 1)")
    rng = np.random.default_rng(args.seed)
    indices = np.arange(len(samples))
    rng.shuffle(indices)
    n_lab = max(1, int(round(len(samples) * args.ratio)))
    labeled_idx = sorted(indices[:n_lab].tolist())
    unlabeled_idx = sorted(indices[n_lab:].tolist())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{int(round(args.ratio * 100)):02d}p"
    labeled_path = out_dir / f"{args.prefix}_labeled_{tag}.txt"
    unlabeled_path = out_dir / f"{args.prefix}_unlabeled_{tag}.txt"
    labeled_path.write_text("\n".join(samples[i] for i in labeled_idx) + "\n", encoding="utf-8")
    unlabeled_path.write_text("\n".join(samples[i] for i in unlabeled_idx) + "\n", encoding="utf-8")
    print(f"wrote {labeled_path} ({len(labeled_idx)} samples)")
    print(f"wrote {unlabeled_path} ({len(unlabeled_idx)} samples)")


if __name__ == "__main__":
    main()
