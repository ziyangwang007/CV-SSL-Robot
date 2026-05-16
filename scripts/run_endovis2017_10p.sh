#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root. The H5 dataset should be arranged as:
#   ../data/robotic/train/*.h5
#   ../data/robotic/val/*.h5
#   ../data/robotic/test/*.h5

python code/train_mac_s3robo.py \
  --root_path ../data/robotic \
  --train_split data/splits/endovis2017_train.txt \
  --val_split data/splits/endovis2017_val.txt \
  --exp EndoVis2017/MAC_S3Robo \
  --num_classes 2 \
  --binary_mask \
  --labeled_ratio 0.10 \
  --batch_size 24 \
  --labeled_bs 12 \
  --max_iterations 30000 \
  --base_lr 0.01 \
  --patch_size 256 256
