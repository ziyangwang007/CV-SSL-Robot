#!/usr/bin/env bash
set -euo pipefail

for ratio in 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90; do
  python code/train_mac_s3robo.py \
    --root_path ../data/robotic \
    --train_split data/splits/endovis2017_train.txt \
    --val_split data/splits/endovis2017_val.txt \
    --exp EndoVis2017/MAC_S3Robo \
    --num_classes 2 \
    --binary_mask \
    --labeled_ratio "${ratio}" \
    --batch_size 24 \
    --labeled_bs 12 \
    --max_iterations 30000 \
    --base_lr 0.01 \
    --patch_size 256 256
 done
