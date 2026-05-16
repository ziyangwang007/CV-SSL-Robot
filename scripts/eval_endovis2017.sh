#!/usr/bin/env bash
set -euo pipefail

python code/test_mac_s3robo.py \
  --root_path ../data/robotic \
  --split test \
  --split_file data/splits/endovis2017_test.txt \
  --num_classes 2 \
  --binary_mask \
  --checkpoint ../model/EndoVis2017/MAC_S3Robo/10p/mac_s3robo_ensemble_best.pth \
  --output_dir ../model/EndoVis2017/MAC_S3Robo/10p/test_predictions \
  --save_predictions
