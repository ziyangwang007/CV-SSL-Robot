# CV-SSL-Robot / MAC-S3Robo

Mamba-driven contrastive semi-supervised surgical robot segmentation for EndoVis-style HDF5 datasets.

This repository combines the Semi-Mamba-UNet training idea with the surgical robotic H5 data pipeline from CV-WSL-Robot. The main method trains two Mamba-UNet branches, exchanges cross pseudo-labels on unlabelled frames, and adds a pixel-level contrastive loss over decoder features.

## What is implemented

- Dual Mamba-UNet branches using VSS blocks from the Mamba-UNet codebase.
- RGB surgical frame support; grayscale input is still accepted and repeated to three channels.
- Semi-supervised labelled/unlabelled mini-batches through a two-stream sampler.
- Supervised CE + Dice loss on labelled samples.
- Cross pseudo-label CE loss on unlabelled samples.
- Symmetric pixel-level InfoNCE contrastive loss with a lightweight projector head.
- Mean Dice, accuracy, precision, sensitivity, and specificity evaluation.
- EndoVis 2017 split templates from the uploaded surgical robot repository.

## Repository layout

```text
MAC-S3Robo/
  code/
    train_mac_s3robo.py                 # main semi-supervised method
    test_mac_s3robo.py                  # test/inference with branch or ensemble checkpoints
    train_fully_supervised_mamba_robot.py
    compute_mamba_cost.py
    dataloaders/robot_h5.py
    networks/mac_s3robo.py              # Mamba-UNet branch + projector
    networks/                           # original Mamba-UNet network modules
    utils/ssl_losses.py
    utils/robot_metrics.py
  data/splits/                          # split templates
  scripts/                              # runnable shell examples
  third_party/                          # local mamba and causal-conv1d sources from uploaded repo
  docs/                                 # original README files and paper copy
```

## Environment

The original paper settings were Ubuntu 20.04, Python 3.8.8, PyTorch 1.10, CUDA 11.3, and one RTX 3090. Newer CUDA/PyTorch combinations can work, but Mamba selective scan is CUDA-sensitive.

```bash
conda create -n mac-s3robo python=3.8 -y
conda activate mac-s3robo
pip install -r requirements.txt

# Install local CUDA extensions from the included source trees.
pip install -e third_party/causal-conv1d
pip install -e third_party/mamba
```

If local extension installation fails on your machine, try the matching wheel versions from `mamba-ssm` and `causal-conv1d` for your CUDA/PyTorch version.

## Data format

The loader expects HDF5 files with at least:

```text
image: H x W x 3 RGB image, or H x W grayscale image
label: H x W integer mask
```

Recommended layout:

```text
../data/robotic/
  train/*.h5
  val/*.h5
  test/*.h5
```

For EndoVis 2017 binary instrument masks, use `--binary_mask`. This converts every non-zero label, including 255-valued foreground masks, to class 1.

For EndoVis 2018 multi-class scene segmentation, do not use `--binary_mask`; instead set the correct `--num_classes` for your processed labels.

## Train MAC-S3Robo on EndoVis 2017, 10% labelled

Run from the repository root:

```bash
bash scripts/run_endovis2017_10p.sh
```

Equivalent explicit command:

```bash
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
```

Outputs are written to:

```text
../model/EndoVis2017/MAC_S3Robo/10p/
  mac_s3robo_branch_a_best.pth
  mac_s3robo_branch_b_best.pth
  mac_s3robo_ensemble_best.pth
  validation_metrics.csv
  labeled_indices.txt
  unlabeled_indices.txt
  log.txt
```

## Test

```bash
python code/test_mac_s3robo.py \
  --root_path ../data/robotic \
  --split test \
  --split_file data/splits/endovis2017_test.txt \
  --num_classes 2 \
  --binary_mask \
  --checkpoint ../model/EndoVis2017/MAC_S3Robo/10p/mac_s3robo_ensemble_best.pth \
  --output_dir ../model/EndoVis2017/MAC_S3Robo/10p/test_predictions \
  --save_predictions
```

The test script writes `metrics.csv` and optional PNG masks.

## Label-ratio sensitivity sweep

```bash
bash scripts/run_ratio_sweep_endovis2017.sh
```

This runs ratios 10%, 20%, ..., 90% using the same split template and writes one experiment folder per ratio.

## EndoVis 2018 example

Prepare an H5 root such as `../data/endovis2018` and provide split files for train/val/test. Then run:

```bash
python code/train_mac_s3robo.py \
  --root_path ../data/endovis2018 \
  --train_split data/splits/endovis2018_train.txt \
  --val_split data/splits/endovis2018_val.txt \
  --exp EndoVis2018/MAC_S3Robo \
  --num_classes <YOUR_CLASS_COUNT> \
  --labeled_ratio 0.10 \
  --batch_size 24 \
  --labeled_bs 12 \
  --max_iterations 30000 \
  --base_lr 0.01 \
  --patch_size 256 256
```

## Fully supervised Mamba-UNet baseline

```bash
python code/train_fully_supervised_mamba_robot.py \
  --root_path ../data/robotic \
  --train_split data/splits/endovis2017_train.txt \
  --val_split data/splits/endovis2017_val.txt \
  --exp EndoVis2017/MambaUNet_Fully \
  --num_classes 2 \
  --binary_mask \
  --batch_size 12 \
  --max_iterations 30000
```

## GitHub publishing

After unzipping this folder locally:

```bash
cd MAC-S3Robo
git init
git add .
git commit -m "Initial MAC-S3Robo code release"
git branch -M main

# Create an empty repo on GitHub, then:
git remote add origin git@github.com:ziyangwang007/CV-SSL-Robot.git
git push -u origin main
```

## Notes

- `code/networks/mac_s3robo.py` is the method-specific wrapper: it exposes final decoder features from VSSM and attaches the projector head.
- `code/train_mac_s3robo.py` contains the full objective: labelled supervision + cross pseudo-label supervision + pixel-level contrastive self-supervision.
- The default config does not require a pretrained VMamba checkpoint. Add `--load_pretrained --pretrained_ckpt /path/to/vmamba_tiny_e292.pth` to use one.
- For high-resolution inputs, reduce `--contrastive_pixels` or `--batch_size` if GPU memory is tight.

