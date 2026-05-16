# Method mapping

MAC-S3Robo is implemented as a surgical EndoVis adaptation of Semi-Mamba-UNet.

| Paper component | Code location |
|---|---|
| Dual Mamba-based U-shaped branches | `code/networks/mac_s3robo.py`, class `MACS3RoboBranch` |
| VSS encoder-decoder | `code/networks/mamba_sys.py` and `code/networks/vision_mamba.py` |
| Labelled CE + Dice supervision | `code/train_mac_s3robo.py`, `supervised_ce_dice_loss` |
| Cross pseudo-label supervision | `code/train_mac_s3robo.py`, `semi_a` and `semi_b` |
| Pixel-level contrastive self-supervision | `code/utils/ssl_losses.py`, `PixelContrastiveLoss` |
| Robotic H5 data loader | `code/dataloaders/robot_h5.py` |
| Evaluation metrics | `code/utils/robot_metrics.py` |
