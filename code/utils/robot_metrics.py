from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from scipy.ndimage import zoom


@dataclass
class MetricAverages:
    dice: float
    acc: float
    precision: float
    sensitivity: float
    specificity: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "dice": self.dice,
            "acc": self.acc,
            "precision": self.precision,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
        }


def per_class_metrics(pred: np.ndarray, gt: np.ndarray, cls: int, eps: float = 1e-7) -> Dict[str, float]:
    pred_c = pred == cls
    gt_c = gt == cls
    tp = np.logical_and(pred_c, gt_c).sum(dtype=np.float64)
    tn = np.logical_and(~pred_c, ~gt_c).sum(dtype=np.float64)
    fp = np.logical_and(pred_c, ~gt_c).sum(dtype=np.float64)
    fn = np.logical_and(~pred_c, gt_c).sum(dtype=np.float64)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    acc = (tp + tn + eps) / (tp + tn + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    sensitivity = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)
    return {
        "dice": float(dice),
        "acc": float(acc),
        "precision": float(precision),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
    }


def mean_multiclass_metrics(pred: np.ndarray, gt: np.ndarray, num_classes: int, include_background: bool = False) -> MetricAverages:
    classes = range(0 if include_background else 1, num_classes)
    values = [per_class_metrics(pred, gt, c) for c in classes]
    if not values:
        values = [per_class_metrics(pred, gt, 1)]
    return MetricAverages(
        dice=float(np.mean([v["dice"] for v in values])),
        acc=float(np.mean([v["acc"] for v in values])),
        precision=float(np.mean([v["precision"] for v in values])),
        sensitivity=float(np.mean([v["sensitivity"] for v in values])),
        specificity=float(np.mean([v["specificity"] for v in values])),
    )


def _prepare_image_tensor(image: torch.Tensor, patch_size: Tuple[int, int], device: torch.device) -> torch.Tensor:
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError(f"Expected CxHxW or BxCxHxW image tensor, got {image.shape}")
    image_np = image.squeeze(0).cpu().numpy()
    c, h, w = image_np.shape
    if (h, w) != tuple(patch_size):
        resized = np.stack([zoom(image_np[ch], (patch_size[0] / h, patch_size[1] / w), order=1) for ch in range(c)], axis=0)
    else:
        resized = image_np
    return torch.from_numpy(resized).unsqueeze(0).float().to(device)


def _resize_pred(pred: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    if pred.shape == target_shape:
        return pred
    return zoom(pred, (target_shape[0] / pred.shape[0], target_shape[1] / pred.shape[1]), order=0)


def predict_single(image: torch.Tensor, model, patch_size: Tuple[int, int], device: torch.device) -> np.ndarray:
    input_tensor = _prepare_image_tensor(image, patch_size, device)
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        if isinstance(output, (tuple, list)):
            output = output[0]
        pred = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0).cpu().numpy()
    _, h, w = image.shape if image.ndim == 3 else image.squeeze(0).shape
    return _resize_pred(pred, (h, w)).astype(np.int64)


def predict_single_pair(image: torch.Tensor, model_a, model_b, patch_size: Tuple[int, int], device: torch.device) -> np.ndarray:
    input_tensor = _prepare_image_tensor(image, patch_size, device)
    model_a.eval(); model_b.eval()
    with torch.no_grad():
        out_a = model_a(input_tensor)
        out_b = model_b(input_tensor)
        if isinstance(out_a, (tuple, list)):
            out_a = out_a[0]
        if isinstance(out_b, (tuple, list)):
            out_b = out_b[0]
        output = 0.5 * (out_a + out_b)
        pred = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0).cpu().numpy()
    _, h, w = image.shape if image.ndim == 3 else image.squeeze(0).shape
    return _resize_pred(pred, (h, w)).astype(np.int64)


def evaluate_loader(model, loader, num_classes: int, patch_size: Tuple[int, int], device: torch.device) -> Dict[str, float]:
    all_metrics: List[MetricAverages] = []
    for batch in loader:
        image = batch["image"][0]
        label = batch["label"][0].cpu().numpy().astype(np.int64)
        pred = predict_single(image, model, patch_size, device)
        all_metrics.append(mean_multiclass_metrics(pred, label, num_classes))
    return _average_metric_objects(all_metrics)


def evaluate_loader_pair(model_a, model_b, loader, num_classes: int, patch_size: Tuple[int, int], device: torch.device) -> Dict[str, float]:
    all_metrics: List[MetricAverages] = []
    for batch in loader:
        image = batch["image"][0]
        label = batch["label"][0].cpu().numpy().astype(np.int64)
        pred = predict_single_pair(image, model_a, model_b, patch_size, device)
        all_metrics.append(mean_multiclass_metrics(pred, label, num_classes))
    return _average_metric_objects(all_metrics)


def _average_metric_objects(metrics: List[MetricAverages]) -> Dict[str, float]:
    if not metrics:
        return {"dice": 0.0, "acc": 0.0, "precision": 0.0, "sensitivity": 0.0, "specificity": 0.0}
    keys = metrics[0].as_dict().keys()
    return {k: float(np.mean([m.as_dict()[k] for m in metrics])) for k in keys}
