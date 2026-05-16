"""HDF5 dataloaders for semi-supervised EndoVis robotic segmentation.

Expected H5 keys:
    image: H x W x 3 RGB image, or H x W grayscale image
    label: H x W integer mask

The loader supports either train/val/test directories under root_path, e.g.
    root_path/train/*.h5
    root_path/val/*.h5
    root_path/test/*.h5
or explicit split files containing names or paths. A line such as `data_0001`
will be resolved as `<root>/<split>/data_0001.h5` first.
"""

from __future__ import annotations

import itertools
import os
import random
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler


class RoboticH5Dataset(Dataset):
    def __init__(
        self,
        base_dir: str,
        split: str = "train",
        transform=None,
        split_file: Optional[str] = None,
        image_key: str = "image",
        label_key: str = "label",
        binary_mask: bool = False,
        ignore_label: int = 255,
        normalize: bool = True,
        return_name: bool = True,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.split = split
        self.transform = transform
        self.image_key = image_key
        self.label_key = label_key
        self.binary_mask = binary_mask
        self.ignore_label = int(ignore_label)
        self.normalize = normalize
        self.return_name = return_name
        self.sample_list = self._build_sample_list(split_file)
        if len(self.sample_list) == 0:
            raise RuntimeError(
                f"No H5 samples found for split='{split}'. Checked base_dir={self.base_dir} "
                f"and split_file={split_file}."
            )
        print(f"[{split}] total {len(self.sample_list)} samples")

    def _read_lines(self, split_file: str) -> List[str]:
        with open(split_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]

    def _resolve_line(self, line: str) -> Path:
        p = Path(line)
        candidates: List[Path] = []
        if p.is_absolute():
            candidates.append(p)
            if p.suffix == "":
                candidates.append(Path(str(p) + ".h5"))
        else:
            if p.suffix == ".h5":
                candidates.extend([
                    self.base_dir / p,
                    self.base_dir / self.split / p.name,
                ])
            else:
                candidates.extend([
                    self.base_dir / self.split / f"{line}.h5",
                    self.base_dir / f"{line}.h5",
                    self.base_dir / self.split / line,
                    self.base_dir / line,
                ])
        for c in candidates:
            if c.exists():
                return c
        # Return the most likely path so the eventual error is informative.
        return candidates[0]

    def _build_sample_list(self, split_file: Optional[str]) -> List[Path]:
        if split_file:
            return [self._resolve_line(line) for line in self._read_lines(split_file)]
        split_dir = self.base_dir / self.split
        if split_dir.exists():
            return sorted(split_dir.glob("*.h5"))
        return sorted(self.base_dir.glob(f"{self.split}/*.h5"))

    def __len__(self) -> int:
        return len(self.sample_list)

    def _load_case(self, path: Path) -> Tuple[np.ndarray, np.ndarray]:
        if not path.exists():
            raise FileNotFoundError(f"Cannot find H5 sample: {path}")
        with h5py.File(path, "r") as h5f:
            if self.image_key not in h5f:
                raise KeyError(f"Missing key '{self.image_key}' in {path}. Available keys: {list(h5f.keys())}")
            if self.label_key not in h5f:
                raise KeyError(f"Missing key '{self.label_key}' in {path}. Available keys: {list(h5f.keys())}")
            image = np.asarray(h5f[self.image_key][:])
            label = np.asarray(h5f[self.label_key][:])
        image = _to_float_image(image, normalize=self.normalize)
        label = _prepare_label(label, binary_mask=self.binary_mask, ignore_label=self.ignore_label)
        return image, label

    def __getitem__(self, idx: int):
        path = self.sample_list[idx]
        image, label = self._load_case(path)
        sample = {"image": image, "label": label}
        if self.split == "train" and self.transform is not None:
            sample = self.transform(sample)
        else:
            sample = to_tensor(sample)
        sample["idx"] = idx
        if self.return_name:
            sample["name"] = path.stem
        return sample


class RandomGenerator:
    def __init__(self, output_size: Sequence[int], ignore_label: int = 255) -> None:
        self.output_size = tuple(int(v) for v in output_size)
        self.ignore_label = int(ignore_label)

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label, cval=self.ignore_label)
        image, label = resize_pair(image, label, self.output_size)
        return to_tensor({"image": image, "label": label})


class CenterResize:
    def __init__(self, output_size: Sequence[int]) -> None:
        self.output_size = tuple(int(v) for v in output_size)

    def __call__(self, sample):
        image, label = resize_pair(sample["image"], sample["label"], self.output_size)
        return to_tensor({"image": image, "label": label})


def _to_float_image(image: np.ndarray, normalize: bool = True) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        image = image[..., None]
    if image.ndim != 3:
        raise ValueError(f"Expected image shape HxW or HxWxC, got {image.shape}")
    image = image.astype(np.float32)
    if normalize:
        max_val = float(np.nanmax(image)) if image.size else 1.0
        if max_val > 1.5:
            image = image / 255.0
    return image


def _prepare_label(label: np.ndarray, binary_mask: bool, ignore_label: int) -> np.ndarray:
    label = np.asarray(label)
    if label.ndim == 3 and label.shape[-1] == 1:
        label = label[..., 0]
    if label.ndim != 2:
        raise ValueError(f"Expected label shape HxW, got {label.shape}")
    if binary_mask:
        # EndoVis 2017 binary masks are often stored as {0, 255}; all non-zero labels become foreground.
        label = (label > 0).astype(np.uint8)
    else:
        label = label.astype(np.int64)
    return label


def random_rot_flip(image: np.ndarray, label: np.ndarray):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k, axes=(0, 1)).copy()
    label = np.rot90(label, k, axes=(0, 1)).copy()
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image: np.ndarray, label: np.ndarray, cval: int = 255):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, axes=(0, 1), order=1, reshape=False, mode="reflect")
    label = ndimage.rotate(label, angle, axes=(0, 1), order=0, reshape=False, mode="constant", cval=cval)
    return image, label


def resize_pair(image: np.ndarray, label: np.ndarray, output_size: Sequence[int]):
    out_h, out_w = int(output_size[0]), int(output_size[1])
    h, w = image.shape[:2]
    image_zoom = (out_h / h, out_w / w, 1) if image.ndim == 3 else (out_h / h, out_w / w)
    image = zoom(image, image_zoom, order=1)
    label = zoom(label, (out_h / h, out_w / w), order=0)
    return image, label


def to_tensor(sample):
    image, label = sample["image"], sample["label"]
    if image.ndim == 2:
        image = image[..., None]
    image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1).contiguous()
    label = torch.from_numpy(label.astype(np.int64)).long()
    return {"image": image, "label": label}


class TwoStreamBatchSampler(Sampler):
    """Iterate one labelled stream and one unlabelled stream in each mini-batch."""

    def __init__(self, primary_indices: Sequence[int], secondary_indices: Sequence[int], batch_size: int, secondary_batch_size: int):
        self.primary_indices = list(primary_indices)
        self.secondary_indices = list(secondary_indices)
        self.secondary_batch_size = int(secondary_batch_size)
        self.primary_batch_size = int(batch_size) - self.secondary_batch_size
        if not (len(self.primary_indices) >= self.primary_batch_size > 0):
            raise ValueError("Need at least one labelled sample per batch; reduce labelled_bs or increase labelled set.")
        if not (len(self.secondary_indices) >= self.secondary_batch_size > 0):
            raise ValueError("Need at least one unlabelled sample per batch; reduce unlabelled batch size or labelled ratio.")

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable: Iterable[int]):
    return np.random.permutation(list(iterable)).tolist()


def iterate_eternally(indices: Sequence[int]):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices).tolist()
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable: Iterable[int], n: int):
    args = [iter(iterable)] * n
    return zip(*args)


def split_labeled_unlabeled(total: int, labeled_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    if not (0.0 < labeled_ratio < 1.0):
        raise ValueError("labeled_ratio must be between 0 and 1 for semi-supervised training.")
    rng = np.random.default_rng(seed)
    indices = np.arange(total)
    rng.shuffle(indices)
    labeled_count = max(1, int(round(total * labeled_ratio)))
    labeled_count = min(labeled_count, total - 1)
    labeled = sorted(indices[:labeled_count].tolist())
    unlabeled = sorted(indices[labeled_count:].tolist())
    return labeled, unlabeled
