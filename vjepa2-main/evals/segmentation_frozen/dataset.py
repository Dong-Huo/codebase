# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Segmentation datasets for probe-training on top of a frozen V-JEPA 2.1 encoder.

Supported dataset layouts
--------------------------
ADE20K   (dataset_name="ade20k")
    <root>/
        images/training/   *.jpg
        images/validation/ *.jpg
        annotations/training/   *.png   (pixel values = class ids, 0 = background)
        annotations/validation/ *.png

Pascal VOC 2012  (dataset_name="voc2012")
    <root>/
        JPEGImages/      *.jpg
        SegmentationClass/ *.png  (pixel values = class ids, 255 = ignore)
        ImageSets/Segmentation/train.txt
        ImageSets/Segmentation/val.txt

Cityscapes  (dataset_name="cityscapes")
    <root>/
        leftImg8bit/{train,val}/<city>/*.png
        gtFine/{train,val}/<city>/*_gtFine_labelTrainIds.png

Generic / custom  (dataset_name="generic")
    split_file: text file, one entry per line:
        <img_rel_path> <mask_rel_path>
    Paths are relative to <root>.
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler

import torchvision.transforms as T
import torchvision.transforms.functional as TF

from src.utils.logging import get_logger

logger = get_logger("segmentation_frozen.dataset")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Dataset base
# ---------------------------------------------------------------------------

class SegmentationDataset(Dataset):
    """
    Base class for image segmentation datasets.

    Subclasses must implement `_collect_samples()` which populates
    `self.samples` as a list of (image_path, mask_path) tuples.

    Augmentation
    ------------
    Training:
      - Random scale & crop (multiscale, scale ∈ [min_scale, max_scale])
      - Random horizontal flip
    Validation:
      - Resize shorter side to `resolution`, then center-crop
    """

    def __init__(
        self,
        resolution: int,
        training: bool,
        ignore_index: int = 255,
        min_scale: float = 0.5,
        max_scale: float = 2.0,
    ):
        self.resolution = resolution
        self.training = training
        self.ignore_index = ignore_index
        self.min_scale = min_scale
        self.max_scale = max_scale

        self.img_norm = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self.samples = []           # filled by subclasses
        self._collect_samples()
        logger.info(f"{'Train' if training else 'Val'} set: {len(self.samples)} samples")

    def _collect_samples(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, msk_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(msk_path)          # palette or grayscale PNG

        if self.training:
            image, mask = self._random_scale_crop(image, mask)
            if torch.rand(1).item() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
        else:
            image, mask = self._val_resize_crop(image, mask)

        image = TF.to_tensor(image)         # [3, H, W], float in [0, 1]
        image = self.img_norm(image)

        mask = torch.as_tensor(np.array(mask, dtype=np.int64))  # [H, W]
        mask = self._remap_ignore(mask)
        return image, mask

    # ------------------------------------------------------------------
    # Augmentation helpers
    # ------------------------------------------------------------------

    def _random_scale_crop(self, image, mask):
        r = self.resolution
        ratio = self.min_scale + torch.rand(1).item() * (self.max_scale - self.min_scale)
        scaled = int(r * ratio)

        image = TF.resize(image, [scaled, scaled],
                          interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [scaled, scaled],
                         interpolation=TF.InterpolationMode.NEAREST)

        # Pad if smaller than crop size
        pad_h = max(r - scaled, 0)
        pad_w = max(r - scaled, 0)
        if pad_h > 0 or pad_w > 0:
            image = TF.pad(image, [0, 0, pad_w, pad_h], fill=0)
            mask = TF.pad(mask, [0, 0, pad_w, pad_h], fill=self.ignore_index)

        i, j, h, w = T.RandomCrop.get_params(image, (r, r))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)
        return image, mask

    def _val_resize_crop(self, image, mask):
        r = self.resolution
        image = TF.resize(image, [r, r], interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [r, r], interpolation=TF.InterpolationMode.NEAREST)
        return image, mask

    def _remap_ignore(self, mask: torch.Tensor) -> torch.Tensor:
        """Override in subclasses to remap dataset-specific ignore values."""
        return mask


# ---------------------------------------------------------------------------
# ADE20K
# ---------------------------------------------------------------------------

class ADE20KDataset(SegmentationDataset):
    """
    ADE20K (150 stuff + thing classes, background = class 0).

    Mask pixel values are 1-indexed class IDs; 0 means "no object" (ignore).
    We subtract 1 so class IDs become 0-based and set 0→ignore_index.

    Folder layout::
        <root>/images/{training,validation}/*.jpg
        <root>/annotations/{training,validation}/*.png
    """

    NUM_CLASSES = 150

    def __init__(self, root: str, training: bool, **kwargs):
        self.root = root
        self.split = "training" if training else "validation"
        super().__init__(training=training, **kwargs)

    def _collect_samples(self):
        img_dir = os.path.join(self.root, "images", self.split)
        msk_dir = os.path.join(self.root, "annotations", self.split)
        for fname in sorted(os.listdir(img_dir)):
            stem = os.path.splitext(fname)[0]
            img_path = os.path.join(img_dir, fname)
            msk_path = os.path.join(msk_dir, stem + ".png")
            if os.path.isfile(msk_path):
                self.samples.append((img_path, msk_path))

    def _remap_ignore(self, mask: torch.Tensor) -> torch.Tensor:
        # ADE20K masks: 0 = background (ignore), 1-150 = classes → 0-149
        out = mask - 1                          # 0-149, background → -1
        out[out < 0] = self.ignore_index        # background → 255
        return out


# ---------------------------------------------------------------------------
# Pascal VOC 2012
# ---------------------------------------------------------------------------

class VOC2012Dataset(SegmentationDataset):
    """
    Pascal VOC 2012 semantic segmentation (21 classes including background).

    Mask pixel values: 0=background, 1-20=object classes, 255=ignore.

    Folder layout::
        <root>/JPEGImages/*.jpg
        <root>/SegmentationClass/*.png
        <root>/ImageSets/Segmentation/{train,val}.txt
    """

    NUM_CLASSES = 21

    def __init__(self, root: str, training: bool, **kwargs):
        self.root = root
        self.split = "train" if training else "val"
        super().__init__(training=training, **kwargs)

    def _collect_samples(self):
        split_file = os.path.join(
            self.root, "ImageSets", "Segmentation", self.split + ".txt"
        )
        with open(split_file) as f:
            names = [l.strip() for l in f if l.strip()]
        for name in names:
            img_path = os.path.join(self.root, "JPEGImages", name + ".jpg")
            msk_path = os.path.join(self.root, "SegmentationClass", name + ".png")
            if os.path.isfile(img_path) and os.path.isfile(msk_path):
                self.samples.append((img_path, msk_path))


# ---------------------------------------------------------------------------
# Cityscapes
# ---------------------------------------------------------------------------

class CityscapesDataset(SegmentationDataset):
    """
    Cityscapes semantic segmentation (19 train classes).

    Uses `*_gtFine_labelTrainIds.png` masks where:
      - 0-18 are the 19 train classes
      - 255 means ignore

    Folder layout::
        <root>/leftImg8bit/{train,val}/<city>/*_leftImg8bit.png
        <root>/gtFine/{train,val}/<city>/*_gtFine_labelTrainIds.png
    """

    NUM_CLASSES = 19

    def __init__(self, root: str, training: bool, **kwargs):
        self.root = root
        self.split = "train" if training else "val"
        super().__init__(training=training, **kwargs)

    def _collect_samples(self):
        img_base = os.path.join(self.root, "leftImg8bit", self.split)
        msk_base = os.path.join(self.root, "gtFine", self.split)
        for city in sorted(os.listdir(img_base)):
            city_img_dir = os.path.join(img_base, city)
            city_msk_dir = os.path.join(msk_base, city)
            for fname in sorted(os.listdir(city_img_dir)):
                if not fname.endswith("_leftImg8bit.png"):
                    continue
                stem = fname.replace("_leftImg8bit.png", "")
                msk_fname = stem + "_gtFine_labelTrainIds.png"
                img_path = os.path.join(city_img_dir, fname)
                msk_path = os.path.join(city_msk_dir, msk_fname)
                if os.path.isfile(msk_path):
                    self.samples.append((img_path, msk_path))


# ---------------------------------------------------------------------------
# Generic / custom (text-file pair list)
# ---------------------------------------------------------------------------

class GenericSegDataset(SegmentationDataset):
    """
    Generic segmentation dataset driven by a text file.

    Each line of `split_file` must contain two whitespace-separated fields::
        <img_relative_path>  <mask_relative_path>

    Both paths are joined with `root`.
    """

    def __init__(self, root: str, split_file: str, training: bool, **kwargs):
        self.root = root
        self.split_file = split_file
        super().__init__(training=training, **kwargs)

    def _collect_samples(self):
        with open(self.split_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_path = os.path.join(self.root, parts[0])
                    msk_path = os.path.join(self.root, parts[1])
                    self.samples.append((img_path, msk_path))


# ---------------------------------------------------------------------------
# Factory + DataLoader
# ---------------------------------------------------------------------------

_DATASETS = {
    "ade20k": ADE20KDataset,
    "voc2012": VOC2012Dataset,
    "cityscapes": CityscapesDataset,
    "generic": GenericSegDataset,
}


def make_segmentation_dataloader(
    dataset_name: str,
    root: str,
    resolution: int,
    batch_size: int,
    world_size: int,
    rank: int,
    training: bool,
    num_workers: int = 8,
    ignore_index: int = 255,
    drop_last: bool = True,
    # generic dataset only
    split_file: str = None,
):
    """
    Build a DistributedSampler-backed DataLoader for segmentation.

    Args:
        dataset_name: one of "ade20k", "voc2012", "cityscapes", "generic"
        root:         dataset root directory
        resolution:   square crop/resize resolution (e.g. 384)
        split_file:   required only for dataset_name="generic"
    """
    dataset_name = dataset_name.lower()
    if dataset_name not in _DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Choose from: {list(_DATASETS.keys())}"
        )

    cls = _DATASETS[dataset_name]
    kwargs = dict(
        root=root,
        training=training,
        resolution=resolution,
        ignore_index=ignore_index,
    )
    if dataset_name == "generic":
        if split_file is None:
            raise ValueError("split_file is required for dataset_name='generic'")
        kwargs["split_file"] = split_file

    dataset = cls(**kwargs)

    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=training
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last and training,
        persistent_workers=num_workers > 0,
    )
    return loader, sampler
