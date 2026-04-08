# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Segmentation visualization utilities.

Saves a side-by-side PNG for each sample in a batch:
    [ original image | ground-truth mask | predicted mask ]

Color palettes
--------------
  - VOC2012   (21 classes)  : standard Pascal VOC colors
  - ADE20K    (150 classes) : MIT ADE20K colors
  - Cityscapes (19 classes) : standard Cityscapes train-id colors
  - Generic   (any N)       : auto-generated palette via HSV spacing
"""

import os

import numpy as np
import torch
from PIL import Image

# ImageNet normalization constants (must match dataset.py)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Palettes
# ---------------------------------------------------------------------------

def _voc_palette():
    """Standard Pascal VOC 21-class palette (same as used in torchvision)."""
    palette = [
        (0,   0,   0),    # 0  background
        (128, 0,   0),    # 1  aeroplane
        (0,   128, 0),    # 2  bicycle
        (128, 128, 0),    # 3  bird
        (0,   0,   128),  # 4  boat
        (128, 0,   128),  # 5  bottle
        (0,   128, 128),  # 6  bus
        (128, 128, 128),  # 7  car
        (64,  0,   0),    # 8  cat
        (192, 0,   0),    # 9  chair
        (64,  128, 0),    # 10 cow
        (192, 128, 0),    # 11 dining table
        (64,  0,   128),  # 12 dog
        (192, 0,   128),  # 13 horse
        (64,  128, 128),  # 14 motorbike
        (192, 128, 128),  # 15 person
        (0,   64,  0),    # 16 potted plant
        (128, 64,  0),    # 17 sheep
        (0,   192, 0),    # 18 sofa
        (128, 192, 0),    # 19 train
        (0,   64,  128),  # 20 tv/monitor
    ]
    return palette


def _cityscapes_palette():
    """Standard Cityscapes 19 train-id colors."""
    palette = [
        (128, 64,  128),  # 0  road
        (244, 35,  232),  # 1  sidewalk
        (70,  70,  70),   # 2  building
        (102, 102, 156),  # 3  wall
        (190, 153, 153),  # 4  fence
        (153, 153, 153),  # 5  pole
        (250, 170, 30),   # 6  traffic light
        (220, 220, 0),    # 7  traffic sign
        (107, 142, 35),   # 8  vegetation
        (152, 251, 152),  # 9  terrain
        (70,  130, 180),  # 10 sky
        (220, 20,  60),   # 11 person
        (255, 0,   0),    # 12 rider
        (0,   0,   142),  # 13 car
        (0,   0,   70),   # 14 truck
        (0,   60,  100),  # 15 bus
        (0,   80,  100),  # 16 train
        (0,   0,   230),  # 17 motorcycle
        (119, 11,  32),   # 18 bicycle
    ]
    return palette


def _ade20k_palette():
    """150-color ADE20K palette (MIT scene parsing benchmark)."""
    # Standard palette shipped with the ADE20K devkit
    palette = [
        (120, 120, 120), (180, 120, 120), (6, 230, 230), (80, 50, 50),
        (4, 200, 3), (120, 120, 80), (140, 140, 140), (204, 5, 255),
        (230, 230, 230), (4, 250, 7), (224, 5, 255), (235, 255, 7),
        (150, 5, 61), (120, 120, 70), (8, 255, 51), (255, 6, 82),
        (143, 255, 140), (204, 255, 4), (255, 51, 7), (204, 70, 3),
        (0, 102, 200), (61, 230, 250), (255, 6, 51), (11, 102, 255),
        (255, 7, 71), (255, 9, 224), (9, 7, 230), (220, 220, 220),
        (255, 9, 92), (112, 9, 255), (8, 255, 214), (7, 255, 224),
        (255, 184, 6), (10, 255, 71), (255, 41, 10), (7, 255, 255),
        (224, 255, 8), (102, 8, 255), (255, 61, 6), (255, 194, 7),
        (255, 122, 8), (0, 255, 20), (255, 8, 41), (255, 5, 153),
        (6, 51, 255), (235, 12, 255), (160, 150, 20), (0, 163, 255),
        (140, 140, 140), (250, 10, 15), (20, 255, 0), (31, 255, 0),
        (255, 31, 0), (255, 224, 0), (153, 255, 0), (0, 0, 255),
        (255, 71, 0), (0, 235, 255), (0, 173, 255), (31, 0, 255),
        (11, 200, 200), (255, 82, 0), (0, 255, 245), (0, 61, 255),
        (0, 255, 112), (0, 255, 133), (255, 0, 0), (255, 163, 0),
        (255, 102, 0), (194, 255, 0), (0, 143, 255), (51, 255, 0),
        (0, 82, 255), (0, 255, 41), (0, 255, 173), (10, 0, 255),
        (173, 255, 0), (0, 255, 153), (255, 92, 0), (255, 0, 255),
        (255, 0, 245), (255, 0, 102), (255, 173, 0), (255, 0, 20),
        (255, 184, 184), (0, 31, 255), (0, 255, 61), (0, 71, 255),
        (255, 0, 204), (0, 255, 194), (0, 255, 82), (0, 10, 255),
        (0, 112, 255), (51, 0, 255), (0, 194, 255), (0, 122, 255),
        (0, 255, 163), (255, 153, 0), (0, 255, 10), (255, 112, 0),
        (143, 255, 0), (82, 0, 255), (163, 255, 0), (255, 235, 0),
        (8, 184, 170), (133, 0, 255), (0, 255, 92), (184, 0, 255),
        (255, 0, 31), (0, 184, 255), (0, 214, 255), (255, 0, 112),
        (92, 255, 0), (0, 224, 255), (112, 224, 255), (70, 184, 160),
        (163, 0, 255), (153, 0, 255), (71, 255, 0), (255, 0, 163),
        (255, 204, 0), (255, 0, 143), (0, 255, 235), (133, 255, 0),
        (255, 0, 235), (245, 0, 255), (255, 0, 122), (255, 245, 0),
        (10, 190, 212), (214, 255, 0), (0, 204, 255), (20, 0, 255),
        (255, 255, 0), (0, 153, 255), (0, 41, 255), (0, 255, 204),
        (41, 0, 255), (41, 255, 0), (173, 0, 255), (0, 245, 255),
        (71, 0, 255), (122, 0, 255), (0, 255, 184), (0, 92, 255),
        (184, 255, 0), (0, 133, 255), (255, 214, 0), (25, 194, 194),
        (102, 255, 0), (92, 0, 255),
    ]
    return palette


def _hsv_palette(n: int):
    """Auto-generate n visually distinct colors via HSV hue spacing."""
    import colorsys
    palette = []
    for i in range(n):
        h = i / n
        r, g, b = colorsys.hsv_to_rgb(h, 0.8, 0.9)
        palette.append((int(r * 255), int(g * 255), int(b * 255)))
    return palette


_PALETTE_REGISTRY = {
    "voc2012":    _voc_palette,
    "cityscapes": _cityscapes_palette,
    "ade20k":     _ade20k_palette,
}


def get_palette(dataset_name: str, num_classes: int):
    """
    Return a list of (R, G, B) tuples, one per class.
    Falls back to HSV-generated colors for unknown datasets or if num_classes
    exceeds the built-in palette length.
    """
    fn = _PALETTE_REGISTRY.get(dataset_name.lower())
    if fn is not None:
        pal = fn()
        if len(pal) >= num_classes:
            return pal[:num_classes]
    return _hsv_palette(num_classes)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _denormalize(img_tensor: torch.Tensor) -> np.ndarray:
    """
    Reverse ImageNet normalization.
    img_tensor: [3, H, W] float
    Returns:    [H, W, 3] uint8
    """
    img = img_tensor.cpu().float().numpy().transpose(1, 2, 0)  # [H, W, 3]
    img = img * _IMAGENET_STD + _IMAGENET_MEAN
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def _mask_to_rgb(mask: np.ndarray, palette: list, ignore_index: int = 255) -> np.ndarray:
    """
    Map a [H, W] integer class-index array to an [H, W, 3] uint8 RGB image.
    Ignore pixels (ignore_index) are rendered in white.
    """
    H, W = mask.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for c, color in enumerate(palette):
        rgb[mask == c] = color
    rgb[mask == ignore_index] = (255, 255, 255)   # white = ignore
    return rgb


def _hstack(*arrays: np.ndarray, gap: int = 4) -> np.ndarray:
    """Horizontally stack [H, W, 3] arrays with a white gap between them."""
    H = arrays[0].shape[0]
    divider = np.full((H, gap, 3), 255, dtype=np.uint8)
    parts = []
    for i, a in enumerate(arrays):
        parts.append(a)
        if i < len(arrays) - 1:
            parts.append(divider)
    return np.concatenate(parts, axis=1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_seg_visualizations(
    out_dir: str,
    epoch: int,
    images: torch.Tensor,
    gt_masks: torch.Tensor,
    pred_logits: torch.Tensor,
    palette: list,
    ignore_index: int = 255,
    max_samples: int = 8,
):
    """
    Save side-by-side visualizations: [image | ground truth | prediction].

    Args:
        out_dir:      directory to write PNGs into (created if missing)
        epoch:        used for file naming
        images:       [B, 3, H, W]  normalized image tensors
        gt_masks:     [B, H, W]     integer ground-truth class indices
        pred_logits:  [B, C, H, W]  raw logits from the model
        palette:      list of (R,G,B) tuples, one per class
        ignore_index: pixel value treated as unlabeled
        max_samples:  how many samples from the batch to save
    """
    os.makedirs(out_dir, exist_ok=True)

    pred_labels = pred_logits.argmax(dim=1).cpu().numpy()   # [B, H, W]
    gt_np = gt_masks.cpu().numpy()                          # [B, H, W]

    B = min(images.shape[0], max_samples)
    for i in range(B):
        img_rgb  = _denormalize(images[i])
        gt_rgb   = _mask_to_rgb(gt_np[i],   palette, ignore_index)
        pred_rgb = _mask_to_rgb(pred_labels[i], palette, ignore_index)

        combined = _hstack(img_rgb, gt_rgb, pred_rgb)
        fname = os.path.join(out_dir, f"epoch{epoch:04d}_sample{i:02d}.png")
        Image.fromarray(combined).save(fname)


def make_legend(palette: list, class_names: list = None, swatch_size: int = 20) -> Image.Image:
    """
    Build a legend image mapping color swatches to class names.

    Args:
        palette:      list of (R,G,B) tuples
        class_names:  optional list of strings; defaults to "class 0", "class 1", …
        swatch_size:  height/width of each color swatch in pixels
    Returns:
        PIL Image
    """
    from PIL import ImageDraw, ImageFont

    n = len(palette)
    if class_names is None:
        class_names = [f"class {i}" for i in range(n)]

    row_h = swatch_size + 4
    width = 200
    height = row_h * n + 4

    legend = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(legend)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i, (color, name) in enumerate(zip(palette, class_names)):
        y = 2 + i * row_h
        draw.rectangle([2, y, 2 + swatch_size, y + swatch_size], fill=color)
        draw.text((2 + swatch_size + 6, y + 2), name, fill=(0, 0, 0), font=font)

    return legend
