from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


@dataclass
class TransformConfig:
    img_size: int = 512
    grayscale_to_rgb: bool = True
    normalize: str = "imagenet"  # "imagenet" | "none"
    augment_enabled: bool = True
    rotate_deg: float = 5
    brightness: float = 0.05
    contrast: float = 0.05
    hflip: bool = False


def _to_rgb_if_needed(img: Image.Image, grayscale_to_rgb: bool) -> Image.Image:
    # images loaded as "L" (1-channel). For torchvision pretrained backbones, 3-channel is typical.
    if grayscale_to_rgb:
        return img.convert("RGB")
    return img  # keep "L"


def build_train_transform(cfg: TransformConfig) -> T.Compose:
    ops = []
    if cfg.grayscale_to_rgb:
        ops.append(T.Grayscale(num_output_channels=3))
    else:
        ops.append(T.Grayscale(num_output_channels=1))

    # Resize to square deterministically (simple baseline). You can later switch to pad+resize.
    ops.append(T.Resize((cfg.img_size, cfg.img_size), interpolation=T.InterpolationMode.BILINEAR))

    if cfg.augment_enabled:
        # Light augmentations appropriate for hand radiographs
        ops.append(T.RandomApply([T.RandomRotation(degrees=cfg.rotate_deg)], p=0.5))
        ops.append(T.ColorJitter(brightness=cfg.brightness, contrast=cfg.contrast))
        if cfg.hflip:
            ops.append(T.RandomHorizontalFlip(p=0.5))

    ops.append(T.ToTensor())

    if cfg.normalize.lower() == "imagenet":
        ops.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))

    return T.Compose(ops)


def build_eval_transform(cfg: TransformConfig) -> T.Compose:
    ops = []
    
    if cfg.grayscale_to_rgb:
        ops.append(T.Grayscale(num_output_channels=3))
    else:
        ops.append(T.Grayscale(num_output_channels=1))

    ops.append(T.Resize((cfg.img_size, cfg.img_size), interpolation=T.InterpolationMode.BILINEAR))
    ops.append(T.ToTensor())

    if cfg.normalize.lower() == "imagenet":
        ops.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))

    return T.Compose(ops)
