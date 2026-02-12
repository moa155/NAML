"""Detection transforms implementing the paper's data augmentation strategy.

The paper uses:
  1. Horizontal and vertical flips (applied randomly during training)
  2. Luminance augmentation (random brightness changes)
  3. Random cropping

All transforms preserve bounding-box consistency.
"""

import random
from typing import Dict, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image


# ---------------------------------------------------------------------------
# Composable transform pair (image, target) -> (image, target)
# ---------------------------------------------------------------------------

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    """Convert numpy HWC image to CHW float tensor."""

    def __call__(self, image, target):
        if isinstance(image, np.ndarray):
            image = Image.fromarray((image * 255).astype(np.uint8))
        image = F.to_tensor(image)  # CHW, float32 in [0,1]
        return image, target


class RandomHorizontalFlip:
    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            if isinstance(image, torch.Tensor):
                _, _, width = image.shape
                image = image.flip(-1)
            else:
                height, width = image.shape[:2]
                image = np.flip(image, axis=1).copy()

            boxes = target["boxes"]
            if len(boxes) > 0:
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                target["boxes"] = boxes
        return image, target


class RandomVerticalFlip:
    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            if isinstance(image, torch.Tensor):
                _, height, _ = image.shape
                image = image.flip(-2)
            else:
                height = image.shape[0]
                image = np.flip(image, axis=0).copy()

            boxes = target["boxes"]
            if len(boxes) > 0:
                boxes[:, [1, 3]] = height - boxes[:, [3, 1]]
                target["boxes"] = boxes
        return image, target


class RandomBrightness:
    """Luminance augmentation as described in the paper."""

    def __init__(self, delta: float = 0.2):
        self.delta = delta

    def __call__(self, image, target):
        if isinstance(image, np.ndarray):
            factor = 1.0 + random.uniform(-self.delta, self.delta)
            image = np.clip(image * factor, 0.0, 1.0)
        elif isinstance(image, torch.Tensor):
            factor = 1.0 + random.uniform(-self.delta, self.delta)
            image = torch.clamp(image * factor, 0.0, 1.0)
        return image, target


class RandomCrop:
    """Random crop that preserves bounding boxes (keeps boxes with sufficient overlap)."""

    def __init__(self, min_scale: float = 0.8, max_scale: float = 1.0):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, image, target):
        if random.random() < 0.5:
            return image, target  # apply 50% of the time

        if isinstance(image, torch.Tensor):
            _, h, w = image.shape
        else:
            h, w = image.shape[:2]

        scale = random.uniform(self.min_scale, self.max_scale)
        new_h, new_w = int(h * scale), int(w * scale)
        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)

        # Crop image
        if isinstance(image, torch.Tensor):
            image = image[:, top : top + new_h, left : left + new_w]
        else:
            image = image[top : top + new_h, left : left + new_w].copy()

        boxes = target["boxes"]
        if len(boxes) > 0:
            # Shift boxes
            boxes[:, 0] -= left
            boxes[:, 1] -= top
            boxes[:, 2] -= left
            boxes[:, 3] -= top

            # Clip to crop bounds
            boxes[:, 0] = boxes[:, 0].clamp(min=0, max=new_w)
            boxes[:, 1] = boxes[:, 1].clamp(min=0, max=new_h)
            boxes[:, 2] = boxes[:, 2].clamp(min=0, max=new_w)
            boxes[:, 3] = boxes[:, 3].clamp(min=0, max=new_h)

            # Remove degenerate boxes
            keep = (boxes[:, 2] - boxes[:, 0] > 2) & (boxes[:, 3] - boxes[:, 1] > 2)
            target["boxes"] = boxes[keep]
            target["labels"] = target["labels"][keep]
            target["area"] = target["area"][keep] if "area" in target else None
            target["iscrowd"] = target["iscrowd"][keep]

        return image, target


# ---------------------------------------------------------------------------
# Pre-built transform pipelines
# ---------------------------------------------------------------------------

def get_train_transforms(use_augmentation: bool = True) -> Compose:
    """Training transforms following the paper's augmentation strategy."""
    transforms = [ToTensor()]
    if use_augmentation:
        transforms += [
            RandomHorizontalFlip(prob=0.5),
            RandomVerticalFlip(prob=0.5),
            RandomBrightness(delta=0.2),
            RandomCrop(min_scale=0.8, max_scale=1.0),
        ]
    return Compose(transforms)


def get_val_transforms() -> Compose:
    """Validation/test transforms (no augmentation)."""
    return Compose([ToTensor()])
