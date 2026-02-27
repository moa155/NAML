"""FCOS: Anchor-free detector — the paper's proposed method.

This implements the anchor-free detection framework from:
  Wu et al., "Pneumonia detection based on RSNA dataset and anchor-free
  deep learning detector", Scientific Reports (2024).

Architecture:
  - ResNet-50 backbone
  - Feature Pyramid Network (FPN) with 5 levels (strides 8–128)
  - Two-branch detection head (center classification + scale regression)
  - Focal loss for class imbalance
  - GroupNorm in head for training stability (matching v2 detector quality)

Fine-tuned from COCO-pretrained backbone + FPN for faster convergence.
"""

from functools import partial

import torch.nn as nn
from torchvision.models.detection import fcos_resnet50_fpn
from torchvision.models.detection.fcos import FCOSClassificationHead, FCOSRegressionHead


def build_fcos(num_classes: int = 2, pretrained_backbone: bool = True,
               min_size: int = 512, max_size: int = 512):
    """Build FCOS model with ResNet-50 FPN backbone.

    When pretrained_backbone=True, loads COCO-pretrained weights to get
    a pretrained backbone + FPN. The detection heads are then replaced
    with GroupNorm versions (torchvision has no fcos_resnet50_fpn_v2).
    """
    if pretrained_backbone:
        # Load COCO-pretrained model (gets pretrained backbone + FPN)
        model = fcos_resnet50_fpn(
            weights="DEFAULT",
            min_size=min_size,
            max_size=max_size,
        )
    else:
        model = fcos_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            num_classes=num_classes,
            min_size=min_size,
            max_size=max_size,
        )

    # Replace heads with GroupNorm versions for our number of classes.
    # The v2 RetinaNet/Faster R-CNN baselines use GroupNorm in their heads,
    # which is critical for stable training with small batch sizes.
    in_channels = model.backbone.out_channels  # 256 for FPN
    num_anchors = model.anchor_generator.num_anchors_per_location()[0]  # 1 for FCOS
    norm_layer = partial(nn.GroupNorm, 32)
    model.head.classification_head = FCOSClassificationHead(
        in_channels, num_anchors, num_classes, norm_layer=norm_layer,
    )
    model.head.regression_head = FCOSRegressionHead(
        in_channels, num_anchors, norm_layer=norm_layer,
    )

    model.score_thresh = 0.05
    model.nms_thresh = 0.5
    return model
