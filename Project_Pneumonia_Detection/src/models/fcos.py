"""FCOS: Anchor-free detector — the paper's proposed method.

This implements the anchor-free detection framework from:
  Wu et al., "Pneumonia detection based on RSNA dataset and anchor-free
  deep learning detector", Scientific Reports (2024).

Architecture:
  - ResNet-50 backbone
  - Feature Pyramid Network (FPN) with 5 levels (strides 8–128)
  - Two-branch detection head (center classification + scale regression)
  - Focal loss for class imbalance
"""

from torchvision.models.detection import fcos_resnet50_fpn


def build_fcos(num_classes: int = 2, pretrained_backbone: bool = True,
               min_size: int = 512, max_size: int = 512):
    """Build FCOS model with ResNet-50 FPN backbone."""
    if pretrained_backbone:
        model = fcos_resnet50_fpn(
            weights=None,
            weights_backbone="IMAGENET1K_V1",
            num_classes=num_classes,
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

    model.score_thresh = 0.1
    model.nms_thresh = 0.5
    return model
