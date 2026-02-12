"""Faster R-CNN: Two-stage anchor-based detector (comparison method).

Faster R-CNN (Ren et al., 2015) is included as a comparison method from the
paper's Table 3.  It uses:
  - ResNet-50 + FPN backbone
  - Region Proposal Network (RPN) for candidate generation
  - ROI pooling + classification/regression heads
"""

from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2


def build_faster_rcnn(num_classes: int = 2, pretrained_backbone: bool = True,
                      min_size: int = 512, max_size: int = 512):
    """Build Faster R-CNN with ResNet-50 FPN v2 backbone."""
    if pretrained_backbone:
        model = fasterrcnn_resnet50_fpn_v2(
            weights=None,
            weights_backbone="IMAGENET1K_V1",
            num_classes=num_classes,
            min_size=min_size,
            max_size=max_size,
        )
    else:
        model = fasterrcnn_resnet50_fpn_v2(
            weights=None,
            weights_backbone=None,
            num_classes=num_classes,
            min_size=min_size,
            max_size=max_size,
        )

    model.roi_heads.score_thresh = 0.1
    model.roi_heads.nms_thresh = 0.5
    return model
