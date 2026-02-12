"""RetinaNet: One-stage anchor-based detector (comparison method).

RetinaNet (Lin et al., 2017) is included as a comparison method from the
paper's Table 3.  It uses:
  - ResNet-50 + FPN backbone
  - Anchor-based detection with focal loss
  - Separate classification and regression sub-networks
"""

from torchvision.models.detection import retinanet_resnet50_fpn_v2


def build_retinanet(num_classes: int = 2, pretrained_backbone: bool = True,
                    min_size: int = 512, max_size: int = 512):
    """Build RetinaNet with ResNet-50 FPN v2 backbone."""
    if pretrained_backbone:
        model = retinanet_resnet50_fpn_v2(
            weights=None,
            weights_backbone="IMAGENET1K_V1",
            num_classes=num_classes,
            min_size=min_size,
            max_size=max_size,
        )
    else:
        model = retinanet_resnet50_fpn_v2(
            weights=None,
            weights_backbone=None,
            num_classes=num_classes,
            min_size=min_size,
            max_size=max_size,
        )

    model.score_thresh = 0.1
    model.nms_thresh = 0.5
    return model
