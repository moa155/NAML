"""COCO-style evaluation metrics for object detection.

Computes AP, AR at various IoU thresholds, matching the metrics reported
in the paper (Table 2 and Table 3).
"""

from typing import Dict, List

import numpy as np
import torch


def compute_iou_matrix(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU between two sets of boxes (xyxy format).

    Args:
        boxes1: (N, 4) tensor
        boxes2: (M, 4) tensor

    Returns:
        (N, M) IoU matrix
    """
    x1 = torch.max(boxes1[:, 0].unsqueeze(1), boxes2[:, 0].unsqueeze(0))
    y1 = torch.max(boxes1[:, 1].unsqueeze(1), boxes2[:, 1].unsqueeze(0))
    x2 = torch.min(boxes1[:, 2].unsqueeze(1), boxes2[:, 2].unsqueeze(0))
    y2 = torch.min(boxes1[:, 3].unsqueeze(1), boxes2[:, 3].unsqueeze(0))

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1.unsqueeze(1) + area2.unsqueeze(0) - inter

    return inter / (union + 1e-6)


def compute_ap_at_iou(
    predictions: List[Dict],
    targets: List[Dict],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute AP and AR at a single IoU threshold.

    Uses the standard VOC/COCO matching: each ground-truth box is matched
    to at most one prediction (highest IoU, greedy).

    Returns dict with keys: AP, precision, recall (arrays), num_gt, num_pred.
    """
    # Collect all predictions across images, sorted by score descending
    all_scores = []
    all_tp = []
    total_gt = 0

    for pred, gt in zip(predictions, targets):
        gt_boxes = gt["boxes"]
        pred_boxes = pred["boxes"]
        pred_scores = pred["scores"]

        total_gt += len(gt_boxes)

        if len(pred_boxes) == 0:
            continue

        # Sort predictions by score descending
        order = pred_scores.argsort(descending=True)
        pred_boxes = pred_boxes[order]
        pred_scores = pred_scores[order]

        matched_gt = set()

        for i in range(len(pred_boxes)):
            all_scores.append(pred_scores[i].item())

            if len(gt_boxes) == 0:
                all_tp.append(0)
                continue

            ious = compute_iou_matrix(pred_boxes[i:i+1], gt_boxes)[0]
            best_iou, best_idx = ious.max(0) if ious.numel() > 0 else (torch.tensor(0.0), torch.tensor(-1))

            # Handle single-element tensors
            if ious.dim() == 0:
                best_iou_val = ious.item()
                best_idx_val = 0
            else:
                best_iou_val = best_iou.item()
                best_idx_val = best_idx.item()

            if best_iou_val >= iou_threshold and best_idx_val not in matched_gt:
                all_tp.append(1)
                matched_gt.add(best_idx_val)
            else:
                all_tp.append(0)

    if total_gt == 0:
        return {"AP": 0.0, "num_gt": 0, "num_pred": len(all_scores)}

    # Sort all predictions globally by score
    indices = np.argsort(-np.array(all_scores))
    tp = np.array(all_tp)[indices]
    fp = 1 - tp

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    recalls = tp_cumsum / total_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    # Compute AP using 101-point interpolation (COCO style)
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        prec_at_recall = precisions[recalls >= t]
        if len(prec_at_recall) > 0:
            ap += prec_at_recall.max()
    ap /= 101

    return {
        "AP": float(ap),
        "precisions": precisions.tolist(),
        "recalls": recalls.tolist(),
        "num_gt": total_gt,
        "num_pred": len(all_scores),
    }


def compute_ar(
    predictions: List[Dict],
    targets: List[Dict],
    iou_threshold: float = 0.5,
    max_dets: int = 100,
) -> float:
    """Compute Average Recall at a given IoU with a max detection limit."""
    total_recalled = 0
    total_gt = 0

    for pred, gt in zip(predictions, targets):
        gt_boxes = gt["boxes"]
        pred_boxes = pred["boxes"]
        pred_scores = pred["scores"]

        total_gt += len(gt_boxes)
        if len(gt_boxes) == 0 or len(pred_boxes) == 0:
            continue

        # Keep top-K by score
        order = pred_scores.argsort(descending=True)[:max_dets]
        pred_boxes = pred_boxes[order]

        ious = compute_iou_matrix(pred_boxes, gt_boxes)
        matched_gt = set()

        for i in range(len(pred_boxes)):
            if len(gt_boxes) == 0:
                break
            row_ious = ious[i]
            best_idx = row_ious.argmax().item()
            if row_ious[best_idx].item() >= iou_threshold and best_idx not in matched_gt:
                matched_gt.add(best_idx)
                total_recalled += 1

    return total_recalled / max(total_gt, 1)


def _box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def compute_metrics(
    predictions: List[Dict],
    targets: List[Dict],
) -> Dict[str, float]:
    """Compute the full set of detection metrics reported in the paper.

    Returns dict with:
        AP@0.5, AP@0.5:0.95, AP_M, AP_L, AR@10, AR_M, AR_L,
        patient_accuracy, patient_precision, patient_recall, patient_f1
    """
    # --- Object detection metrics ---
    result_50 = compute_ap_at_iou(predictions, targets, iou_threshold=0.5)
    ap50 = result_50["AP"]

    # AP@[0.5:0.95] (COCO standard)
    ap_sum = 0.0
    for iou_t in np.arange(0.5, 1.0, 0.05):
        ap_sum += compute_ap_at_iou(predictions, targets, iou_threshold=iou_t)["AP"]
    ap_5095 = ap_sum / 10

    # Size-based AP (medium: area 32^2 to 96^2, large: area > 96^2)
    # Following COCO: medium = 32^2..96^2, large > 96^2
    MEDIUM_AREA = (32 ** 2, 96 ** 2)
    LARGE_AREA = 96 ** 2

    pred_m, tgt_m = [], []
    pred_l, tgt_l = [], []

    for pred, gt in zip(predictions, targets):
        areas = gt["area"] if len(gt["boxes"]) > 0 else torch.zeros(0)

        m_mask = (areas >= MEDIUM_AREA[0]) & (areas < MEDIUM_AREA[1])
        l_mask = areas >= LARGE_AREA

        # Medium targets
        gt_m = {
            "boxes": gt["boxes"][m_mask] if len(gt["boxes"]) > 0 else gt["boxes"],
            "labels": gt["labels"][m_mask] if len(gt["labels"]) > 0 else gt["labels"],
            "area": areas[m_mask] if len(areas) > 0 else areas,
            "iscrowd": gt["iscrowd"][m_mask] if len(gt["iscrowd"]) > 0 else gt["iscrowd"],
        }
        tgt_m.append(gt_m)
        pred_m.append(pred)

        # Large targets
        gt_l = {
            "boxes": gt["boxes"][l_mask] if len(gt["boxes"]) > 0 else gt["boxes"],
            "labels": gt["labels"][l_mask] if len(gt["labels"]) > 0 else gt["labels"],
            "area": areas[l_mask] if len(areas) > 0 else areas,
            "iscrowd": gt["iscrowd"][l_mask] if len(gt["iscrowd"]) > 0 else gt["iscrowd"],
        }
        tgt_l.append(gt_l)
        pred_l.append(pred)

    ap_m = compute_ap_at_iou(pred_m, tgt_m, iou_threshold=0.5)["AP"]
    ap_l = compute_ap_at_iou(pred_l, tgt_l, iou_threshold=0.5)["AP"]

    # Recall metrics
    ar_10 = compute_ar(predictions, targets, iou_threshold=0.5, max_dets=10)
    ar_m = compute_ar(pred_m, tgt_m, iou_threshold=0.5, max_dets=100)
    ar_l = compute_ar(pred_l, tgt_l, iou_threshold=0.5, max_dets=100)

    # --- Patient-level classification ---
    # A patient is predicted positive if the model outputs at least one detection
    tp, fp, tn, fn = 0, 0, 0, 0
    for pred, gt in zip(predictions, targets):
        has_gt = len(gt["boxes"]) > 0
        has_pred = len(pred["boxes"]) > 0 and (pred["scores"] > 0.5).any()

        if has_gt and has_pred:
            tp += 1
        elif has_gt and not has_pred:
            fn += 1
        elif not has_gt and has_pred:
            fp += 1
        else:
            tn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "AP@0.5": ap50,
        "AP@0.5:0.95": ap_5095,
        "AP_M": ap_m,
        "AP_L": ap_l,
        "AR@10": ar_10,
        "AR_M": ar_m,
        "AR_L": ar_l,
        "patient_accuracy": accuracy,
        "patient_precision": precision,
        "patient_recall": recall,
        "patient_f1": f1,
        "precisions": result_50.get("precisions", []),
        "recalls": result_50.get("recalls", []),
    }
