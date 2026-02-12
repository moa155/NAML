"""Training and evaluation engine for detection models."""

import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch, returning average losses."""
    model.train()
    running_losses = {}
    num_batches = 0

    pbar = tqdm(data_loader, desc=f"Epoch {epoch}", file=sys.stdout)
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Skip batches with no valid targets (can cause errors)
        valid = True
        for t in targets:
            if t["boxes"].numel() > 0 and (
                (t["boxes"][:, 2] <= t["boxes"][:, 0]).any()
                or (t["boxes"][:, 3] <= t["boxes"][:, 1]).any()
            ):
                valid = False
                break
        if not valid:
            continue

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        if not math.isfinite(losses.item()):
            print(f"WARNING: non-finite loss {losses.item()}, skipping batch")
            continue

        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Accumulate losses
        for k, v in loss_dict.items():
            running_losses[k] = running_losses.get(k, 0.0) + v.item()
        running_losses["total_loss"] = running_losses.get("total_loss", 0.0) + losses.item()
        num_batches += 1

        pbar.set_postfix(loss=f"{losses.item():.4f}")

    # Average
    avg_losses = {k: v / max(num_batches, 1) for k, v in running_losses.items()}
    return avg_losses


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Tuple[List[Dict], List[Dict]]:
    """Run inference and collect predictions + ground truth in COCO-compatible format.

    Returns:
        all_predictions: list of dicts with keys
            image_id, boxes (xyxy), scores, labels
        all_targets: list of dicts with keys
            image_id, boxes (xyxy), labels, area, iscrowd
    """
    model.eval()
    all_predictions = []
    all_targets = []

    for images, targets in tqdm(data_loader, desc="Evaluating", file=sys.stdout):
        images = [img.to(device) for img in images]
        outputs = model(images)

        for output, target in zip(outputs, targets):
            img_id = target["image_id"].item()

            pred = {
                "image_id": img_id,
                "boxes": output["boxes"].cpu(),
                "scores": output["scores"].cpu(),
                "labels": output["labels"].cpu(),
            }
            all_predictions.append(pred)

            gt = {
                "image_id": img_id,
                "boxes": target["boxes"],
                "labels": target["labels"],
                "area": target["area"],
                "iscrowd": target["iscrowd"],
            }
            all_targets.append(gt)

    return all_predictions, all_targets


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config,
    model_name: str,
) -> Dict:
    """Full training loop with validation.

    Returns a history dict with per-epoch losses and metrics.
    """
    device = config.device
    model.to(device)

    # Optimizer: Adam (as in the paper)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=config.learning_rate, weight_decay=config.weight_decay)

    # LR scheduler: step decay (paper: reduce by 0.1x at milestones)
    # Scale milestones if using fewer epochs than default
    milestones = list(config.lr_milestones)
    if config.num_epochs < max(milestones):
        # Scale milestones proportionally to num_epochs
        ratio = config.num_epochs / 20.0
        milestones = sorted(set(max(1, int(m * ratio)) for m in milestones))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=config.lr_gamma
    )

    history = {
        "train_losses": [],
        "val_metrics": [],
        "epoch_times": [],
        "learning_rates": [],
    }

    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ap = -1.0

    for epoch in range(1, config.num_epochs + 1):
        t0 = time.time()

        # --- Train ---
        avg_losses = train_one_epoch(model, optimizer, train_loader, device, epoch)
        history["train_losses"].append(avg_losses)
        history["learning_rates"].append(optimizer.param_groups[0]["lr"])

        # --- Validate ---
        predictions, targets = evaluate(model, val_loader, device)
        from src.evaluate import compute_metrics
        metrics = compute_metrics(predictions, targets)
        history["val_metrics"].append(metrics)

        scheduler.step()
        elapsed = time.time() - t0
        history["epoch_times"].append(elapsed)

        ap50 = metrics.get("AP@0.5", 0.0)
        print(
            f"[{model_name}] Epoch {epoch}/{config.num_epochs}  "
            f"loss={avg_losses['total_loss']:.4f}  "
            f"AP@0.5={ap50:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.6f}  "
            f"time={elapsed:.1f}s"
        )

        # Save best checkpoint
        if ap50 > best_ap:
            best_ap = ap50
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "ap50": ap50},
                ckpt_dir / f"{model_name}_best.pth",
            )

    # Save final checkpoint
    torch.save(
        {"epoch": config.num_epochs, "model_state_dict": model.state_dict()},
        ckpt_dir / f"{model_name}_final.pth",
    )

    # Save history
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Convert tensors for JSON serialization
    serializable = _make_serializable(history)
    with open(out_dir / f"{model_name}_history.json", "w") as f:
        json.dump(serializable, f, indent=2)

    return history


def _make_serializable(obj):
    """Recursively convert torch tensors and numpy arrays for JSON."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (torch.Tensor, np.ndarray)):
        return obj.tolist() if hasattr(obj, 'tolist') else float(obj)
    elif isinstance(obj, (np.float32, np.float64, np.int64, np.int32)):
        return float(obj)
    return obj
