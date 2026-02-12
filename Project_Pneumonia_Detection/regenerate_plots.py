#!/usr/bin/env python3
"""One-time script: evaluate all models and regenerate all plots with full data (including PR curves)."""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.dataset import RSNAPneumoniaDataset, collate_fn, load_rsna_dataframes
from src.transforms import get_val_transforms
from src.models import build_model
from src.engine import evaluate
from src.evaluate import compute_metrics
from src.visualize import generate_all_plots, plot_detection_samples

MODELS = ["fcos", "retinanet", "faster_rcnn"]

def main():
    config = Config(
        data_dir="data",
        output_dir="results",
        checkpoint_dir="checkpoints",
        num_workers=0,
        max_samples=500,
        force_device="cpu",
        seed=42,
    )

    # Seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Build val loader
    df = load_rsna_dataframes(str(config.labels_path), str(config.detail_labels_path))
    patient_ids = np.array(df["patientId"].unique())
    np.random.seed(config.seed)
    np.random.shuffle(patient_ids)
    patient_ids = patient_ids[:config.max_samples]
    split_idx = int(len(patient_ids) * (1 - config.val_split))
    val_ids = set(patient_ids[split_idx:])
    val_df = df[df["patientId"].isin(val_ids)].reset_index(drop=True)

    val_dataset = RSNAPneumoniaDataset(
        image_dir=str(config.images_path),
        annotations_df=val_df,
        transforms=get_val_transforms(),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, collate_fn=collate_fn,
    )

    print(f"Val set: {len(val_df)} annotations from {len(val_ids)} patients")

    # Evaluate all models
    all_metrics = {}
    all_predictions = {}
    all_targets = None

    for name in MODELS:
        ckpt_path = Path(config.checkpoint_dir) / f"{name}_best.pth"
        if not ckpt_path.exists():
            ckpt_path = Path(config.checkpoint_dir) / f"{name}_final.pth"
        if not ckpt_path.exists():
            print(f"  Skipping {name} (no checkpoint)")
            continue

        print(f"\nEvaluating: {name.upper()}")
        model = build_model(name, num_classes=config.num_classes, pretrained_backbone=False,
                            min_size=config.image_min_size, max_size=config.image_max_size)
        ckpt = torch.load(ckpt_path, map_location=config.device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(config.device)

        predictions, targets = evaluate(model, val_loader, config.device)
        metrics = compute_metrics(predictions, targets)
        all_metrics[name] = metrics
        all_predictions[name] = predictions
        if all_targets is None:
            all_targets = targets

        print(f"  AP@0.5: {metrics['AP@0.5']*100:.1f}%  AR@10: {metrics['AR@10']*100:.1f}%")
        print(f"  PR data: {len(metrics.get('precisions', []))} points")

    # Load histories
    out_dir = Path(config.output_dir)
    histories = {}
    for name in MODELS:
        hist_path = out_dir / f"{name}_history.json"
        if hist_path.exists():
            with open(hist_path) as f:
                histories[name] = json.load(f)

    # Collect sample images for detection visualization
    sample_images = []
    sample_targets = []
    for i in range(min(4, len(val_dataset))):
        img, tgt = val_dataset[i]
        sample_images.append(img)
        sample_targets.append(tgt)

    # Get predictions for sample images
    sample_preds_by_model = {}
    for name in MODELS:
        ckpt_path = Path(config.checkpoint_dir) / f"{name}_best.pth"
        if not ckpt_path.exists():
            ckpt_path = Path(config.checkpoint_dir) / f"{name}_final.pth"
        if not ckpt_path.exists():
            continue

        model = build_model(name, num_classes=config.num_classes, pretrained_backbone=False,
                            min_size=config.image_min_size, max_size=config.image_max_size)
        ckpt = torch.load(ckpt_path, map_location=config.device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(config.device)
        model.eval()

        preds = []
        with torch.no_grad():
            for img in sample_images:
                out = model([img.to(config.device)])[0]
                preds.append({k: v.cpu() for k, v in out.items()})
        sample_preds_by_model[name] = preds

    # Generate ALL plots (with full metrics including PR data)
    generate_all_plots(
        histories=histories,
        all_metrics=all_metrics,
        output_dir=config.output_dir,
        predictions_by_model=all_predictions,
        targets=all_targets,
        images=sample_images,
    )

    # Also generate detection samples
    plot_detection_samples(sample_images, sample_targets, sample_preds_by_model, out_dir)

    print("\nAll plots regenerated with full data!")


if __name__ == "__main__":
    main()
