#!/usr/bin/env python3
"""Pneumonia Detection: Anchor-Free vs. Anchor-Based Object Detection.

Reproduces and extends the method from:
  Wu et al., "Pneumonia detection based on RSNA dataset and anchor-free
  deep learning detector", Scientific Reports (2024).

Three detection models are compared:
  1. FCOS  — anchor-free detector (paper's proposed method)
  2. RetinaNet — one-stage, anchor-based (comparison from paper Table 3)
  3. Faster R-CNN — two-stage, anchor-based (comparison from paper Table 3)

All models use a ResNet-50 backbone with Feature Pyramid Network (FPN).

Usage:
  python main.py --mode train --model all
  python main.py --mode evaluate --model all
  python main.py --mode compare
  python main.py --mode full          # train + evaluate + compare (all models)
  python main.py --mode visualize     # detection visualization on samples
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.dataset import RSNAPneumoniaDataset, collate_fn, load_rsna_dataframes
from src.transforms import get_train_transforms, get_val_transforms
from src.models import build_model
from src.engine import train_model, evaluate
from src.evaluate import compute_metrics
from src.visualize import generate_all_plots


MODELS = ["fcos", "retinanet", "faster_rcnn"]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_data_loaders(config: Config):
    """Create train/val data loaders from the RSNA dataset."""
    df = load_rsna_dataframes(
        str(config.labels_path),
        str(config.detail_labels_path),
    )

    # Unique patient IDs for splitting
    patient_ids = np.array(df["patientId"].unique())
    np.random.seed(config.seed)
    np.random.shuffle(patient_ids)

    # Optionally limit number of patients for faster training
    if config.max_samples is not None and config.max_samples < len(patient_ids):
        patient_ids = patient_ids[:config.max_samples]
        print(f"Using subset: {len(patient_ids)} patients (--max-samples={config.max_samples})")

    split_idx = int(len(patient_ids) * (1 - config.val_split))
    train_ids = set(patient_ids[:split_idx])
    val_ids = set(patient_ids[split_idx:])

    train_df = df[df["patientId"].isin(train_ids)].reset_index(drop=True)
    val_df = df[df["patientId"].isin(val_ids)].reset_index(drop=True)

    print(f"Dataset split: {len(train_ids)} train / {len(val_ids)} val patients")
    print(f"  Train annotations: {len(train_df)} rows ({train_df['Target'].sum()} positive)")
    print(f"  Val   annotations: {len(val_df)} rows ({val_df['Target'].sum()} positive)")

    train_dataset = RSNAPneumoniaDataset(
        image_dir=str(config.images_path),
        annotations_df=train_df,
        transforms=get_train_transforms(config.use_augmentation),
    )
    val_dataset = RSNAPneumoniaDataset(
        image_dir=str(config.images_path),
        annotations_df=val_df,
        transforms=get_val_transforms(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=config.device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=config.device.type == "cuda",
    )

    return train_loader, val_loader, val_dataset


def run_train(model_names: list, config: Config):
    """Train the specified models."""
    train_loader, val_loader, _ = build_data_loaders(config)
    histories = {}

    for name in model_names:
        print(f"\n{'='*60}")
        print(f"  Training: {name.upper()}")
        print(f"{'='*60}")

        model = build_model(name, num_classes=config.num_classes,
                            pretrained_backbone=config.pretrained_backbone,
                            min_size=config.image_min_size, max_size=config.image_max_size)
        history = train_model(model, train_loader, val_loader, config, name)
        histories[name] = history

    return histories


def run_evaluate(model_names: list, config: Config):
    """Evaluate trained models and return metrics."""
    _, val_loader, _ = build_data_loaders(config)
    all_metrics = {}
    all_predictions = {}
    all_targets = None

    for name in model_names:
        ckpt_path = Path(config.checkpoint_dir) / f"{name}_best.pth"
        if not ckpt_path.exists():
            ckpt_path = Path(config.checkpoint_dir) / f"{name}_final.pth"
        if not ckpt_path.exists():
            print(f"WARNING: No checkpoint found for {name}, skipping evaluation.")
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

        print(f"  AP@0.5:       {metrics['AP@0.5']*100:.1f}")
        print(f"  AP@[.5:.95]:  {metrics['AP@0.5:0.95']*100:.1f}")
        print(f"  AP_M:         {metrics['AP_M']*100:.1f}")
        print(f"  AP_L:         {metrics['AP_L']*100:.1f}")
        print(f"  AR@10:        {metrics['AR@10']*100:.1f}")
        print(f"  Patient Acc:  {metrics['patient_accuracy']*100:.1f}")
        print(f"  Patient F1:   {metrics['patient_f1']*100:.1f}")

    return all_metrics, all_predictions, all_targets


def run_compare(config: Config):
    """Load saved histories and metrics, generate comparison plots."""
    out_dir = Path(config.output_dir)

    # Load histories
    histories = {}
    for name in MODELS:
        hist_path = out_dir / f"{name}_history.json"
        if hist_path.exists():
            with open(hist_path) as f:
                histories[name] = json.load(f)

    # Load or recompute metrics
    metrics_path = out_dir / "all_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            all_metrics = json.load(f)
    else:
        print("No saved metrics found. Running evaluation first...")
        all_metrics, _, _ = run_evaluate(list(histories.keys()), config)

    if not histories and not all_metrics:
        print("ERROR: No training histories or metrics found. Train models first.")
        return

    generate_all_plots(
        histories=histories,
        all_metrics=all_metrics,
        output_dir=config.output_dir,
    )


def run_visualize(config: Config, num_samples: int = 6):
    """Generate detection visualizations on sample images."""
    _, val_loader, val_dataset = build_data_loaders(config)

    # Collect sample images and targets
    sample_images = []
    sample_targets = []
    for i in range(min(num_samples, len(val_dataset))):
        img, tgt = val_dataset[i]
        sample_images.append(img)
        sample_targets.append(tgt)

    # Load predictions for each model
    predictions_by_model = {}
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
        predictions_by_model[name] = preds

    from src.visualize import plot_detection_samples
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_detection_samples(sample_images, sample_targets, predictions_by_model, out_dir)


def run_full(config: Config):
    """Run the complete pipeline: train, evaluate, compare, visualize."""
    print("=" * 60)
    print("  PNEUMONIA DETECTION — FULL PIPELINE")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Models: {MODELS}")
    print()

    # Train all models
    histories = run_train(MODELS, config)

    # Evaluate all models
    all_metrics, all_predictions, all_targets = run_evaluate(MODELS, config)

    # Collect sample images for visualization
    _, _, val_dataset = build_data_loaders(config)
    sample_images = []
    sample_targets = []
    for i in range(min(6, len(val_dataset))):
        img, tgt = val_dataset[i]
        sample_images.append(img)
        sample_targets.append(tgt)

    # Generate all plots
    generate_all_plots(
        histories=histories,
        all_metrics=all_metrics,
        output_dir=config.output_dir,
        predictions_by_model=all_predictions,
        targets=all_targets,
        images=sample_images,
    )

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {config.output_dir}/")
    print(f"Checkpoints saved to: {config.checkpoint_dir}/")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pneumonia Detection: Anchor-Free vs. Anchor-Based",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "compare", "visualize", "full"],
        default="full",
        help="Execution mode (default: full)",
    )
    parser.add_argument(
        "--model",
        choices=MODELS + ["all"],
        default="all",
        help="Model to train/evaluate (default: all)",
    )
    parser.add_argument("--data-dir", default="data", help="Path to RSNA dataset")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=4, help="Data loader workers")
    parser.add_argument("--image-size", type=int, default=512, help="Image size for model input")
    parser.add_argument("--no-augmentation", action="store_true", help="Disable data augmentation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of patients for faster training (default: all)")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda", "mps"],
                        help="Force device (default: auto-detect)")

    return parser.parse_args()


def main():
    args = parse_args()

    config = Config(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_workers=args.num_workers,
        image_min_size=args.image_size,
        image_max_size=args.image_size,
        use_augmentation=not args.no_augmentation,
        seed=args.seed,
        max_samples=args.max_samples,
        force_device=args.device,
    )

    set_seed(config.seed)

    # Validate data directory
    if args.mode in ("train", "evaluate", "full", "visualize"):
        if not config.images_path.exists():
            print(f"ERROR: Dataset not found at {config.images_path}")
            print(f"Download the RSNA Pneumonia Detection Challenge dataset from:")
            print(f"  https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data")
            print(f"Extract it into: {config.data_dir}/")
            print(f"Expected structure:")
            print(f"  {config.data_dir}/stage_2_train_labels.csv")
            print(f"  {config.data_dir}/stage_2_train_images/{{patientId}}.dcm")
            sys.exit(1)

    model_names = MODELS if args.model == "all" else [args.model]

    if args.mode == "train":
        run_train(model_names, config)
    elif args.mode == "evaluate":
        run_evaluate(model_names, config)
    elif args.mode == "compare":
        run_compare(config)
    elif args.mode == "visualize":
        run_visualize(config)
    elif args.mode == "full":
        run_full(config)


if __name__ == "__main__":
    main()
