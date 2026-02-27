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
import os
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.config import Config
from src.dataset import RSNAPneumoniaDataset, collate_fn, load_rsna_dataframes
from src.transforms import get_train_transforms, get_val_transforms
from src.models import build_model
from src.engine import train_model, evaluate_model
from src.evaluate import compute_metrics
from src.visualize import generate_all_plots


MODELS = ["fcos", "retinanet", "faster_rcnn"]


# ── GPU discovery ──────────────────────────────────────────────────────

def discover_gpus() -> list:
    """Return list of available CUDA device indices."""
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


def print_gpu_info(gpu_ids: list):
    """Print info about all available GPUs."""
    if not gpu_ids:
        device = "MPS" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "CPU"
        print(f"No CUDA GPUs found. Using {device}.")
        return
    print(f"Found {len(gpu_ids)} GPU(s):")
    for idx in gpu_ids:
        props = torch.cuda.get_device_properties(idx)
        mem_gb = props.total_memory / 1e9
        print(f"  GPU {idx}: {props.name} ({mem_gb:.1f} GB)")


# ── Seed ───────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True   # auto-tune convolution algorithms
        torch.backends.cuda.matmul.allow_tf32 = True   # TF32 matmul (Turing+)
        torch.backends.cudnn.allow_tf32 = True          # TF32 convolutions (Turing+)
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)


# ── Data ───────────────────────────────────────────────────────────────

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

    # Optionally limit number of patients
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

    n_workers = config.effective_num_workers
    pf = getattr(config, "prefetch_factor", 2)

    # WeightedRandomSampler for class imbalance
    use_weighted = getattr(config, "use_weighted_sampler", False)
    sampler = None
    if use_weighted:
        positive_mask = train_dataset.get_positive_mask()
        n_pos = sum(positive_mask)
        n_neg = len(positive_mask) - n_pos
        weight = getattr(config, "positive_sample_weight", 3.0)
        weights = [weight if is_pos else 1.0 for is_pos in positive_mask]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        print(f"  WeightedRandomSampler: {n_pos} positive ({weight}x), {n_neg} negative (1x)")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=(sampler is None),  # shuffle only if no sampler
        sampler=sampler,
        num_workers=n_workers,
        collate_fn=collate_fn,
        pin_memory=config.pin_memory,
        persistent_workers=n_workers > 0,
        prefetch_factor=pf if n_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size * 4,  # 4x batch for eval (no gradients stored)
        shuffle=False,
        num_workers=n_workers,
        collate_fn=collate_fn,
        pin_memory=config.pin_memory,
        persistent_workers=n_workers > 0,
        prefetch_factor=pf if n_workers > 0 else None,
    )

    return train_loader, val_loader, val_dataset


# ── Training (single model, single GPU) ───────────────────────────────

def _train_single(name: str, config: Config):
    """Train a single model. Used as target for multiprocessing."""
    set_seed(config.seed)
    train_loader, val_loader, _ = build_data_loaders(config)

    print(f"\n{'='*60}")
    print(f"  Training: {name.upper()} on {config.device}")
    print(f"{'='*60}")

    model = build_model(name, num_classes=config.num_classes,
                        pretrained_backbone=config.pretrained_backbone,
                        min_size=config.image_min_size, max_size=config.image_max_size)
    history = train_model(model, train_loader, val_loader, config, name)

    # Clear GPU cache after training
    if config.device.type == "cuda":
        torch.cuda.empty_cache()

    return history


def _train_on_gpu(name: str, gpu_id: int, config: Config):
    """Train a model on a specific GPU (for multiprocessing.Process)."""
    adjusted = replace(config, force_device=f"cuda:{gpu_id}")
    try:
        _train_single(name, adjusted)
    except Exception as e:
        print(f"ERROR training {name} on GPU {gpu_id}: {e}")
        import traceback
        traceback.print_exc()


# ── Multi-GPU parallel training ───────────────────────────────────────

def run_train(model_names: list, config: Config):
    """Train models, using multi-GPU parallelism when available."""
    gpu_ids = discover_gpus()
    print_gpu_info(gpu_ids)

    # If user forced a specific device, or only 1 model, train sequentially
    if config.force_device is not None or len(model_names) <= 1 or len(gpu_ids) <= 1:
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
            # Clear GPU cache between models
            if config.device.type == "cuda":
                torch.cuda.empty_cache()
        return histories

    # --- Multi-GPU: pool-based scheduling (no idle GPUs) ---
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    n_gpus = len(gpu_ids)
    total_workers = config.effective_num_workers
    workers_per_gpu = max(1, total_workers // n_gpus)
    adjusted_config = replace(config, num_workers=workers_per_gpu)

    print(f"\nGPU pool: {n_gpus} GPUs, {workers_per_gpu} workers/GPU")
    print(f"  Models queued: {[m.upper() for m in model_names]}")

    pending = list(model_names)       # models waiting to start
    running = {}                       # gpu_id -> (model_name, process)
    histories = {}

    while pending or running:
        # Launch on every free GPU
        free_gpus = [g for g in gpu_ids if g not in running]
        while pending and free_gpus:
            gpu_id = free_gpus.pop(0)
            name = pending.pop(0)
            print(f"\n{'='*60}")
            print(f"  Starting: {name.upper()} on GPU {gpu_id}")
            print(f"  Queue remaining: {[m.upper() for m in pending]}")
            print(f"{'='*60}")
            p = mp.Process(target=_train_on_gpu, args=(name, gpu_id, adjusted_config))
            p.start()
            running[gpu_id] = (name, p)

        if not running:
            break

        # Poll running processes (check every 5 seconds)
        import time as _time
        _time.sleep(5)

        for gpu_id in list(running.keys()):
            name, p = running[gpu_id]
            if not p.is_alive():
                p.join()
                del running[gpu_id]
                if p.exitcode == 0:
                    print(f"\n  {name.upper()} completed on GPU {gpu_id}")
                else:
                    print(f"\n  WARNING: {name.upper()} exited with code {p.exitcode} on GPU {gpu_id}")

                # Load history from saved JSON
                hist_path = Path(config.output_dir) / f"{name}_history.json"
                if hist_path.exists():
                    with open(hist_path) as f:
                        histories[name] = json.load(f)

                # GPU is now free — next iteration will fill it immediately

    return histories


# ── Evaluate ───────────────────────────────────────────────────────────

def run_evaluate(model_names: list, config: Config):
    """Evaluate trained models and return metrics."""
    _, val_loader, _ = build_data_loaders(config)
    all_metrics = {}
    all_predictions = {}
    all_targets = None

    use_amp = hasattr(config, "use_amp") and config.use_amp
    use_tta = getattr(config, "use_tta", False)
    use_soft_nms = getattr(config, "use_soft_nms", False)

    for name in model_names:
        ckpt_path = Path(config.checkpoint_dir) / f"{name}_best.pth"
        if not ckpt_path.exists():
            ckpt_path = Path(config.checkpoint_dir) / f"{name}_final.pth"
        if not ckpt_path.exists():
            print(f"WARNING: No checkpoint found for {name}, skipping evaluation.")
            continue

        print(f"\nEvaluating: {name.upper()}" + (" (TTA+Soft-NMS)" if use_tta or use_soft_nms else ""))
        model = build_model(name, num_classes=config.num_classes, pretrained_backbone=False,
                            min_size=config.image_min_size, max_size=config.image_max_size)
        ckpt = torch.load(ckpt_path, map_location=config.device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(config.device)

        predictions, targets = evaluate_model(
            model, val_loader, config.device,
            use_amp=use_amp,
            use_tta=use_tta,
            use_soft_nms=use_soft_nms,
        )
        metrics = compute_metrics(predictions, targets, patient_threshold=config.patient_threshold)
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
        opt_thresh = metrics.get("optimal_threshold", 0.3)
        roc_auc = metrics.get("roc_auc", 0.0)
        print(f"  ROC AUC:      {roc_auc*100:.1f}  (optimal threshold={opt_thresh:.3f})")

        # Clear GPU cache between models
        if config.device.type == "cuda":
            torch.cuda.empty_cache()

    return all_metrics, all_predictions, all_targets


# ── Compare / Visualize / Full ─────────────────────────────────────────

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

    # Evaluate to get predictions for AP-vs-IoU plot
    model_names = list(histories.keys()) or MODELS
    all_metrics, all_predictions, all_targets = run_evaluate(model_names, config)

    if not histories and not all_metrics:
        print("ERROR: No training histories or metrics found. Train models first.")
        return

    generate_all_plots(
        histories=histories,
        all_metrics=all_metrics,
        output_dir=config.output_dir,
        predictions_by_model=all_predictions,
        targets=all_targets,
    )


def run_visualize(config: Config, num_samples: int = 6):
    """Generate detection visualizations on sample images."""
    _, val_loader, val_dataset = build_data_loaders(config)

    use_tta = getattr(config, "use_tta", False)
    use_soft_nms = getattr(config, "use_soft_nms", False)

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

        if config.device.type == "cuda":
            torch.cuda.empty_cache()

    from src.visualize import plot_detection_samples
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if predictions_by_model:
        plot_detection_samples(sample_images, sample_targets, predictions_by_model, out_dir)
    else:
        print("WARNING: No model predictions available, skipping detection samples plot.")


def run_full(config: Config):
    """Run the complete pipeline: train, evaluate, compare, visualize."""
    gpu_ids = discover_gpus()
    print("=" * 60)
    print("  PNEUMONIA DETECTION — FULL PIPELINE")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"GPUs:   {len(gpu_ids)} ({', '.join(torch.cuda.get_device_name(i) for i in gpu_ids) if gpu_ids else 'none'})")
    print(f"Models: {MODELS}")
    print(f"Epochs: {config.num_epochs}, Batch: {config.batch_size}, Val freq: {getattr(config, 'val_frequency', 1)}")
    print(f"Resume: {getattr(config, 'resume', False)}, Compile: {getattr(config, 'use_compile', False)}")
    print(f"EMA: {getattr(config, 'use_ema', False)}, Freeze: {getattr(config, 'freeze_backbone_epochs', 0)}ep")
    print(f"TTA: {getattr(config, 'use_tta', False)}, Soft-NMS: {getattr(config, 'use_soft_nms', False)}")
    print(f"Scheduler: {getattr(config, 'scheduler_type', 'cosine')}, Weighted sampler: {getattr(config, 'use_weighted_sampler', False)}")
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


# ── CLI ────────────────────────────────────────────────────────────────

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
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=4, help="Data loader workers")
    parser.add_argument("--image-size", type=int, default=512, help="Image size for model input")
    parser.add_argument("--no-augmentation", action="store_true", help="Disable data augmentation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of patients for faster training (default: all)")
    parser.add_argument("--device", default=None,
                        help="Force device: cpu, cuda, cuda:0, cuda:1, mps (default: auto-detect)")
    parser.add_argument("--patient-threshold", type=float, default=0.3,
                        help="Score threshold for patient-level classification (default: 0.3)")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable Automatic Mixed Precision on CUDA")
    parser.add_argument("--threads", type=int, default=0,
                        help="OpenMP threads for CPU parallelism (0=auto)")

    # Performance args
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile() optimization")
    parser.add_argument("--val-frequency", type=int, default=2,
                        help="Validate every N epochs (default: 2)")
    parser.add_argument("--early-stopping", type=int, default=5,
                        help="Stop after N validations without AP improvement (0=disabled)")
    parser.add_argument("--prefetch-factor", type=int, default=4,
                        help="DataLoader prefetch factor (default: 4)")

    # Advanced training
    parser.add_argument("--freeze-epochs", type=int, default=3,
                        help="Freeze backbone for N epochs (default: 3, 0=disabled)")
    parser.add_argument("--no-ema", action="store_true",
                        help="Disable Exponential Moving Average")
    parser.add_argument("--ema-decay", type=float, default=0.999,
                        help="EMA decay factor (default: 0.999)")
    parser.add_argument("--scheduler", choices=["cosine", "step"], default="cosine",
                        help="LR scheduler type (default: cosine)")
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="Gradient accumulation steps (default: 1)")
    parser.add_argument("--multi-scale", action="store_true",
                        help="Enable multi-scale training [448..576]")

    # Advanced evaluation
    parser.add_argument("--no-tta", action="store_true",
                        help="Disable Test-Time Augmentation")
    parser.add_argument("--no-soft-nms", action="store_true",
                        help="Disable Gaussian Soft-NMS")

    # Advanced data
    parser.add_argument("--no-weighted-sampler", action="store_true",
                        help="Disable WeightedRandomSampler")
    parser.add_argument("--pos-weight", type=float, default=3.0,
                        help="Weight for positive patients in sampler (default: 3.0)")

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
        patient_threshold=args.patient_threshold,
        use_amp=not args.no_amp,
        num_threads=args.threads,
        # Performance
        resume=args.resume,
        use_compile=not args.no_compile,
        val_frequency=args.val_frequency,
        early_stopping_patience=args.early_stopping,
        prefetch_factor=args.prefetch_factor,
        # Advanced training
        freeze_backbone_epochs=args.freeze_epochs,
        use_ema=not args.no_ema,
        ema_decay=args.ema_decay,
        scheduler_type=args.scheduler,
        gradient_accumulation=args.grad_accum,
        multi_scale=args.multi_scale,
        # Advanced evaluation
        use_tta=not args.no_tta,
        use_soft_nms=not args.no_soft_nms,
        # Advanced data
        use_weighted_sampler=not args.no_weighted_sampler,
        positive_sample_weight=args.pos_weight,
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
