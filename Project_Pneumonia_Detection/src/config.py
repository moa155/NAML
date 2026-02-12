"""Configuration for pneumonia detection experiments."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch


@dataclass
class Config:
    # --- Paths ---
    data_dir: str = "data"
    output_dir: str = "results"
    checkpoint_dir: str = "checkpoints"

    # Dataset files (RSNA Pneumonia Detection Challenge)
    train_csv: str = "stage_2_train_labels.csv"
    detail_csv: str = "stage_2_detailed_class_info.csv"
    train_images_dir: str = "stage_2_train_images"

    # --- Dataset ---
    val_split: float = 0.2
    seed: int = 42
    max_samples: Optional[int] = None  # Limit number of patients (None = use all)

    # --- Training ---
    batch_size: int = 4
    num_workers: int = 4
    num_epochs: int = 20
    learning_rate: float = 1e-3
    lr_milestones: tuple = (12, 16)
    lr_gamma: float = 0.1
    weight_decay: float = 1e-4
    momentum: float = 0.9

    # --- Model ---
    num_classes: int = 2  # background + pneumonia
    pretrained_backbone: bool = True

    # --- Detection ---
    nms_threshold: float = 0.5
    score_threshold: float = 0.1

    # --- Data augmentation (paper's method) ---
    use_augmentation: bool = True
    image_min_size: int = 512
    image_max_size: int = 512

    # --- Device override ---
    force_device: Optional[str] = None  # None = auto-detect, "cpu", "cuda", "mps"

    @property
    def device(self) -> torch.device:
        if self.force_device is not None:
            return torch.device(self.force_device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @property
    def images_path(self) -> Path:
        return Path(self.data_dir) / self.train_images_dir

    @property
    def labels_path(self) -> Path:
        return Path(self.data_dir) / self.train_csv

    @property
    def detail_labels_path(self) -> Path:
        return Path(self.data_dir) / self.detail_csv
