# Pneumonia Detection: Anchor-Free vs. Anchor-Based Object Detection

NAML (Numerical Analysis and Machine Learning) course project — Politecnico di Milano.

This project reproduces and extends the method from:

> Wu et al., *"Pneumonia detection based on RSNA dataset and anchor-free deep learning detector"*,
> Scientific Reports **14**, 1929 (2024). DOI: [10.1038/s41598-024-52156-7](https://doi.org/10.1038/s41598-024-52156-7)

Three object detection frameworks are implemented and compared for pneumonia
detection on chest X-ray images from the RSNA Pneumonia Detection Challenge
dataset.

## Methods

| # | Model | Type | Description |
|---|-------|------|-------------|
| 1 | **FCOS** | Anchor-free, one-stage | Paper's proposed method. Per-pixel prediction with Feature Pyramid Network (FPN), two-branch detection head, and focal loss. |
| 2 | **RetinaNet** | Anchor-based, one-stage | Comparison method from paper Table 3. Uses FPN with anchor boxes and focal loss. |
| 3 | **Faster R-CNN** | Anchor-based, two-stage | Comparison method from paper Table 3. Region Proposal Network + ROI classification/regression. |

All three models share a **ResNet-50** backbone with **FPN** for fair comparison,
following the paper's experimental setup.

### Data Augmentation (Paper's Method)

The paper's augmentation strategy is applied during training:
- Horizontal and vertical flips
- Luminance (brightness) augmentation
- Random cropping

### Evaluation Metrics

**Object Detection** (COCO-style, as in the paper):
- AP@0.5 (Average Precision at IoU=0.5)
- AP@[0.5:0.95] (COCO standard)
- AP\_M, AP\_L (medium and large objects)
- AR@10, AR\_M, AR\_L (Average Recall)

**Patient-Level Classification**:
- Accuracy, Precision, Recall, F1-Score

## Dataset

**RSNA Pneumonia Detection Challenge** (Kaggle, 2018)

- ~26,684 chest X-ray images (1024x1024, DICOM format)
- ~6,012 patients with pneumonia (bounding box annotations)
- ~20,672 patients without pneumonia
- Collected by 18 radiologists from 16 institutions

### Download

1. Install the Kaggle CLI (if not installed):
   ```bash
   pip install kaggle
   ```

2. Set up Kaggle API credentials:
   - Go to https://www.kaggle.com/settings → API → Create New Token
   - Place the downloaded `kaggle.json` in `~/.kaggle/`

3. Download and extract the dataset:
   ```bash
   kaggle competitions download -c rsna-pneumonia-detection-challenge
   unzip rsna-pneumonia-detection-challenge.zip -d data/
   ```

   Alternatively, download manually from:
   https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data

### Expected Directory Structure

```
data/
├── stage_2_train_labels.csv
├── stage_2_detailed_class_info.csv
├── stage_2_train_images/
│   ├── 0004cfab-14fd-4e49-80ba-63a80b6bddd6.dcm
│   ├── ...
│   └── ffff8f2e-5765-4278-9e63-4414e0ae7a8e.dcm
└── stage_2_test_images/
    └── ...
```

## Installation

### Local

```bash
# Clone or navigate to the project directory
cd Project_Pneumonia_Detection

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Kaggle (Recommended for GPU Training)

Kaggle provides free GPUs (P100 or T4 x2) with the RSNA dataset already available — no download needed.

Use the provided `Pneumonia_Detection_Kaggle.ipynb` notebook:

1. Create a new Kaggle Notebook and upload the `.ipynb` file
2. **Add Data** → search `rsna-pneumonia-detection-challenge` (Competition tab) → Add
3. Upload `Project_Pneumonia_Detection.zip` as a Kaggle Dataset, then add it via **Add Data**
4. Settings → Accelerator → **GPU T4 x2** (parallel training) or **GPU P100**
5. Settings → Internet → **On**
6. Run all cells

With **T4 x2**, two models train in parallel on separate GPUs, reducing total time by ~1/3.

### Google Colab (Alternative)

Use `Pneumonia_Detection_Colab.ipynb` for Colab. Requires uploading the project and downloading the dataset via Kaggle API.

**Colab tips:**
- Use `Runtime > Change runtime type > T4 GPU`
- Copy data to local disk before preprocessing (Drive I/O is slow)
- AMP is enabled by default on CUDA, providing ~30% speedup

### Requirements

- Python >= 3.9
- PyTorch >= 2.0
- torchvision >= 0.15
- CUDA GPU recommended (also works on Apple Silicon MPS or CPU)

## Usage

### Full Pipeline (Recommended)

Train all three models, evaluate, and generate comparison plots:

```bash
python main.py --mode full --data-dir data/
```

### Running on Different Devices

The program automatically detects the best available device (CUDA > MPS > CPU).
You can override this with the `--device` flag:

```bash
# CUDA (NVIDIA GPU) — fastest, enables Automatic Mixed Precision (AMP)
python main.py --mode full --device cuda

# Multi-GPU — train specific models on specific GPUs
python main.py --mode train --model fcos --device cuda:0
python main.py --mode train --model retinanet --device cuda:1

# MPS (Apple Silicon M1/M2/M3) — GPU acceleration on macOS
python main.py --mode full --device mps

# CPU — no GPU required, slowest but universally compatible
python main.py --mode full --device cpu

# Auto-detect (default) — uses the best available device
python main.py --mode full
```

**Device-specific notes:**

| Device | AMP Support | Recommended Batch Size | Notes |
|--------|-------------|------------------------|-------|
| CUDA   | Yes (default on) | 4–8 | Fastest option. Use `--no-amp` to disable mixed precision |
| MPS    | No | 2–4 | Good performance on Apple Silicon. Some models (e.g., Faster R-CNN) may have limited MPS support |
| CPU    | No | 1–2 | Use `--threads N` to set OpenMP parallelism. Reduce `--epochs` and `--max-samples` for faster iteration |

### Individual Steps

```bash
# Train a single model
python main.py --mode train --model fcos --data-dir data/
python main.py --mode train --model retinanet --data-dir data/
python main.py --mode train --model faster_rcnn --data-dir data/

# Train all models
python main.py --mode train --model all --data-dir data/

# Evaluate trained models
python main.py --mode evaluate --model all --data-dir data/

# Generate comparison plots (from saved results)
python main.py --mode compare

# Generate detection visualizations
python main.py --mode visualize --data-dir data/
```

### Quick Test Run

To verify the setup works before a full training run:

```bash
# Train on a small subset (50 patients, 2 epochs)
python main.py --mode full --max-samples 50 --epochs 2
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `full` | `train`, `evaluate`, `compare`, `visualize`, or `full` |
| `--model` | `all` | `fcos`, `retinanet`, `faster_rcnn`, or `all` |
| `--data-dir` | `data` | Path to the RSNA dataset |
| `--output-dir` | `results` | Directory for plots and metrics |
| `--checkpoint-dir` | `checkpoints` | Directory for model checkpoints |
| `--epochs` | `20` | Number of training epochs |
| `--batch-size` | `4` | Training batch size |
| `--lr` | `0.001` | Learning rate (Adam optimizer) |
| `--num-workers` | `4` | DataLoader worker processes |
| `--image-size` | `512` | Input image size (resized to NxN) |
| `--no-augmentation` | off | Disable data augmentation |
| `--seed` | `42` | Random seed for reproducibility |
| `--device` | auto | Force device: `cpu`, `cuda`, `cuda:0`, `cuda:1`, or `mps` |
| `--max-samples` | all | Limit number of patients (for quick testing) |
| `--patient-threshold` | `0.3` | Confidence threshold for patient-level classification |
| `--no-amp` | off | Disable Automatic Mixed Precision (CUDA only) |
| `--threads` | `0` | OpenMP threads for CPU parallelism (0 = auto) |

### Regenerating Plots

To regenerate all plots from existing checkpoints without retraining:

```bash
python regenerate_plots.py                    # auto-detect device
python regenerate_plots.py --device cuda      # force CUDA
python regenerate_plots.py --device mps       # force MPS
python regenerate_plots.py --device cpu       # force CPU
```

### Preprocessing (Optional)

Convert DICOM images to PNG for ~10-50x faster data loading:

```bash
python -m src.preprocess --data-dir data/
```

The dataset loader automatically uses PNG files if available.

### Resource Requirements

- **GPU Memory**: ~4–8 GB recommended (batch size 4 at 512px images)
- **Disk**: ~3 GB for the RSNA dataset, ~500 MB for checkpoints
- For machines with less GPU memory, reduce `--batch-size` to 2 or 1
- For CPU-only training, consider using `--max-samples` to limit dataset size

## Output

After running the full pipeline, the following outputs are generated:

### `results/`

| File | Description |
|------|-------------|
| `training_loss.png/pdf` | Training loss curves for all models |
| `val_ap_over_epochs.png/pdf` | Validation AP@0.5 across epochs |
| `ap_comparison.png/pdf` | Bar chart comparing AP metrics |
| `ar_comparison.png/pdf` | Bar chart comparing AR metrics |
| `pr_curve.png/pdf` | Precision-Recall curves |
| `ap_vs_iou.png/pdf` | AP as a function of IoU threshold |
| `classification_metrics.png/pdf` | Patient-level accuracy, precision, recall, F1 |
| `detection_samples.png/pdf` | Sample images with GT and predicted boxes |
| `epoch_times.png/pdf` | Training speed comparison |
| `learning_rate.png` | LR schedule visualization |
| `comparison_table.tex` | LaTeX table (paper Table 3 format) |
| `all_metrics.json` | All metrics in machine-readable format |
| `{model}_history.json` | Per-epoch training history for each model |

### `checkpoints/`

| File | Description |
|------|-------------|
| `{model}_best.pth` | Best checkpoint (highest val AP@0.5) |
| `{model}_final.pth` | Checkpoint after the last epoch |

## Project Structure

```
Project_Pneumonia_Detection/
├── main.py                      # Entry point — CLI for all operations
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── Pneumonia_Detection_Kaggle.ipynb   # Kaggle notebook (multi-GPU support)
├── Pneumonia_Detection_Colab.ipynb    # Google Colab notebook
├── src/
│   ├── __init__.py
│   ├── config.py                # Configuration dataclass
│   ├── dataset.py               # RSNA dataset loader (DICOM support)
│   ├── transforms.py            # Detection-aware data augmentation
│   ├── models/
│   │   ├── __init__.py          # Model factory
│   │   ├── fcos.py              # FCOS anchor-free detector (paper method)
│   │   ├── retinanet.py         # RetinaNet (comparison)
│   │   └── faster_rcnn.py       # Faster R-CNN (comparison)
│   ├── engine.py                # Training and inference loops
│   ├── evaluate.py              # COCO-style AP/AR computation
│   └── visualize.py             # All plotting and LaTeX table generation
├── data/                        # RSNA dataset (not tracked in git)
├── results/                     # Generated plots and metrics
├── checkpoints/                 # Saved model weights
├── report/                      # LaTeX report (generated separately)
├── presentation/                # LaTeX presentation (generated separately)
├── docs/
│   └── Pneumonia_detection.pdf  # Reference paper
├── lectures/                    # NAML course lecture materials
└── labs/                        # NAML course lab materials
```

## Reference

```bibtex
@article{wu2024pneumonia,
  title={Pneumonia detection based on RSNA dataset and anchor-free deep learning detector},
  author={Wu, Linghua and Zhang, Jing and Wang, Yilin and Ding, Rong and Cao, Yueqin
          and Liu, Guiqin and Liufu, Changsheng and Xie, Baowei and Kang, Shanping
          and Liu, Rui and Li, Wenle and Guan, Furen},
  journal={Scientific Reports},
  volume={14},
  pages={1929},
  year={2024},
  publisher={Nature Publishing Group},
  doi={10.1038/s41598-024-52156-7}
}
```
