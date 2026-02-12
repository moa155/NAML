"""Preprocess RSNA DICOM files to PNG for faster training.

DICOM loading is ~10-50x slower than PNG loading.  This script converts
all training DICOM images to 8-bit PNG files, enabling much faster
data loading during training.

Usage:
    python -m src.preprocess --data-dir data/
"""

import argparse
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import pydicom
from PIL import Image
from tqdm import tqdm


def convert_one(args):
    dcm_path, out_path = args
    try:
        dcm = pydicom.dcmread(str(dcm_path))
        pixel_array = dcm.pixel_array.astype(np.float32)
        pmin, pmax = pixel_array.min(), pixel_array.max()
        if pmax - pmin > 0:
            pixel_array = (pixel_array - pmin) / (pmax - pmin)
        pixel_array = (pixel_array * 255).astype(np.uint8)
        Image.fromarray(pixel_array).save(str(out_path))
        return True
    except Exception as e:
        print(f"Error converting {dcm_path}: {e}")
        return False


def preprocess(data_dir: str):
    data_dir = Path(data_dir)
    dcm_dir = data_dir / "stage_2_train_images"
    png_dir = data_dir / "stage_2_train_images_png"
    png_dir.mkdir(exist_ok=True)

    dcm_files = sorted(dcm_dir.glob("*.dcm"))
    print(f"Found {len(dcm_files)} DICOM files")

    # Skip already converted
    tasks = []
    for dcm_path in dcm_files:
        out_path = png_dir / (dcm_path.stem + ".png")
        if not out_path.exists():
            tasks.append((dcm_path, out_path))

    if not tasks:
        print("All files already converted.")
        return

    print(f"Converting {len(tasks)} files to PNG (skipping {len(dcm_files) - len(tasks)} already done)...")
    n_workers = min(cpu_count(), 8)

    with Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(convert_one, tasks),
            total=len(tasks),
            desc="DICOM -> PNG",
        ))

    success = sum(results)
    print(f"Done: {success}/{len(tasks)} converted successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess DICOM to PNG")
    parser.add_argument("--data-dir", default="data", help="Path to RSNA data directory")
    args = parser.parse_args()
    preprocess(args.data_dir)
