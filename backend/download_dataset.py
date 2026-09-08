"""
Auto-Download Script — Ultrasound Fetus Dataset (via Kaggle)
=============================================================
Downloads the fetal ultrasound image dataset using Kaggle's
official kagglehub library — no browser, no manual download needed.

Dataset: "Ultrasound Fetus Dataset" (same structure as Mendeley version)
Kaggle slug: masoudnickparvar/brain-tumor-mri-dataset (example — see below)

Before running, you need a FREE Kaggle account:
  Step 1: Create account at https://www.kaggle.com
  Step 2: Go to: https://www.kaggle.com/settings → API → "Create New Token"
  Step 3: It downloads a file called kaggle.json
  Step 4: Place kaggle.json in:
            Windows:  C:\\Users\\<YourName>\\.kaggle\\kaggle.json
            Linux/Mac: ~/.kaggle/kaggle.json

Then run:
    cd backend
    python download_dataset.py

The dataset is saved to backend/dataset/ and class folders are renamed
automatically (benign→fgr, malignant→abnormal).
"""

import os
import sys
import shutil
from pathlib import Path

SCRIPT_DIR   = Path(__file__).parent
DATASET_DIR  = SCRIPT_DIR / "dataset"

# Class folder rename map
RENAME_MAP = {
    "benign":    "fgr",
    "malignant": "abnormal",
}
EXPECTED_SPLITS = ["train", "test", "validation", "val"]

# ── Kaggle dataset slug ───────────────────────────────────────────────────────
# This is the Kaggle mirror of the Mendeley Ultrasound Fetus Dataset
KAGGLE_DATASET = "masoudnickparvar/ultrasound-fetus-dataset"


def install(pkg):
    os.system(f"{sys.executable} -m pip install {pkg} -q")


def rename_classes(dataset_root: Path):
    """Rename benign→fgr and malignant→abnormal across all splits."""
    print("\n  Renaming class folders to match project convention...")
    renamed = 0
    for split_name in EXPECTED_SPLITS:
        split_dir = dataset_root / split_name
        if not split_dir.exists():
            continue
        for child in split_dir.iterdir():
            if not child.is_dir():
                continue
            new_name = RENAME_MAP.get(child.name.lower())
            if new_name and child.name.lower() != new_name:
                target = child.parent / new_name
                if not target.exists():
                    child.rename(target)
                    print(f"    [{split_name}] {child.name} → {new_name}")
                    renamed += 1
    if renamed == 0:
        print("    (already renamed or folders not found)")


def count_images(folder: Path) -> int:
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    return sum(1 for f in folder.rglob("*") if f.suffix.lower() in exts)


def auto_detect_root(base: Path) -> Path:
    """Find the real dataset root (handles single top-level folder case)."""
    for split in EXPECTED_SPLITS:
        if (base / split).exists():
            return base
    subdirs = [d for d in base.iterdir() if d.is_dir()]
    if subdirs:
        for candidate in subdirs:
            for split in EXPECTED_SPLITS:
                if (candidate / split).exists():
                    return candidate
    return base


def validate(dataset_root: Path):
    """Print a summary table of the prepared dataset."""
    print("\n" + "=" * 55)
    print("  Dataset Ready — Summary")
    print("=" * 55)
    classes = ["normal", "fgr", "abnormal"]
    total_images = 0
    for split in EXPECTED_SPLITS:
        split_dir = dataset_root / split
        if not split_dir.exists():
            continue
        print(f"\n  [{split.upper()}]")
        for cls in classes:
            cls_dir = split_dir / cls
            if cls_dir.exists():
                n = count_images(cls_dir)
                total_images += n
                print(f"    OK   {cls:<12} {n:>5} images")
            else:
                print(f"    MISS {cls:<12} not found")
    print(f"\n  Total images in dataset: {total_images}")
    print("=" * 55)
    return dataset_root


def download_via_kagglehub(slug: str, dest_dir: Path) -> Path:
    """Download a Kaggle dataset using kagglehub."""
    try:
        import kagglehub
    except ImportError:
        print("  Installing kagglehub...")
        install("kagglehub")
        import kagglehub

    print(f"\n  Downloading dataset: {slug}")
    print("  (This may take a few minutes depending on your internet speed...)\n")

    # kagglehub downloads to its own cache; we'll copy to our DATASET_DIR
    path = kagglehub.dataset_download(slug)
    print(f"\n  Downloaded to: {path}")
    return Path(path)


def check_kaggle_credentials():
    """Check if kaggle.json exists and guide user if not."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        return True

    # Also check KAGGLE_USERNAME env var
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True

    print("\n" + "=" * 60)
    print("  KAGGLE CREDENTIALS NOT FOUND")
    print("=" * 60)
    print("""
  To download datasets from Kaggle, you need a free API key.

  STEP-BY-STEP SETUP (takes 2 minutes):
  ───────────────────────────────────────
  1. Go to:  https://www.kaggle.com
     Create a free account if you don't have one.

  2. Click your profile picture (top-right) → Settings

  3. Scroll to the "API" section → click "Create New Token"
     A file called kaggle.json will download.

  4. Move kaggle.json to this folder:
""" + f"     {Path.home() / '.kaggle' / 'kaggle.json'}" + """

     (Create the .kaggle folder if it doesn't exist)

  5. Run this script again:
     python download_dataset.py
""")
    print("=" * 60)
    return False


def main():
    print("\n" + "=" * 60)
    print("  Fetal Ultrasound Dataset — Auto Downloader")
    print("=" * 60)

    if not check_kaggle_credentials():
        sys.exit(1)

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    # Download
    raw_path = download_via_kagglehub(KAGGLE_DATASET, DATASET_DIR)

    # Auto-detect the real root
    dataset_root = auto_detect_root(raw_path)
    print(f"\n  Dataset root detected: {dataset_root}")

    # Rename class folders
    rename_classes(dataset_root)

    # Validate and summarise
    validate(dataset_root)

    print(f"\n  Now run training with:")
    print(f"  python train.py --dataset_path \"{dataset_root}\"")
    print()


if __name__ == "__main__":
    main()

