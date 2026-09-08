"""
Dataset Preparation Script
===========================
Renames the Mendeley "Ultrasound Fetus Dataset" class folders
so they match this project's 3-class convention:

  Mendeley name   →   Project name
  ─────────────────────────────────
  normal          →   normal       (unchanged)
  benign          →   fgr          (Fetal Growth Restriction)
  malignant       →   abnormal     (Other Fetal Abnormalities)

Usage:
    python prepare_dataset.py --dataset_path "C:/path/to/downloaded/dataset"

After running this script, the dataset will be ready for train.py.
"""

import os
import argparse
from pathlib import Path


# ── Rename map ────────────────────────────────────────────────────────────────
RENAME_MAP = {
    "benign":    "fgr",
    "malignant": "abnormal",
    "normal":    "normal",   # listed explicitly for validation purposes
}

# Sub-splits that may exist inside the downloaded archive
EXPECTED_SPLITS = ["train", "test", "validation", "val"]


def rename_class_folders(split_dir: Path, dry_run: bool = False) -> dict:
    """
    Rename class folders inside a single split directory (train / test / val).

    Returns a summary dict of {old_name: new_name} renames performed.
    """
    summary = {}
    if not split_dir.exists():
        print(f"  [SKIP] Split folder not found: {split_dir}")
        return summary

    for child in sorted(split_dir.iterdir()):
        if not child.is_dir():
            continue

        folder_name = child.name.lower()
        new_name = RENAME_MAP.get(folder_name)

        if new_name is None:
            print(f"  [WARN] Unknown class folder '{child.name}' in {split_dir.name} — skipped")
            continue

        if folder_name == new_name:
            print(f"  [OK]   '{child.name}' already has the correct name — no change")
            summary[child.name] = new_name
            continue

        target = child.parent / new_name
        if target.exists():
            print(f"  [WARN] Target '{new_name}' already exists in {split_dir.name} — skipping rename")
            continue

        if dry_run:
            print(f"  [DRY]  Would rename: {split_dir.name}/{child.name} -> {new_name}")
        else:
            child.rename(target)
            print(f"  [OK]   Renamed: {split_dir.name}/{child.name} -> {new_name}")

        summary[child.name] = new_name

    return summary


def count_images(folder: Path) -> int:
    """Count image files recursively in a folder."""
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    return sum(1 for f in folder.rglob("*") if f.suffix.lower() in extensions)


def validate_dataset(dataset_root: Path) -> bool:
    """
    Validate that the dataset has the correct structure after renaming.
    Prints a summary table and returns True if valid.
    """
    print("\n" + "=" * 55)
    print("  Dataset Validation Summary")
    print("=" * 55)

    expected_classes = ["normal", "fgr", "abnormal"]
    all_ok = True

    for split in EXPECTED_SPLITS:
        split_dir = dataset_root / split
        if not split_dir.exists():
            continue

        print(f"\n  [{split.upper()}]")
        for cls in expected_classes:
            cls_dir = split_dir / cls
            if cls_dir.exists():
                n = count_images(cls_dir)
                print(f"    OK  {cls:<12} {n:>5} images")
            else:
                print(f"    MISS {cls:<12} MISSING")
                all_ok = False

    print("=" * 55)
    if all_ok:
        print("  Dataset structure is VALID — ready for training!")
    else:
        print("  Some class folders are MISSING — check the dataset path.")
    print("=" * 55 + "\n")
    return all_ok


def auto_detect_root(base: Path) -> Path:
    """
    If the ZIP was extracted into a single top-level folder,
    find and return the actual root containing train/test splits.
    """
    # Check if splits exist directly under base
    for split in EXPECTED_SPLITS:
        if (base / split).exists():
            return base

    # Otherwise, look one level deeper
    subdirs = [d for d in base.iterdir() if d.is_dir()]
    if len(subdirs) == 1:
        candidate = subdirs[0]
        for split in EXPECTED_SPLITS:
            if (candidate / split).exists():
                print(f"  [AUTO] Detected dataset root: {candidate}")
                return candidate

    return base  # fallback


def prepare(dataset_path: str, dry_run: bool = False):
    """Main preparation function."""
    dataset_root = auto_detect_root(Path(dataset_path).resolve())

    print("\n" + "=" * 55)
    print("  Fetal Ultrasound Dataset Preparation")
    print(f"  Root: {dataset_root}")
    print("=" * 55)

    found_splits = []
    for split_name in EXPECTED_SPLITS:
        split_dir = dataset_root / split_name
        if split_dir.exists():
            found_splits.append(split_name)
            print(f"\n  Processing split: [{split_name.upper()}]")
            rename_class_folders(split_dir, dry_run=dry_run)

    if not found_splits:
        print("\n  [ERROR] No split folders (train/test/validation) found.")
        print("  Make sure --dataset_path points to the folder that contains")
        print("  'train/', 'test/', and/or 'validation/' subfolders.")
        return False

    if not dry_run:
        validate_dataset(dataset_root)

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare the Mendeley Ultrasound Fetus Dataset for this project."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the downloaded and extracted dataset folder."
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Preview what would be renamed without making any changes."
    )
    args = parser.parse_args()
    prepare(args.dataset_path, dry_run=args.dry_run)
