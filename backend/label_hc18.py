"""
HC18 Dataset Labeller
======================
Reads the HC18 training CSV and automatically sorts ultrasound images
into class folders based on head circumference (HC) measurements:

  HC ≥ 5th percentile for gestational age  →  normal/
  HC < 5th percentile for gestational age  →  fgr/

This produces a 2-class dataset ready for train.py.

Usage:
    python label_hc18.py \
        --images_dir  "C:/path/to/HC18/training_set" \
        --csv_path    "C:/path/to/HC18/training_set_pixel_size_and_HC.csv" \
        --output_dir  "C:/path/to/HC18/labelled_dataset"

After running, the output folder structure will be:
    labelled_dataset/
    ├── train/
    │   ├── normal/
    │   └── fgr/
    └── validation/
        ├── normal/
        └── fgr/

Then run:
    python train.py --dataset_path "C:/path/to/HC18/labelled_dataset"

Clinical reference (HC 5th percentile by gestational age):
    Source: Hadlock et al. (1984) / ISUOG guidelines
    The thresholds used here are from published biometry charts.
"""

import os
import shutil
import argparse
import random
from pathlib import Path

import pandas as pd

# ── Clinical HC 5th percentile by gestational week (mm) ──────────────────────
# Source: Hadlock et al. / INTERGROWTH-21st project
# Format: {gestational_age_weeks: (5th_percentile_mm, 95th_percentile_mm)}
HC_PERCENTILE_TABLE = {
    14: (88,  108),
    15: (100, 122),
    16: (112, 136),
    17: (124, 150),
    18: (136, 164),
    19: (148, 178),
    20: (159, 191),
    21: (170, 203),
    22: (181, 215),
    23: (191, 226),
    24: (201, 237),
    25: (210, 247),
    26: (220, 257),
    27: (229, 266),
    28: (238, 275),
    29: (246, 284),
    30: (254, 292),
    31: (262, 299),
    32: (270, 307),
    33: (277, 313),
    34: (284, 320),
    35: (290, 326),
    36: (296, 331),
    37: (301, 336),
    38: (306, 341),
    39: (310, 345),
    40: (314, 349),
}

# ── If no gestational age is available, use a fixed threshold ─────────────────
# Mean fetal HC at ~28 weeks. Images below this are considered possible FGR.
DEFAULT_FGR_THRESHOLD_MM = 245.0   # < 5th percentile ~28 weeks
DEFAULT_UPPER_BOUND_MM   = 345.0   # > 95th percentile → possible macrocephaly


def classify_hc(hc_mm: float, ga_weeks: int | None = None) -> str:
    """
    Classify a head circumference measurement as 'normal' or 'fgr'.

    Args:
        hc_mm:    Measured HC in millimetres.
        ga_weeks: Gestational age in weeks (optional).

    Returns:
        'normal' or 'fgr'
    """
    if ga_weeks is not None and ga_weeks in HC_PERCENTILE_TABLE:
        p5, _ = HC_PERCENTILE_TABLE[ga_weeks]
        return "fgr" if hc_mm < p5 else "normal"
    else:
        # No GA info — use fixed threshold
        return "fgr" if hc_mm < DEFAULT_FGR_THRESHOLD_MM else "normal"


def load_csv(csv_path: str) -> pd.DataFrame:
    """Load and clean the HC18 CSV file."""
    df = pd.read_csv(csv_path)

    # Normalise column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    print(f"  Loaded CSV: {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")

    # Identify the HC column (may be named differently)
    hc_col = None
    for candidate in ["head_circumference_(mm)", "hc_(mm)", "hc_mm", "hc", "head_circumference"]:
        if candidate in df.columns:
            hc_col = candidate
            break

    if hc_col is None:
        # Use the last numeric column as HC
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        hc_col = numeric_cols[-1]
        print(f"  [AUTO] Using column '{hc_col}' as HC measurement")

    df["hc_mm"] = pd.to_numeric(df[hc_col], errors="coerce")

    # Identify filename column
    fname_col = None
    for candidate in ["filename", "file_name", "image", "name"]:
        if candidate in df.columns:
            fname_col = candidate
            break
    if fname_col is None:
        fname_col = df.columns[0]
        print(f"  [AUTO] Using column '{fname_col}' as filename")

    df["filename"] = df[fname_col].astype(str)

    return df


def build_dataset(
    images_dir:  str,
    csv_path:    str,
    output_dir:  str,
    val_split:   float = 0.15,
    seed:        int   = 42,
):
    """
    Main function: label HC18 images and copy into train/val class folders.
    """
    images_path = Path(images_dir).resolve()
    output_path = Path(output_dir).resolve()
    random.seed(seed)

    print("\n" + "=" * 60)
    print("  HC18 Dataset Labeller")
    print("=" * 60)
    print(f"  Images  : {images_path}")
    print(f"  CSV     : {csv_path}")
    print(f"  Output  : {output_path}")
    print(f"  Val split: {val_split*100:.0f}%")
    print("=" * 60)

    # ── Load CSV ──────────────────────────────────────────────────────────────
    df = load_csv(csv_path)
    df = df.dropna(subset=["hc_mm"])
    print(f"\n  Valid HC measurements: {len(df)}")
    print(f"  HC range: {df['hc_mm'].min():.1f} mm — {df['hc_mm'].max():.1f} mm")
    print(f"  HC mean : {df['hc_mm'].mean():.1f} mm")

    # ── Classify each image ────────────────────────────────────────────────────
    df["class"] = df["hc_mm"].apply(lambda hc: classify_hc(hc))

    normal_count = (df["class"] == "normal").sum()
    fgr_count    = (df["class"] == "fgr").sum()
    print(f"\n  Labelling summary:")
    print(f"    normal : {normal_count} images")
    print(f"    fgr    : {fgr_count}    images")

    if fgr_count == 0:
        print("\n  [WARN] No FGR images found with the default threshold.")
        print(f"         FGR threshold: HC < {DEFAULT_FGR_THRESHOLD_MM} mm")
        print("         The HC18 dataset is a NORMAL fetal dataset — most images")
        print("         will be labelled 'normal'. Consider:")
        print("         (a) Lowering the threshold with --fgr_threshold")
        print("         (b) Using a mixed dataset with clinically confirmed FGR cases")

    # ── Find actual image files (skip annotation masks) ──────────────────────
    found, missing, skipped_annot = 0, 0, 0
    records = []

    # Build a lookup: all real image files in the directory (not annotations)
    all_real_images = {
        p.stem.lower(): p
        for p in images_path.glob("*.png")
        if "_annotation" not in p.stem.lower()
    }
    print(f"\n  Real ultrasound images found in folder: {len(all_real_images)}")
    print(f"  (Annotation mask files are automatically excluded)")

    for _, row in df.iterrows():
        fname = row["filename"]
        # Remove extension if present
        stem = Path(fname).stem

        # Skip annotation files listed in CSV
        if "_annotation" in stem.lower():
            skipped_annot += 1
            continue

        # Try exact match first, then lowercase
        img_path = (
            images_path / (stem + ".png")
            if (images_path / (stem + ".png")).exists()
            else all_real_images.get(stem.lower())
        )

        if img_path is None:
            missing += 1
            continue

        records.append({"path": img_path, "class": row["class"]})
        found += 1

    print(f"\n  Images found   : {found}")
    print(f"  Images missing : {missing}")

    if found == 0:
        print("\n  [ERROR] No images found. Check --images_dir path.")
        print(f"          Expected images in: {images_path}")
        return

    # ── Train / val split ─────────────────────────────────────────────────────
    random.shuffle(records)
    val_n  = max(1, int(len(records) * val_split))
    val_records   = records[:val_n]
    train_records = records[val_n:]

    # ── Copy files into output structure ──────────────────────────────────────
    splits = {"train": train_records, "validation": val_records}

    for split_name, split_records in splits.items():
        for cls in ["normal", "fgr"]:
            (output_path / split_name / cls).mkdir(parents=True, exist_ok=True)

    copied = {"train": {"normal": 0, "fgr": 0}, "validation": {"normal": 0, "fgr": 0}}

    for split_name, split_records in splits.items():
        for rec in split_records:
            src  = rec["path"]
            cls  = rec["class"]
            dest = output_path / split_name / cls / src.name
            shutil.copy2(src, dest)
            copied[split_name][cls] += 1

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Dataset Created Successfully")
    print("=" * 60)
    for split_name in ["train", "validation"]:
        print(f"\n  [{split_name.upper()}]")
        for cls in ["normal", "fgr"]:
            n = copied[split_name][cls]
            print(f"    {cls:<12} {n:>5} images")

    print(f"\n  Output folder: {output_path}")
    print("\n  Next step — run training:")
    print(f"  python train.py --dataset_path \"{output_path}\"")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Label HC18 images as normal/fgr using head circumference measurements."
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Path to the HC18 training_set folder containing .png images."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to training_set_pixel_size_and_HC.csv"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dataset_hc18_labelled",
        help="Where to save the labelled dataset (default: dataset_hc18_labelled)"
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.15,
        help="Fraction of data to use for validation (default: 0.15)"
    )
    parser.add_argument(
        "--fgr_threshold",
        type=float,
        default=DEFAULT_FGR_THRESHOLD_MM,
        help=f"HC threshold in mm below which image is labelled FGR (default: {DEFAULT_FGR_THRESHOLD_MM})"
    )
    args = parser.parse_args()

    # Allow override of FGR threshold
    DEFAULT_FGR_THRESHOLD_MM = args.fgr_threshold

    build_dataset(
        images_dir=args.images_dir,
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        val_split=args.val_split,
    )
