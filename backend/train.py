"""
Training Script -- Fetal Ultrasound CNN
========================================
Fine-tunes ResNet50 on the prepared Ultrasound Fetus Dataset.

Classes:
    0 -- normal    (Normal Fetus)
    1 -- fgr       (Fetal Growth Restriction)
    2 -- abnormal  (Other Fetal Abnormalities)

Usage:
    python train.py --dataset_path "C:/path/to/dataset" --epochs 30

Output:
    model/checkpoint/best_model.pth   <- best validation accuracy checkpoint
    model/checkpoint/final_model.pth  <- final epoch checkpoint

After training, the app will automatically load best_model.pth at startup.
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms, models

# -- Paths ---------------------------------------------------------------------
SCRIPT_DIR    = Path(__file__).parent
CHECKPOINT_DIR = SCRIPT_DIR / "model" / "checkpoint"

# -- Class order MUST match folder names in the dataset ------------------------
CLASS_ORDER = ["normal", "fgr", "abnormal"]


def get_transforms(image_size: int = 224):
    """
    Returns train / val transforms.
    Training uses aggressive augmentation suited for ultrasound images.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    return train_tf, val_tf


def build_model(model_name: str = "resnet50", num_classes: int = 3):
    """Load ResNet50 with ImageNet weights, replace the classification head."""
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, num_classes)
        )
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model


def purge_missing_files(split_dir: Path):
    """
    Scan all image files in a split directory and remove any that are
    missing on disk (broken symlinks, bad entries).
    Returns count of removed entries.
    """
    removed = 0
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    for cls_dir in split_dir.iterdir():
        if not cls_dir.is_dir():
            continue
        for img in cls_dir.iterdir():
            if img.suffix.lower() not in exts:
                continue
            if not img.exists() or img.stat().st_size == 0:
                print(f"  [REMOVE] Bad file: {img}")
                try:
                    img.unlink()
                except Exception:
                    pass
                removed += 1
    return removed


class SafeImageFolder(datasets.ImageFolder):
    """
    ImageFolder subclass that pre-validates all image paths at init time
    and silently drops any that are missing or unreadable from disk.
    Prevents FileNotFoundError during training iteration.
    """
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        original_count = len(self.samples)
        valid = []
        for path, label in self.samples:
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                valid.append((path, label))
        dropped = original_count - len(valid)
        if dropped > 0:
            print(f"  [CLEAN] Dropped {dropped} missing/broken files from {Path(root).name}/")
        self.samples = valid
        self.targets = [s[1] for s in valid]


def load_datasets(dataset_path: str, train_tf, val_tf, batch_size: int = 32):
    """
    Load train / val / test splits from the prepared dataset folder.
    Expects subfolders: train/, validation/ (or val/), test/
    """
    root = Path(dataset_path).resolve()

    # Locate split folders
    train_dir = root / "train"
    val_dir   = root / "validation" if (root / "validation").exists() else root / "val"
    test_dir  = root / "test"

    if not train_dir.exists():
        raise FileNotFoundError(
            f"Training folder not found at: {train_dir}\n"
            "Make sure you ran prepare_dataset.py first and the path is correct."
        )

    # Pre-validate: remove missing files and build safe datasets
    if train_dir.exists():
        removed = purge_missing_files(train_dir)
        if removed:
            print(f"  Removed {removed} bad file(s) from train/")

    # Build datasets using SafeImageFolder to skip any remaining broken files
    train_ds = SafeImageFolder(str(train_dir), transform=train_tf)
    val_ds   = SafeImageFolder(str(val_dir),   transform=val_tf) if val_dir.exists() else None
    test_ds  = SafeImageFolder(str(test_dir),  transform=val_tf) if test_dir.exists() else None

    # Verify class ordering matches our expected order
    detected_classes = train_ds.classes
    print(f"\n  Detected classes: {detected_classes}")
    if sorted(detected_classes) != sorted(CLASS_ORDER):
        print(f"  [WARN] Expected classes {CLASS_ORDER}, got {detected_classes}")
        print("  Make sure you ran prepare_dataset.py to rename the folders.")

    # Compute class weights to handle class imbalance
    class_counts = [0] * len(CLASS_ORDER)
    for _, label in train_ds.samples:
        class_counts[label] += 1
    total = sum(class_counts)
    class_weights = [total / (len(CLASS_ORDER) * c) if c > 0 else 1.0 for c in class_counts]
    print(f"  Class counts (train): {dict(zip(detected_classes, class_counts))}")
    print(f"  Class weights:        {dict(zip(detected_classes, [round(w, 3) for w in class_weights]))}")

    # Loaders
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False
    ) if val_ds else None
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False
    ) if test_ds else None

    return train_loader, val_loader, test_loader, class_weights, detected_classes


def train_epoch(model, loader, criterion, optimizer, device):
    """Run one training epoch. Returns (loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total * 100


def evaluate(model, loader, criterion, device):
    """Evaluate on validation/test set. Returns (loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total * 100


def train(
    dataset_path: str,
    model_name:   str   = "resnet50",
    epochs:       int   = 30,
    batch_size:   int   = 32,
    lr:           float = 1e-4,
    device_name:  str   = "auto",
    resume:       bool  = False,
):
    # -- Device ----------------------------------------------------------------
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    print(f"\n  Device: {device}")

    if device.type == "cpu":
        threads = min(12, os.cpu_count() or 4)
        torch.set_num_threads(threads)
        print(f"  Configured PyTorch CPU threads: {threads}")

    # -- Checkpoint directory --------------------------------------------------
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_ckpt  = CHECKPOINT_DIR / "best_model.pth"
    final_ckpt = CHECKPOINT_DIR / "final_model.pth"
    history_file = CHECKPOINT_DIR / "training_history.json"

    # -- Data ------------------------------------------------------------------
    train_tf, val_tf = get_transforms()
    train_loader, val_loader, test_loader, class_weights, class_names = load_datasets(
        dataset_path, train_tf, val_tf, batch_size=batch_size
    )

    # -- Model -----------------------------------------------------------------
    model = build_model(model_name, num_classes=len(class_names))
    start_epoch = 1
    best_val_acc = 0.0
    history = []

    if resume and best_ckpt.exists():
        print(f"\n  [RESUME] Found existing checkpoint: {best_ckpt}")
        try:
            ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["state_dict"])
            best_val_acc = float(ckpt.get("val_accuracy", 0.0))
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            print(f"  [RESUME] Loaded weights from Epoch {ckpt.get('epoch', 0)} with Best Val Acc: {best_val_acc:.2f}%")
            print(f"  [RESUME] Continuing training from Epoch {start_epoch} to {epochs}...")
        except Exception as e:
            print(f"  [WARN] Failed to load resume checkpoint ({e}). Starting fresh.")

    if history_file.exists() and resume:
        try:
            with open(history_file, "r") as f:
                history = json.load(f)
        except Exception:
            history = []

    model = model.to(device)

    # -- Loss with class weights -----------------------------------------------
    weights_tensor = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    # -- Optimizer: two parameter groups (different LR for backbone vs head) ---
    if model_name == "resnet50":
        head_params     = list(model.fc.parameters())
        backbone_params = [p for p in model.parameters() if not any(
            p is q for q in head_params
        )]
    else:
        head_params     = list(model.classifier.parameters())
        backbone_params = [p for p in model.parameters() if not any(
            p is q for q in head_params
        )]

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": lr * 0.1},   # fine-tune backbone slowly
        {"params": head_params,     "lr": lr},          # train head faster
    ], weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Fast-forward scheduler if resuming
    for _ in range(1, start_epoch):
        scheduler.step()

    # -- Training loop ---------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Starting training: {model_name} | Epochs {start_epoch}->{epochs} | batch={batch_size}")
    print(f"{'='*60}")
    print(f"  {'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>9}  {'Val Acc':>8}  {'LR':>8}")
    print(f"  {'-'*60}")

    vl_acc = 0.0
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()

        # Train
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        if val_loader:
            vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)
        else:
            vl_loss, vl_acc = 0.0, 0.0

        scheduler.step()
        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"  {epoch:>5}  {tr_loss:>10.4f}  {tr_acc:>8.2f}%"
            f"  {vl_loss:>9.4f}  {vl_acc:>7.2f}%  {current_lr:>8.2e}"
            f"  ({elapsed:.1f}s)"
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(tr_loss, 4),
            "train_acc": round(tr_acc, 2),
            "val_loss": round(vl_loss, 4),
            "val_acc": round(vl_acc, 2),
            "lr": current_lr,
            "elapsed_seconds": round(elapsed, 1)
        })
        try:
            with open(history_file, "w") as f:
                json.dump(history, f, indent=2)
        except Exception:
            pass

        # Save best checkpoint
        if val_loader and vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save({
                "epoch":        epoch,
                "model_name":   model_name,
                "class_names":  class_names,
                "num_classes":  len(class_names),
                "val_accuracy": vl_acc,
                "state_dict":   model.state_dict(),
            }, best_ckpt)
            print(f"           >> [SAVED] New best model: {vl_acc:.2f}% val accuracy")

    # -- Save final model ------------------------------------------------------
    torch.save({
        "epoch":        epochs,
        "model_name":   model_name,
        "class_names":  class_names,
        "num_classes":  len(class_names),
        "val_accuracy": vl_acc if val_loader else 0.0,
        "state_dict":   model.state_dict(),
    }, final_ckpt)

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best val accuracy : {best_val_acc:.2f}%")
    print(f"  Best checkpoint   : {best_ckpt}")
    print(f"  Final checkpoint  : {final_ckpt}")

    # -- Test set evaluation ---------------------------------------------------
    if test_loader and best_ckpt.exists():
        print(f"\n  Evaluating best model on test set...")
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        print(f"  Test accuracy: {te_acc:.2f}%  |  Test loss: {te_loss:.4f}")

    print(f"{'='*60}\n")
    print("  The app will automatically load best_model.pth on next startup.")
    print("  Run:  python app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the fetal ultrasound CNN classifier.")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the prepared dataset folder (with train/, validation/, test/).")
    parser.add_argument("--model",        type=str, default="resnet50",
                        choices=["resnet50", "mobilenet_v2"],
                        help="Model architecture (default: resnet50).")
    parser.add_argument("--epochs",       type=int,   default=30,
                        help="Number of training epochs (default: 30).")
    parser.add_argument("--batch_size",   type=int,   default=32,
                        help="Batch size (default: 32). Reduce if out of memory.")
    parser.add_argument("--lr",           type=float, default=1e-4,
                        help="Learning rate for the classification head (default: 1e-4).")
    parser.add_argument("--device",       type=str,   default="auto",
                        help="'cpu', 'cuda', or 'auto' (default: auto).")
    parser.add_argument("--resume",       action="store_true",
                        help="Resume from existing best_model.pth if found.")
    args = parser.parse_args()

    train(
        dataset_path=args.dataset_path,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device_name=args.device,
        resume=args.resume,
    )
