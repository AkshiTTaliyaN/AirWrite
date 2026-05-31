"""
Train EMNIST Balanced CNN.

Usage:
    python -m ml.training.train_cnn --epochs 5

Resuming after an interruption:
    python -m ml.training.train_cnn --epochs 5 --resume
    (picks up from the last completed epoch checkpoint)
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import config
from ml.models.cnn import EMNISTCNN, NUM_CLASSES


# ── Dataset helpers ────────────────────────────────────────────────────────────

def _clean_corrupt_download(dataset_dir: Path) -> None:
    """
    Remove partially downloaded EMNIST files so torchvision re-downloads cleanly.
    Called when a dataset load fails, indicating a corrupt/incomplete download.
    """
    emnist_dir = dataset_dir / "EMNIST"
    raw_dir = emnist_dir / "raw"
    if raw_dir.exists():
        print(f"[!] Removing corrupt dataset files from {raw_dir} ...")
        shutil.rmtree(raw_dir)
        print("[!] Deleted. Will re-download on next attempt.")


def get_dataloaders(batch_size: int = 128):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1751,), (0.3332,)),
        ]
    )

    # Try loading — if corrupt from a previous interrupted download, clean and retry once.
    for attempt in range(2):
        try:
            train_set = datasets.EMNIST(
                root=str(config.ML_DATASETS_DIR),
                split="balanced",
                train=True,
                download=True,
                transform=transform,
            )
            test_set = datasets.EMNIST(
                root=str(config.ML_DATASETS_DIR),
                split="balanced",
                train=False,
                download=True,
                transform=transform,
            )
            break  # success
        except Exception as e:
            if attempt == 0:
                print(f"[!] Dataset load failed ({e}). Cleaning corrupt files and retrying...")
                _clean_corrupt_download(config.ML_DATASETS_DIR)
            else:
                raise RuntimeError(
                    "Dataset download failed twice. Check your internet connection."
                ) from e

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


# ── Training / evaluation ──────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, pred = outputs.max(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, pred = outputs.max(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)
    return correct / total


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def _checkpoint_path(output_path: Path, epoch: int) -> Path:
    """e.g.  ml/models/emnist_cnn_epoch3.ckpt"""
    return output_path.parent / f"{output_path.stem}_epoch{epoch}.ckpt"


def save_checkpoint(model, optimizer, epoch: int, output_path: Path) -> None:
    path = _checkpoint_path(output_path, epoch)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )
    print(f"  [checkpoint] saved → {path.name}")


def load_latest_checkpoint(model, optimizer, output_path: Path, total_epochs: int):
    """
    Scan for the highest-epoch checkpoint and load it.
    Returns the epoch number to resume FROM (next epoch to train).
    Returns 1 if no checkpoint found.
    """
    best_epoch = 0
    best_path = None
    for epoch in range(total_epochs, 0, -1):
        p = _checkpoint_path(output_path, epoch)
        if p.exists():
            best_epoch = epoch
            best_path = p
            break

    if best_path is None:
        print("[resume] No checkpoint found — starting from scratch.")
        return 1

    print(f"[resume] Loading checkpoint: {best_path.name}")
    ckpt = torch.load(best_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    print(f"[resume] Resuming from epoch {best_epoch + 1}")
    return best_epoch + 1


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train AirWrite EMNIST CNN")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", type=str, default=str(config.DEFAULT_MODEL_PATH))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest per-epoch checkpoint.",
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    config.ML_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.ML_DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, test_loader = get_dataloaders(args.batch_size)

    model = EMNISTCNN(NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 1
    if args.resume:
        start_epoch = load_latest_checkpoint(model, optimizer, out_path, args.epochs)

    if start_epoch > args.epochs:
        print("All epochs already completed. Nothing to do.")
        return

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_acc = evaluate(model, test_loader, device)
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"loss={train_loss:.4f}  train_acc={train_acc:.4f}  test_acc={test_acc:.4f}"
        )
        # Save a checkpoint after every epoch — safe to interrupt any time
        save_checkpoint(model, optimizer, epoch, out_path)

    # Save final clean weights (just model state, no optimizer bloat)
    torch.save(model.state_dict(), out_path)
    print(f"\nFinal model saved → {out_path}")
    print("You can delete the .ckpt files now if you want.")


if __name__ == "__main__":
    main()
