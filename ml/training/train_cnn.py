"""
Train EMNIST Balanced CNN.

Usage:
    python -m ml.training.train_cnn --epochs 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import config
from ml.models.cnn import EMNISTCNN, NUM_CLASSES


def get_dataloaders(batch_size: int = 128):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1751,), (0.3332,)),
        ]
    )
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
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


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


def main():
    parser = argparse.ArgumentParser(description="Train AirWrite EMNIST CNN")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", type=str, default=str(config.DEFAULT_MODEL_PATH))
    args = parser.parse_args()

    config.ML_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.ML_DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, test_loader = get_dataloaders(args.batch_size)
    model = EMNISTCNN(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_acc = evaluate(model, test_loader, device)
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"loss={train_loss:.4f} train_acc={train_acc:.4f} test_acc={test_acc:.4f}"
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Saved model to {out_path}")


if __name__ == "__main__":
    main()
