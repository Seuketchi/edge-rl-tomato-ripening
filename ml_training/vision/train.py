"""Training script for tomato ripeness classifier.

Trains MobileNetV2 (or alternative) on tomato dataset with:
- Cosine annealing LR schedule with warmup
- Early stopping on validation accuracy
- Label smoothing
- Mixed-precision training (if GPU available)
- Checkpoint saving + metrics logging

Usage:
    python -m ml_training.vision.train --config ml_training/config.yaml
    python -m ml_training.vision.train --config ml_training/config.yaml --epochs 2 --smoke-test
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import classification_report, confusion_matrix

from ml_training.vision.dataset import create_dataloaders
from ml_training.vision.model import create_model, get_model_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tomato ripeness classifier")
    parser.add_argument("--config", type=str, default="ml_training/config.yaml", help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--smoke-test", action="store_true", help="Quick test with 2 epochs, small batch")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
) -> tuple[float, float]:
    """Train for one epoch.

    Returns:
        Tuple of (average loss, accuracy).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model on a dataloader.

    Returns:
        Tuple of (average loss, accuracy).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def generate_evaluation_artifacts(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    class_names: list[str],
    device: torch.device,
    output_dir: Path,
) -> dict:
    """Generate confusion matrix, classification report, and plots.

    Args:
        model: Trained model.
        test_loader: Test set dataloader.
        class_names: List of class names.
        device: Torch device.
        output_dir: Directory to save artifacts.

    Returns:
        Dict with evaluation metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []

    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Classification report
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True,
    )
    report_text = classification_report(
        all_labels, all_preds,
        target_names=class_names,
    )
    print("\n=== Test Set Classification Report ===")
    print(report_text)

    # Save report as JSON
    report_path = output_dir / "classification_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Confusion matrix plot
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Test Set")
    plt.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix to {cm_path}")

    # --- Per-class metrics bar chart ---
    metrics_names = ["precision", "recall", "f1-score"]
    x = np.arange(len(class_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, metric in enumerate(metrics_names):
        vals = [report[c][metric] for c in class_names]
        bars = ax.bar(x + i * width, vals, width, label=metric.capitalize())
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title(f"Per-Class Metrics — Test Set (Accuracy: {report['accuracy']:.1%})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "per_class_metrics.png", dpi=150)
    plt.close(fig)
    print(f"Saved per-class metrics to {output_dir / 'per_class_metrics.png'}")

    return {
        "test_accuracy": report["accuracy"],
        "per_class_f1": {
            name: report[name]["f1-score"] for name in class_names
        },
        "confusion_matrix": cm.tolist(),
    }


@torch.no_grad()
def plot_sample_predictions(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    class_names: list[str],
    device: torch.device,
    output_dir: Path,
    num_samples: int = 16,
) -> None:
    """Plot a grid of sample predictions with true vs predicted labels.

    Correct predictions are shown in green, incorrect in red.
    """
    model.eval()
    # ImageNet de-normalisation for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    images_collected = []
    labels_collected = []
    preds_collected = []

    for images, labels in test_loader:
        outputs = model(images.to(device))
        _, predicted = outputs.max(1)
        for i in range(images.size(0)):
            images_collected.append(images[i])
            labels_collected.append(labels[i].item())
            preds_collected.append(predicted[i].cpu().item())
            if len(images_collected) >= num_samples:
                break
        if len(images_collected) >= num_samples:
            break

    cols = 4
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.5 * rows))
    axes = axes.flatten() if num_samples > 1 else [axes]

    for idx in range(len(axes)):
        ax = axes[idx]
        if idx < len(images_collected):
            img = images_collected[idx] * std + mean  # de-normalise
            img = img.clamp(0, 1).permute(1, 2, 0).numpy()
            ax.imshow(img)
            true_name = class_names[labels_collected[idx]]
            pred_name = class_names[preds_collected[idx]]
            correct = labels_collected[idx] == preds_collected[idx]
            colour = "#2ecc71" if correct else "#e74c3c"
            symbol = "✓" if correct else "✗"
            ax.set_title(
                f"True: {true_name}\nPred: {pred_name} {symbol}",
                color=colour, fontsize=10, fontweight="bold",
            )
        ax.axis("off")

    plt.suptitle("Sample Predictions — Test Set", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = output_dir / "sample_predictions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved sample predictions to {path}")


def plot_training_curves(metrics_log: list[dict], output_dir: Path) -> None:
    """Plot training and validation loss/accuracy curves."""
    epochs = [m["epoch"] for m in metrics_log]
    train_loss = [m["train_loss"] for m in metrics_log]
    val_loss = [m["val_loss"] for m in metrics_log]
    train_acc = [m["train_acc"] for m in metrics_log]
    val_acc = [m["val_acc"] for m in metrics_log]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_loss, label="Train", linewidth=2)
    ax1.plot(epochs, val_loss, label="Validation", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_acc, label="Train", linewidth=2)
    ax2.plot(epochs, val_acc, label="Validation", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    curves_path = output_dir / "training_curves.png"
    fig.savefig(curves_path, dpi=150)
    plt.close(fig)
    print(f"Saved training curves to {curves_path}")


def main() -> None:
    args = parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    vision_cfg = config["vision"]
    train_cfg = vision_cfg["training"]
    seed = config.get("project", {}).get("seed", 42)

    # Override for smoke test
    if args.smoke_test:
        train_cfg["epochs"] = 2
        train_cfg["batch_size"] = 8
        print("=== SMOKE TEST MODE ===")

    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs

    # Setup
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.get("project", {}).get("output_dir", "outputs")) / f"vision_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")

    # Save config snapshot
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # Data
    print("\n--- Loading Data ---")
    train_loader, val_loader, test_loader = create_dataloaders(config, seed=seed)

    # Model
    print("\n--- Creating Model ---")
    model = create_model(config)
    model = model.to(device)

    summary = get_model_summary(model, vision_cfg.get("input_size", 224))
    print(f"  FP32 size:  {summary['fp32_size_mb']} MB")
    print(f"  INT8 size:  {summary['int8_size_kb']} KB (estimated)")
    with open(output_dir / "model_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Loss, optimizer, scheduler
    label_smoothing = train_cfg.get("label_smoothing", 0.0)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )

    epochs = train_cfg["epochs"]
    warmup_epochs = train_cfg.get("warmup_epochs", 5)

    scheduler_type = train_cfg.get("scheduler", "cosine")
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6,
        )
    elif scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=5, factor=0.5,
        )

    # Mixed precision (GPU only)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    print(f"\n--- Training for {epochs} epochs ---")
    best_val_acc = 0.0
    patience_counter = 0
    early_stopping_patience = train_cfg.get("early_stopping_patience", 15)
    metrics_log = []

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()

        # Warmup: linear LR increase
        if epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group["lr"] = train_cfg["learning_rate"] * warmup_factor

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Step scheduler (after warmup)
        if epoch >= warmup_epochs:
            if scheduler_type == "plateau":
                scheduler.step(val_acc)
            else:
                scheduler.step()

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        metrics = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "lr": round(current_lr, 8),
            "time_s": round(epoch_time, 1),
        }
        metrics_log.append(metrics)

        print(
            f"Epoch {epoch+1:3d}/{epochs} | "
            f"Train: {train_acc:.4f} (loss {train_loss:.4f}) | "
            f"Val: {val_acc:.4f} (loss {val_loss:.4f}) | "
            f"LR: {current_lr:.2e} | "
            f"{epoch_time:.1f}s"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            ckpt_path = output_dir / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "config": config,
            }, ckpt_path)
            print(f"  ✓ New best model saved (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch+1} (patience={early_stopping_patience})")
                break

    # Save metrics log
    metrics_path = output_dir / "metrics.csv"
    if metrics_log:
        with open(metrics_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics_log[0].keys())
            writer.writeheader()
            writer.writerows(metrics_log)
        print(f"\nSaved metrics to {metrics_path}")

    # Plot training curves
    if len(metrics_log) > 1:
        plot_training_curves(metrics_log, output_dir)

    # Final evaluation on test set
    print("\n--- Final Evaluation on Test Set ---")
    best_ckpt = torch.load(output_dir / "best_model.pth", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    class_names = vision_cfg["dataset"]["class_names"]
    eval_results = generate_evaluation_artifacts(
        model, test_loader, class_names, device, output_dir,
    )
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print(f"Test accuracy: {eval_results['test_accuracy']:.4f}")

    # Sample predictions grid
    plot_sample_predictions(model, test_loader, class_names, device, output_dir)

    # Save final results
    final_results = {
        "best_val_acc": best_val_acc,
        "test_accuracy": eval_results["test_accuracy"],
        "per_class_f1": eval_results["per_class_f1"],
        "model_summary": summary,
        "total_epochs_trained": len(metrics_log),
        "training_time_s": sum(m["time_s"] for m in metrics_log),
    }
    with open(output_dir / "final_results.json", "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"\n✅ Training complete. All artifacts saved to {output_dir}/")


if __name__ == "__main__":
    main()
