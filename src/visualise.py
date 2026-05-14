from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image


def ensure_results_dir(results_dir: str | Path) -> Path:
    path = Path(results_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def plot_class_distribution(metadata: dict, results_dir: Path) -> None:
    labels = sorted(metadata["train_distribution"].keys())
    train_counts = [metadata["train_distribution"][label] for label in labels]
    val_counts = [metadata["val_distribution"][label] for label in labels]

    x_positions = range(len(labels))
    plt.figure(figsize=(8, 5))
    plt.bar([x - 0.2 for x in x_positions], train_counts, width=0.4, label="Train")
    plt.bar([x + 0.2 for x in x_positions], val_counts, width=0.4, label="Validation")
    plt.xticks(list(x_positions), labels)
    plt.xlabel("Story position label")
    plt.ylabel("Number of images")
    plt.title("Class distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "class_distribution.png", dpi=160)
    plt.close()


def plot_histories(histories: dict[str, dict], results_dir: Path) -> None:
    plt.figure(figsize=(10, 6))
    for name, history in histories.items():
        plt.plot(history["train_loss"], linestyle="--", label=f"{name} train")
        plt.plot(history["val_loss"], label=f"{name} val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross entropy loss")
    plt.title("Training and validation loss")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(results_dir / "loss_curves.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 6))
    for name, history in histories.items():
        plt.plot(history["train_accuracy"], linestyle="--", label=f"{name} train")
        plt.plot(history["val_accuracy"], label=f"{name} val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and validation accuracy")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(results_dir / "accuracy_curves.png", dpi=160)
    plt.close()


def save_results_table(rows: list[dict], results_dir: Path) -> pd.DataFrame:
    table = pd.DataFrame(rows)
    table.to_csv(results_dir / "results_table.csv", index=False)
    return table


def save_sample_predictions(model: torch.nn.Module, val_loader, device: torch.device, results_dir: Path) -> None:
    model.eval()
    images, labels = next(iter(val_loader))
    images = images.to(device)
    labels = labels.to(device)
    with torch.no_grad():
        predictions = model(images).argmax(dim=1)

    images = images.cpu()
    labels = labels.cpu()
    predictions = predictions.cpu()

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    count = min(12, images.size(0))
    plt.figure(figsize=(12, 8))
    for index in range(count):
        image = (images[index] * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        plt.subplot(3, 4, index + 1)
        plt.imshow(image)
        plt.axis("off")
        plt.title(f"true {labels[index].item() + 1} / pred {predictions[index].item() + 1}")
    plt.tight_layout()
    plt.savefig(results_dir / "sample_predictions.png", dpi=160)
    plt.close()


def collect_predictions(model: torch.nn.Module, val_loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_labels: list[torch.Tensor] = []
    all_predictions: list[torch.Tensor] = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits = model(images)
            all_predictions.append(logits.argmax(dim=1).cpu())
            all_labels.append(labels.cpu())

    labels_array = torch.cat(all_labels).numpy()
    predictions_array = torch.cat(all_predictions).numpy()
    return labels_array, predictions_array


def plot_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    results_dir: Path,
    num_classes: int = 5,
) -> None:
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, predicted_label in zip(labels, predictions):
        matrix[true_label, predicted_label] += 1

    plt.figure(figsize=(7, 6))
    plt.imshow(matrix, cmap="Blues")
    plt.colorbar(label="Number of images")
    plt.xticks(range(num_classes), range(1, num_classes + 1))
    plt.yticks(range(num_classes), range(1, num_classes + 1))
    plt.xlabel("Predicted position")
    plt.ylabel("True position")
    plt.title("Validation confusion matrix")

    threshold = matrix.max() / 2 if matrix.max() else 0
    for row in range(num_classes):
        for col in range(num_classes):
            color = "white" if matrix[row, col] > threshold else "black"
            plt.text(col, row, str(matrix[row, col]), ha="center", va="center", color=color)

    plt.tight_layout()
    plt.savefig(results_dir / "confusion_matrix.png", dpi=160)
    plt.close()


def plot_per_class_accuracy(
    labels: np.ndarray,
    predictions: np.ndarray,
    results_dir: Path,
    num_classes: int = 5,
) -> None:
    class_accuracies = []
    for label in range(num_classes):
        mask = labels == label
        accuracy = float((predictions[mask] == label).mean()) if mask.any() else 0.0
        class_accuracies.append(accuracy)

    plt.figure(figsize=(8, 5))
    plt.bar(range(1, num_classes + 1), class_accuracies, color="#3b82f6")
    plt.axhline(1 / num_classes, color="#ef4444", linestyle="--", label="Chance level")
    plt.ylim(0, 1)
    plt.xlabel("True story position")
    plt.ylabel("Accuracy")
    plt.title("Per-class validation accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "per_class_accuracy.png", dpi=160)
    plt.close()


def plot_prediction_distribution(
    labels: np.ndarray,
    predictions: np.ndarray,
    results_dir: Path,
    num_classes: int = 5,
) -> None:
    true_counts = np.bincount(labels, minlength=num_classes)
    predicted_counts = np.bincount(predictions, minlength=num_classes)
    x_positions = np.arange(num_classes)

    plt.figure(figsize=(8, 5))
    plt.bar(x_positions - 0.2, true_counts, width=0.4, label="True labels")
    plt.bar(x_positions + 0.2, predicted_counts, width=0.4, label="Predictions")
    plt.xticks(x_positions, range(1, num_classes + 1))
    plt.xlabel("Story position")
    plt.ylabel("Number of images")
    plt.title("Validation prediction distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "prediction_distribution.png", dpi=160)
    plt.close()


def plot_validation_story_strip(metadata: dict, results_dir: Path) -> None:
    val_items = metadata.get("val_items", [])
    if not val_items:
        return

    story_id = val_items[0].story_id
    story_items = sorted(
        [item for item in val_items if item.story_id == story_id],
        key=lambda item: item.position,
    )
    if not story_items:
        return

    plt.figure(figsize=(14, 4))
    for index, item in enumerate(story_items):
        image = Image.open(item.image_path).convert("RGB")
        plt.subplot(1, len(story_items), index + 1)
        plt.imshow(image)
        plt.axis("off")
        plt.title(f"Position {item.position}")
    plt.suptitle(f"Example validation story {story_id}")
    plt.tight_layout()
    plt.savefig(results_dir / "validation_story_strip.png", dpi=160)
    plt.close()
