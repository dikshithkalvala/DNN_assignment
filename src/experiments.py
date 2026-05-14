from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from src.dataset import build_dataloaders
from src.model import build_model
from src.train import train_model
from src.visualise import (
    collect_predictions,
    ensure_results_dir,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_histories,
    plot_per_class_accuracy,
    plot_prediction_distribution,
    plot_validation_story_strip,
    save_json,
    save_results_table,
    save_sample_predictions,
)


EXPERIMENTS = {
    "baseline": {
        "modification": "reference CNN",
        "model": {},
    },
    "dropout_030": {
        "modification": "add dropout p=0.30",
        "model": {"dropout": 0.30},
    },
    "larger_filters": {
        "modification": "increase filters from [16, 32, 64] to [32, 64, 128]",
        "model": {"filters": [32, 64, 128]},
    },
    "kernel_5": {
        "modification": "change convolution kernel size from 3 to 5",
        "model": {"kernel_size": 5},
    },
    "batch_norm": {
        "modification": "add batch normalization",
        "model": {"batch_norm": True},
    },
    "four_conv_layers": {
        "modification": "add a fourth convolution block",
        "model": {"filters": [16, 32, 64, 128]},
    },
}


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def merge_model_config(base_config: dict, updates: dict) -> dict:
    config = copy.deepcopy(base_config)
    config.update(updates)
    return config


def run_experiments(config: dict, selected: list[str] | None = None) -> tuple[list[dict], dict[str, dict]]:
    set_seed(config["dataset"]["seed"])
    results_dir = ensure_results_dir(config["output"]["results_dir"])
    train_loader, val_loader, metadata = build_dataloaders(config)
    plot_class_distribution(metadata, results_dir)

    experiment_names = selected or list(EXPERIMENTS.keys())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict] = []
    histories: dict[str, dict] = {}
    best_model = None
    best_accuracy = -1.0

    for name in experiment_names:
        experiment = EXPERIMENTS[name]
        model_config = merge_model_config(config["model"], experiment["model"])
        model = build_model(model_config).to(device)
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config["training"]["epochs"],
            learning_rate=config["training"]["learning_rate"],
            device=device,
        )

        history_dict = {
            "train_loss": history.train_loss,
            "val_loss": history.val_loss,
            "train_accuracy": history.train_accuracy,
            "val_accuracy": history.val_accuracy,
        }
        histories[name] = history_dict
        final_val_accuracy = history.val_accuracy[-1]

        rows.append(
            {
                "experiment": name,
                "modification": experiment["modification"],
                "train_loss": round(history.train_loss[-1], 4),
                "validation_loss": round(history.val_loss[-1], 4),
                "train_accuracy": round(history.train_accuracy[-1], 4),
                "validation_accuracy": round(final_val_accuracy, 4),
            }
        )

        if final_val_accuracy > best_accuracy:
            best_accuracy = final_val_accuracy
            best_model = model

    table = save_results_table(rows, results_dir)
    plot_histories(histories, results_dir)
    plot_validation_story_strip(metadata, results_dir)
    save_json(
        {
            "dataset": {key: value for key, value in metadata.items() if key != "val_items"},
            "device": str(device),
            "histories": histories,
            "results": rows,
        },
        results_dir / "metrics.json",
    )
    if best_model is not None:
        save_sample_predictions(best_model, val_loader, device, results_dir)
        labels, predictions = collect_predictions(best_model, val_loader, device)
        num_classes = config["model"]["num_classes"]
        plot_confusion_matrix(labels, predictions, results_dir, num_classes)
        plot_per_class_accuracy(labels, predictions, results_dir, num_classes)
        plot_prediction_distribution(labels, predictions, results_dir, num_classes)

    print("\nDataset summary")
    print({key: value for key, value in metadata.items() if key != "val_items"})
    print("\nResults")
    print(table.to_string(index=False))
    return rows, histories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run visual story position classification experiments.")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML file.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=list(EXPERIMENTS.keys()),
        default=None,
        help="Optional subset of experiments to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    run_experiments(config, args.experiments)


if __name__ == "__main__":
    main()
