from __future__ import annotations

import torch
from torch import nn


class StoryPositionCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        filters: list[int] | tuple[int, ...] = (16, 32, 64),
        kernel_size: int = 3,
        dropout: float = 0.0,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = 3
        padding = kernel_size // 2

        for out_channels in filters:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding))
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.extend([nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2)])
            in_channels = out_channels

        self.features = nn.Sequential(*layers)
        classifier_layers: list[nn.Module] = [nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()]
        if dropout > 0:
            classifier_layers.append(nn.Dropout(dropout))
        classifier_layers.append(nn.Linear(filters[-1], num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        return self.classifier(features)


def build_model(model_config: dict) -> StoryPositionCNN:
    return StoryPositionCNN(
        num_classes=model_config["num_classes"],
        filters=model_config["filters"],
        kernel_size=model_config["kernel_size"],
        dropout=model_config["dropout"],
        batch_norm=model_config["batch_norm"],
    )
