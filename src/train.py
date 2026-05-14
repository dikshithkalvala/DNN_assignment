from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_accuracy: list[float] = field(default_factory=list)
    val_accuracy: list[float] = field(default_factory=list)


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
    predictions = logits.argmax(dim=1)
    correct = (predictions == labels).sum().item()
    return correct, labels.numel()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            correct, count = accuracy_from_logits(logits, labels)
            total_loss += loss.item() * batch_size
            total_correct += correct
            total_count += count

    return total_loss / total_count, total_correct / total_count


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    progress: bool = True,
) -> History:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = History()

    iterator = range(1, epochs + 1)
    if progress:
        iterator = tqdm(iterator, desc="epochs")

    for _epoch in iterator:
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device)

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.train_accuracy.append(train_accuracy)
        history.val_accuracy.append(val_accuracy)

    return history
