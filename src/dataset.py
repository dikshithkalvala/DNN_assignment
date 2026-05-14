from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


@dataclass(frozen=True)
class StoryItem:
    image_path: Path
    label: int
    story_id: int
    position: int


class StoryPositionDataset(Dataset):
    def __init__(self, items: list[StoryItem], image_size: int, cache_images: bool = True) -> None:
        self.items = items
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.cached_images = [self._load_image(item) for item in items] if cache_images else None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.items[index]
        if self.cached_images is not None:
            return self.cached_images[index], torch.tensor(item.label, dtype=torch.long)
        return self._load_image(item), torch.tensor(item.label, dtype=torch.long)

    def _load_image(self, item: StoryItem) -> torch.Tensor:
        image = Image.open(item.image_path).convert("RGB")
        return self.transform(image)


def discover_story_items(root: str | Path, story_length: int = 5) -> list[StoryItem]:
    root_path = Path(root)
    image_paths = sorted(
        [path for path in root_path.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS],
        key=natural_key,
    )
    usable_count = (len(image_paths) // story_length) * story_length
    image_paths = image_paths[:usable_count]

    items: list[StoryItem] = []
    for index, image_path in enumerate(image_paths):
        story_id = index // story_length
        position = index % story_length
        items.append(StoryItem(image_path=image_path, label=position, story_id=story_id, position=position + 1))
    return items


def split_by_story(
    items: list[StoryItem],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[StoryItem], list[StoryItem]]:
    story_ids = sorted({item.story_id for item in items})
    generator = torch.Generator().manual_seed(seed)
    shuffled = torch.tensor(story_ids)[torch.randperm(len(story_ids), generator=generator)].tolist()
    val_count = max(1, round(len(story_ids) * val_fraction))
    val_ids = set(shuffled[:val_count])

    train_items = [item for item in items if item.story_id not in val_ids]
    val_items = [item for item in items if item.story_id in val_ids]
    return train_items, val_items


def class_distribution(items: list[StoryItem], num_classes: int = 5) -> dict[int, int]:
    counts = {label + 1: 0 for label in range(num_classes)}
    for item in items:
        counts[item.label + 1] += 1
    return counts


def build_dataloaders(config: dict) -> tuple[DataLoader, DataLoader, dict[str, object]]:
    dataset_cfg = config["dataset"]
    training_cfg = config["training"]

    items = discover_story_items(dataset_cfg["root"], dataset_cfg["story_length"])
    train_items, val_items = split_by_story(items, dataset_cfg["val_fraction"], dataset_cfg["seed"])

    train_dataset = StoryPositionDataset(
        train_items,
        dataset_cfg["image_size"],
        dataset_cfg.get("cache_images", True),
    )
    val_dataset = StoryPositionDataset(
        val_items,
        dataset_cfg["image_size"],
        dataset_cfg.get("cache_images", True),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=True,
        num_workers=training_cfg.get("num_workers", 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=False,
        num_workers=training_cfg.get("num_workers", 0),
    )

    metadata = {
        "total_images": len(items),
        "total_stories": len({item.story_id for item in items}),
        "train_images": len(train_items),
        "val_images": len(val_items),
        "train_stories": len({item.story_id for item in train_items}),
        "val_stories": len({item.story_id for item in val_items}),
        "train_distribution": class_distribution(train_items, config["model"]["num_classes"]),
        "val_distribution": class_distribution(val_items, config["model"]["num_classes"]),
        "val_items": val_items,
    }
    return train_loader, val_loader, metadata
