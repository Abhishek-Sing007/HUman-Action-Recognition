from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class PoseSequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray) -> None:
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[index], self.labels[index]


def load_processed_dataset(data_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    files = sorted(data_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(
            f"No processed samples found in {data_dir}. Run prepare_dataset.py first."
        )

    sequences = []
    labels = []
    class_names: set[str] = set()

    for file_path in files:
        payload = np.load(file_path, allow_pickle=False)
        sequences.append(payload["sequence"])
        labels.append(str(payload["label"]))
        class_names.add(str(payload["label"]))

    classes = sorted(class_names)
    class_to_index = {label: idx for idx, label in enumerate(classes)}
    numeric_labels = np.asarray([class_to_index[label] for label in labels], dtype=int)
    stacked = np.stack(sequences).astype(np.float32)
    return stacked, numeric_labels, classes


def split_dataset(
    sequences: np.ndarray,
    labels: np.ndarray,
    train_split: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    counts = Counter(labels.tolist())
    can_stratify = len(counts) > 1 and min(counts.values()) >= 2
    stratify = labels if can_stratify else None
    return train_test_split(
        sequences,
        labels,
        train_size=train_split,
        random_state=seed,
        stratify=stratify,
    )
