from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import AppConfig
from .dataset import PoseSequenceDataset, load_processed_dataset, split_dataset
from .model import ActivityGRU
from .utils import ensure_dir, save_json, set_seed


def train_model(data_dir: Path, config: AppConfig) -> dict[str, object]:
    set_seed(config.seed)
    ensure_dir(config.artifacts_dir)

    sequences, labels, classes = load_processed_dataset(data_dir)
    x_train, x_val, y_train, y_val = split_dataset(
        sequences, labels, config.train_split, config.seed
    )

    train_loader = DataLoader(
        PoseSequenceDataset(x_train, y_train),
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        PoseSequenceDataset(x_val, y_val),
        batch_size=config.batch_size,
        shuffle=False,
    )

    device = torch.device(
        "cuda" if config.device == "cuda" and torch.cuda.is_available() else "cpu"
    )

    model = ActivityGRU(
        input_size=config.feature_dim,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_classes=len(classes),
        dropout=config.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    best_val_acc = 0.0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []

    for epoch in range(config.epochs):
        model.train()
        train_losses = []

        for batch_x, batch_y in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}", leave=False
        ):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        val_metrics = evaluate_model(model, val_loader, device, classes)
        val_acc = float(val_metrics["accuracy"])
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": float(np.mean(train_losses) if train_losses else 0.0),
                "val_accuracy": val_acc,
            }
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state.")

    checkpoint_path = config.artifacts_dir / "activity_gru.pt"
    checkpoint_config = asdict(config)
    checkpoint_config["artifacts_dir"] = str(checkpoint_config["artifacts_dir"])
    torch.save(
        {
            "model_state_dict": best_state,
            "classes": classes,
            "config": checkpoint_config,
        },
        checkpoint_path,
    )

    metadata = {
        "classes": classes,
        "best_val_accuracy": best_val_acc,
        "history": history,
    }
    save_json(config.artifacts_dir / "training_summary.json", metadata)
    return {
        "checkpoint_path": str(checkpoint_path),
        "classes": classes,
        "best_val_accuracy": best_val_acc,
        "history": history,
    }


def evaluate_checkpoint(
    data_dir: Path,
    checkpoint_path: Path,
    batch_size: int = 16,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    classes = checkpoint["classes"]
    config_data = checkpoint["config"]

    sequences, labels, _ = load_processed_dataset(data_dir)
    dataset = PoseSequenceDataset(sequences, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = ActivityGRU(
        input_size=config_data["num_pose_landmarks"] * config_data["landmark_dims"],
        hidden_size=config_data["hidden_size"],
        num_layers=config_data["num_layers"],
        num_classes=len(classes),
        dropout=config_data["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return evaluate_model(model, loader, torch.device("cpu"), classes)


def evaluate_model(
    model: ActivityGRU,
    loader: DataLoader,
    device: torch.device,
    classes: list[str],
) -> dict[str, object]:
    model.eval()
    all_preds = []
    all_targets = []
    label_ids = list(range(len(classes)))

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(batch_y.numpy().tolist())

    accuracy = accuracy_score(all_targets, all_preds)
    report = classification_report(
        all_targets,
        all_preds,
        labels=label_ids,
        target_names=classes,
        zero_division=0,
        output_dict=True,
    )
    matrix = confusion_matrix(all_targets, all_preds, labels=label_ids).tolist()
    return {
        "accuracy": float(accuracy),
        "classification_report": report,
        "confusion_matrix": matrix,
    }
