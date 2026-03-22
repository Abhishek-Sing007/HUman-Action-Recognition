from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from src.activity_recognition.config import AppConfig
from src.activity_recognition.train import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the activity recognition model.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--sequence-length", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AppConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        sequence_length=args.sequence_length,
        artifacts_dir=args.artifacts_dir,
    )
    results = train_model(args.data_dir, config)
    print(f"Saved checkpoint to {results['checkpoint_path']}")
    print(f"Best validation accuracy: {results['best_val_accuracy']:.4f}")
    print(f"Classes: {', '.join(results['classes'])}")


if __name__ == "__main__":
    main()
