from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from src.activity_recognition.inference import run_webcam


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real-time webcam activity recognition."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--camera-id", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_webcam(args.checkpoint, camera_id=args.camera_id)


if __name__ == "__main__":
    main()
