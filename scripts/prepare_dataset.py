from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
from tqdm import tqdm

from src.activity_recognition.pose import PoseExtractor
from src.activity_recognition.utils import ensure_dir


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract pose sequences from class-organized video folders."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    extractor = PoseExtractor(sequence_length=args.sequence_length)

    class_dirs = sorted(path for path in args.input_dir.iterdir() if path.is_dir())
    if not class_dirs:
        raise FileNotFoundError(
            f"No class folders found in {args.input_dir}. Expected folders per activity."
        )

    sample_count = 0
    for class_dir in class_dirs:
        label = class_dir.name
        videos = sorted(
            path for path in class_dir.iterdir() if path.suffix.lower() in VIDEO_EXTENSIONS
        )
        for video_path in tqdm(videos, desc=f"Processing {label}", leave=False):
            sequence = extractor.extract_from_video(video_path)
            output_name = f"{label}_{video_path.stem}.npz"
            np.savez_compressed(
                args.output_dir / output_name,
                sequence=sequence,
                label=label,
            )
            sample_count += 1

    print(f"Saved {sample_count} processed samples to {args.output_dir}")


if __name__ == "__main__":
    main()
