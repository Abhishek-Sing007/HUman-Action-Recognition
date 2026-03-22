from __future__ import annotations

from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from .model import ActivityGRU
from .pose import PoseExtractor


class RealtimeActivityRecognizer:
    def __init__(self, checkpoint_path: Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.classes: list[str] = checkpoint["classes"]
        config = checkpoint["config"]

        self.extractor = PoseExtractor(
            sequence_length=config["sequence_length"],
            num_landmarks=config["num_pose_landmarks"],
            landmark_dims=config["landmark_dims"],
        )
        self.sequence = deque(maxlen=config["sequence_length"])
        self.model = ActivityGRU(
            input_size=config["num_pose_landmarks"] * config["landmark_dims"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            num_classes=len(self.classes),
            dropout=config["dropout"],
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict_frame(self, frame: np.ndarray) -> tuple[str, float]:
        features = self.extractor.extract_from_frame(frame)
        self.sequence.append(features)
        if len(self.sequence) < self.sequence.maxlen:
            return "collecting...", 0.0

        sequence_array = np.stack(self.sequence).astype(np.float32)
        tensor = torch.tensor(sequence_array[None, ...], dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            index = int(torch.argmax(probs).item())
            return self.classes[index], float(probs[index].item())


def run_webcam(checkpoint_path: Path, camera_id: int = 0) -> None:
    recognizer = RealtimeActivityRecognizer(checkpoint_path)
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        label, confidence = recognizer.predict_frame(frame)
        overlay = f"Activity: {label} ({confidence:.2f})"
        cv2.putText(
            frame,
            overlay,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Activity Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
