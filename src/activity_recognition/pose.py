from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError as exc:
    raise ImportError(
        "MediaPipe is not installed. Activate your virtual environment and run: pip install -r requirements.txt"
    ) from exc

try:
    _pose_module = mp.solutions.pose
except AttributeError:
    try:
        from mediapipe.python.solutions import pose as _pose_module
    except ImportError as exc:
        raise ImportError(
            "Your MediaPipe installation does not expose Pose solutions correctly. "
            "Reinstall it with: pip uninstall mediapipe -y && pip install mediapipe==0.10.14"
        ) from exc


class PoseExtractor:
    def __init__(
        self,
        sequence_length: int,
        num_landmarks: int = 33,
        landmark_dims: int = 4,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.sequence_length = sequence_length
        self.num_landmarks = num_landmarks
        self.landmark_dims = landmark_dims
        self._pose = _pose_module.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    @property
    def feature_dim(self) -> int:
        return self.num_landmarks * self.landmark_dims

    def extract_from_frame(self, frame: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)
        if not result.pose_landmarks:
            return np.zeros(self.feature_dim, dtype=np.float32)

        landmarks = []
        for landmark in result.pose_landmarks.landmark[: self.num_landmarks]:
            landmarks.extend(
                [landmark.x, landmark.y, landmark.z, landmark.visibility]
            )
        return np.asarray(landmarks, dtype=np.float32)

    def extract_from_video(self, video_path: Path) -> np.ndarray:
        cap = cv2.VideoCapture(str(video_path))
        frames: list[np.ndarray] = []

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(self.extract_from_frame(frame))

        cap.release()
        return self._to_fixed_sequence(frames)

    def extract_from_frames(self, frames: Iterable[np.ndarray]) -> np.ndarray:
        features = [self.extract_from_frame(frame) for frame in frames]
        return self._to_fixed_sequence(features)

    def _to_fixed_sequence(self, frames: list[np.ndarray]) -> np.ndarray:
        if not frames:
            return np.zeros((self.sequence_length, self.feature_dim), dtype=np.float32)

        if len(frames) >= self.sequence_length:
            indices = np.linspace(
                0, len(frames) - 1, self.sequence_length, dtype=int
            )
            sampled = [frames[index] for index in indices]
        else:
            pad = [np.zeros(self.feature_dim, dtype=np.float32)] * (
                self.sequence_length - len(frames)
            )
            sampled = frames + pad

        return np.stack(sampled).astype(np.float32)
