from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AppConfig:
    sequence_length: int = 30
    num_pose_landmarks: int = 33
    landmark_dims: int = 4
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.25
    batch_size: int = 16
    learning_rate: float = 1e-3
    epochs: int = 25
    train_split: float = 0.8
    seed: int = 42
    device: str = "cpu"
    artifacts_dir: Path = field(default_factory=lambda: Path("artifacts"))

    @property
    def feature_dim(self) -> int:
        return self.num_pose_landmarks * self.landmark_dims
