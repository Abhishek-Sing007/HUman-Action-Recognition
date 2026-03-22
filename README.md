# Deep Learning Based Activity Recognition Application

This project detects human activities from video using pose estimation and a deep learning
sequence classifier. It is designed as a practical academic project that can be trained on
custom videos and also run in real time from a webcam.

## Features

- Pose-based activity recognition using MediaPipe landmarks
- GRU-based temporal classifier built with PyTorch
- Training, validation, and evaluation pipeline
- Real-time webcam inference with on-screen predictions
- Configurable activity classes such as `standing`, `sitting`, `walking`, `running`, `falling`
- Project report material covering objectives, methodology, outcomes, and schedule

## Project Structure

```text
.
├── docs/
│   └── project_report.md
├── requirements.txt
├── scripts/
│   ├── evaluate.py
│   ├── prepare_dataset.py
│   ├── run_webcam.py
│   └── train.py
└── src/
    └── activity_recognition/
        ├── __init__.py
        ├── config.py
        ├── dataset.py
        ├── inference.py
        ├── model.py
        ├── pose.py
        ├── train.py
        └── utils.py
```

## Dataset Layout

Store videos in this format:

```text
data/
├── raw/
│   ├── standing/
│   │   ├── sample_01.mp4
│   │   └── sample_02.mp4
│   ├── sitting/
│   ├── walking/
│   ├── running/
│   └── falling/
└── processed/
```

Each folder name becomes a class label.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Workflow

### 1. Prepare pose sequences from videos

```bash
python scripts/prepare_dataset.py \
  --input-dir data/raw \
  --output-dir data/processed \
  --sequence-length 30
```

This extracts pose landmarks from each video and saves `.npz` sequence files.

### 2. Train the model

```bash
python scripts/train.py \
  --data-dir data/processed \
  --epochs 25 \
  --batch-size 16
```

The trained model and metadata will be stored in `artifacts/`.

### 3. Evaluate the model

```bash
python scripts/evaluate.py \
  --data-dir data/processed \
  --checkpoint artifacts/activity_gru.pt
```

### 4. Run real-time webcam inference

```bash
python scripts/run_webcam.py \
  --checkpoint artifacts/activity_gru.pt
```

## Model Design

- Input: sequences of 2D/3D pose landmarks from MediaPipe
- Temporal encoder: stacked GRU
- Output: activity class probabilities

This design keeps the model lightweight enough for classroom demos and edge-friendly
deployment while still modeling motion over time.

## Evaluation Metrics

The project reports:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

## Notes

- For strong results, collect balanced video samples per class with multiple people,
  backgrounds, and lighting conditions.
- If you want to deploy to web or mobile later, the pose extraction and GRU model can be
  exported or replaced with a smaller classifier.
- A CPU-only workflow is supported, though training will be slower.
