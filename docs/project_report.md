# Project Report

## Title

Deep Learning Based Activity Recognition Application

## Abstract

Human activity detection is increasingly important in workplace safety, smart surveillance,
fitness monitoring, pose estimation, gesture recognition, and interactive systems. Manual
monitoring is slow, expensive, and difficult to scale. This project presents a deep learning
based activity recognition application that uses visual input and pose estimation to detect
human activities in real time.

The proposed system extracts human skeletal keypoints from video using a pose estimation
algorithm and then feeds those temporal pose sequences into a recurrent neural network for
classification. The model learns motion patterns and posture transitions for activities such as
standing, sitting, walking, running, exercising, bending, and falling. The application is built to
operate on both prerecorded videos and live camera feeds.

This project demonstrates how deep learning can improve machine perception of dynamic
actions while remaining practical for real-world use. It also provides a foundation for future
extensions such as mobile deployment, action forecasting, and multimodal activity analysis.

## Objectives

- Develop an intelligent system for identifying human activities such as standing, sitting,
  bending, lying down, walking, running, and falling.
- Achieve real-time activity recognition from video files or live webcam feeds.
- Build an application useful for safety monitoring, surveillance, and gesture-aware systems.
- Evaluate performance using accuracy, precision, recall, F1-score, and confusion matrix.
- Keep the model lightweight enough for future web or mobile deployment.
- Create a baseline that can be improved with new data and user feedback.

## Social And Environmental Justification

- Safety: the system can support fall detection and unsafe posture monitoring.
- Workplace safety: the application can help detect risky body movements in industrial
  settings.
- Fitness and wellness: pose-aware monitoring can support exercise feedback and posture
  correction.
- Environmental impact: camera-based monitoring reduces dependence on extra hardware
  sensors and lowers electronic waste.
- Global societal value: scalable AI monitoring can contribute to smart cities, health support,
  and safer public spaces.

## Methodology

1. Use visual input from prerecorded videos or live camera streams.
2. Extract pose landmarks from each frame using MediaPipe Pose.
3. Convert frame-wise landmarks into fixed-length temporal sequences.
4. Train a GRU-based deep learning classifier on those pose sequences.
5. Evaluate the classifier on unseen samples using standard classification metrics.
6. Integrate the trained model into a real-time inference application.
7. Refine the model based on errors, data quality, and latency constraints.

## Hardware And Software Requirements

### Software

- Python 3.10+
- PyTorch
- MediaPipe
- OpenCV
- NumPy
- scikit-learn
- Matplotlib

### Hardware

- Laptop or desktop with webcam
- Optional GPU for faster training
- Minimum 8 GB RAM recommended

## Expected Outcomes

- Accurate recognition of multiple human activities from video.
- Real-time monitoring through a webcam-based application.
- Robustness across moderate changes in lighting and background.
- A reusable academic project structure for future deployment and research.

## 14-Week Schedule

| Week | Task |
| --- | --- |
| 1 | Finalize objectives, scope, and literature review |
| 2 | Compare deep learning architectures for activity recognition |
| 3 | Collect and organize datasets |
| 4 | Preprocess videos and extract frames or landmarks |
| 5 | Design model architecture |
| 6 | Build initial prototype and run preliminary training |
| 7 | Optimize model and train on full dataset |
| 8 | Evaluate using classification metrics |
| 9 | Integrate the trained model into the application |
| 10 | Add real-time webcam prediction |
| 11 | Test under varying conditions |
| 12 | Improve latency and robustness |
| 13 | Perform final simulations and document findings |
| 14 | Demonstrate the application and finalize deliverables |

## Current Implementation Summary

The current implementation uses pose estimation plus a GRU classifier. This is a strong
baseline because it focuses on skeletal motion rather than raw pixels, which improves
efficiency and helps the model generalize better with smaller datasets.
