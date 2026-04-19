# Emotion Detection using Deep Learning

**B.Tech Final Year Project — G. L. Bajaj Institute of Technology and Management, Greater Noida**  
Electronics and Communication Engineering | Session 2022-23

**Team:** Aman Gangwar · Ankit Raj · Duvesh Chauhan · Nidhish Kumar Singh  
**Guide:** Dr. Mohan Singh (Associate Professor, ECE)

---

## Overview

Real-time facial emotion recognition system that:
- Classifies **7 emotions** — Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- Detects **Age** (8 groups) and **Gender** simultaneously
- Runs on live webcam, video files, or static images

### Architecture
| Component | Model | Input |
|-----------|-------|-------|
| Emotion   | 4-block VGG-style CNN | 48×48 grayscale |
| Age/Gender | Broad-ResNet (dual-head) | 224×224 RGB |
| Face detector | Haar Cascade (OpenCV) | BGR frame |

### Dataset
**FER2013** — 35,887 labelled grayscale face images (48×48 px)
- Train: 28,709 · Val: 3,589 · Test: 3,589
- Source: [Kaggle — msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013)

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the dataset
```bash
# Option A — Kaggle API (recommended)
pip install kaggle
# Place ~/.kaggle/kaggle.json first, then:
python download_dataset.py

# Option B — Manual
# Download fer2013.csv from Kaggle and place it at:  data/fer2013.csv
```

### 3. Train the model
```bash
python -m src.train --data data/fer2013.csv
# With custom options:
python -m src.train --data data/fer2013.csv --epochs 100 --batch_size 64 --lr 0.001
```

Training artefacts are saved to `models/`:
- `emotion_model_best.keras` — best validation checkpoint
- `training_history.png` — accuracy & loss curves
- `class_distribution.png` — dataset balance chart

### 4. Evaluate
```bash
python -m src.evaluate --model models/emotion_model_best.keras --data data/fer2013.csv
```
Outputs: confusion matrix, per-class accuracy bar, Grad-CAM visualisations.

### 5. Run real-time detection
```bash
# Webcam (default)
python app.py

# Single image
python app.py --mode image --source photo.jpg

# Video file
python app.py --mode video --source video.mp4

# Skip age/gender if Caffe models not downloaded
python app.py --no_age_gender

# Save output
python app.py --mode video --source video.mp4 --save output.mp4
```

---

## Project Structure

```
Emotion_Detection/
├── src/
│   ├── model.py          # CNN (emotion) + Broad-ResNet (age/gender) architectures
│   ├── train.py          # Training with augmentation, callbacks, class weights
│   ├── evaluate.py       # Confusion matrix, Grad-CAM, per-class accuracy
│   └── utils.py          # Data loading (CSV & dir), augmentation, visualisation
├── app.py                # Real-time detection app (webcam / image / video)
├── download_dataset.py   # FER2013 + Age/Gender model downloader
├── requirements.txt
├── models/               # Saved model checkpoints (auto-created)
└── data/                 # Dataset files (place fer2013.csv here)
```

---

## Training Details

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Initial LR | 1e-3 |
| LR schedule | ReduceLROnPlateau (×0.5, patience=5) |
| Batch size | 64 |
| Max epochs | 80 (early stopping, patience=15) |
| Regularisation | L2 (1e-4), Dropout 0.25 / 0.5 |
| Class weights | Balanced (sklearn) |

**Data augmentation applied during training:**
- Horizontal flip
- Rotation ±15°
- Width/height shift ±10%
- Zoom ±10%

---

## Age/Gender Models (Optional)

The app integrates pre-trained Caffe models by Gil Levi & Tal Hassner (2015):

```bash
python download_dataset.py --ag_models
```

This downloads four files into `models/`:
`age_deploy.prototxt`, `age_net.caffemodel`, `gender_deploy.prototxt`, `gender_net.caffemodel`

---

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | ~65–70% (FER2013 is hard; human-level ≈ 65%) |
| Best emotions | Happy, Surprise, Neutral |
| Hardest emotions | Disgust, Fear (fewer training samples) |

---

## References

1. Goodfellow et al. — Challenges in Representation Learning: A report on three machine learning contests (FER2013), 2013
2. Levi & Hassner — Age and Gender Classification using Convolutional Neural Networks, CVPR 2015
3. Simonyan & Zisserman — Very Deep Convolutional Networks for Large-Scale Image Recognition (VGGNet), ICLR 2015
4. He et al. — Deep Residual Learning for Image Recognition (ResNet), CVPR 2016
5. Selvaraju et al. — Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization, ICCV 2017
