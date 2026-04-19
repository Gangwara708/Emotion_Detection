"""
Data loading, preprocessing, and augmentation utilities for FER2013.

FER2013 formats supported:
  1. CSV  — fer2013.csv with columns: emotion | pixels | Usage
  2. Dir  — data/train/<emotion>/ and data/test/<emotion>/
"""

import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.model import EMOTIONS, NUM_CLASSES, IMG_SIZE


# ── Dataset loading ────────────────────────────────────────────────────────────

def load_fer2013_csv(csv_path: str):
    """
    Load FER2013 from the original CSV file.
    Returns (X_train, y_train), (X_val, y_val), (X_test, y_test)
    as float32 arrays normalised to [0, 1].
    """
    print(f'Loading FER2013 from {csv_path} ...')
    df = pd.read_csv(csv_path)

    def _parse(row):
        pixels = np.array(row['pixels'].split(), dtype='float32')
        return pixels.reshape(IMG_SIZE, IMG_SIZE, 1) / 255.0

    X = np.array([_parse(r) for _, r in df.iterrows()])
    y = to_categorical(df['emotion'].values, NUM_CLASSES)

    train_mask = df['Usage'] == 'Training'
    val_mask   = df['Usage'] == 'PublicTest'
    test_mask  = df['Usage'] == 'PrivateTest'

    return (
        (X[train_mask], y[train_mask]),
        (X[val_mask],   y[val_mask]),
        (X[test_mask],  y[test_mask]),
    )


def load_fer2013_dirs(data_dir: str, val_split: float = 0.1):
    """
    Load FER2013 from a directory layout:
        data_dir/train/<emotion_name>/*.jpg
        data_dir/test/<emotion_name>/*.jpg
    """
    emotion_map = {name.lower(): i for i, name in enumerate(EMOTIONS)}

    def _load_split(split_dir):
        images, labels = [], []
        for emotion_name in os.listdir(split_dir):
            idx = emotion_map.get(emotion_name.lower())
            if idx is None:
                continue
            folder = os.path.join(split_dir, emotion_name)
            for fname in os.listdir(folder):
                img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(img.reshape(IMG_SIZE, IMG_SIZE, 1) / 255.0)
                labels.append(idx)
        return np.array(images, dtype='float32'), to_categorical(labels, NUM_CLASSES)

    train_dir = os.path.join(data_dir, 'train')
    test_dir  = os.path.join(data_dir, 'test')

    X_all, y_all = _load_split(train_dir)
    X_test, y_test = _load_split(test_dir)

    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=val_split, random_state=42, stratify=y_all.argmax(1)
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def load_dataset(data_path: str, val_split: float = 0.1):
    """Auto-detect format and load dataset."""
    if os.path.isfile(data_path) and data_path.endswith('.csv'):
        return load_fer2013_csv(data_path)
    elif os.path.isdir(data_path):
        return load_fer2013_dirs(data_path, val_split)
    else:
        raise FileNotFoundError(
            f'Dataset not found at "{data_path}". '
            'Run download_dataset.py first or place fer2013.csv here.'
        )


# ── Data augmentation ──────────────────────────────────────────────────────────

def get_train_generator(X_train, y_train, batch_size: int = 64):
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
    )
    datagen.fit(X_train)
    return datagen.flow(X_train, y_train, batch_size=batch_size, shuffle=True)


def get_val_generator(X_val, y_val, batch_size: int = 64):
    datagen = ImageDataGenerator()
    return datagen.flow(X_val, y_val, batch_size=batch_size, shuffle=False)


# ── Face detection helper (for real-time app) ─────────────────────────────────

def get_face_detector(cascade_path: str = None):
    if cascade_path is None:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    return cv2.CascadeClassifier(cascade_path)


def detect_faces(frame_gray, detector, scale=1.3, min_neighbors=5):
    return detector.detectMultiScale(
        frame_gray, scaleFactor=scale, minNeighbors=min_neighbors,
        minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
    )


def preprocess_face(face_roi_gray):
    face = cv2.resize(face_roi_gray, (IMG_SIZE, IMG_SIZE))
    face = face.astype('float32') / 255.0
    return face.reshape(1, IMG_SIZE, IMG_SIZE, 1)


# ── Visualisation helpers ──────────────────────────────────────────────────────

def plot_class_distribution(y_labels: np.ndarray, title: str = 'Class Distribution',
                             save_path: str = None):
    counts = np.bincount(y_labels, minlength=NUM_CLASSES)
    plt.figure(figsize=(10, 5))
    bars = plt.bar(EMOTIONS, counts, color=sns.color_palette('Set2', NUM_CLASSES))
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Emotion')
    plt.ylabel('Count')
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                 str(count), ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_sample_images(X, y, n_per_class: int = 3, save_path: str = None):
    fig, axes = plt.subplots(NUM_CLASSES, n_per_class,
                             figsize=(n_per_class * 2, NUM_CLASSES * 2))
    y_idx = y.argmax(1) if y.ndim == 2 else y
    for cls_i, emotion in enumerate(EMOTIONS):
        idxs = np.where(y_idx == cls_i)[0][:n_per_class]
        for col, idx in enumerate(idxs):
            ax = axes[cls_i][col]
            ax.imshow(X[idx].squeeze(), cmap='gray')
            ax.axis('off')
            if col == 0:
                ax.set_ylabel(emotion, fontsize=9, rotation=0, labelpad=40, va='center')
    plt.suptitle('Sample Images per Emotion', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_training_history(history, save_path: str = None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Model Accuracy', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Model Loss', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def draw_emotion_bar(frame, emotions_prob: np.ndarray, x_offset: int = 10, y_offset: int = 10):
    """Overlay a probability bar chart on a BGR frame (in-place)."""
    bar_w, bar_max_h = 15, 60
    for i, (emotion, prob) in enumerate(zip(EMOTIONS, emotions_prob)):
        bar_h = int(prob * bar_max_h)
        x = x_offset + i * (bar_w + 4)
        y = y_offset + bar_max_h - bar_h
        color = (0, 255, 0) if i == emotions_prob.argmax() else (200, 200, 200)
        cv2.rectangle(frame, (x, y), (x + bar_w, y_offset + bar_max_h), color, -1)
        cv2.putText(frame, emotion[0], (x, y_offset + bar_max_h + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    return frame
