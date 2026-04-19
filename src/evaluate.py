"""
Evaluation script — confusion matrix, per-class accuracy, and grad-CAM visualisation.

Usage:
    python -m src.evaluate --model models/emotion_model_best.keras --data data/fer2013.csv
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)

from src.model import EMOTIONS, NUM_CLASSES, IMG_SIZE
from src.utils import load_dataset


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate Emotion Detection CNN')
    p.add_argument('--model', type=str, default='models/emotion_model_best.keras')
    p.add_argument('--data',  type=str, default='data/fer2013.csv')
    p.add_argument('--output_dir', type=str, default='models/')
    return p.parse_args()


# ── Confusion matrix ───────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ['d', '.2f'],
        ['Confusion Matrix (counts)', 'Confusion Matrix (normalised)']
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=EMOTIONS, yticklabels=EMOTIONS, ax=ax)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_per_class_accuracy(y_true, y_pred, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    plt.figure(figsize=(10, 5))
    colors = ['#2ecc71' if a >= 0.7 else '#e74c3c' for a in per_class_acc]
    bars = plt.bar(EMOTIONS, per_class_acc * 100, color=colors, edgecolor='white')
    plt.axhline(y=70, color='black', linestyle='--', linewidth=1, alpha=0.5, label='70% threshold')
    plt.ylim(0, 105)
    plt.title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Emotion')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    for bar, acc in zip(bars, per_class_acc):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{acc * 100:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# ── Grad-CAM ───────────────────────────────────────────────────────────────────

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        predicted_class = tf.argmax(predictions[0])
        loss = predictions[:, predicted_class]

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(predicted_class)


def _find_last_conv(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError('No Conv2D layer found in model.')


def plot_gradcam(model, X_samples, y_samples, n=5, save_path=None):
    last_conv = _find_last_conv(model)
    fig, axes = plt.subplots(2, n, figsize=(n * 2.5, 5))

    for i in range(n):
        img = X_samples[i:i + 1]
        heatmap, pred = make_gradcam_heatmap(img, model, last_conv)

        img_disp = (img[0, :, :, 0] * 255).astype('uint8')
        heatmap_resized = np.uint8(255 * heatmap)
        import cv2
        heatmap_color = cv2.applyColorMap(
            cv2.resize(heatmap_resized, (IMG_SIZE, IMG_SIZE)), cv2.COLORMAP_JET
        )
        img_rgb = cv2.cvtColor(img_disp, cv2.COLOR_GRAY2RGB)
        superimposed = cv2.addWeighted(img_rgb, 0.6, heatmap_color, 0.4, 0)

        true_lbl = EMOTIONS[y_samples[i].argmax()]
        pred_lbl = EMOTIONS[pred]

        axes[0][i].imshow(img_disp, cmap='gray')
        axes[0][i].set_title(f'True: {true_lbl}', fontsize=8)
        axes[0][i].axis('off')

        axes[1][i].imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
        axes[1][i].set_title(f'Pred: {pred_lbl}', fontsize=8,
                              color='green' if true_lbl == pred_lbl else 'red')
        axes[1][i].axis('off')

    plt.suptitle('Grad-CAM Visualisations', fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# ── Main evaluation ────────────────────────────────────────────────────────────

def evaluate(args):
    print('=' * 60)
    print('  Emotion Detection CNN — Evaluation')
    print('=' * 60)

    model = load_model(args.model)
    print(f'Model loaded from: {args.model}')

    _, _, (X_test, y_test) = load_dataset(args.data)
    print(f'Test samples: {X_test.shape[0]:,}')

    y_true = y_test.argmax(1)
    y_prob = model.predict(X_test, batch_size=64, verbose=1)
    y_pred = y_prob.argmax(1)

    acc = accuracy_score(y_true, y_pred)
    print(f'\nOverall Accuracy : {acc * 100:.2f}%')
    print('\nClassification Report:')
    print(classification_report(y_true, y_pred, target_names=EMOTIONS))

    os.makedirs(args.output_dir, exist_ok=True)

    plot_confusion_matrix(
        y_true, y_pred,
        save_path=os.path.join(args.output_dir, 'confusion_matrix.png')
    )
    plot_per_class_accuracy(
        y_true, y_pred,
        save_path=os.path.join(args.output_dir, 'per_class_accuracy.png')
    )

    # Grad-CAM on a random subset
    idxs = np.random.choice(len(X_test), size=5, replace=False)
    plot_gradcam(
        model, X_test[idxs], y_test[idxs],
        save_path=os.path.join(args.output_dir, 'gradcam.png')
    )

    print(f'\nAll evaluation plots saved to: {args.output_dir}')
    return acc


if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
