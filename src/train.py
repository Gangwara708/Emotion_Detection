"""
Training script for the Emotion Detection CNN on FER2013.

Usage:
    python -m src.train --data data/fer2013.csv
    python -m src.train --data data/ --epochs 100 --batch_size 64
"""

import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    TensorBoard, CSVLogger
)

from src.model import build_emotion_cnn, EMOTIONS
from src.utils import (
    load_dataset, get_train_generator, get_val_generator,
    plot_class_distribution, plot_sample_images, plot_training_history
)


def parse_args():
    p = argparse.ArgumentParser(description='Train Emotion Detection CNN')
    p.add_argument('--data', type=str, default='data/fer2013.csv',
                   help='Path to fer2013.csv or data/ directory')
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--model_dir', type=str, default='models/')
    p.add_argument('--no_augment', action='store_true',
                   help='Disable data augmentation')
    p.add_argument('--resume', type=str, default=None,
                   help='Path to checkpoint to resume training from')
    return p.parse_args()


def build_callbacks(model_dir: str):
    os.makedirs(model_dir, exist_ok=True)
    return [
        ModelCheckpoint(
            filepath=os.path.join(model_dir, 'emotion_model_best.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=os.path.join(model_dir, 'emotion_model_last.keras'),
            save_best_only=False,
            verbose=0,
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        ),
        TensorBoard(log_dir=os.path.join(model_dir, 'logs'), histogram_freq=0),
        CSVLogger(os.path.join(model_dir, 'training_log.csv'), append=True),
    ]


def compute_class_weights(y_train: np.ndarray) -> dict:
    """Handle class imbalance — FER2013 has far more 'Happy' samples."""
    from sklearn.utils.class_weight import compute_class_weight
    labels = y_train.argmax(1)
    weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    return dict(enumerate(weights))


def train(args):
    print('=' * 60)
    print('  Emotion Detection CNN — Training')
    print('=' * 60)

    # ── GPU setup ──────────────────────────────────────────────
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f'GPUs available: {[g.name for g in gpus]}')
    else:
        print('No GPU found — training on CPU (will be slow).')

    # ── Data ───────────────────────────────────────────────────
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_dataset(args.data)

    print(f'\nDataset sizes:')
    print(f'  Train : {X_train.shape[0]:,}')
    print(f'  Val   : {X_val.shape[0]:,}')
    print(f'  Test  : {X_test.shape[0]:,}')

    os.makedirs(args.model_dir, exist_ok=True)
    plot_class_distribution(
        y_train.argmax(1), 'Training Set Class Distribution',
        save_path=os.path.join(args.model_dir, 'class_distribution.png')
    )
    plot_sample_images(
        X_train, y_train,
        save_path=os.path.join(args.model_dir, 'sample_images.png')
    )

    # ── Model ──────────────────────────────────────────────────
    model = build_emotion_cnn()

    if args.resume and os.path.exists(args.resume):
        print(f'\nResuming from checkpoint: {args.resume}')
        model.load_weights(args.resume)

    model.compile(
        optimizer=Adam(learning_rate=args.lr),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    model.summary()

    # ── Training ───────────────────────────────────────────────
    class_weights = compute_class_weights(y_train)
    print(f'\nClass weights: {class_weights}')

    steps_per_epoch = len(X_train) // args.batch_size

    if args.no_augment:
        history = model.fit(
            X_train, y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_data=(X_val, y_val),
            class_weight=class_weights,
            callbacks=build_callbacks(args.model_dir),
        )
    else:
        train_gen = get_train_generator(X_train, y_train, args.batch_size)
        val_gen   = get_val_generator(X_val, y_val, args.batch_size)
        history = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=args.epochs,
            validation_data=val_gen,
            validation_steps=len(X_val) // args.batch_size,
            class_weight=class_weights,
            callbacks=build_callbacks(args.model_dir),
        )

    # ── Post-training ──────────────────────────────────────────
    plot_training_history(
        history,
        save_path=os.path.join(args.model_dir, 'training_history.png')
    )

    print('\nEvaluating on test set ...')
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test Accuracy : {test_acc * 100:.2f}%')
    print(f'Test Loss     : {test_loss:.4f}')

    # Save final weights and metadata
    model.save(os.path.join(args.model_dir, 'emotion_model_final.keras'))
    print(f'\nAll artefacts saved to: {args.model_dir}')

    return model, history


if __name__ == '__main__':
    args = parse_args()
    train(args)
