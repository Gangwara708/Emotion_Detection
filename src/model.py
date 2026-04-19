"""
CNN model for facial emotion detection.
Trained on FER2013 dataset — 7 emotion classes, 48×48 grayscale input.
Architecture: 4 VGG-style conv blocks → GlobalAvgPool → Dense head.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, Activation, MaxPooling2D,
    Dropout, Flatten, Dense, GlobalAveragePooling2D, Input,
    Add, ZeroPadding2D
)
from tensorflow.keras.regularizers import l2

# FER2013 emotion labels
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
NUM_CLASSES = len(EMOTIONS)
IMG_SIZE = 48


def _conv_bn_relu(x, filters, kernel_size=3, strides=1, padding='same', l2_reg=1e-4):
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding,
               kernel_regularizer=l2(l2_reg), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def build_emotion_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=NUM_CLASSES):
    """
    Deep CNN for emotion recognition.
    4 convolutional blocks with increasing filter depth (64→128→256→512),
    followed by a fully-connected classification head.
    """
    inputs = Input(shape=input_shape)

    # Block 1 — 64 filters
    x = _conv_bn_relu(inputs, 64)
    x = _conv_bn_relu(x, 64)
    x = MaxPooling2D(pool_size=2, strides=2)(x)
    x = Dropout(0.25)(x)

    # Block 2 — 128 filters
    x = _conv_bn_relu(x, 128)
    x = _conv_bn_relu(x, 128)
    x = MaxPooling2D(pool_size=2, strides=2)(x)
    x = Dropout(0.25)(x)

    # Block 3 — 256 filters
    x = _conv_bn_relu(x, 256)
    x = _conv_bn_relu(x, 256)
    x = _conv_bn_relu(x, 256)
    x = MaxPooling2D(pool_size=2, strides=2)(x)
    x = Dropout(0.25)(x)

    # Block 4 — 512 filters
    x = _conv_bn_relu(x, 512)
    x = _conv_bn_relu(x, 512)
    x = _conv_bn_relu(x, 512)
    x = MaxPooling2D(pool_size=2, strides=2)(x)
    x = Dropout(0.25)(x)

    # Classification head
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs, name='EmotionCNN')
    return model


# ── Residual block used by the Broad-ResNet head ──────────────────────────────

def _residual_block(x, filters, strides=1, l2_reg=1e-4):
    shortcut = x
    x = _conv_bn_relu(x, filters, strides=strides, l2_reg=l2_reg)
    x = Conv2D(filters, 3, padding='same', kernel_regularizer=l2(l2_reg),
               use_bias=False)(x)
    x = BatchNormalization()(x)

    if strides != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, 1, strides=strides, padding='same',
                          kernel_regularizer=l2(l2_reg), use_bias=False)(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def build_broad_resnet(input_shape=(224, 224, 3), num_age_classes=8, num_gender_classes=2):
    """
    Broad-ResNet for joint age and gender prediction (colour input, 224×224).
    Returns a model with two output heads: age and gender.
    """
    inputs = Input(shape=input_shape)

    x = _conv_bn_relu(inputs, 64, kernel_size=7, strides=2, padding='same')
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # Residual stages
    for filters, strides in [(64, 1), (128, 2), (256, 2), (512, 2)]:
        x = _residual_block(x, filters, strides=strides)
        x = _residual_block(x, filters)

    x = GlobalAveragePooling2D()(x)

    # Shared representation
    shared = Dense(512)(x)
    shared = BatchNormalization()(shared)
    shared = Activation('relu')(shared)
    shared = Dropout(0.4)(shared)

    # Age head (8 groups: 0-2, 4-6, 8-12, 15-20, 25-32, 38-43, 48-53, 60+)
    age_out = Dense(256, activation='relu')(shared)
    age_out = Dropout(0.3)(age_out)
    age_out = Dense(num_age_classes, activation='softmax', name='age')(age_out)

    # Gender head
    gender_out = Dense(128, activation='relu')(shared)
    gender_out = Dropout(0.3)(gender_out)
    gender_out = Dense(num_gender_classes, activation='softmax', name='gender')(gender_out)

    model = Model(inputs, [age_out, gender_out], name='BroadResNet')
    return model


def get_model_summary(model):
    model.summary()


if __name__ == '__main__':
    emotion_model = build_emotion_cnn()
    emotion_model.summary()
    print(f'\nEmotion classes: {EMOTIONS}')

    age_gender_model = build_broad_resnet()
    age_gender_model.summary()
