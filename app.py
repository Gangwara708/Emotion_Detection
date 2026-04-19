"""
Real-time Emotion, Age, and Gender Detection Application.

Modes:
  webcam  — live camera feed (default)
  image   — single image file
  video   — video file

Usage:
    python app.py                                       # webcam
    python app.py --mode image --source photo.jpg
    python app.py --mode video --source video.mp4
    python app.py --mode webcam --camera 0

Prerequisites:
  1. Train the emotion model:  python -m src.train --data data/fer2013.csv
  2. Model saved at:           models/emotion_model_best.keras

Age/Gender:
  Uses pre-trained Caffe models (Gil Levi & Tal Hassner, 2015).
  Download with:  python download_dataset.py --ag_models
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────

EMOTIONS   = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
AGE_GROUPS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
              '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDERS    = ['Male', 'Female']

# Emotion → BGR colour mapping for bounding boxes
EMOTION_COLORS = {
    'Angry':    (0, 0, 255),
    'Disgust':  (0, 140, 255),
    'Fear':     (128, 0, 128),
    'Happy':    (0, 255, 0),
    'Sad':      (255, 0, 0),
    'Surprise': (0, 255, 255),
    'Neutral':  (200, 200, 200),
}

IMG_SIZE = 48


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Emotion, Age & Gender Detection')
    p.add_argument('--mode', choices=['webcam', 'image', 'video'], default='webcam')
    p.add_argument('--source', type=str, default=None,
                   help='Image or video file path (not needed for webcam)')
    p.add_argument('--camera', type=int, default=0, help='Camera index')
    p.add_argument('--emotion_model', type=str,
                   default='models/emotion_model_best.keras',
                   help='Path to trained emotion model')
    p.add_argument('--age_proto',   type=str, default='models/age_deploy.prototxt')
    p.add_argument('--age_model',   type=str, default='models/age_net.caffemodel')
    p.add_argument('--gender_proto',type=str, default='models/gender_deploy.prototxt')
    p.add_argument('--gender_model',type=str, default='models/gender_net.caffemodel')
    p.add_argument('--no_age_gender', action='store_true',
                   help='Skip age/gender prediction (if Caffe models not downloaded)')
    p.add_argument('--save', type=str, default=None,
                   help='Save output to this file (video/image path)')
    p.add_argument('--fps_limit', type=int, default=30)
    return p.parse_args()


# ── Model loading ──────────────────────────────────────────────────────────────

def load_emotion_model(model_path: str):
    if not os.path.exists(model_path):
        sys.exit(
            f'\n[ERROR] Emotion model not found at "{model_path}".\n'
            'Train it first:  python -m src.train --data data/fer2013.csv\n'
        )
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path)
    print(f'[OK] Emotion model loaded from {model_path}')
    return model


def load_age_gender_nets(age_proto, age_model, gender_proto, gender_model):
    try:
        age_net    = cv2.dnn.readNet(age_model,    age_proto)
        gender_net = cv2.dnn.readNet(gender_model, gender_proto)
        print('[OK] Age/Gender Caffe models loaded.')
        return age_net, gender_net
    except cv2.error as e:
        print(f'[WARN] Could not load Age/Gender models: {e}')
        print('       Run: python download_dataset.py --ag_models')
        return None, None


# ── Inference helpers ──────────────────────────────────────────────────────────

def predict_emotion(model, face_gray):
    face = cv2.resize(face_gray, (IMG_SIZE, IMG_SIZE)).astype('float32') / 255.0
    face = face.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    probs = model.predict(face, verbose=0)[0]
    return probs


def predict_age_gender(age_net, gender_net, face_bgr):
    MODEL_MEAN = (78.4263377603, 87.7689143744, 114.895847746)
    blob = cv2.dnn.blobFromImage(
        cv2.resize(face_bgr, (227, 227)), 1.0, (227, 227), MODEL_MEAN, swapRB=False
    )
    gender_net.setInput(blob)
    gender_probs = gender_net.forward()
    gender = GENDERS[gender_probs[0].argmax()]

    age_net.setInput(blob)
    age_probs = age_net.forward()
    age = AGE_GROUPS[age_probs[0].argmax()]

    return age, gender


# ── Frame rendering ────────────────────────────────────────────────────────────

def draw_face_info(frame, x, y, w, h, emotion, probs, age=None, gender=None):
    color = EMOTION_COLORS.get(emotion, (255, 255, 255))

    # Bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Main label
    label = emotion
    if gender and age:
        label = f'{gender}, {age} | {emotion}'

    label_y = y - 10 if y > 20 else y + h + 20
    cv2.rectangle(frame, (x, label_y - 18), (x + len(label) * 9, label_y + 2), color, -1)
    cv2.putText(frame, label, (x + 2, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    # Mini probability bars (drawn to the right of the face)
    bar_x = x + w + 8
    bar_y = y
    bar_w, bar_max = 80, 14
    for i, (emo, p) in enumerate(zip(EMOTIONS, probs)):
        fill_w = int(p * bar_max * 5)
        c = color if emo == emotion else (100, 100, 100)
        cv2.rectangle(frame, (bar_x, bar_y + i * 15),
                      (bar_x + fill_w, bar_y + i * 15 + 12), c, -1)
        cv2.putText(frame, f'{emo[:3]} {p * 100:.0f}%',
                    (bar_x + fill_w + 3, bar_y + i * 15 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (220, 220, 220), 1)

    return frame


def draw_fps(frame, fps):
    cv2.putText(frame, f'FPS: {fps:.1f}', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


# ── Core detection loop ────────────────────────────────────────────────────────

def process_frame(frame, emotion_model, face_cascade,
                  age_net=None, gender_net=None, use_ag=True):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        face_gray = gray[y:y + h, x:x + w]
        face_bgr  = frame[y:y + h, x:x + w]

        probs   = predict_emotion(emotion_model, face_gray)
        emotion = EMOTIONS[probs.argmax()]

        age, gender = None, None
        if use_ag and age_net and gender_net:
            age, gender = predict_age_gender(age_net, gender_net, face_bgr)

        draw_face_info(frame, x, y, w, h, emotion, probs, age, gender)

    return frame


# ── Run modes ─────────────────────────────────────────────────────────────────

def run_webcam(args, emotion_model, face_cascade, age_net, gender_net):
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f'[ERROR] Cannot open camera {args.camera}')

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps    = cap.get(cv2.CAP_PROP_FPS) or 20
        w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.save, fourcc, fps, (w, h))

    print('\nPress Q to quit, S to save a screenshot.\n')
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame, emotion_model, face_cascade,
                              age_net, gender_net, not args.no_age_gender)

        cur_time = time.time()
        fps = 1.0 / (cur_time - prev_time + 1e-9)
        prev_time = cur_time
        draw_fps(frame, fps)

        cv2.imshow('Emotion Detection — Press Q to quit', frame)
        if writer:
            writer.write(frame)

        key = cv2.waitKey(max(1, int(1000 / args.fps_limit))) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            cv2.imwrite('screenshot.png', frame)
            print('[Screenshot saved as screenshot.png]')

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


def run_image(args, emotion_model, face_cascade, age_net, gender_net):
    frame = cv2.imread(args.source)
    if frame is None:
        sys.exit(f'[ERROR] Cannot read image: {args.source}')

    frame = process_frame(frame, emotion_model, face_cascade,
                          age_net, gender_net, not args.no_age_gender)

    out = args.save or 'output_' + os.path.basename(args.source)
    cv2.imwrite(out, frame)
    print(f'[OK] Result saved to {out}')
    cv2.imshow('Emotion Detection', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_video(args, emotion_model, face_cascade, age_net, gender_net):
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        sys.exit(f'[ERROR] Cannot open video: {args.source}')

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps    = cap.get(cv2.CAP_PROP_FPS) or 25
        w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.save, fourcc, fps, (w, h))

    print('Processing video ... Press Q to quit.')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = process_frame(frame, emotion_model, face_cascade,
                              age_net, gender_net, not args.no_age_gender)
        cv2.imshow('Emotion Detection — Press Q to quit', frame)
        if writer:
            writer.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    emotion_model = load_emotion_model(args.emotion_model)
    face_cascade  = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    age_net, gender_net = None, None
    if not args.no_age_gender:
        age_net, gender_net = load_age_gender_nets(
            args.age_proto, args.age_model,
            args.gender_proto, args.gender_model
        )

    mode_fn = {'webcam': run_webcam, 'image': run_image, 'video': run_video}
    mode_fn[args.mode](args, emotion_model, face_cascade, age_net, gender_net)


if __name__ == '__main__':
    main()
