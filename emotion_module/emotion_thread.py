import threading
import time
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque, Counter

# ✅ Load CNN model
model = load_model("models/emotion_model.h5")

# ✅ Haar Cascade
face_cascade = cv2.CascadeClassifier(
    "haarcascade/haarcascade_frontalface_default.xml"
)

# 7-class labels (FER2013)
emotion_labels = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

# Map → 5 final categories
EMOTION_MAP = {
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fear/surprise",
    "surprise": "fear/surprise",
    "happy": "happy",
    "neutral": "sad/neutral",
    "sad": "sad/neutral"
}

current_emotion = "sad/neutral"
emotion_history = deque(maxlen=10)

latest_frame = None
lock = threading.Lock()
thread_running = False


def get_most_common(emotions):
    if not emotions:
        return "sad/neutral"
    return Counter(emotions).most_common(1)[0][0]


def emotion_worker():
    global current_emotion, latest_frame, thread_running

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Camera not accessible")
        return

    while thread_running:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        for (x, y, w, h) in faces:

            face_img = gray[y:y+h, x:x+w]

            if face_img.size > 0:
                # ✅ Preprocess
                face_img = cv2.resize(face_img, (48, 48))
                face_img = face_img / 255.0
                face_img = np.reshape(face_img, (1, 48, 48, 1))

                # ✅ Predict
                preds = model.predict(face_img, verbose=0)
                raw_emotion = emotion_labels[np.argmax(preds)]

                mapped_emotion = EMOTION_MAP.get(raw_emotion, "sad/neutral")

                emotion_history.append(mapped_emotion)

                with lock:
                    current_emotion = get_most_common(emotion_history)
                    display_emotion = current_emotion
            else:
                display_emotion = current_emotion

            # 🎯 Draw box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            label_y = y - 10 if y > 20 else y + 20

            cv2.putText(
                frame,
                display_emotion,
                (x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        # Store frame
        with lock:
            latest_frame = frame.copy()

        time.sleep(0.03)

    cap.release()


def get_latest_frame():
    with lock:
        return None if latest_frame is None else latest_frame.copy()


def get_current_emotion():
    with lock:
        return current_emotion


def start_emotion_thread():
    global thread_running
    if not thread_running:
        thread_running = True
        threading.Thread(target=emotion_worker, daemon=True).start()
