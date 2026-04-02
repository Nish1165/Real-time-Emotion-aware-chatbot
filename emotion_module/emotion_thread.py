import threading
import time
import cv2
import mediapipe as mp
from hsemotion.facial_emotions import HSEmotionRecognizer
from collections import deque, Counter

predictor = HSEmotionRecognizer(model_name='enet_b0_8_best_afew')


# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# -------- Emotion Mapping (8 classes → 5 research classes) --------
# -------- Corrected Emotion Mapping for HSEmotion --------
EMOTION_MAP = {
    "anger": "angry",
    "contempt": "angry",
    "disgust": "disgust",
    "fear": "fear/surprise",
    "happiness": "happy",      # Was "happy", HSEmotion returns "Happiness"
    "neutral": "sad/neutral",
    "sadness": "sad/neutral",  # Was "sad", HSEmotion returns "Sadness"
    "surprise": "fear/surprise"
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

    while thread_running:

        ret, frame = cap.read()
        if not ret:
            continue

        # Mirror webcam (more natural)
        frame = cv2.flip(frame, 1)

        # Store frame for streaming
        with lock:
            latest_frame = frame.copy()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:

                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape

                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)

                face_img = frame[max(0, y):y+h, max(0, x):x+w]

                if face_img.size > 0:

                    # Raw prediction from HSEmotion
                    raw_emotion, _ = predictor.predict_emotions(face_img)

                    # Convert to research paper classes
                    mapped_emotion = EMOTION_MAP.get(raw_emotion.lower(), "sad/neutral")

                    emotion_history.append(mapped_emotion)

                    with lock:
                        current_emotion = get_most_common(emotion_history)

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
