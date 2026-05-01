import os
import cv2
import time
from google import genai
from google.genai import types
from flask import Flask, render_template, request, jsonify, Response
from dotenv import load_dotenv

# Import emotion module
from emotion_module.emotion_thread import (
    start_emotion_thread,
    get_current_emotion,
    get_latest_frame
)

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure Gemini Client
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(api_version='v1beta')
)

# Start emotion detection thread
start_emotion_thread()


# ✅ SIMPLE + CLEAN PROMPT
def build_system_prompt(emotion):
    # Normalize combined emotions → 5 final categories
    clean_emotion = emotion.split('/')[0] if '/' in emotion else emotion

    return (
        f"You are an empathetic and helpful tutor. "
        f"The user is feeling '{clean_emotion}'. "
        f"Acknowledge their feeling briefly and then help them clearly."
    )


# Routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/get_emotion")
def get_emotion():
    return jsonify({"emotion": get_current_emotion() or "neutral"})


@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            frame = get_latest_frame()

            if frame is not None:
                # ✅ Improved performance (compression)
                _, buffer = cv2.imencode(
                    ".jpg",
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                )

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" +
                    buffer.tobytes() +
                    b"\r\n"
                )

            time.sleep(0.04)  # ~25 FPS

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("user_message", "").strip()
    detected_emotion = get_current_emotion() or "neutral"

    if not user_message:
        return jsonify({
            "reply": "I'm listening!",
            "emotion": detected_emotion
        })

    system_prompt = build_system_prompt(detected_emotion)

    reply = "⚠️ I'm a bit busy right now. Please try again in a moment."

    # ✅ Retry logic (handles Gemini failures)
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-3.1-flash-lite-preview",
                contents=f"{system_prompt}\n\nUser: {user_message}"
            )

            reply = response.text
            break

        except Exception as e:
            print(f"[Retry {attempt+1}] Gemini Error: {e}")
            time.sleep(2)

    return jsonify({
        "reply": reply,
        "emotion": detected_emotion
    })


if __name__ == "__main__":
    app.run(
        debug=False,
        threaded=True,
        host="0.0.0.0",
        port=5000
    )
