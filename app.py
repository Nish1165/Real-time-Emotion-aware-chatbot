import os
import cv2
import time
from google import genai
from google.genai import types
from flask import Flask, render_template, request, jsonify, Response
from dotenv import load_dotenv

# Import your custom emotion module
from emotion_module.emotion_thread import start_emotion_thread, get_current_emotion, get_latest_frame

# 1. Load Environment Variables
load_dotenv()
app = Flask(__name__)

# 2. Configure Gemini Client (Using v1beta for Preview Model access)
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(api_version='v1beta')
)

# 3. Start the Background Emotion Detection Thread
start_emotion_thread()

def build_system_prompt(emotion):
    """Generates a high-empathy persona that prioritizes emotional validation."""
    vibe_guide = {
        "angry": "use a very de-escalating, soft, and calm tone. Acknowledge their frustration first.",
        "sad": "be deeply encouraging, warm, and patient. Offer comfort before help.",
        "neutral": "be professional, steady, and clear.",
        "fear": "be highly reassuring and protective. Use phrases like 'It's okay' or 'We've got this'.",
        "surprise": "match their energy with curiosity and wonder.",
        "happy": "be high-energy, celebratory, and share in their excitement!",
        "disgust": "be polite but stay focused on redirecting them toward a positive task."
    }
    selected_vibe = vibe_guide.get(emotion, "be a helpful tutor")
    return (
        f"CRITICAL INSTRUCTION: The student is currently feeling '{emotion}'. "
        f"You MUST {selected_vibe} "
        "1. Always start your response by briefly acknowledging how the student seems to be feeling. "
        "2. Match your vocabulary to their emotional state. "
        "3. Do NOT give direct answers immediately. Instead, lead them with questions. "
        "4. Your goal is to be a mentor, not just a search engine."
    )

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
                _, buffer = cv2.imencode(".jpg", frame)
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
            time.sleep(0.03) 
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("user_message", "").strip()
    detected_emotion = get_current_emotion() or "neutral"

    if not user_message:
        return jsonify({"reply": "I'm listening!", "emotion": detected_emotion})

    try:
        system_instructions = build_system_prompt(detected_emotion)
        
        # Using the specific Preview ID for the 500 RPD quota
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview", 
            contents=f"{system_instructions}\n\nStudent asks: {user_message}"
        )
        reply = response.text

    except Exception as e:
        print(f"\n[BACKEND ERROR] Critical Failure: {e}\n")
        # Fallback message for the UI
        reply = "⚠️ I'm catching my breath! My reasoning engine is a bit busy. Please try sending that again in a second."

    return jsonify({
        "reply": reply, 
        "emotion": detected_emotion
    })

if __name__ == "__main__":
    app.run(debug=False, threaded=True, host="0.0.0.0", port=5000)