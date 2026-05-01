# 🎭 Real-Time Emotion-Aware Conversational Chatbot

## 📌 Overview

The **Real-Time Emotion-Aware Conversational Chatbot** is an intelligent system that integrates **computer vision**, **deep learning**, and **natural language interaction** to detect human emotions in real time and respond accordingly.

The system uses a webcam to capture facial expressions, processes them using a trained **CNN model** [Downliad link : https://drive.google.com/file/d/1ZfdeLMf5SV06yvjLimA_pT_eWEEk7tXs/view?usp=drive_link], and generates emotionally aware responses through a conversational AI module.

---

## 🚀 Features

* 🎥 Real-time face detection using webcam
* 😊 Emotion recognition (5 classes: Angry, Disgust, Surprise/Fear, Happy, Neutral/Sad.)
* 🤖 Emotion-aware chatbot responses
* ⚡ Live video streaming with predictions
* 🌐 Web-based interface using Flask
* 📊 CNN model trained on FER2013 for real-time emotion recognition.

---

Emotion Design Strategy

To improve both model robustness and chatbot response relevance, similar emotion classes are merged:

Fear + Surprise → Surprise/Fear
Sad + Neutral → Neutral/Sad

This design choice:

Reduces classification ambiguity
Improves real-time prediction stability
Aligns with chatbot behavior (similar emotional responses required)

---

## 🧠 Tech Stack

* **Frontend:** HTML, CSS
* **Backend:** Flask (Python)
* **Machine Learning:** TensorFlow / Keras
* **Computer Vision:** OpenCV
* **Dataset:** AffectNet
* **Other Tools:** NumPy, Pandas

---

## 🏗️ System Architecture

1. Webcam captures live video feed
2. Face detection using OpenCV
3. Preprocessing (grayscale, resizing, normalization)
4. HSEmotion model predicts emotion
5. Emotion passed to chatbot module
6. Chatbot generates context-aware response
7. Output displayed on web interface

---

##  Required Directory Structure:-

Real-time-Emotion-aware-chatbot/
│
├── app.py
├── requirements.txt
├── .env
│
├── emotion_module/
│   └── emotion_thread.py
│
├── models/
│   └── emotion_model.h5   (download separately)
│
├── haarcascade/
│   └── haarcascade_frontalface_default.xml
│
├── templates/
│   └── index.html
│
└── static/
     └── style.css
     

##  Add API key for LLM:-

Create a .env file in root:

GEMINI_API_KEY=your_api_key_here

.env file is required for chatbot functionality

---
## ⚙️ Setup

### 1. Create virtual environment (optional but recommended)

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
python app.py
```

### 4. Open in browser

```
http://127.0.0.1:5000/
```

---

## Validation Score of the trained CNN Emotion Classifier:-

Accuracy 75%

Precision (Macro Avg) 0.75

Recall (Macro Avg) 0.67

F1 Score (Macro Avg) 0.70 


