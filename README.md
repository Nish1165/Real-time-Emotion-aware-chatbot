# 🎭 Real-Time Emotion-Aware Conversational Chatbot

## 📌 Overview

The **Real-Time Emotion-Aware Conversational Chatbot** is an intelligent system that integrates **computer vision**, **deep learning**, and **natural language interaction** to detect human emotions in real time and respond accordingly.

The system uses a webcam to capture facial expressions, processes them using a trained **CNN model**, and generates emotionally aware responses through a conversational AI module.

---

## 🚀 Features

* 🎥 Real-time face detection using webcam
* 😊 Emotion recognition (7 classes: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise)
* 🤖 Emotion-aware chatbot responses
* ⚡ Live video streaming with predictions
* 🌐 Web-based interface using Flask
* 📊 HSEmotion model deployed for real-time emotion recognition.

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


⭐ *If you like this project, consider giving it a star!*
