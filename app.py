import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)

# -------- LOAD MODEL --------
MODEL_PATH = "crack_model.keras"  # update to your .keras path
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully")

# -------- CONFIG --------
CNN_THRESHOLD = 0.25       # lowered to catch thin cracks
CONFIRM_FRAMES = 1         # single-frame confirmation for demo
recent_probs = []
confirm_count = 0

# -------- FRONTEND --------
@app.route("/")
def home():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    frontend_dir = os.path.join(base_dir, "..", "frontend")
    return send_from_directory(frontend_dir, "index.html")

# -------- PREPROCESSING --------
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    resized = cv2.resize(rgb, (224, 224))
    normalized = resized / 255.0

    return np.expand_dims(normalized, axis=0)

# -------- PREDICT API --------
@app.route("/predict", methods=["POST"])
def predict():
    global recent_probs, confirm_count

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img_bytes = request.files["image"].read()
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    # Preprocess for CNN
    input_img = preprocess(img)
    prob = float(model.predict(input_img, verbose=0)[0][0])

    # Temporal smoothing
    recent_probs.append(prob)
    recent_probs = recent_probs[-3:]
    avg_prob = sum(recent_probs) / len(recent_probs)

    # Single-frame confirmation
    if avg_prob > CNN_THRESHOLD:
        confirm_count += 1
    else:
        confirm_count = 0

    crack_detected = confirm_count >= CONFIRM_FRAMES

    return jsonify({
        "crack": crack_detected,
        "confidence": round(avg_prob, 3),
        "model": "Hackathon Thin Crack Build"
    })

# -------- RUN --------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
