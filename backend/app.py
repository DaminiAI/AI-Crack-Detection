
import os

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# -------- PATH CONFIGURATION --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
FRONTEND_DIR = os.path.join(PROJECT_DIR, "frontend")
MODEL_PATH = os.path.join(BASE_DIR, "crack_model.keras")

# -------- FLASK APP --------
app = Flask(__name__)
CORS(app)

# -------- LOAD MODEL --------
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at: {MODEL_PATH}\n"
        "Please check that backend/crack_model.keras exists."
    )

model = tf.keras.models.load_model(MODEL_PATH)
print("Crack detection model loaded successfully.")

# -------- DETECTION CONFIGURATION --------
CNN_THRESHOLD = 0.25
CONFIRM_FRAMES = 1
recent_probs = []
confirm_count = 0


# -------- FRONTEND ROUTE --------
@app.route("/", methods=["GET"])
def home():
    if not os.path.isfile(os.path.join(FRONTEND_DIR, "index.html")):
        return jsonify({
            "error": "Frontend file not found.",
            "expected_file": os.path.join(FRONTEND_DIR, "index.html")
        }), 404

    return send_from_directory(FRONTEND_DIR, "index.html")


# -------- IMAGE PREPROCESSING --------
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Improve local contrast using CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    enhanced = clahe.apply(gray)

    # Convert grayscale image to 3 channels
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

    # Resize and normalize
    resized = cv2.resize(rgb, (224, 224))
    normalized = resized.astype(np.float32) / 255.0

    # Add batch dimension
    return np.expand_dims(normalized, axis=0)


# -------- PREDICTION API --------
@app.route("/predict", methods=["POST"])
def predict():
    global recent_probs, confirm_count

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img_bytes = request.files["image"].read()

    if not img_bytes:
        return jsonify({"error": "Uploaded image is empty"}), 400

    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image file"}), 400

    try:
        input_img = preprocess(img)
        prediction = model.predict(input_img, verbose=0)
        prob = float(np.asarray(prediction).reshape(-1)[0])
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Could not process this image"}), 500

    # Smooth predictions over recent frames
    recent_probs.append(prob)
    recent_probs = recent_probs[-3:]
    avg_prob = sum(recent_probs) / len(recent_probs)

    # Apply configured threshold
    if avg_prob > CNN_THRESHOLD:
        confirm_count += 1
    else:
        confirm_count = 0

    crack_detected = confirm_count >= CONFIRM_FRAMES

    return jsonify({
        "crack": bool(crack_detected),
        "confidence": round(avg_prob, 3),
        "model": "Crack Detection CNN"
    })


# -------- HEALTH CHECK --------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None
    })


# -------- RUN SERVER --------
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
