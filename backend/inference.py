import tensorflow as tf
import cv2
import numpy as np

MODEL_PATH = "model/crack_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)

def predict_crack(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Invalid image")

    img_resized = cv2.resize(img, (224, 224)) / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)

    pred = model.predict(img_resized)[0][0]

    result = {
        "crack": bool(pred > 0.5),
        "confidence": float(pred if pred > 0.5 else 1 - pred)
    }

    return result
