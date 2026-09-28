import cv2
import numpy as np

def estimate_severity(image_path, confidence):
    img = cv2.imread(image_path, 0)
    edges = cv2.Canny(img, 100, 200)

    crack_pixels = np.sum(edges > 0)
    total_pixels = edges.size
    ratio = crack_pixels / total_pixels

    if confidence < 0.6:
        return "Rejected", "Low", "No action"

    if ratio < 0.01:
        return "Minor", "Low", "Monitor"
    elif ratio < 0.03:
        return "Moderate", "Medium", "Schedule inspection"
    else:
        return "Severe", "High", "Immediate maintenance"
