# AI-Crack-Detection
AI-powered real-time crack detection system for infrastructure, using computer vision to identify cracks in images instantly.
# AI-Based Real-Time Crack Detection System

## Project Overview
Infrastructure safety is a major challenge. Manual inspection of cracks in roads, walls, and buildings is time-consuming and error-prone. This project presents an **AI-based real-time crack detection system** that automatically detects surface cracks using **computer vision and deep learning**.

The system uses a **pre-trained CNN model** to analyze images captured from a live camera feed and highlights cracks instantly. This approach helps in faster inspection, early damage detection, and improved maintenance planning.

---

## Project Story
Crack detection is usually done through manual inspection, which is slow and highly dependent on human judgment. Our idea was to automate this process using AI so that cracks can be detected **instantly and in real time**.

We built a system that captures live video, processes each frame, enhances crack visibility, and then passes it to a trained deep learning model. The model predicts whether a crack exists and returns the confidence score. This system can be used in smart infrastructure monitoring and safety analysis.

---

## Technologies Used
- **Python 3.x**
- **TensorFlow / Keras** – Deep learning model
- **OpenCV** – Video capture and image processing
- **Ngrok** – Secure live video streaming for demo/testing
- **NumPy** – Numerical operations

---

## System Working
1. Live video is captured from a camera using OpenCV.
2. Each frame is converted to grayscale and enhanced using CLAHE.
3. The image is resized and normalized.
4. The processed frame is passed to a CNN model.
5. The system predicts crack presence with confidence.
6. The live video feed is streamed using Ngrok for demo.

---

## Setup & Execution Steps

### Step 1: Install Python
Ensure Python is installed:
```bash
python --version

Step 2: Clone the Repository
git clone https://github.com/DaminiAI/AI-Crack-Detection.git
cd AI-Crack-Detection

Step 3: Create & Activate Virtual Environment (Optional)
python -m venv venv
Windows
venv\Scripts\activate
Linux
source venv/bin/activate

Step 4: Install Required Libraries
pip install tensorflow
pip install opencv-python
pip install numpy
pip install flask
pip install flask-cors
pip install pyngrok

Step 5: Add the Trained Model
Place the trained model file in the project root folder:
crack_model.keras

Step 6: Run the Application
python app.py
Your webcam will open and start detecting cracks in real-time.
In the terminal, Ngrok will display a public URL like:
https://xxxx.ngrok.io

Step 7: Open the Live Demo
Copy the Ngrok URL from the terminal and open it in a browser.
You can test with live webcam feed or send images to the /predict API.

