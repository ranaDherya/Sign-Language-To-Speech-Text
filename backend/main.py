from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector
from io import BytesIO
from typing import Any

# Initialize FastAPI
app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow requests from your frontend
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Load the trained model
model = tf.keras.models.load_model(r"../model/word_model/keras_model.h5")

# Load labels
with open("../model/word_model/labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize Hand Detector
detector = HandDetector(maxHands=1)

# Image preprocessing parameters
offset = 20
imgSize = 224

def preprocess_image(img):
    """Preprocess the uploaded image for the model."""
    aspectRatio = img.shape[0] / img.shape[1]
    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

    if aspectRatio > 1:
        scale = imgSize / img.shape[0]
        new_width = int(scale * img.shape[1])
        imgResize = cv2.resize(img, (new_width, imgSize))
        wGap = (imgSize - new_width) // 2
        imgWhite[:, wGap:wGap + new_width] = imgResize
    else:
        scale = imgSize / img.shape[1]
        new_height = int(scale * img.shape[0])
        imgResize = cv2.resize(img, (imgSize, new_height))
        hGap = (imgSize - new_height) // 2
        imgWhite[hGap:hGap + new_height, :] = imgResize

    imgWhite = imgWhite.astype("float32") / 255.0
    imgWhite = np.expand_dims(imgWhite, axis=0)  # Add batch dimension
    return imgWhite

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Receives an image, processes it, and returns the detected sign."""
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    hands, _ = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
            imgWhite = preprocess_image(imgCrop)
            predictions = model.predict(imgWhite)
            index = np.argmax(predictions)
            detected_word = labels[index]
            confidence = float(predictions[0][index])
            print(detected_word)

            return {"word": detected_word, "confidence": confidence}

    return {"word": "No hand detected", "confidence": 0.0}

# Run server with: uvicorn main:app --reload
