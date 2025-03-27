import cv2
import numpy as np
import tensorflow as tf
import math
from cvzone.HandTrackingModule import HandDetector

# Load the trained model
model = tf.keras.models.load_model("5h model/keras_model.h5")

# Load labels
with open("5h model/labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize Hand Detector
detector = HandDetector(maxHands=1)

# Camera Setup
cap = cv2.VideoCapture(0)

# Image preprocessing parameters
offset = 20
imgSize = 224

def preprocess_image(imgCrop):
    """Resize and normalize the cropped hand image."""
    aspectRatio = imgCrop.shape[0] / imgCrop.shape[1]

    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

    if aspectRatio > 1:  # Tall image
        scale = imgSize / imgCrop.shape[0]
        new_width = math.ceil(scale * imgCrop.shape[1])
        imgResize = cv2.resize(imgCrop, (new_width, imgSize))
        wGap = (imgSize - new_width) // 2
        imgWhite[:, wGap:wGap + new_width] = imgResize
    else:  # Wide image
        scale = imgSize / imgCrop.shape[1]
        new_height = math.ceil(scale * imgCrop.shape[0])
        imgResize = cv2.resize(imgCrop, (imgSize, new_height))
        hGap = (imgSize - new_height) // 2
        imgWhite[hGap:hGap + new_height, :] = imgResize

    # Normalize image for model
    imgWhite = imgWhite.astype("float32") / 255.0  
    imgWhite = np.expand_dims(imgWhite, axis=0)  # Add batch dimension

    return imgWhite

def predict(imgWhite):
    """Make a prediction using the loaded model."""
    predictions = model.predict(imgWhite)
    index = np.argmax(predictions)
    return labels[index], predictions[0][index]

def main():
    while True:
        success, img = cap.read()
        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
                imgWhite = preprocess_image(imgCrop)
                label, confidence = predict(imgWhite)

                # Adjusted Bounding Box
                box_height = 40  # Reduced height for smaller font
                cv2.rectangle(imgOutput, (x - offset, y - offset - box_height), 
                              (x + w + offset, y - offset), 
                              (0, 255, 0), cv2.FILLED)

                # Adjusted Font Size and Position
                cv2.putText(imgOutput, f"{label} ({confidence:.2f})", 
                            (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.8, (0, 0, 0), 2)

                # Draw Hand Bounding Box
                cv2.rectangle(imgOutput, (x - offset, y - offset), 
                              (x + w + offset, y + h + offset), 
                              (0, 255, 0), 2)  # Reduced thickness

                cv2.imshow('ImageCrop', imgCrop)
                cv2.imshow('Image with White Background', imgWhite[0])  # Remove batch dimension for display

        cv2.imshow('Hand Sign Recognition', imgOutput)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
