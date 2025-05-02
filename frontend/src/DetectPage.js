import React, { useRef, useState } from "react";
import Webcam from "react-webcam";
import axios from "axios";
import "./DetectPage.css";

const DetectPage = () => {
  const webcamRef = useRef(null);
  const [detectedWord, setDetectedWord] = useState("");
  const [sentence, setSentence] = useState("");

  const resetHandler = () => {
    setSentence("");
    setDetectedWord("");
  };

  const captureAndPredict = async () => {
    const imageSrc = webcamRef.current.getScreenshot();
    if (!imageSrc) return;

    const blob = await fetch(imageSrc).then((res) => res.blob());
    const formData = new FormData();
    formData.append("file", blob, "image.jpg");

    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/predict/",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      const word = response.data.word;
      setDetectedWord(word);
      if (word !== "No hand detected") {
        setSentence((prev) => (prev + " " + word).trim());
      }
    } catch (error) {
      console.error("Error detecting sign:", error);
    }
  };

  const speak = (text) => {
    const synth = window.speechSynthesis;
    const utterance = new SpeechSynthesisUtterance(text);
    synth.speak(utterance);
  };

  return (
    <div className="detect-container">
      <h1>🖐️ Sign Language Detection</h1>

      <Webcam
        ref={webcamRef}
        screenshotFormat="image/jpeg"
        width={400}
        height={300}
        className="webcam"
      />

      <div className="button-group">
        <button className="action-button" onClick={captureAndPredict}>
          📸 Capture & Detect
        </button>
        <button className="action-button" onClick={() => speak(sentence)}>
          🔊 Speak Sentence
        </button>
        <button className="reset-button" onClick={resetHandler}>
          🔁 Reset
        </button>
      </div>

      <div className="output">
        <p>
          <strong>Detected Word:</strong>{" "}
          {detectedWord || "Waiting for input..."}
        </p>
        <p>
          <strong>Sentence:</strong> {sentence || "No sentence yet."}
        </p>
      </div>
    </div>
  );
};

export default DetectPage;
