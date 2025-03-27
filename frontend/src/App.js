import React, { useRef, useState } from "react";
import Webcam from "react-webcam";
import axios from "axios";

const App = () => {
  const webcamRef = useRef(null);
  const [detectedWord, setDetectedWord] = useState("");

  const captureAndPredict = async () => {
    const imageSrc = webcamRef.current.getScreenshot();

    if (!imageSrc) return;

    // Convert base64 to Blob
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

      setDetectedWord(response.data.word);
      speak(response.data.word); // Convert to speech
    } catch (error) {
      console.error("Error detecting sign:", error);
    }
  };

  async function uploadImage(file) {
    const formData = new FormData();
    formData.append("file", file);

    const response = await axios.post(
      "http://127.0.0.1:8000/predict/",
      formData,
      {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      }
    );

    const data = await response.json();
    console.log("Response:", data); // Debugging

    if (data.word) {
      setDetectedWord(data.word); // Update UI state
    }
  }

  const speak = (text) => {
    const synth = window.speechSynthesis;
    const utterance = new SpeechSynthesisUtterance(text);
    synth.speak(utterance);
  };

  return (
    <div style={{ textAlign: "center", marginTop: "20px" }}>
      <h1>Sign Language Detection</h1>
      <Webcam
        ref={webcamRef}
        screenshotFormat="image/jpeg"
        width={400}
        height={300}
      />
      <br />
      <button
        onClick={captureAndPredict}
        style={{ padding: "10px", marginTop: "10px", fontSize: "16px" }}
      >
        Capture & Detect
      </button>
      <h2>Detected Word: {detectedWord}</h2>
    </div>
  );
};

export default App;
