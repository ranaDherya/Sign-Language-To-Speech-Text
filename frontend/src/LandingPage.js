import React from "react";
import { Link } from "react-router-dom";
import "./App.css";

const LandingPage = () => {
  return (
    <div className="app">
      <header className="hero">
        <h1>SignSpeak</h1>
        <p>Bridging the gap between gestures and speech using AI</p>
        <Link to="/detect" className="cta-button">
          Get Started
        </Link>
      </header>

      <section className="features">
        <div className="feature-card">
          <h2>Real-time Detection</h2>
          <p>Detect sign language gestures instantly using your webcam.</p>
        </div>
        <div className="feature-card">
          <h2>Speech Output</h2>
          <p>
            Recognized gestures are converted into clear speech in real-time.
          </p>
        </div>
        <div className="feature-card">
          <h2>Accessible & Free</h2>
          <p>
            Designed to assist the hearing impaired community with zero cost.
          </p>
        </div>
      </section>

      <footer className="footer">
        <p>© 2025 SignSpeak. All rights reserved.</p>
      </footer>
    </div>
  );
};

export default LandingPage;
