import React, { useState } from "react";
import Header from "./components/Header";
import CameraView from "./components/CameraView";
import "./App.css";

function App() {
  const [cameraEnabled, setCameraEnabled] = useState(false);
  const [translatedText, setTranslatedText] = useState("");

  // Toggle the camera on/off
  const toggleCamera = () => setCameraEnabled(!cameraEnabled);

  return (
    <div className="App">
      {/* Header */}
      <Header onBack={() => console.log("Back button clicked")} />

      {/* Main content */}
      <CameraView
        cameraEnabled={cameraEnabled}
        translatedText={translatedText}
        toggleCamera={toggleCamera}
        setTranslatedText={setTranslatedText}
      />
    </div>
  );
}

export default App;
