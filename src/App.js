import React, { useState } from "react";
import Header from "./components/Header";
import CameraView from "./components/CameraView";
import "./App.css";

function App() {
  const [cameraEnabled, setCameraEnabled] = useState(false);
  const [translatedText, setTranslatedText] = useState("");

  const toggleCamera = () => setCameraEnabled(!cameraEnabled);

  return (
    <div className="App">
      <Header onBack={() => console.log("Back button clicked")} />

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
