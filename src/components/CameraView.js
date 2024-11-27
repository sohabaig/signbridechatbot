import React from "react";
import styled from "styled-components";

const CameraView = ({ cameraEnabled, translatedText, toggleCamera, setTranslatedText }) => {
  return (
    <CameraContainer>
      {!cameraEnabled ? (
        <Placeholder>
          <p>Click below to enable the camera!</p>
          <CameraButton onClick={toggleCamera}>Enable Camera</CameraButton>
        </Placeholder>
      ) : (
        <LiveCamera>
          {/* Translation overlay */}
          {translatedText && <TranslatedText>{translatedText}</TranslatedText>}

          {/* ASL Guidance overlay */}
          <ASLGuide>
            <p>ASL Guidance will appear here dynamically.</p>
          </ASLGuide>

          {/* Mock toggle for translation */}
          <CaptureButton onClick={() => setTranslatedText("Hello! This is a test translation.")}>
            Mock Translate
          </CaptureButton>
        </LiveCamera>
      )}
    </CameraContainer>
  );
};

export default CameraView;

const CameraContainer = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  height: 80vh; /* Occupy most of the screen */
  padding: 1rem;
  background-color: #f8f9fa;
`;

const Placeholder = styled.div`
  text-align: center;
  font-family: "Quicksand", sans-serif;
`;

const CameraButton = styled.button`
  background-color: #4a53ff;
  color: white;
  border: none;
  padding: 0.5rem 1rem;
  font-family: "Quicksand", sans-serif;
  font-size: 1rem;
  cursor: pointer;
  border-radius: 8px;
`;

const LiveCamera = styled.div`
  position: relative;
  width: 90%;
  height: 90%;
  background-color: #e0e0e0; /* Mock camera feed */
  border-radius: 16px;
  overflow: hidden;
`;

const TranslatedText = styled.div`
  position: absolute;
  top: 1rem;
  left: 1rem;
  background-color: #4a53ff;
  color: white;
  padding: 0.5rem 1rem;
  font-family: "Quicksand", sans-serif;
  font-size: 1rem;
  border-radius: 8px;
`;

const ASLGuide = styled.div`
  position: absolute;
  bottom: 1rem;
  right: 1rem;
  background-color: rgba(0, 0, 0, 0.5);
  color: white;
  padding: 0.5rem;
  font-family: "Quicksand", sans-serif;
  font-size: 0.875rem;
  border-radius: 8px;
`;

const CaptureButton = styled.button`
  position: absolute;
  bottom: 1rem;
  left: 50%;
  transform: translateX(-50%);
  background-color: #ffffff;
  color: #4a53ff;
  border: none;
  padding: 0.5rem 1rem;
  font-family: "Quicksand", sans-serif;
  font-size: 1rem;
  cursor: pointer;
  border-radius: 50%;
`;
