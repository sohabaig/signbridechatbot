from flask import Flask, render_template, request, jsonify, session, Response
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import random
import time

app = Flask(__name__)
app.secret_key = "temp-key"  # Secret key for session management

alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

class ImprovedNet(nn.Module):
    def __init__(self):
        super(ImprovedNet, self).__init__()
        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Second convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Third convolutional block
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Fourth convolutional block
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Calculate the size of flattened features
        self._to_linear = None
        
        # Initialize size and create fully connected layers
        self._initialize_size()

    def _initialize_size(self):
         # Pass a dummy input through conv layers to get the correct size
        x = torch.randn(1, 3, 224, 224)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        
        # Create the fc layers with correct input size
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self._to_linear, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 25)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Initialize model
model = ImprovedNet()
try:
    # Load the checkpoint
    checkpoint = torch.load('asl_model_final.pth', map_location=torch.device('cpu'))
    # Extract just the model state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded successfully! Accuracy: {checkpoint['accuracy']:.2f}%")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

model.eval() 

# Define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

current_prediction = "None"  
current_confidence = 0.0  

def predict_frame(frame):
    try:
        # Convert frame to PIL Image
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) 
        
        # Ensure image is resized to 224x224
        img = img.resize((224, 224), Image.Resampling.BILINEAR)
        
        # Preprocess image
        img_tensor = transform(img).unsqueeze(0) 
        
        # Make prediction
        with torch.no_grad():
            outputs = model(img_tensor)  # Forward pass
            _, predicted = torch.max(outputs, 1)
            confidence = float(torch.softmax(outputs, 1).max())
            
        # Get prediction (accounting for removed J and Z)
        prediction = predicted.item()
        if prediction >= 9:  # After J was removed
            prediction += 1
        letter = chr(65 + prediction)  
        
        return letter, confidence
    
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return None, 0

def generate_frames(quiz_state=None):
    try:
        camera = cv2.VideoCapture(0) 
        if not camera.isOpened():
            print("Error: Unable to access the camera.")
            return

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        frame_skip = 3  
        frame_count = 0
        last_prediction_time = time.time()

        while True:
            success, frame = camera.read()
            if not success:
                print("Failed to read from camera. Restarting...")
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            current_time = time.time()
            if current_time - last_prediction_time > 1:
                global current_prediction, current_confidence
                letter, confidence = predict_frame(frame)
                if letter:
                    current_prediction = letter
                    current_confidence = confidence
                    if quiz_state and quiz_state['current_letter'] == current_prediction:
                        quiz_state['score'] += 1
                        quiz_state['current_letter'] = random.choice(
                            list(set(alphabet) - set(quiz_state['completed_letters']))
                        )
                        quiz_state['completed_letters'].append(current_prediction)
                last_prediction_time = current_time

            ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
    except Exception as e:
        print(f"Error in video feed: {e}")
    finally:
        if 'camera' in locals() and camera.isOpened():
            camera.release()
            print("Camera resource released.")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/quiz")
def quiz():
    session['quiz_state'] = {
        "score": 0,
        "completed_letters": [],
        "current_letter": random.choice(alphabet)
    }
    return render_template("quiz.html", current_letter=session['quiz_state']["current_letter"], score=session['quiz_state']["score"])

@app.route("/video_feed")
def video_feed():
    quiz_state = session.get("quiz_state", {"score": 0, "completed_letters": [], "current_letter": random.choice(alphabet)})
    return Response(generate_frames(quiz_state), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/prediction", methods=["GET"])
def prediction():
    global current_prediction, current_confidence
    return {
        "prediction": current_prediction,
        "confidence": current_confidence
    }

if __name__ == "__main__":
    app.run(debug=True)
