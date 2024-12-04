from flask import Flask, Response, render_template
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
import io
import numpy as np
import time

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

app = Flask(__name__)

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
    print("Please ensure you have a trained model saved as 'asl_model_final.pth'")
    exit(1)

model.eval()

# Define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

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
            outputs = model(img_tensor)
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

def generate_frames():
    camera = cv2.VideoCapture(0)
    last_prediction_time = time.time()
    current_prediction = "None"
    current_confidence = 0
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Make prediction every 0.5 seconds
        current_time = time.time()
        if current_time - last_prediction_time > 0.5:
            letter, confidence = predict_frame(frame)
            if letter is not None:
                current_prediction = letter
                current_confidence = confidence
            last_prediction_time = current_time
        
        # Add prediction text to frame
        text = f"Letter: {current_prediction} ({current_confidence:.2f})"
        cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Convert frame to jpg
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)