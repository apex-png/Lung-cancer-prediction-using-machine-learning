from torchvision import models, transforms
import torch.nn as nn
import torch
from PIL import Image
from flask import Flask, request, jsonify, render_template
import os

app = Flask(__name__)

# Define model architecture
model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 128),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(128, 4)
)
model.load_state_dict(torch.load("resnet50_weights.pth", map_location=torch.device('cpu')))
model.eval()

class_names = ['adenocarcinoma', 'large cell carcinoma', 'normal', 'squamous cell carcinoma']

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    os.makedirs('uploads', exist_ok=True)
    img_path = os.path.join('uploads', file.filename)
    file.save(img_path)

    img_tensor = preprocess_image(img_path)
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        predicted_idx = torch.argmax(probs).item()
        predicted_class = class_names[predicted_idx]
        confidence = round(probs[predicted_idx].item() * 100, 2)

    os.remove(img_path)

    return jsonify({
        'prediction': predicted_class,
        'probability': confidence,
        'cancer_stage': 'N/A' if predicted_class == 'normal' else 'Cancer Detected'
    })

if __name__ == '__main__':
    app.run(debug=True)
