from flask import Flask, request, render_template, jsonify

import os
import numpy as np
import cv2

import torch
import torch.nn as nn

from torchvision import models
import torchvision.transforms as transforms

from PIL import Image

app = Flask(__name__)

# CONFIG
BASE_PATH = "model/"
DEVICE = torch.device("cpu")

# LOAD PYTORCH MODEL
net_model = None
idx_to_class = None

class SimpleNet(nn.Module):
    def __init__(self, num_classes):
        super(SimpleNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)

try:
    checkpoint = torch.load(
        os.path.join(BASE_PATH, "garbage_model.pth"),
        map_location="cpu"
    )

    net_model = models.efficientnet_b0(weights=None)

    in_features = net_model.classifier[1].in_features
    net_model.classifier[1] = nn.Linear(
        in_features,
        len(checkpoint["class_names"])
    )

    net_model.load_state_dict(checkpoint["model_state_dict"])
    net_model.eval()

    idx_to_class = checkpoint["class_names"]

    print("CNN model loaded successfully.")

except Exception as e:
    print(f"Model load error: {e}")

# IMAGE TRANSFORM
transform_net = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ROUTES
@app.route("/")
def index():
    return render_template("index.html")

# PREDICT
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file"})

    file = request.files["file"]

    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image"})

    if net_model is None:
        return jsonify({"error": "CNN model not loaded on server"})

    try:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_pil = Image.fromarray(img_rgb)

        img_tensor = transform_net(img_pil)
        img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            outputs = net_model(img_tensor)
            probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

        top_idx = np.argsort(probs)[-3:][::-1]

        top_list = []

        for idx in top_idx:
            top_list.append({
                "label": idx_to_class[idx],
                "confidence": round(float(probs[idx]) * 100, 2)
            })

        return jsonify({
            "model": "Convolutional Neural Network (CNN)",
            "prediction": top_list[0]["label"],
            "top_3": top_list
        })

    except Exception as e:
        return jsonify({
            "error": f"Prediction Error: {str(e)}"
        })

# MAIN
if __name__ == "__main__":
    app.run(debug=True)