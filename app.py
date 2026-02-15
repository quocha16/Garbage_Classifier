from flask import Flask, request, render_template, jsonify

import os
import numpy as np
import cv2
import joblib

from sklearn.preprocessing import normalize

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import models
import torchvision.transforms as transforms

from PIL import Image
app = Flask(__name__)

# CONFIG
WIDTH, HEIGHT = 128, 128
IMG_SIZE = (WIDTH, HEIGHT)
VOCAB_SIZE = 700
BASE_PATH = "model/"
DEVICE = torch.device("cpu")

# LOAD SVM PIPELINE
svm_model = None
pca_model = None
le_model = None
kmeans_model = None
scaler_model = None

try:
    svm_model = joblib.load(os.path.join(BASE_PATH, "svm_garbage_model.pkl"))
    print("SVM loaded")
except Exception as e:
    print("SVM error:", e)

try:
    pca_model = joblib.load(os.path.join(BASE_PATH, "pca_garbage_model.pkl"))
    print("PCA loaded")
except Exception as e:
    print("PCA error:", e)

try:
    le_model = joblib.load(os.path.join(BASE_PATH, "le_garbage_model.pkl"))
    print("LE loaded")
except Exception as e:
    print("LE error:", e)

try:
    kmeans_model = joblib.load(os.path.join(BASE_PATH, "kmeans_garbage_model.pkl"))
    print("KMeans loaded")
except Exception as e:
    print("KMeans error:", e)

try:
    scaler_model = joblib.load(os.path.join(BASE_PATH, "scaler_garbage_model.pkl"))
    print("Scaler loaded")
except Exception as e:
    print("Scaler error:", e)


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
    net_model.classifier[1] = nn.Linear(in_features, len(checkpoint["class_names"]))

    net_model.load_state_dict(checkpoint["model_state_dict"])
    net_model.eval()

    class_names = checkpoint["class_names"]
    idx_to_class = class_names

    print("Net loaded successfully.")

except Exception as e:
    print(f"Net load error: {e}")

# ORB
orb = cv2.ORB_create(nfeatures=800)

# UTIL
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def extract_features_combined(img):
    img_resized = cv2.resize(img, IMG_SIZE)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(gray, None)

    bow_hist = np.zeros(VOCAB_SIZE, dtype=np.float32)

    if des is not None:
        words = kmeans_model.predict(des.astype(np.float64))
        unique, counts = np.unique(words, return_counts=True)
        for u, c in zip(unique, counts):
            if u < VOCAB_SIZE:
                bow_hist[u] = c

    if bow_hist.sum() > 0:
        bow_hist = normalize(bow_hist.reshape(1, -1), norm='l2').flatten()

    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()

    color_feat = np.concatenate([h_hist, s_hist, v_hist])
    color_feat = normalize(color_feat.reshape(1, -1), norm='l2').flatten()

    return np.concatenate([bow_hist, color_feat])

transform_net = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

#ROUTES
@app.route("/")
def index():
    return render_template("index.html")

#PREDICT
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file"})

    file = request.files["file"]
    # Nhận tham số model_type từ Frontend
    model_type = request.form.get("model_type", "svm")

    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image"})

    #SVM
    if model_type == "svm":
        if svm_model is None:
            return jsonify({"error": "SVM model not loaded on server"})

        try:
            feat = extract_features_combined(img).reshape(1, -1)
            feat_scaled = scaler_model.transform(feat)
            feat_pca = pca_model.transform(feat_scaled)

            raw_scores = svm_model.decision_function(feat_pca)
            scores = raw_scores if raw_scores.ndim == 1 else raw_scores[0]

            if len(scores) == 1:  # Binary classification case handle
                val = scores[0]
                prob_pos = 1 / (1 + np.exp(-val))
                probs = np.array([1 - prob_pos, prob_pos])
            else:
                probs = softmax(scores)

            top_idx = np.argsort(probs)[-3:][::-1]
            top_list = []
            for idx in top_idx:
                label_name = str(le_model.inverse_transform([idx])[0])
                top_list.append({
                    "label": label_name,
                    "confidence": round(float(probs[idx]) * 100, 2)
                })

            return jsonify({
                "model": "Support Vector Machine (SVM)",
                "prediction": top_list[0]["label"],
                "top_3": top_list
            })
        except Exception as e:
            return jsonify({"error": f"SVM Processing Error: {str(e)}"})

    #NEURAL NET
    elif model_type == "net":
        if net_model is None:
            return jsonify({"error": "Neural Network not loaded on server"})

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
                label_name = idx_to_class[idx]
                top_list.append({
                    "label": label_name,
                    "confidence": round(float(probs[idx]) * 100, 2)
                })

            return jsonify({
                "model": "Convolutional Neural Network (CNN)",
                "prediction": top_list[0]["label"],
                "top_3": top_list
            })
        except Exception as e:
            return jsonify({"error": f"Net Processing Error: {str(e)}"})

    else:
        return jsonify({"error": "Invalid model type selected"})

# MAIN
if __name__ == "__main__":
<<<<<<< HEAD
    app.run(debug=True)
=======
    app.run(debug=True)
>>>>>>> 2707eb4 (Update files)
