from flask import Flask, request, render_template, jsonify
import os
import numpy as np
import cv2
import joblib
import torch
import torch.nn as nn
from sklearn.preprocessing import normalize
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

#GLOBAL VARIABLE
svm_model = None
pca_model = None
le_model = None
kmeans_model = None
scaler_model = None
net_model = None
idx_to_class = None

# Định nghĩa cấu trúc mạng (nạp checkpoint)
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

# ORB 
orb = cv2.ORB_create(nfeatures=800)

# TOP 3 CALC FUNCTIONS
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# FEATURES EXTRACTION 
def extract_features_combined(img):
    img_resized = cv2.resize(img, IMG_SIZE)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(gray, None)
    bow_hist = np.zeros(VOCAB_SIZE, dtype=np.float32)

    if des is not None:
        global kmeans_model
        if kmeans_model is None:
            kmeans_model = joblib.load(os.path.join(BASE_PATH, "kmeans_garbage_model.pkl"))
        
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
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ROUTES
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    global svm_model, pca_model, le_model, kmeans_model, scaler_model, net_model, idx_to_class
    
    if "file" not in request.files:
        return jsonify({"error": "No file"})

    file = request.files["file"]
    model_type = request.form.get("model_type", "svm")
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image"})

    # SVM
    if model_type == "svm":
        try:
            if svm_model is None:
                print("Lazy Loading SVM Pipeline...")
                svm_model = joblib.load(os.path.join(BASE_PATH, "svm_garbage_model.pkl"))
                pca_model = joblib.load(os.path.join(BASE_PATH, "pca_garbage_model.pkl"))
                le_model = joblib.load(os.path.join(BASE_PATH, "le_garbage_model.pkl"))
                kmeans_model = joblib.load(os.path.join(BASE_PATH, "kmeans_garbage_model.pkl"))
                scaler_model = joblib.load(os.path.join(BASE_PATH, "scaler_garbage_model.pkl"))

            feat = extract_features_combined(img).reshape(1, -1)
            feat_scaled = scaler_model.transform(feat)
            feat_pca = pca_model.transform(feat_scaled)

            raw_scores = svm_model.decision_function(feat_pca)
            scores = raw_scores if raw_scores.ndim == 1 else raw_scores[0]
            
            probs = softmax(scores) if len(scores) > 1 else np.array([1/(1+np.exp(-scores[0])), 1-(1/(1+np.exp(-scores[0])))])

            top_idx = np.argsort(probs)[-3:][::-1]
            top_list = [{"label": str(le_model.inverse_transform([i])[0]), 
                         "confidence": round(float(probs[i]) * 100, 2)} for i in top_idx]

            return jsonify({"model": "SVM", "prediction": top_list[0]["label"], "top_3": top_list})
        except Exception as e:
            return jsonify({"error": f"SVM Error: {str(e)}"})

    # CNN
    elif model_type == "net":
        try:
            # Lazy Load CNN Model
            if net_model is None:
                print("Lazy Loading EfficientNet...")
                checkpoint = torch.load(os.path.join(BASE_PATH, "garbage_model.pth"), map_location="cpu")
                net_model = models.efficientnet_b0(weights=None)
                in_features = net_model.classifier[1].in_features
                net_model.classifier[1] = nn.Linear(in_features, len(checkpoint["class_names"]))
                net_model.load_state_dict(checkpoint["model_state_dict"])
                net_model.eval()
                idx_to_class = checkpoint["class_names"]

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tensor = transform_net(img_pil).unsqueeze(0)

            with torch.no_grad():
                outputs = net_model(img_tensor)
                probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

            top_idx = np.argsort(probs)[-3:][::-1]
            top_list = [{"label": idx_to_class[i], "confidence": round(float(probs[i]) * 100, 2)} for i in top_idx]

            return jsonify({"model": "CNN", "prediction": top_list[0]["label"], "top_3": top_list})
        except Exception as e:
            return jsonify({"error": f"CNN Error: {str(e)}"})

    return jsonify({"error": "Invalid model type"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
