from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import cv2
import os
from sklearn.preprocessing import normalize

app = Flask(__name__)

WIDTH, HEIGHT = 128, 128
IMG_SIZE = (WIDTH, HEIGHT)
VOCAB_SIZE = 800

# LOAD MODELS
base_path = "model/"

try:
    dnn_model = joblib.load(os.path.join(base_path, "dnn_garbage_model.pkl"))
    pca_model = joblib.load(os.path.join(base_path, "pca_garbage_model.pkl"))
    le_model = joblib.load(os.path.join(base_path, "le_garbage_model.pkl"))
    gmm_model = joblib.load(os.path.join(base_path, "gmm_garbage_model.pkl"))
    scaler_model = joblib.load(os.path.join(base_path, "scaler_garbage_model.pkl"))
except Exception as e:
    print(f"Error: {e}")

orb = cv2.ORB_create(nfeatures=800)


def extract_features_combined(img):
    img_resized = cv2.resize(img, IMG_SIZE)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(gray, None)

    #GMM Bag of Visual Words
    bow_hist = np.zeros(VOCAB_SIZE, dtype=np.float32)
    if des is not None:
        probs = gmm_model.predict_proba(des.astype(np.float32))
        bow_hist = probs.sum(axis=0)

    #BoW
    if bow_hist.sum() > 0:
        bow_hist = normalize(bow_hist.reshape(1, -1), norm='l2').flatten()

    #Color Histogram
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()

    color_feat = np.concatenate([h_hist, s_hist, v_hist])
    color_feat = normalize(color_feat.reshape(1, -1), norm='l2').flatten()

    return np.concatenate([bow_hist, color_feat])


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part"})

        file = request.files["file"]
        if file.filename == '':
            return jsonify({"error": "No selected file"})

        #Đọc và tiền xử lý ảnh
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Lỗi đọc ảnh"})

        #Trích xuất đặc trưng
        feat = extract_features_combined(img).reshape(1, -1)
        feat_scaled = scaler_model.transform(feat)
        feat_pca = pca_model.transform(feat_scaled)

        probs = dnn_model.predict_proba(feat_pca)[0]

        top_4_idx = np.argsort(probs)[-4:][::-1]

        top_4_list = []

        for idx in top_4_idx:
            top_4_list.append({
                "label": le_model.inverse_transform([idx])[0],
                "confidence": round(float(probs[idx]) * 100, 2)
            })


        return jsonify({
            "prediction": top_4_list[0]["label"],
            "top_4": top_4_list
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)