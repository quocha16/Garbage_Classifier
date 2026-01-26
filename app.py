from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import cv2
import os
from sklearn.preprocessing import normalize

app = Flask(__name__)

WIDTH, HEIGHT = 128, 128
IMG_SIZE = (WIDTH, HEIGHT)
VOCAB_SIZE = 700

svm_model = None
pca_model = None
le_model = None
kmeans_model = None
scaler_model = None

#Load
base_path = "model/"

try:

    svm_model = joblib.load(os.path.join(base_path, "svm_garbage_model.pkl"))
    pca_model = joblib.load(os.path.join(base_path, "pca_garbage_model.pkl"))
    le_model = joblib.load(os.path.join(base_path, "le_garbage_model.pkl"))
    kmeans_model = joblib.load(os.path.join(base_path, "kmeans_garbage_model.pkl"))
    scaler_model = joblib.load(os.path.join(base_path, "scaler_garbage_model.pkl"))

except Exception as e:
    print(f"Error: {e}")

orb = cv2.ORB_create(nfeatures=800)


#Softmax để tính xác suất
def softmax(x):
    """Tính softmax cho mảng 1D"""
    e_x = np.exp(x - np.max(x))  # Trừ max để ổn định số học
    return e_x / e_x.sum(axis=0)


def extract_features_combined(img):
    img_resized = cv2.resize(img, IMG_SIZE)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(gray, None)

    #KMeans Bag of Visual Words
    bow_hist = np.zeros(VOCAB_SIZE, dtype=np.float32)

    if des is not None:
        #Predict cụm
        words = kmeans_model.predict(des.astype(np.float64))
        unique, counts = np.unique(words, return_counts=True)
        for u, c in zip(unique, counts):
            if u < VOCAB_SIZE:
                bow_hist[u] = c

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
    #Kiểm tra xem model đã load được chưa
    if svm_model is None:
        return jsonify({"error"})

    try:
        if "file" not in request.files:
            return jsonify({"error"})

        file = request.files["file"]
        if file.filename == '':
            return jsonify({"error"})

        #Đọc ảnh
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error"})

        #Trích xuất đặc trưng
        try:
            feat = extract_features_combined(img).reshape(1, -1)
            feat_scaled = scaler_model.transform(feat)
            feat_pca = pca_model.transform(feat_scaled)
        except Exception as e_feat:
            print(f"Lỗi Extract Feature: {e_feat}")
            return jsonify({"error": f"Lỗi xử lý ảnh: {str(e_feat)}"})

        #Predict
        try:
            #Lấy raw scores (khoảng cách)
            raw_scores = svm_model.decision_function(feat_pca)

            #Xử lý output shape của decision_function
            if raw_scores.ndim == 1:
                scores = raw_scores  # Trường hợp mảng 1 chiều
            else:
                scores = raw_scores[0]  # Lấy dòng đầu tiên

            probs = []

            #Logic phân biệt Binary (2 class) và Multiclass (>2 class)
            if len(scores) == 1:
                #Phân loại nhị phân
                #Score là 1 số thực. <0 là class 0, >0 là class 1
                val = scores[0] if isinstance(scores, (list, np.ndarray)) else scores
                prob_pos = 1 / (1 + np.exp(-val))  # Sigmoid
                probs = np.array([1 - prob_pos, prob_pos])
            else:
                #Phân loại đa lớp dùng softmax
                probs = softmax(scores)

        except Exception as e_svm:
            print(f"Lỗi tính toán SVM: {e_svm}")
            return jsonify({"error": f"Lỗi nội bộ SVM: {str(e_svm)}"})

        #Trả về kết quả
        #Top 3
        num_classes = len(probs)
        top_k = min(3, num_classes)

        top_idx = np.argsort(probs)[-top_k:][::-1]

        top_list = []
        for idx in top_idx:
            #Lấy tên nhãn từ LabelEncoder
            label_name = str(le_model.inverse_transform([idx])[0])
            confidence_score = float(probs[idx]) * 100

            top_list.append({
                "label": label_name,
                "confidence": round(confidence_score, 2)
            })

        return jsonify({
            "prediction": top_list[0]["label"],
            "top_3": top_list
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)