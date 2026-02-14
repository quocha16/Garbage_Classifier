# Garbage Classifier Project

This project explores both Classical Machine Learning and Deep Learning approaches for real-world waste classification.

It is designed as a practical learning project that demonstrates:

- Feature engineering with traditional computer vision techniques
- Classical ML pipeline construction
- CNN-based image classification with PyTorch
- Deployment via a Flask web interface

#  Project Overview

The goal of this project is to bridge theoretical Machine Learning knowledge with real-world implementation.

Unlike many projects that rely purely on pre-trained deep models, this system implements two independent pipelines:

1. **Classical ML Pipeline (SVM-based)**
2. **Deep Learning Pipeline (CNN-based – EfficientNet-B0)**

Users can upload an image via a web interface and select which model to use for prediction.

# Model Architectures

## Classical Machine Learning (SVM)

This pipeline is built entirely from scratch using handcrafted features:

- ORB Feature Extraction
- Bag of Visual Words (BoW)
- K-Means Clustering (Visual Vocabulary)
- Color Histogram Features
- Feature Normalization (Scaler)
- Dimensionality Reduction (PCA)
- Support Vector Machine (SVM) Classifier

### Performance:
~60% on Test Set

This approach emphasizes transparency and interpretability of the ML pipeline.

---

## Deep Learning Model (CNN)

A convolutional neural network built using:

- EfficientNet-B0 architecture
- Fully retrained classifier layer
- Softmax output for multi-class prediction

This model learns features automatically instead of relying on handcrafted feature engineering.

### Performance:
99% on Validation Set

## Web Interface Usage

1.  Access the web interface at: https://garbage-classifier-l35k.onrender.com
  > *Note: The application is hosted on a free Render instance. It may take up to a minute to load initially while the server spins up from inactivity.*
2.  Drag and drop an image, or click to upload.
3.  Click **"Analyze"**
4.  Wait for the processing pipeline to finish.
5.  View the result (Users can select an alternative prediction if the current result is incorrect, as the system provides the top 3 probabilities).

## Requirements

To run this project locally, ensure you have Python installed.

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/quocha16/garbage-classifier.git
    cd garbage-classifier
    ```
    
3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```
  
3.  **Run the application:**

    ```bash
    python app.py
    ```
    
## License

This project is licensed under the **MIT License**.
