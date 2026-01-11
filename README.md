# Garbage Classifier using Machine Learning only

This project is a Machine Learning implementation designed to explore computer vision techniques and apply them to the real-world problem of waste classification.

## Overview

The primary goal of this project is to bridge the gap between theoretical Machine Learning concepts and practical application. Instead of using pre-trained Deep Learning models, this project constructs a classification pipeline from scratch using classical algorithms to understand the underlying mechanics of feature extraction and image classification.

The project helps users identify the disposal category of waste items and how to dispose of them properly. While the current model serves as a fundamental proof-of-concept, it demonstrates the potential of applying code to environmental challenges.

**Current Performance:**
- **Training Accuracy:** ~65%
- **Testing Accuracy:** ~54%
> ***Note on Development:** This project adopts a modern AI-augmented development workflow. While the algorithms and model pipeline were manually implemented for educational purposes, to prioritize the core Machine Learning algorithms, the web interface was developed using rapid prototyping techniques.
Website link in Web Interface Usage section.*

## Features

This project moves away from "Black Box" deep learning to use explicit feature extraction and classical classification methods:

- **Image Preprocessing**: Resizes and normalizes images to ensure consistency and reduce noise before processing.
- **Color Histogram:** Extracts color distribution features to distinguish objects based on their color profiles.
- **Bag of Visual Words (BoW):** Adapts NLP concepts to computer vision by treating image features as "words" to create frequency histograms.
- **K-Means Clustering:** Used within the BoW model to cluster descriptors and construct a "visual vocabulary."
- **Principal Component Analysis (PCA):** Reduces dimensionality to focus on the most significant features and improve processing speed.
- **Support Vector Machine (SVM):** The main classifier that categorizes the waste based on the processed feature vectors.

## Data Structure
The link to the dataset can be found in `dataset-source.txt`.

The model utilizes a labeled dataset containing images across these categories:
<pre>
garbage-dataset/
  trash/
  shoes/
  plastic/
  paper/ 
  metal/ 
  glass/ 
  clothes/
  cardboard/
  biological/
  battery/
 </pre>
 
## Limitations

Given the use of classical ML techniques on a complex dataset, the current accuracy reflects the challenges of the approach:
1.  **Low Accuracy:** The model currently achieves ~54% on unseen data, indicating room for improvement in feature engineering.
2.  **Background Sensitivity:** The algorithms are sensitive to cluttered backgrounds and lighting variations.
3.  **Feature Limit:** Hand-crafted features (Color Histograms, BoW) may not capture high-level semantic details as effectively as CNNs.
4.  **Algorithm Limit:** The model is strictly limited to classifying items into the specific categories defined in the training dataset. It does not perform object detection (to locate the object) and **cannot distinguish between "trash" and "non-trash" objects**. If an undefined item is uploaded, the model will force a prediction into one of the known classes based on mathematical probability.

## Web Interface Usage

1.  Access the web interface at: https://garbage-classifier-l35k.onrender.com
  > *Note: The application is hosted on a free Render instance. It may take up to a minute to load initially while the server spins up from inactivity.*
2.  Drag and drop an image, or click to upload.
3.  Click **"Analyze"**
4.  Wait for the processing pipeline to finish.
5.  View the classification result.

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
