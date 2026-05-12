# About the Project
This project focuses on Deep Learning-based waste classification using Computer Vision techniques and PyTorch.

It is designed as a practical learning project to explore how modern CNN architectures can solve real-world image classification problems and how trained models can be deployed through a web application.

The project includes:

- CNN-based image classification with PyTorch
- Transfer Learning using EfficientNet-B0
- Model Quantization for lightweight deployment
- Image preprocessing and augmentation
- Flask web deployment (This part relies heavily on AI generation)

# Overview

The goal of this project is to bridge theoretical Deep Learning knowledge with real-world implementation while understanding how image classification systems are trained, evaluated, and deployed.

The system allows users to upload an image through a web interface and receive predictions from the trained neural network model.

To make deployment feasible on low-resource environments and free hosting platforms, the model is optimized using quantization techniques to reduce memory usage and improve inference efficiency.

# Model Architecture

## Dataset Structure

Dataset source is provided in `dataset-source.txt`.

The dataset contains the following labeled categories:

```bash
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
```

# Deep Learning Model (CNN)

The classification model is built using:

- EfficientNet-B0 architecture
- Transfer Learning with PyTorch
- Fully retrained classifier layer
- Softmax output for multi-class prediction
- Quantized model optimization for lightweight inference

Instead of relying on handcrafted feature engineering, the model automatically learns visual patterns and discriminative features directly from training images.

## Performance

- ~97% accuracy on Validation Set

# Deployment Optimization

To support deployment on free-tier cloud hosting services with limited CPU and memory resources, the model uses quantization techniques.

Benefits include:

- Reduced model size
- Lower RAM consumption
- Faster CPU inference
- Better compatibility with free web hosting instances

This allows the application to run on lightweight environments such as Railway free instances without requiring GPU acceleration.

# Web Interface Usage

1. Access the web interface at:

    https://garbage-classifier-v85s.onrender.com

> Note: The application is hosted on a free hosting instance. It may take up to a minute to load initially while the server spins up from inactivity.

2. Drag and drop an image, or click to upload.

3. Click **"Analyze"**

4. Wait for the prediction process to finish.

5. View the result and prediction probabilities.

# Requirements

To run this project locally, ensure Python is installed.

## 1. Clone the repository

```bash
git clone https://github.com/quocha16/garbage-classifier.git
cd garbage-classifier
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Run the application

```bash
python app.py
```

# License

This project is licensed under the **MIT License**.
