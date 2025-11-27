# Brain Tumor Detection Using CNN (MRI Classification)
# 1. Problem Statement

Brain tumors require early and accurate diagnosis to improve patient outcomes. MRI scans are one of the most reliable imaging techniques, but manual interpretation is time-consuming and subject to human error.
This project aims to build an automated system that classifies MRI images as either Tumor or No Tumor using a Convolutional Neural Network (CNN).
The system also includes:

A trained deep learning model

A FastAPI backend for real-time inference

A Streamlit interface for user-friendly interaction

# 2. Methodology
Data Collection

Dataset used: Brain MRI Images for Tumor Detection

Contains MRI scans labelled as tumor and no tumor

Data Pre-processing

Image resizing (e.g., 150x150 or 224x224)

Normalization (pixel values scaled to [0,1])

Train/Validation/Test split

Data augmentation:

Rotation

Horizontal/vertical flips

Zoom

Brightness adjustments

Model Training

CNN architecture built using TensorFlow/Keras

EarlyStopping to prevent overfitting

ReduceLROnPlateau for adaptive learning rates

ModelCheckpoint to save the best model (best_balanced_model.keras)

Evaluation

Accuracy, Precision, Recall and F1 Score

Confusion Matrix

Grad-CAM heatmaps for interpretability
