
# Brain Tumor Detection Using CNN on MRI Images

## **1. Project Overview**

This repository contains an end-to-end deep learning system for detecting brain tumors from MRI images. A custom Convolutional Neural Network (CNN) is trained to classify MRI scans into **tumor** and **no_tumor** categories. The project includes dataset preparation, model design, training, evaluation, explainability with Grad-CAM, and deployment using **FastAPI** and **Streamlit**.

---

## **2. Problem Statement**

Early detection of brain tumors significantly improves treatment success. However, MRI analysis is time-consuming and requires expert interpretation. This project aims to build a reliable automated classifier that assists in identifying tumors in MRI scans using deep learning, thereby reducing workload and improving consistency in diagnosis.

---

## **3. Dataset**

**Source:** Brain MRI Images for Brain Tumor Detection (Kaggle)

**Classes:**

* tumor
* no_tumor

**Preprocessing:**

* Resize to 150×150
* Normalize pixel values (0–1)
* Data augmentation (rotation, horizontal flip, zoom, shear)
* 80% training and 20% validation split

---

## **4. Methodology**

### **4.1 Data Processing**

* Checked labels and cleaned inconsistent class folders
* Applied augmentations to prevent overfitting
* Created training and validation generators

### **4.2 Model Training**

The model was built using TensorFlow/Keras with:

* Conv2D layers with ReLU and L2 regularization
* BatchNormalization for stable convergence
* MaxPooling for spatial downsampling
* Dropout to avoid overfitting
* Fully connected layers ending in a sigmoid neuron
<img width="592" height="766" alt="Screenshot 2025-11-27 224629" src="https://github.com/user-attachments/assets/7d89a069-3220-4efc-8968-0aad75674d4d" />


**Training configuration:**

* Optimizer: Adam (learning rate: 0.0007)
* Loss: Binary cross-entropy
* Metrics: Accuracy, Precision, Recall, AUC
* Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

---

## **5. Model Architecture**

```
Conv2D → BatchNorm → MaxPool  
Conv2D → BatchNorm → MaxPool  
Conv2D → BatchNorm → MaxPool  
Flatten  
Dense → Dropout  
Dense → Sigmoid
```

---

## **6. Streamlit Interface**

A user-friendly interface was built with Streamlit.

**Features:**

* Upload MRI image
* See model prediction (tumor or no_tumor)
* View Grad-CAM heatmap
* Confidence score display

**Screenshot :**

<img width="1536" height="1024" alt="c1160b3d-c11d-4b52-b70e-e79a258f74bc" src="https://github.com/user-attachments/assets/3353f4dd-20b6-44e6-9657-e7ad0f486d1a" />




---

## **7. Results**

**Final Accuracy Achieved:** 82%

**Evaluation Outputs:**

* Validation accuracy and loss curves
* Confusion matrix
* Classification report
* Grad-CAM visualizations highlighting tumor regions
<img width="1308" height="481" alt="image" src="https://github.com/user-attachments/assets/9ad6e2ca-1633-457b-9605-acb442ba0cbd" />
<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/0bda0428-3665-4dec-a16e-8b98a6d9ea57" />



**Example Grad-CAM output:**
<img width="1390" height="472" alt="image" src="https://github.com/user-attachments/assets/3fb46d15-692f-4a8c-80b9-28973439f3f9" />
0<img width="1520" height="503" alt="image" src="https://github.com/user-attachments/assets/9d8607e0-7da6-47c5-8a11-d6480ae443ce" />


---

## **8. Deployment**

* **FastAPI** used for serving the model through a `/predict` endpoint.
* **Streamlit** front-end interacts with FastAPI for live inference.
* Allows real-time classification of MRI scans uploaded by users.

---



## **9. Future Work**

* Hyperparameter tuning
* Transfer learning integration
* Multi-class tumor subtype classification
* Improved Grad-CAM overlays
* Dockerized deployment

---

