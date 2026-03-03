# Skin Disease Classification using Deep Learning

This project implements a multi-class skin disease classification system using deep learning and transfer learning techniques. The goal is to automatically classify dermoscopic images into different disease categories, simulating a real-world tele-dermatology workflow.

---

## 📌 Dataset

- HAM10000 Dataset
- Multi-class skin lesion dataset
- Stratified train-validation-test split (70-15-15)

---

## 🧠 Models Implemented

- ResNet50 (Transfer Learning)
- VGG16 (Transfer Learning)
- MobileNetV2 (Transfer Learning)

Final fully connected layers were modified to match the number of disease classes.

---

## ⚙️ Training Details

- Image size: 224x224
- Optimizer: Adam
- Loss: Weighted Cross-Entropy Loss (to handle class imbalance)
- Data augmentation: Brightness/Contrast, Horizontal Flip
- Epochs: 10

---

## 📊 Evaluation Metrics

- Accuracy
- Precision (Weighted)
- Recall (Weighted)
- F1-Score (Weighted)
- Confusion Matrix
- Misclassification Analysis
- Group-wise Fairness Analysis

---

## 📈 Model Comparison

Three models were compared based on:

- Overall F1-score
- Minority class detection
- Confusion matrix patterns

ResNet50 demonstrated strong feature extraction and superior performance compared to lightweight architectures.

---

## 📁 Project Structure
Skin-Disease-Classification/
│
├── data
├── config.py
├── dataset.py
├── train.py
├── utils.py
├── experiments.py
├── models/
├── results/
├── roc_curve.py
├── roc_3modelcomparison.py
├── gradcam.py
├── requirements.txt
├── .gitignore
└── README.md

---
## Model Interpretability

- Grad-CAM visualization for understanding prediction regions
- Per-class ROC curves for evaluating classification separability

## Future Improvements

->Perform cross-dataset validation

->Deploy as web-based diagnostic tool

## 🚀 How to Run
```bash
pip install -r requirements.txt
python train.py 