# 🌿 Tomato Leaf Disease & Severity Prediction using Multi-Task Learning

##  Overview

This project presents a **multi-task learning (MTL) framework** for simultaneous:

*  Disease Classification (6 classes)
*  Disease Severity Prediction (3 stages)

from a **single tomato leaf image**.

Unlike traditional approaches that focus only on classification, this work jointly models both tasks to provide a more **comprehensive understanding of plant health**, which is essential for precision agriculture.

---

##  Motivation

Most deep learning models in plant disease analysis are limited to **disease identification**, ignoring **severity estimation**, which is crucial for:

* Treatment planning
* Yield prediction
* Agricultural decision-making

This project addresses this gap by:

* Learning **shared representations** across tasks
* Modeling **inter-task relationships**
* Handling **partial supervision** (missing severity labels)

---

##  Model Architecture

### 🔹 Backbone

* Pretrained **ResNet50**

### 🔹 Feature Refinement

* **CBAM (Convolutional Block Attention Module)**

  * Channel attention
  * Spatial attention

### 🔹 Multi-Task Design

* Disease Classification Head
* Severity Prediction Head

### 🔹 Cross-Task Attention

* Severity prediction is guided using disease features
* Disease features are **detached from gradient flow** to avoid interference

This enables effective **inter-task feature sharing** while maintaining stability.

---

##  Dataset

* Public **Kaggle Tomato Leaf Dataset**

https://www.kaggle.com/datasets/janiruwalisingha/tomato-leaf-disease-severity-dataset

### Classes

**Disease (6 classes):**

* Bacterial Spot
* Early Blight
* Late Blight
* Leaf Mold
* Septoria Leaf Spot
* Healthy

**Severity (3 stages):**

* Early
* Mid
* Late

Severity labels are available **only for diseased samples**, making this a **partial supervision problem** 

---

##  Preprocessing & Augmentation

* Resize → 224 × 224
* Normalize → ImageNet statistics

### Augmentations

* Geometric transformations
* Color perturbations
* MixUp
* CutMix

These improve generalization under real-world variations.

---

## Loss Function

[
\mathcal{L} = 0.6 \mathcal{L}*{disease} + 0.4 \mathcal{L}*{severity}
]

* Cross-entropy with label smoothing
* **Masked severity loss** for missing labels

---

##  Training Details

| Parameter       | Value            |
| --------------- | ---------------- |
| Optimizer       | AdamW            |
| Batch Size      | 32               |
| Epochs          | 50               |
| Scheduler       | Cosine Annealing |
| Early Stopping  | Yes              |
| Backbone Freeze | First 5 epochs   |

---

##  Results

| Task                   | Accuracy   | F1 Score | AUC    |
| ---------------------- | ---------- | -------- | ------ |
| Disease Classification | **97.85%** | 0.9786   | 0.9984 |
| Severity Prediction    | **77.66%** | 0.7766   | 0.9297 |

* High accuracy for disease classification
* Strong improvement in severity prediction compared to single-task learning
* High AUC values indicate excellent discriminative capability across both tasks

<img width="991" height="731" alt="cm_disease_resnet50" src="https://github.com/user-attachments/assets/b2d06f6d-e21e-4b5b-a8a9-7bdcf70f0c4e" />

<img width="968" height="731" alt="cm_severity_resnet50" src="https://github.com/user-attachments/assets/af3642cb-36bd-4d4a-96cd-d563b43c631d" />



---

## 🔬 Experiments

### 1. Multi-Task vs Single-Task Learning

| Model       | Severity F1 |
| ----------- | ----------- |
| Single Task | 0.3248      |
| Multi-Task  | 0.7488      |

 Demonstrates effectiveness of shared learning

---

### 2. Ablation Study: Cross-Task Attention

| Model                   | Disease Accuracy | Severity F1 |
| ----------------------- | ---------------- | ----------- |
| Full Model              | 97.85%           | 0.7766      |
| Without Cross-Attention | 96.16%           | 0.7551      |

 Cross-task interaction improves performance, especially for severity prediction 

---

##  Requirements

* Python 3.8+
* PyTorch
* Albumentations
* NumPy
* Scikit-learn

---

##  Key Contributions

* Multi-task framework for joint disease and severity prediction
* Cross-task attention for modeling inter-task dependencies
* Effective handling of **partial labels via masking**
* Significant improvement in severity prediction over single-task models

---
