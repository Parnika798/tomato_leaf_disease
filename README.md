

# 🌿 Tomato Leaf Disease & Severity Prediction (Multi-Task Learning)

## 📌 Overview

This project presents a **multi-task deep learning framework** for automated plant disease analysis. The model simultaneously performs:

* 🌱 **Disease Classification** (6 classes)
* 📊 **Severity Prediction** (3 stages)

from a **single tomato leaf image**, improving decision-making for precision agriculture.

---

## 🚀 Key Features

* ✅ Multi-task learning (MTL) architecture
* ✅ Handles **partial labels** (severity only for diseased samples)
* ✅ **Cross-task attention mechanism** (disease → severity guidance)
* ✅ Advanced augmentations (Albumentations + MixUp + CutMix)
* ✅ Class imbalance handling using **WeightedRandomSampler**
* ✅ BMC-level research implementation

---

## 🧠 Problem Formulation

Given an input image:

[
x \in \mathbb{R}^{3 \times 224 \times 224}
]

The model predicts:

* Disease: ( y_d \in {1,...,6} )
* Severity: ( y_s \in {1,...,3} )

Each sample:
[
(x, y_d, y_s, m)
]

* ( y_s = -1 ) if unavailable
* ( m \in {0,1} ) → severity mask

---

## 📂 Dataset

* Source: Public Kaggle dataset
* Link : https://www.kaggle.com/datasets/janiruwalisingha/tomato-leaf-disease-severity-dataset
* Combined:

  * Disease classification dataset
  * Severity prediction dataset

Unified into a custom **multi-task dataset pipeline** 

### ⚙️ Preprocessing

* Resize → 224×224
* Normalize → ImageNet stats

### 🔁 Augmentations

* Random crop, flip
* Color jitter
* Gaussian noise, blur
* Elastic transform
* Coarse dropout
* MixUp & CutMix

---

## 🏗️ Model Architecture

### 🔹 Backbone

* Pretrained **ResNet50**

### 🔹 Attention Module

* **CBAM (Channel + Spatial Attention)**

### 🔹 Heads

* Disease Classification Head
* Severity Prediction Head

### 🔹 Cross-Task Attention

* Severity branch uses disease features
* Disease gradients **detached** to prevent interference

---

## 📉 Loss Function

[
\mathcal{L} = \lambda_d \mathcal{L}_d + \lambda_s \mathcal{L}_s
]

* ( \lambda_d = 0.6 ), ( \lambda_s = 0.4 )
* Cross-entropy with label smoothing

### ⚠️ Masked Severity Loss

[
\mathcal{L}_s =
\begin{cases}
\text{CE}(\hat{y}_s, y_s) & m=1 \
0 & m=0
\end{cases}
]

---

## 🏋️ Training Details

| Parameter         | Value           |
| ----------------- | --------------- |
| Optimizer         | AdamW           |
| LR                | 1e-4            |
| Batch Size        | 32              |
| Epochs            | 50              |
| Scheduler         | Warmup + Cosine |
| Early Stopping    | Patience = 10   |
| Gradient Clipping | 1.0             |

### 🧊 Training Strategy

* Freeze backbone → first 5 epochs
* Then full fine-tuning

---

## 📊 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score (weighted)
* AUC

Evaluated separately for:

* Disease classification
* Severity prediction

---

## 🧪 Ablation Studies

* MTL vs Single-task
* Loss weight tuning
* Backbone comparison
* Attention module impact


## 🧾 Requirements

* Python 3.8+
* PyTorch
* Albumentations
* NumPy
* Scikit-learn
* OpenCV

---

## 💡 Key Contributions

* Novel **cross-task attention mechanism**
* Effective **handling of missing labels via masking**
* Strong generalization via hybrid augmentation strategy
* Real-world applicable precision agriculture solution


