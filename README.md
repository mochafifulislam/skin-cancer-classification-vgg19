# 🩺 Skin Lesion Image Classification Using VGG19 Transfer Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project focuses on the classification of **seven types of skin lesions**, including cancerous and non-cancerous categories, using the **HAM10000 dataset** and a **Transfer Learning** approach based on the **VGG19** deep learning architecture.

---

## 📌 Table of Contents

* [Background](#-background)
* [Dataset](#-dataset)
* [Model Architecture](#-model-architecture)
* [Project Structure](#-project-structure)
* [Installation & Usage](#-installation--usage)
* [Results & Evaluation](#-results--evaluation)
* [License](#-license)
* [Contact](#-contact)

---

## 🔬 Background

Early detection of skin cancer, particularly **Melanoma**, is important for improving the chances of successful treatment.

Deep Learning applied to **dermoscopic medical images** can support the analysis of visual patterns in skin lesions and potentially assist dermatologists in the early assessment process.

This project uses **VGG19 Transfer Learning** to learn visual representations from dermoscopic images and classify them into seven predefined skin-lesion categories.

> ⚠️ **Disclaimer:** This project is intended for research and educational purposes only. The model is not a medical diagnostic tool and should not be used as a substitute for professional medical examination or diagnosis.

---

# 📊 Dataset: HAM10000

The project uses the **HAM10000 (Human Against Machine with 10,000 training images)** dataset, which contains **10,015 labeled dermoscopic images** covering seven diagnostic categories.

## 🩻 Seven Skin Lesion Categories

| Code    | Category                            | Description                           |
| :------ | :---------------------------------- | :------------------------------------ |
| `nv`    | Melanocytic nevi                    | Benign melanocytic lesions            |
| `mel`   | Melanoma                            | Malignant melanocytic tumor           |
| `bkl`   | Benign keratosis-like lesions       | Benign keratotic lesions              |
| `bcc`   | Basal cell carcinoma                | Malignant skin tumor                  |
| `akiec` | Actinic keratoses / Bowen's disease | Precancerous / intraepithelial lesion |
| `vasc`  | Vascular lesions                    | Lesions of vascular origin            |
| `df`    | Dermatofibroma                      | Benign fibrous skin lesion            |

The classification task is therefore a **7-class image classification problem**.

---

# 🏗️ Model Architecture

The project applies **Transfer Learning using VGG19**, a convolutional neural network pretrained on the **ImageNet** dataset.

### Base Model

```text
VGG19
├── Pretrained on ImageNet
├── Convolutional Feature Extractor
└── Initial weights frozen
```

The pretrained VGG19 network is used as the feature extraction backbone, while a custom classification head is added for the seven skin-lesion categories.

### Custom Classification Head

The classification head consists of:

* `GlobalAveragePooling2D` / `Flatten`
* `BatchNormalization`
* `Dense` layer with 256/512 units
* `ReLU` activation
* `Dropout` with a rate of 0.3–0.5
* Final `Dense` layer with 7 units
* `Softmax` activation

The overall architecture can be represented as:

```text
Dermoscopic Image
        │
        ▼
   Preprocessing
        │
        ▼
      VGG19
  ImageNet Weights
        │
        ▼
Feature Extraction
        │
        ▼
GlobalAveragePooling
       / \
    Flatten
        │
        ▼
Batch Normalization
        │
        ▼
Dense Layer
256 / 512 Units
        │
        ▼
      ReLU
        │
        ▼
     Dropout
    0.3 – 0.5
        │
        ▼
Dense Layer
     7 Units
        │
        ▼
     Softmax
        │
        ▼
Skin Lesion Class
```

---

## ⚙️ Training Configuration

| Component          | Configuration                   |
| :----------------- | :------------------------------ |
| Architecture       | VGG19                           |
| Pretrained Weights | ImageNet                        |
| Learning Strategy  | Transfer Learning / Fine-Tuning |
| Optimizer          | Adam                            |
| Learning Rate      | `1e-4`                          |
| Loss Function      | Categorical Crossentropy        |
| Output Classes     | 7                               |
| Output Activation  | Softmax                         |
| Regularization     | Dropout                         |
| Dropout Rate       | 0.3–0.5                         |

The initial training stage uses the pretrained VGG19 weights as the foundation for feature extraction before adapting the model to the HAM10000 classification task.

---

# 🧠 Transfer Learning Workflow

The training workflow can be summarized as:

```text
HAM10000 Dataset
       │
       ▼
Image Preprocessing
       │
       ▼
Train / Validation / Test Data
       │
       ▼
Pretrained VGG19
       │
       ▼
Freeze Base Layers
       │
       ▼
Add Custom Classification Head
       │
       ▼
Train Classification Layers
       │
       ▼
Fine-Tuning
       │
       ▼
7-Class Prediction
       │
       ▼
Model Evaluation
```

Transfer Learning allows the model to leverage visual representations learned from the large-scale ImageNet dataset rather than training the entire convolutional network from scratch.

---

# 📈 Results & Evaluation

Model performance should be evaluated using standard multiclass classification metrics, including:

### Accuracy

```text
Accuracy =
Correct Predictions
───────────────────
Total Predictions
```

### Precision

```text
Precision =
TP
──────────
TP + FP
```

### Recall

```text
Recall =
TP
──────────
TP + FN
```

### F1-Score

```text
F1-Score =
2 × Precision × Recall
─────────────────────
Precision + Recall
```

### Confusion Matrix

A **confusion matrix** can be used to analyze the classification behavior of the model across all seven lesion categories and identify classes that are frequently confused with one another.

> **Note:** The original project description does not provide specific numerical evaluation results. Therefore, no performance values are stated here to avoid introducing unsupported results.


---

# 🛠️ Technology Stack

```text
Python
│
├── TensorFlow / Keras
│   └── VGG19
│
├── NumPy
│   └── Numerical computation
│
├── Pandas
│   └── Dataset management
│
├── Matplotlib
│   └── Visualization
│
└── Scikit-learn
    └── Model evaluation
```

---

# 📊 Output

After training and evaluation, the project can generate:

* Training and validation loss curves
* Training and validation accuracy curves
* Classification report
* Confusion matrix
* Per-class precision
* Per-class recall
* Per-class F1-score
* Final model weights

Example workflow:

```text
Training
   │
   ├── Accuracy
   ├── Loss
   │
   ▼
Validation
   │
   ├── Accuracy
   └── Loss
   │
   ▼
Test Set
   │
   ├── Classification Report
   ├── Confusion Matrix
   └── Per-Class Metrics
```

---

# ⚠️ Limitations

Several considerations are important when interpreting the results of a skin-lesion classification model:

* HAM10000 is a curated dermoscopic image dataset and may not represent every real-world clinical setting.
* Differences in image acquisition conditions can affect model performance.
* Some lesion categories may be more difficult to distinguish than others.
* Class distribution should be considered when interpreting overall accuracy.
* A high classification score does not imply clinical diagnostic reliability.
* The model should not be used independently for medical diagnosis.

---

# 🔐 Medical & Data Disclaimer

This repository is intended for **research, experimentation, and educational purposes**.

The model should **not** be used to diagnose skin cancer, determine treatment, or make clinical decisions.

Any medical interpretation should be performed by qualified healthcare professionals using appropriate clinical examination, patient history, and diagnostic procedures.

---

# 📌 Project Summary

This project demonstrates the application of **Deep Learning and Transfer Learning** to multiclass skin-lesion image classification.

The overall pipeline is:

```text
HAM10000
   ↓
7 Skin Lesion Classes
   ↓
Image Preprocessing
   ↓
Pretrained VGG19
   ↓
Custom Classification Head
   ↓
Fine-Tuning
   ↓
Softmax Prediction
   ↓
7-Class Skin Lesion Classification
```

The project provides an example of how pretrained convolutional neural networks can be adapted to specialized medical-image classification tasks.

---

# 👨‍💻 Author

**Moch Afiful Islam**

Undergraduate Mathematics Student
Universitas Brawijaya, Indonesia

### Research Interests

* Artificial Intelligence
* Machine Learning
* Deep Learning
* Computer Vision
* Medical Image Analysis
* Image Classification
* Data Science
