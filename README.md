# AI-Based Cow & Buffalo Breed Classification Using Custom CNN

> **Published Research** — 2026 IEEE International Conference  
> Co-authored by Prajakta Kuila, Swaraj Kumar Behera, Amiya Ranjan Panda et al., KIIT Deemed to be University

[![IEEE](https://img.shields.io/badge/IEEE-Published-blue)](https://ieeexplore.ieee.org/document/11496187)
[![Accuracy](https://img.shields.io/badge/Accuracy-92.5%25-brightgreen)]()
[![Breeds](https://img.shields.io/badge/Breeds-64-orange)]()
[![Python](https://img.shields.io/badge/Python-3.10+-yellow?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14+-orange?logo=tensorflow)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.24+-red?logo=streamlit)](https://streamlit.io)

---

## Overview

An end-to-end deep learning system that classifies **64 Indian cow and buffalo breeds** from images using a custom-built Convolutional Neural Network (CNN). The model achieves **92.5% overall accuracy** on the validation set, outperforming ResNet50, VGG16, DenseNet121, MobileNetV2, and EfficientNetB0, and is deployed as a Streamlit web application with Google OAuth login.

Accurate breed identification is critical for precision livestock management — enabling selective breeding, health monitoring, and dairy yield prediction. Conventional manual inspection is labor-intensive and error-prone; this system automates that process end-to-end.

---

## Key Results

| Metric | Value |
|---|---|
| Overall Accuracy | **92.5%** |
| Weighted Precision | 0.928 |
| Weighted Recall | 0.931 |
| Weighted F1-Score | **0.929** |
| Specificity | 0.941 |
| No. of Breeds | 64 |
| Dataset Size | ~8,700 images (after preprocessing) |

### Comparison with Pretrained Architectures

| Model | Accuracy | F1-Score |
|---|---|---|
| VGG16 | 0.852 | 0.84 |
| ResNet50 | 0.887 | 0.88 |
| DenseNet121 | 0.895 | 0.89 |
| MobileNetV2 | 0.901 | 0.90 |
| EfficientNetB0 | 0.912 | 0.91 |
| **Custom CNN (Ours)** | **0.925** | **0.929** |

---

## Model Architecture

Custom CNN with **5 convolutional blocks** followed by dense layers:

```
Input (128×128×3)
  → Conv2D(32) + MaxPool
  → Conv2D(64) + MaxPool
  → Conv2D(128) + MaxPool
  → Conv2D(256) + MaxPool
  → Conv2D(512) + MaxPool
  → Dense(512) + Dropout(0.5)
  → Dense(256) + Dropout(0.5)
  → Dense(64, softmax)
```

- **Regularization**: Dropout (rate = 0.5) after each dense layer to prevent overfitting
- **Training**: Adam optimizer, categorical cross-entropy loss, learning rate 0.001, 100 epochs, batch size 32, early stopping

---

## Repository Structure

```
cattle_breed_classification/
├── AI_Breed_Classification_Complete.ipynb  # Full pipeline: preprocessing → training → export
├── app.py                                  # Streamlit web application
├── requirements.txt                        # Python dependencies
├── models/                                 # Saved model files
│   ├── animal_classifier_savedmodel/       # TF SavedModel (used by app.py)
│   └── model.json                          # Class index → breed name mapping
└── .devcontainer/                          # GitHub Codespaces config
```

---

## Pipeline

| Phase | Description |
|---|---|
| **1 — Setup** | Mount Drive, install dependencies, define unified paths |
| **2 — Dataset Organization** | Flatten `species/breed` folders, 80/10/10 train/val/test split |
| **3 — Preprocessing** | Duplicate removal (imagehash), blur detection (OpenCV), corrupt image cleanup |
| **4 — Augmentation** | Flip, rotate, zoom, brightness & contrast variation (3× effective dataset size) |
| **5 — Model Training** | 5-block Custom CNN, 100 epochs, Adam optimizer, early stopping |
| **6 — Evaluation** | Confusion matrix, classification report, Grad-CAM visualizations |
| **7 — Export** | `.keras` model, TF SavedModel, `model.json` class mapping |

---

## Dataset

- **~9,200 raw images** → 8,700 after preprocessing, across 64 Indian cattle breeds (e.g., Gir, Sahiwal, Murrah, Jafarabadi)
- Sources: ICAR-NDRI, DAHD, Kaggle Cattle Breeds dataset, Dairy DigiD, and field-level farm photographs from Odisha
- **3x augmentation** applied — random rotation, flipping, brightness/saturation adjustments, zoom, shear, and Gaussian noise injection
- Images collected across varied lighting, pose, and background conditions for real-world generalization

---

## Interpretability — Grad-CAM

Grad-CAM visualizations confirmed the model correctly focuses on breed-specific features:
- Horn curvature and orientation
- Coat patterns and markings
- Facial structure and ear shape

Misclassifications were primarily limited to visually similar or rare breeds (e.g., Punganur and Toda) due to limited training samples.

---

## Tech Stack

- **Language**: Python 3.10+
- **Deep Learning**: TensorFlow, Keras
- **Image Processing**: OpenCV, Pillow (PIL), imagehash
- **Data Handling**: NumPy, Pandas, Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **App Interface**: Streamlit, SQLite
- **Auth**: Google OAuth 2.0

---

## Running the App

```bash
# Clone the repository
git clone https://github.com/prajakta-k19/cattle_breed_classification
cd cattle_breed_classification

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

The app supports:
- Image upload (JPG, PNG, JPEG)
- Live camera capture
- Google OAuth + email/password login
- Top-3 breed predictions with confidence scores
- Prediction history per session

---

## Applications

- Precision livestock management and selective breeding
- Automated dairy farm monitoring
- Veterinary health tracking and yield prediction
- Integration with IoT devices for real-time edge inference

---

## Publication

> Amiya Ranjan Panda, Swaraj Kumar Behera, **Prajakta Kuila**, Vidya Mohanty, Subhashree Mishra, Manoj Kumar Mishra.  
> *"AI-Based Cow & Buffalo Breed Classification Using Custom CNN"*  
> 2026 IEEE International Conference — [View on IEEE Xplore](https://ieeexplore.ieee.org/document/11496187)

---

## Authors

| Name | Institution |
|---|---|
| Amiya Ranjan Panda | KIIT Deemed to be University |
| **Swaraj Kumar Behera** | KIIT Deemed to be University |
| **Prajakta Kuila** | KIIT Deemed to be University |
| Vidya Mohanty | Aryan Institute of Engineering and Technology |
| Subhashree Mishra | KIIT Deemed to be University |
| Manoj Kumar Mishra | KIIT Deemed to be University |

---

## License

This project is licensed under the MIT License.