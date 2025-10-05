# CIFAR-10: A Step-by-Step Deep Learning Journey

### Author: Pranav A Kumar  
### Role: Data & ML Enthusiast | Allianz Technology


## Overview
This repository documents a hands-on learning journey through image classification using the **CIFAR-10 dataset**.

The goal was to understand how model complexity and data preprocessing techniques influence performance — progressing gradually from simple networks to advanced transfer learning.

---

## Project Structure
Each notebook represents a distinct learning milestone:

| Notebook | Focus | Description |
|-----------|--------|-------------|
| **01_ANN_on_CIFAR10.ipynb** | Artificial Neural Network | Baseline fully connected model on flattened images |
| **02_KMeans_Clustering.ipynb** | Unsupervised Learning | K-Means clustering on PCA-reduced CIFAR-10 features |
| **03_CNN_Basic.ipynb** | Convolutional Neural Network | Custom CNN from scratch for image classification |
| **04_CNN_Data_Augmentation.ipynb** | Data Augmentation | Techniques like rotation, flipping, and shifting to improve generalization |
| **05_Transfer_Learning.ipynb** | Transfer Learning | Feature extraction and fine-tuning using MobileNetV2 pretrained on ImageNet |

---

## Key Learnings
- The **ANN baseline** achieves moderate accuracy (~50%) but struggles to capture spatial information.  
- **Clustering** revealed how similar classes (like cars vs trucks) overlap in feature space.  
- The **CNN model** improved accuracy (~84%) by leveraging spatial hierarchies.  
- **Data augmentation** reduced overfitting and improved model robustness.  
- **Transfer learning** (MobileNetV2) delivered strong performance (~87% accuracy) with significantly less training time.

---

## Results Summary
| Model | Method | Validation Accuracy |
|--------|--------|----------------------|
| ANN | Fully connected layers | ~50% |
| CNN | Basic convolutional model | ~84% |
| CNN + Augmentation | Data augmentation enabled | ~75% |
| Transfer Learning | MobileNetV2 pretrained on ImageNet | ~87% |

---

## Dataset
The CIFAR-10 dataset is automatically loaded using Keras:
python
from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


Dependencies

Python 3.10+
TensorFlow / Keras
NumPy
Matplotlib
scikit-learn
Seaborn
JupyterLab


Reflection

This repository represents more than just model building — it reflects a structured approach to learning deep learning fundamentals, visual understanding, and practical experimentation.

The progression from ANN → CNN → Transfer Learning mirrors how real-world ML problems evolve — from building a foundation to applying state-of-the-art methods.
