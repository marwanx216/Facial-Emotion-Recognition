# 🧠 Facial Emotion Recognition using CNN and ANN

This project applies **Deep Learning** techniques to automatically recognize **human emotions** from facial expressions using **Convolutional Neural Networks (CNNs)** and **Artificial Neural Networks (ANNs)**, implemented in **PyTorch**.

The system classifies faces into one of seven emotion categories:  
😃 **Happy**, 😔 **Sad**, 😡 **Angry**, 😨 **Fear**, 😲 **Surprise**, 😐 **Neutral**, 🤢 **Disgust**

---

## ⚙️ Project Overview

Facial emotion recognition plays a vital role in **human–computer interaction, psychology, and intelligent systems**.  
This project builds and compares two models:
- 🧩 **ANN Model** – a fully connected network to classify extracted facial features.  
- 🧠 **CNN Model** – a convolutional model that automatically learns spatial and visual patterns directly from images.

Both models are trained and evaluated on a labeled facial expression dataset, with GPU acceleration used to boost performance.

<img width="1200" height="400" alt="cnn_history" src="https://github.com/user-attachments/assets/57eaf91d-e185-4cce-ae7e-c9cb1f0d2d01" />

---

## 🚀 Features

- End-to-end emotion recognition pipeline  
- Custom CNN and ANN architectures built from scratch  
- GPU acceleration using **CUDA (NVIDIA GTX 1650)**  
- Real-time emotion inference support  
- Visualization tools for **accuracy, loss curves, and confusion matrices**  
- Clean modular structure with reusable training and testing functions  

---

<img width="3000" height="1800" alt="per_class_accuracy" src="https://github.com/user-attachments/assets/945f737a-8e0d-4e26-9548-f27c8de64200" />

## 🧩 Dataset

The models are trained on a **labeled facial emotion dataset** (compatible with datasets like FER-2013 or custom datasets structured as image folders).  
Each image belongs to one of seven emotion classes:
`['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']`



