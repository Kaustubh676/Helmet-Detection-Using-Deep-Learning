# 🧠 Helmet Detection Using Deep Learning

This project focuses on detecting whether bike riders are **wearing helmets** or **not**, using a deep learning-based image classification model built with TensorFlow and MobileNetV2.

![Helmet Detection](https://img.shields.io/badge/model-MobileNetV2-brightgreen) ![Python](https://img.shields.io/badge/python-3.9-blue) ![License](https://img.shields.io/badge/license-MIT-blue)

---

## 📌 Project Overview

The goal of this project is to contribute to road safety by identifying riders without helmets in images. The model was trained using a custom dataset of annotated images.

### ✨ Key Features:
- Parsed `.xml` annotations to classify images into `With Helmet` and `Without Helmet`.
- Preprocessed and split data into training and validation sets.
- Used **MobileNetV2** for transfer learning with a custom classification head.
- Applied **image augmentation** to improve model generalization.
- Achieved ~92% validation accuracy.
- Evaluated with confusion matrix and tested on custom real-world images.

---

## 🧰 Tech Stack

- **Python**
- **TensorFlow / Keras**
- **MobileNetV2 (Transfer Learning)**
- **ImageDataGenerator (Augmentation)**
- **Google Colab**
- **XML Parsing** with `xml.etree.ElementTree`
- **Matplotlib / Seaborn** for visualization

---

## 📂 Dataset

- Consists of rider images with `.xml` annotation files.
- Classes: `With Helmet` and `Without Helmet`.
- Splitting: 80% training, 20% validation.
- Custom annotation parsing was used to organize the dataset.

---

## 📈 Model Training

```python
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    callbacks=[early_stop]
)
