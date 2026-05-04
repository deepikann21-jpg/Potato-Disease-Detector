# 🥔 Potato Leaf Disease Detector

A deep learning web app that detects potato leaf diseases using a CNN model built with TensorFlow and deployed with Streamlit.

## 📄 Description
This app classifies potato leaf images into 3 categories:
- 🟡 Early Blight
- 🔴 Late Blight
- 🟢 Healthy

## 🚀 Demo
Live app: [Click here](https://share.streamlit.io)

## 🛠️ Tech Stack
- Python
- TensorFlow / Keras
- Streamlit
- Pillow
- NumPy

## 📂 Project Structure
```
potato-disease-detector/
├── app.py
├── requirements.txt
├── potato_disease_model.h5
└── README.md
```

## ▶️ How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📊 Model Details
- Architecture: Custom CNN
- Input Size: 224x224
- Classes: 3
- Test Accuracy: ~87%
