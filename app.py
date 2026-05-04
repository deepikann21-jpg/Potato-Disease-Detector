import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
import os

# Load TFLite model using absolute path
model_path = os.path.join(os.path.dirname(__file__), "potato_disease_model.tflite")
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

st.title("🥔 Potato Leaf Disease Detector")
st.write("Upload a potato leaf image to detect the disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((224, 224))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    predictions = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    st.success(f"**Prediction:** {predicted_class}")
    st.info(f"**Confidence:** {confidence:.2f}%")
