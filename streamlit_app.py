import streamlit as st
import numpy as np
import joblib
from PIL import Image
import tensorflow as tf
import os
from tensorflow.keras.models import Model, load_model


st.title("Breast Cancer Hybrid Classification")
st.write("Upload a breast image and the model will classify it.")


# CNN model yükle
@st.cache_resource
def load_cnn_model():
    model_path = "CNN_Model.h5"
    cnn_model = load_model(model_path)
    return cnn_model

cnn_model = load_cnn_model()

# Görüntü yükleme
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    IMG_SIZE = 128
    img_array = np.array(image.resize((IMG_SIZE, IMG_SIZE))).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # CNN ile tahmin
    prediction = cnn_model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]

    labels = {0: "Benign", 1: "Malignant"}
    st.subheader("Prediction Result")
    st.write(f"Predicted Class: **{labels[predicted_class]}**")

    st.subheader("Class Probabilities")
    st.write({"Benign": float(prediction[0][0]), "Malignant": float(prediction[0][1])})
