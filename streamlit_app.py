import streamlit as st
import numpy as np
import joblib
from PIL import Image
import tensorflow as tf
import os
from tensorflow.keras.models import Model

st.title("Breast Cancer Hybrid Classification")
st.write("Upload a breast image and the model will classify it.")

# ----------------------
# CNN Model Yükleme
# ----------------------
@st.cache_resource
def load_cnn_model():
    model_path = "CNN_Model.h5"
    cnn_model = tf.keras.models.load_model(model_path)
    # Flatten sonrası Dense(558) katmanı output olarak
    # Son 3. katmanı kullanıyoruz
    feature_layer = cnn_model.layers[-3]
    
    # Functional wrapper: feature extraction için
    feature_extractor = tf.keras.Model(inputs=cnn_model.input, outputs=feature_layer.output)
    return feature_extractor

cnn_model = load_cnn_model()

# ----------------------
# GBM Model Yükleme
# ----------------------
@st.cache_resource
def load_gbm_model():
    gbm_path = "CNN_GBM_model.joblib"
    return joblib.load(gbm_path)

gbm_model = load_gbm_model()

# ----------------------
# Görüntü Yükleme ve Ön İşleme
# ----------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    IMG_SIZE = 128
    img_array = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_array).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1,128,128,3)

    # CNN ile feature extraction
    features = cnn_model.predict(img_array)

    # GBM ile tahmin
    prediction = gbm_model.predict(features)
    
    st.subheader("Prediction Result")
    labels = {0: "Benign", 1: "Malignant"}
    predicted_label = labels.get(prediction[0], str(prediction[0]))
    st.write(f"Predicted Class: **{predicted_label}**")

    # Olasılıkları göster
    if hasattr(gbm_model, "predict_proba"):
        proba = gbm_model.predict_proba(features)[0]
        st.subheader("Class Probabilities")
        st.write({"Benign": proba[0], "Malignant": proba[1]})
