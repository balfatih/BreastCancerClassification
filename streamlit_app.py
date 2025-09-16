import streamlit as st
import numpy as np
import joblib
from PIL import Image
import tensorflow as tf
import os

st.title("Breast Cancer Hybrid Classification")
st.write("Upload a breast image and the model will classify it.")


# ----------------------
# CNN Model Yükleme
# ----------------------
@st.cache_resource
def load_cnn_model():
    model_path = "CNN_Model.h5"  # aynı klasörde olmalı
    cnn_model = tf.keras.models.load_model(model_path)
    
    # Feature extraction için ara katmanı seçin
    # Örn. flatten veya global average pooling layer
    feature_layer = cnn_model.get_layer("flatten")  # kendi katman adınızı kontrol edin
    feature_extractor = Model(inputs=cnn_model.input, outputs=feature_layer.output)
    return feature_extractor

cnn_model = load_cnn_model()

# ----------------------
# GBM Model Yükleme
# ----------------------
@st.cache_resource
def load_gbm_model():
    return joblib.load("CNN_GBM_model.joblib")  # aynı klasörde olmalı

gbm_model = load_gbm_model()

# ----------------------
# Görüntü Yükleme ve İşleme
# ----------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

