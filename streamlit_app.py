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
    model_path = "CNN_Model.h5"  # model dosyası aynı klasörde olmalı
    cnn_model = tf.keras.models.load_model(model_path)
    
    # Flatten sonrası Dense 322 katmanını feature olarak alıyoruz
    feature_layer = cnn_model.get_layer("dense_1")
    feature_extractor = Model(inputs=cnn_model.input, outputs=feature_layer.output)
    return feature_extractor

cnn_model = load_cnn_model()

# ----------------------
# GBM Model Yükleme
# ----------------------
@st.cache_resource
def load_gbm_model():
    gbm_path = "CNN_GBM_model.joblib"  # GBM model dosyası aynı klasörde
    return joblib.load(gbm_path)

gbm_model = load_gbm_model()

# ----------------------
# Görüntü Yükleme ve Ön İşleme
# ----------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
