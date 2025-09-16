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
