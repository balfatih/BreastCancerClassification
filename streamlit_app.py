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
    model_path = "cnn_model_original_dataset.keras"
    return tf.keras.models.load_model(model_path)

cnn_model = load_cnn_model()
