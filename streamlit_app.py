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

model_path = tf.keras.models.load_model("CNN_Model.h5")
