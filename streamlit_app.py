import streamlit as st 
import streamlit as st
import numpy as np
import joblib
from PIL import Image
import tensorflow as tf
import gdown
import os

st.title('Breast Cancer Classification')
st.write("Upload a breast image and the model will classify it.")
 

# ----------------------
# CNN Model Yükleme
# ----------------------
def load_cnn_model():
    model_path = "cnn_model.h5"
    if not os.path.exists(model_path):
        url = "https://drive.google.com/file/d/18TuFKSnLCySlrdiZ8M-ySIRqI5adu3oW/"
        import gdown
        gdown.download(url, model_path, quiet=False)
    import tensorflow as tf
    return tf.keras.models.load_model(model_path)

cnn_model = load_cnn_model()
