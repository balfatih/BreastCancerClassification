import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

st.title('Breast Cancer Classification')
st.write("Upload a breast image and the model will classify it.")

model = joblib.load("CNN_GBM_model.joblib")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
    # Görüntüyü aç
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Ön işleme
    image = image.resize((128, 128))  # Eğitim boyutuna uygun hale getir
    img_array = np.array(image).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1,128,128,3)

    # Tahmin
    prediction = model.predict(img_array)

    # Eğer çıktı sınıf ise:
    st.write("### Prediction Result:")
    st.write(prediction)

    # Eğer çıktı olasılıklar ise:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(img_array)
        st.write("### Prediction Probabilities:")
        st.write(proba)
