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

# ----------------------
# GBM Model Yükleme
# ----------------------
@st.cache_resource
def load_gbm_model():
    return joblib.load("CNN_GBM_model.joblib")

gbm_model = load_gbm_model()
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
    # Görüntüyü aç ve göster
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Ön İşleme: 128x128 ve normalize
    image = image.resize((128, 128))
    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1,128,128,3)

    # CNN ile feature çıkar
    features = cnn_model.predict(img_array)

    # GBM ile tahmin
    prediction = gbm_model.predict(features)

    # Tahmini ekrana yaz
    st.subheader("Prediction Result")
    
    # Örnek etiketler: 0 = Benign, 1 = Malignant
    labels = {0: "Benign", 1: "Malignant"}
    predicted_label = labels.get(prediction[0], str(prediction[0]))
    
    st.write(f"Predicted Class: **{predicted_label}**")

    # Olasılıkları göstermek isterseniz
    if hasattr(gbm_model, "predict_proba"):
        proba = gbm_model.predict_proba(features)
        st.subheader("Class Probabilities")
        st.write(proba)
