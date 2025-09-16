import streamlit as st
import numpy as np
from PIL import Image
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------
# Sayfa Ayarları
# --------------------------
st.set_page_config(page_title="Breast Cancer Classification", layout="centered")
st.title("Breast Cancer Classification")
st.write("This app classifies breast cancer images using precomputed features and a GBM model.")

# --------------------------
# GBM Modelini Yükle
# --------------------------
@st.cache_resource
def load_gbm_model():
    return joblib.load("CNN_GBM_model.joblib")  # GBM model dosyanızı buraya koyun

gbm_model = load_gbm_model()

# --------------------------
# Feature ve Label Yükle
# --------------------------
@st.cache_data
def load_features():
    x_test = np.load("x_test.npy")  # Precomputed features
    y_test = np.load("y_test.npy")  # Corresponding labels
    return x_test, y_test

x_test, y_test = load_features()

# --------------------------
# Kullanıcıdan Görüntü Yükleme
# --------------------------
uploaded_file = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Görüntüyü aç ve göster
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Görüntüyü 128x128 boyutuna getir ve normalize et
    IMG_SIZE = 128
    img_array = np.array(image.resize((IMG_SIZE, IMG_SIZE))).astype("float32") / 255.0
    img_array = img_array.flatten().reshape(1, -1)  # Flatten ve 2D array (1, feature_dim)

    # --------------------------
    # En yakın feature'ı bul
    # --------------------------
    similarities = cosine_similarity(img_array, x_test)
    closest_idx = np.argmax(similarities)
    selected_feature = x_test[closest_idx].reshape(1, -1)
    true_label = y_test[closest_idx]

    # --------------------------
    # GBM ile Tahmin
    # --------------------------
    prediction = gbm_model.predict(selected_feature)
    predicted_label = {0: "Benign", 1: "Malignant"}[prediction[0]]

    st.subheader("Prediction Result")
    st.write(f"Predicted Class: **{predicted_label}**")
    st.write(f"Closest Match True Class: **{ {0:'Benign', 1:'Malignant'}[true_label] }**")

    # Eğer GBM modelinde predict_proba varsa, olasılıkları göster
    if hasattr(gbm_model, "predict_proba"):
        proba = gbm_model.predict_proba(selected_feature)[0]
        st.subheader("Class Probabilities")
        st.write({"Benign": float(proba[0]), "Malignant": float(proba[1])})
