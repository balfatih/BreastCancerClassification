import streamlit as st
import numpy as np
from PIL import Image
import joblib
from sklearn.metrics.pairwise import cosine_similarity

st.title("Breast Cancer Classification (Upload Image + Precomputed Features)")

# ----------------------
# GBM Model Yükleme
# ----------------------
@st.cache_resource
def load_gbm_model():
    return joblib.load("CNN_GBM_model.joblib")

gbm_model = load_gbm_model()

# ----------------------
# Feature ve Label Yükleme
# ----------------------
x_test = np.load("x_test.npy")  # (num_samples, 558)
y_test = np.load("y_test.npy")  # (num_samples, )

# ----------------------
# Görüntü Yükleme
# ----------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    IMG_SIZE = 128
    img_array = np.array(image.resize((IMG_SIZE, IMG_SIZE))).astype("float32") / 255.0
    img_array = img_array.flatten().reshape(1, -1)  # (1, 128*128*3)

    # ----------------------
    # En yakın feature bulma
    # ----------------------
    # Bu örnekte euclidean distance yerine cosine similarity kullanıyoruz
    similarities = cosine_similarity(img_array, x_test)
    closest_idx = np.argmax(similarities)  # En benzer feature index
    selected_feature = x_test[closest_idx].reshape(1, -1)
    true_label = y_test[closest_idx]

    # ----------------------
    # GBM ile Tahmin
    # ----------------------
    prediction = gbm_model.predict(selected_feature)
    predicted_label = {0:"Benign",1:"Malignant"}[prediction[0]]

    st.subheader("Prediction Result")
    st.write(f"Predicted Class: **{predicted_label}**")
    st.write(f"Closest Match True Class: **{ {0:'Benign',1:'Malignant'}[true_label] }**")

    # Olasılıkları göster
    if hasattr(gbm_model, "predict_proba"):
        proba = gbm_model.predict_proba(selected_feature)[0]
        st.subheader("Class Probabilities")
        st.write({"Benign": float(proba[0]), "Malignant": float(proba[1])})
