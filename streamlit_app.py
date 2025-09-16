import streamlit as st
import numpy as np
import joblib

st.title("Breast Cancer Classification using Precomputed Features")

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
x_test = np.load("x_test.npy")  # Shape: (num_samples, 558)
y_test = np.load("y_test.npy")  # Shape: (num_samples, )

# Kullanıcıya test setinden bir örnek seçtirebiliriz
sample_index = st.number_input(
    "Select a test sample index (0 - {})".format(len(x_test)-1),
    min_value=0,
    max_value=len(x_test)-1,
    value=0,
    step=1
)

features = x_test[sample_index].reshape(1, -1)  # Tek örnek için (1,558)
true_label = y_test[sample_index]

# ----------------------
# GBM ile Tahmin
# ----------------------
prediction = gbm_model.predict(features)
predicted_label = {0: "Benign", 1: "Malignant"}[prediction[0]]

st.subheader("Prediction Result")
st.write(f"Predicted Class: **{predicted_label}**")
st.write(f"True Class: **{ {0:'Benign',1:'Malignant'}[true_label] }**")

# Olasılıkları göster
if hasattr(gbm_model, "predict_proba"):
    proba = gbm_model.predict_proba(features)[0]
    st.subheader("Class Probabilities")
    st.write({"Benign": float(proba[0]), "Malignant": float(proba[1])})
