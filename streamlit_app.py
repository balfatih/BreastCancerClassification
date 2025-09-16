import streamlit as st
import numpy as np
import joblib
from PIL import Image

# Başlık
st.title("Breast Cancer Hybrid Classification Demo")

# Modeli yükle
@st.cache_resource
def load_model():
    model = joblib.load("CNN_GBM_model.joblib")
    return model

model = load_model()

# Görüntü yükleme
uploaded_file = st.file_uploader("Bir görüntü yükleyin", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Görüntüyü göster
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Yüklenen Görüntü", use_column_width=True)

    # Görüntüyü ön işleme
    image = image.resize((128, 128))
    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1,128,128,3)

    # Tahmin yap
    prediction = model.predict(img_array)

    # Eğer model çıktı sınıf indexiyse
    if prediction.ndim == 1 or prediction.shape[1] == 1:
        st.write(f"Tahmin edilen sınıf: {prediction[0]}")
    else:
        class_names = ["ESCA", "Healthy", "Black Rot", "Leaf Blight"]  # sınıflarını buraya yaz
        predicted_class = np.argmax(prediction, axis=1)[0]
        st.write(f"Tahmin edilen sınıf: {class_names[predicted_class]}")
        st.bar_chart(prediction[0])
