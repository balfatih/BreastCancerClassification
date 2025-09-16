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

