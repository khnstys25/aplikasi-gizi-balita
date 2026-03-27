
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

model = load_model("model_gizi.keras")
scaler = joblib.load("scaler.save")
le = joblib.load("label.save")

st.title("Aplikasi Status Gizi Balita")

umur = st.number_input("Umur (bulan)", 0, 60)
berat = st.number_input("Berat (kg)", 0.0)
tinggi = st.number_input("Tinggi (meter)", 0.0)

if st.button("Prediksi"):
    bmi = berat / (tinggi ** 2)
    
    data = pd.DataFrame({
        "UMUR BULAN/TAHUN": [umur],
        "BERAT(KG)": [berat],
        "TINGGI(TB)": [tinggi],
        "BMI": [bmi]
    })
    
    data_scaled = scaler.transform(data)
    pred = model.predict(data_scaled)
    hasil = le.inverse_transform([np.argmax(pred)])[0]
    
    st.success(f"Status Gizi: {hasil}")
    st.write(f"BMI: {round(bmi,2)}")
