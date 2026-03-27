
import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("model_gizi.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label.pkl")

st.title("Aplikasi Status Gizi Balita")

umur = st.number_input("Umur (bulan)", 0, 60)
berat = st.number_input("Berat (kg)", 0.0)
tinggi = st.number_input("Tinggi (meter)", 0.0)

if st.button("Prediksi"):
    if tinggi > 0:
        bmi = berat / (tinggi ** 2)
        
        data = np.array([[umur, berat, tinggi, bmi]])
        data_scaled = scaler.transform(data)
        
        pred = model.predict(data_scaled)
        hasil = le.inverse_transform(pred)[0]
        
        st.success(f"Status Gizi: {hasil}")
        st.write(f"BMI: {round(bmi,2)}")
    else:
        st.error("Tinggi tidak boleh 0")

