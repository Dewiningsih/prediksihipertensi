import streamlit as st
import numpy as np
import pickle

# Load pipeline
with open('pipeline.pkl', 'rb') as file:
    pipeline = pickle.load(file)

st.set_page_config(page_title="Prediksi Risiko Hipertensi", layout="centered")
st.title("Prediksi Risiko Hipertensi")

st.markdown("Masukkan data pasien untuk memprediksi apakah berisiko hipertensi.")

# Input data
usia = st.number_input("Usia", min_value=1, max_value=120, value=30)
berat = st.number_input("Berat Badan (kg)", min_value=1.0, value=60.0)
tinggi = st.number_input("Tinggi Badan (cm)", min_value=50.0, value=160.0)
lingkar_pinggang = st.number_input("Lingkar Pinggang (cm)", min_value=40.0, value=80.0)
tekanan_darah = st.number_input("Tekanan Darah", min_value=60.0, value=120.0)
imt = st.number_input("IMT (Indeks Massa Tubuh)", min_value=10.0, value=22.0)
aktivitas_total = st.number_input("Aktivitas Total (menit/hari)", min_value=0.0, value=30.0)

# Button untuk prediksi
if st.button("Prediksi"):
    # Susun input dalam urutan fitur
    user_input = np.array([[usia, berat, tinggi, lingkar_pinggang, tekanan_darah, imt, aktivitas_total]])
    
    # Prediksi
    prediksi = pipeline.predict(user_input)[0]
    probabilitas = pipeline.predict_proba(user_input)[0][1]  # Probabilitas kelas risiko

    if prediksi == 1:
        st.error(f"Pasien Diprediksi Berisiko Hipertensi (Probabilitas: {probabilitas:.4f})")
    else:
        st.success(f"Pasien Diprediksi Sehat (Probabilitas Risiko: {probabilitas:.4f})")
