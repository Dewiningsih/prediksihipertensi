import streamlit as st
import numpy as np
import pickle

# Load pipeline
with open('pipeline.pkl', 'rb') as file:
    pipeline = pickle.load(file)

# Judul aplikasi
st.title("Prediksi Risiko Hipertensi")
st.markdown("Model ini memprediksi apakah seseorang berisiko hipertensi berdasarkan beberapa fitur kesehatan.")

# Input dari user
usia = st.number_input("Usia (tahun)", min_value=1, max_value=100, value=30)
berat = st.number_input("Berat Badan (kg)", min_value=30, max_value=200, value=70)
tinggi = st.number_input("Tinggi Badan (cm)", min_value=100, max_value=220, value=170)
lingkar_pinggang = st.number_input("Lingkar Pinggang (cm)", min_value=50, max_value=150, value=80)
tekanan_darah = st.number_input("Tekanan Darah (sistolik)", min_value=80, max_value=200, value=120)
imt = st.number_input("Indeks Massa Tubuh (IMT)", min_value=10.0, max_value=50.0, value=24.2)
aktivitas_total = st.number_input("Aktivitas Total (jam per minggu)", min_value=0, max_value=100, value=5)

# Tombol Prediksi
if st.button("Prediksi"):
    # Masukkan data ke model
    input_data = np.array([[usia, berat, tinggi, lingkar_pinggang, tekanan_darah, imt, aktivitas_total]])
    
    # Prediksi
    prediksi = pipeline.predict(input_data)[0]
    probabilitas = pipeline.predict_proba(input_data)[0][1]
    
    # Tampilkan hasil
    if prediksi == 1:
        st.error(f"❗Pasien berisiko hipertensi. Probabilitas risiko: {probabilitas:.4f}")
    else:
        st.success(f"✅ Pasien tidak berisiko hipertensi. Probabilitas risiko: {probabilitas:.4f}")
