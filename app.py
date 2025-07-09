import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load pipeline model
model_pipeline = joblib.load('model_pipeline.pkl')

# Judul aplikasi
st.title("Prediksi Klasifikasi dengan Feature Selection RFE")

st.markdown("""
Masukkan data pasien sesuai fitur berikut untuk mendapatkan prediksi:
- Usia (tahun)
- Berat Badan (kg)
- Lingkar Pinggang (cm)
- Lingkar Pinggang (Ulang) (cm)
- Tekanan Darah (mmHg)
- IMT (Indeks Massa Tubuh)
- Aktivitas Total (misal jumlah aktivitas fisik per minggu)
""")

# Fungsi validasi input numerik
def input_number(label, min_value=None, max_value=None, step=1.0, format="%f"):
    while True:
        value = st.number_input(label, step=step, format=format)
        if min_value is not None and value < min_value:
            st.error(f"Nilai harus >= {min_value}")
        elif max_value is not None and value > max_value:
            st.error(f"Nilai harus <= {max_value}")
        else:
            return value

# Input user
usia = st.number_input("Usia (tahun)", min_value=0, max_value=120, step=1)
berat_badan = st.number_input("Berat Badan (kg)", min_value=1, max_value=300, step=0.1, format="%.1f")
lingkar_pinggang = st.number_input("Lingkar Pinggang (cm)", min_value=10, max_value=200, step=0.1, format="%.1f")
lingkar_pinggang_ulang = st.number_input("Lingkar Pinggang (Ulang) (cm)", min_value=10, max_value=200, step=0.1, format="%.1f")
tekanan_darah = st.number_input("Tekanan Darah (mmHg)", min_value=50, max_value=250, step=1)
imt = st.number_input("IMT (Indeks Massa Tubuh)", min_value=10.0, max_value=60.0, step=0.1, format="%.1f")
aktivitas_total = st.number_input("Aktivitas Total (per minggu)", min_value=0, max_value=100, step=1)

# Tombol Prediksi
if st.button("Prediksi"):
    # Buat dataframe input model (urutan harus sama dengan fitur training)
    input_data = pd.DataFrame({
        'Usia': [usia],
        'Berat Badan': [berat_badan],
        'Lingkar Pinggang': [lingkar_pinggang],
        'Lingkar Pinggang (Ulang)': [lingkar_pinggang_ulang],
        'Tekanan Darah': [tekanan_darah],
        'IMT': [imt],
        'Aktivitas Total': [aktivitas_total]
    })

    try:
        # Prediksi kelas
        prediksi = model_pipeline.predict(input_data)[0]
        # Probabilitas prediksi
        proba = model_pipeline.predict_proba(input_data)[0]

        st.success(f"Prediksi kelas: {prediksi}")
        st.info(f"Probabilitas: Negatif={proba[0]:.4f}, Positif={proba[1]:.4f}")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
