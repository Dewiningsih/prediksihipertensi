# app.py
import streamlit as st
import numpy as np
import pickle

# ================================
# 🔹 Load Model dan Scaler
# ================================
model = pickle.load(open('model/best_model.pkl', 'rb'))
scaler = pickle.load(open('model/scaler.pkl', 'rb'))

# ================================
# 🔹 Set Judul dan Deskripsi
# ================================
st.set_page_config(page_title="Prediksi Risiko Hipertensi", layout="centered")

st.title("🩺 Prediksi Risiko Hipertensi")
st.markdown("Masukkan data kesehatan Anda untuk mengetahui risiko hipertensi berdasarkan model pembelajaran mesin.")

st.header("📋 Masukkan Data Kesehatan Anda")

# ================================
# 🔹 Input Fitur
# ================================
usia = st.number_input("Usia (tahun)", 1, 120, 30)
ldl = st.number_input("Kadar Kolesterol LDL (mg/dL)", 0.0, 300.0, 100.0)
trigliserida = st.number_input("Kadar Trigliserida (mg/dL)", 0.0, 500.0, 90.0)
berat = st.number_input("Berat Badan (kg)", 1.0, 200.0, 58.0)
tinggi = st.number_input("Tinggi Badan (cm)", 50.0, 250.0, 170.0)
lingkar_pinggang = st.number_input("Lingkar Pinggang (cm)", 20.0, 200.0, 75.0)
lingkar_pinggang_ulang = st.number_input("Lingkar Pinggang (Ulang) (cm)", 20.0, 200.0, 75.0)
tekanan_darah = st.number_input("Tekanan Darah (mmHg)", 50.0, 250.0, 115.0)
imt = st.number_input("Indeks Massa Tubuh (IMT)", 10.0, 60.0, 20.1)
aktivitas = st.number_input("Aktivitas Total (menit/hari)", 0.0, 1440.0, 60.0)

# ================================
# 🔹 Prediksi
# ================================
if st.button("🔍 Prediksi"):
    # Susun urutan fitur sesuai training
    input_data = np.array([[usia, ldl, trigliserida, berat, tinggi, lingkar_pinggang,
                            lingkar_pinggang_ulang, tekanan_darah, imt, aktivitas]])
    
    # Scaling data
    input_scaled = scaler.transform(input_data)

    # Prediksi
    proba = model.predict_proba(input_scaled)[0][1]
    prediksi = model.predict(input_scaled)[0]

    # ========================
    # 🔹 Tampilkan Hasil
    # ========================
    st.subheader("🔎 Hasil Prediksi:")

    if prediksi == 1:
        st.error(f"⚠️ Risiko Tinggi Terkena Hipertensi\nProbabilitas: {proba:.2f}")
        st.markdown("**Saran:**")
        st.markdown("- Jaga pola makan rendah garam dan kolesterol")
        st.markdown("- Lakukan aktivitas fisik rutin")
        st.markdown("- Hindari rokok dan alkohol")
        st.markdown("- Periksa tekanan darah secara berkala")
    else:
        st.success(f"✅ Risiko Rendah Terkena Hipertensi\nProbabilitas: {proba:.2f}")
        st.markdown("**Tetap jaga pola hidup sehat untuk mencegah risiko di masa depan.**")
