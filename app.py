import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("model_final.pkl")

# Judul
st.title("Prediksi Risiko Hipertensi")
st.write("Masukkan data di bawah ini:")

# Input dengan validasi rentang
usia = st.number_input("Usia (tahun)", min_value=18, max_value=100, value=25)
berat = st.number_input("Berat Badan (kg)", min_value=30.0, max_value=200.0, value=60.0)
lp1 = st.number_input("Lingkar Pinggang (cm)", min_value=40.0, max_value=200.0, value=70.0)
lp2 = st.number_input("Lingkar Pinggang (Ulang) (cm)", min_value=40.0, max_value=200.0, value=70.0)
td = st.number_input("Tekanan Darah (mmHg)", min_value=60.0, max_value=250.0, value=120.0)
imt = st.number_input("IMT (Indeks Massa Tubuh)", min_value=10.0, max_value=50.0, value=22.0)
aktivitas = st.number_input("Aktivitas Total (menit/minggu)", min_value=0.0, max_value=10000.0, value=300.0)

# Prediksi saat tombol ditekan
if st.button("Prediksi"):
    input_data = np.array([[usia, berat, lp1, lp2, td, imt, aktivitas]])
    # Prediksi
    prob = model.predict_proba(user_input)[0][1]
    pred = model.predict(user_input)[0]

    st.write(f"Probabilitas Risiko Hipertensi: {prob:.4f}")
    st.write("Prediksi:", "✅ Risiko Hipertensi" if pred == 1 else "❌ Tidak Berisiko")


    if pred == 1:
        st.error(f"⚠️ Risiko Hipertensi! (Probabilitas: {prob:.2%})")
    else:
        st.success(f"✅ Tidak Berisiko Hipertensi (Probabilitas: {prob:.2%})")
