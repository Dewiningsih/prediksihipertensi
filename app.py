import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model_final (1).pkl")

# Judul aplikasi
st.title("Prediksi Risiko Hipertensi")
st.write("Masukkan data berikut untuk memprediksi apakah Anda berisiko hipertensi:")

# Form input
usia = st.number_input("Usia", min_value=18, max_value=100, value=30)
berat = st.number_input("Berat Badan", min_value=30.0, max_value=200.0, value=60.0)
lingkar_pinggang = st.number_input("Lingkar Pinggang", min_value=40.0, max_value=150.0, value=70.0)
lingkar_pinggang_ulang = st.number_input("Lingkar Pinggang (Ulang)", min_value=40.0, max_value=150.0, value=70.0)
tekanan_darah = st.number_input("Tekanan Darah", min_value=60, max_value=250, value=120)
imt = st.number_input("IMT", min_value=10.0, max_value=50.0, value=22.0)
aktivitas_total = st.number_input("Aktivitas Total", min_value=0.0, max_value=10000.0, value=300.0)

# Gabungkan ke dalam DataFrame
user_input = pd.DataFrame([{
    "Usia": usia,
    "Berat Badan": berat,
    "Lingkar Pinggang": lingkar_pinggang,
    "Lingkar Pinggang (Ulang)": lingkar_pinggang_ulang,
    "Tekanan Darah": tekanan_darah,
    "IMT": imt,
    "Aktivitas Total": aktivitas_total
}])

# Prediksi saat tombol ditekan
if st.button("Prediksi"):
    prob = model.predict_proba(user_input)[0][1]
    pred = model.predict(user_input)[0]

    st.write(f"🔍 Probabilitas Risiko Hipertensi: **{prob:.2%}**")
    if pred == 1:
        st.error("⚠️ Hasil: Anda **berisiko hipertensi**.")
    else:
        st.success("✅ Hasil: Anda **tidak berisiko hipertensi**.")
