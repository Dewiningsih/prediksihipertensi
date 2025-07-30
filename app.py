import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model_hipertensi_lgbm_rfe.pkl")

# Judul Aplikasi
st.title("🩺 Prediksi Risiko Hipertensi")
st.write("Isi data berikut untuk memprediksi apakah seseorang berisiko mengalami hipertensi.")

# Fungsi Input Pengguna
def user_input_features():
    usia = st.slider("Usia (tahun)", 10, 100, 40)
    berat = st.number_input("Berat Badan (kg)", min_value=20.0, max_value=200.0, value=60.0)
    tinggi = st.number_input("Tinggi Badan (cm)", min_value=100.0, max_value=250.0, value=165.0)
    lingkar_pinggang = st.number_input("Lingkar Pinggang (cm)", min_value=40.0, max_value=200.0, value=80.0)
    tekanan_darah = st.slider("Tekanan Darah (mmHg)", 60, 200, 120)
    imt = st.number_input("Indeks Massa Tubuh (IMT)", min_value=10.0, max_value=50.0, value=22.0)
    aktivitas = st.slider("Total Aktivitas (menit/hari)", 0, 300, 60)

    # Buat DataFrame
    data = {
        "Usia": usia,
        "Berat Badan": berat,
        "Tinggi Badan": tinggi,
        "Lingkar Pinggang": lingkar_pinggang,
        "Tekanan Darah": tekanan_darah,
        "IMT": imt,
        "Aktivitas Total": aktivitas
    }
    return pd.DataFrame([data])

# Ambil input dari user
input_df = user_input_features()

# Tampilkan input
st.subheader("Data yang Diberikan")
st.write(input_df)

# Prediksi saat tombol ditekan
if st.button("Prediksi Risiko Hipertensi"):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][int(pred)]

    st.subheader("Hasil Prediksi")
    if pred == 1:
        st.error(f"⚠️ Risiko Hipertensi! (Probabilitas: {prob:.2f})")
    else:
        st.success(f"✅ Tidak Berisiko Hipertensi (Probabilitas: {prob:.2f})")
