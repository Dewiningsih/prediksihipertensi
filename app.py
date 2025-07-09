import streamlit as st
import pandas as pd
import joblib

# Load model dan preprocessor
model = joblib.load("model_final.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# Form input
usia = st.number_input("Usia", 18, 100, 30)
berat = st.number_input("Berat Badan", 30.0, 200.0, 60.0)
lingkar_pinggang = st.number_input("Lingkar Pinggang", 40.0, 150.0, 70.0)
lingkar_pinggang_ulang = st.number_input("Lingkar Pinggang (Ulang)", 40.0, 150.0, 70.0)
tekanan_darah = st.number_input("Tekanan Darah", 60, 250, 120)
imt = st.number_input("IMT", 10.0, 50.0, 22.0)
aktivitas_total = st.number_input("Aktivitas Total", 0.0, 10000.0, 300.0)

# Buat DataFrame input user
user_input = pd.DataFrame([{
    "Usia": usia,
    "Berat Badan": berat,
    "Lingkar Pinggang": lingkar_pinggang,
    "Lingkar Pinggang (Ulang)": lingkar_pinggang_ulang,
    "Tekanan Darah": tekanan_darah,
    "IMT": imt,
    "Aktivitas Total": aktivitas_total
}])

if st.button("Prediksi"):
    user_input_prep = preprocessor.transform(user_input)
    prob = model.predict_proba(user_input_prep)[0][1]
    pred = model.predict(user_input_prep)[0]

    st.write(f"🔍 Probabilitas Risiko Hipertensi: **{prob:.2%}**")
    if pred == 1:
        st.error("⚠️ Hasil: Anda **berisiko hipertensi**.")
    else:
        st.success("✅ Hasil: Anda **tidak berisiko hipertensi**.")
