import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# Load model pipeline
model = joblib.load('model_hipertensi_lgbm_rfe.pkl')

# Judul dan sidebar
st.set_page_config(page_title="Prediksi Hipertensi", layout="wide")

st.sidebar.title("ℹ️ Tentang Aplikasi")
st.sidebar.write("""
Aplikasi ini memprediksi risiko hipertensi berdasarkan data kesehatan pribadi menggunakan model Machine Learning **LGBM**.
""")
st.sidebar.write("📈 Model dilatih menggunakan fitur: Usia, Berat Badan, Tinggi Badan, Lingkar Pinggang, Tekanan Darah, IMT, dan Aktivitas Total.")

# Formulir input pengguna
st.title("📝 Masukkan Data Anda")

with st.form("form_user"):
    usia = st.number_input("Usia", min_value=1, max_value=120, value=25)
    berat = st.number_input("Berat Badan (kg)", min_value=1.0, value=50.0)
    tinggi = st.number_input("Tinggi Badan (cm)", min_value=50.0, value=170.0)
    lingkar_pinggang = st.number_input("Lingkar Pinggang (cm)", min_value=30.0, value=70.0)
    tekanan_darah = st.number_input("Tekanan Darah (mmHg)", min_value=80.0, max_value=200.0, value=117.0)
    imt = st.number_input("IMT", min_value=10.0, max_value=50.0, value=berat / ((tinggi / 100) ** 2))
    aktivitas = st.number_input("Aktivitas Total (MET/week)", min_value=0.0, value=240.0)

    submitted = st.form_submit_button("Prediksi")

if submitted:
    # Buat dataframe untuk prediksi
    data = pd.DataFrame({
        'Usia': [usia],
        'Berat Badan': [berat],
        'Tinggi Badan': [tinggi],
        'Lingkar Pinggang': [lingkar_pinggang],
        'Tekanan Darah': [tekanan_darah],
        'IMT': [imt],
        'Aktivitas Total': [aktivitas]
    })

    # Prediksi
    pred_proba = model.predict_proba(data)[0][1]
    pred_label = model.predict(data)[0]

    st.header("🩺 Prediksi Risiko Hipertensi")
    if pred_label == 1:
        st.markdown("**Risiko Hipertensi: 🟥 Ya**")
    else:
        st.markdown("**Risiko Hipertensi: 🟩 Tidak**")
    st.markdown(f"**Probabilitas: {pred_proba:.2f}**")

    # Visualisasi probabilitas
    st.subheader("📊 Visualisasi Probabilitas")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pred_proba * 100,
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "red" if pred_proba >= 0.5 else "green"},
               'steps': [
                   {'range': [0, 50], 'color': "lightgreen"},
                   {'range': [50, 100], 'color': "lightcoral"}]},
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Persentase Risiko Hipertensi"}
    ))
    st.plotly_chart(fig)

    # Saran
    st.subheader("💡 Saran dan Tips")
    if pred_label == 1:
        st.markdown("""
        - 🧂 Kurangi konsumsi garam
        - 🏃‍♂️ Tingkatkan aktivitas fisik mingguan
        - 🥗 Terapkan pola makan sehat (DASH diet)
        - 🚭 Hindari merokok dan alkohol
        - 🧘 Kelola stres dengan baik
        """)
    else:
        st.markdown("""
        - ✅ Pertahankan gaya hidup sehat
        - 🏃 Rutin berolahraga minimal 150 menit per minggu
        - 🧂 Tetap kontrol asupan garam dan lemak
        - 🩺 Lakukan pemeriksaan tekanan darah secara berkala
        """)

