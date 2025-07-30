import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

# Load model
model = joblib.load("model_hipertensi_lgbm_rfe.pkl")

# Judul aplikasi
st.set_page_config(page_title="Prediksi Hipertensi", layout="wide")
st.title("🩺 Prediksi Risiko Hipertensi")
st.markdown("Masukkan data kesehatan Anda untuk memprediksi kemungkinan risiko hipertensi.")

# Sidebar Input
st.sidebar.header("📝 Masukkan Data Anda")
usia = st.sidebar.slider("Usia (tahun)", 10, 100, 30)
berat = st.sidebar.slider("Berat Badan (kg)", 30, 150, 65)
tinggi = st.sidebar.slider("Tinggi Badan (cm)", 130, 200, 165)
lingkar_pinggang = st.sidebar.slider("Lingkar Pinggang (cm)", 50, 150, 80)
tekanan_darah = st.sidebar.slider("Tekanan Darah (mmHg)", 80, 200, 120)
imt = round(berat / ((tinggi/100)**2), 2)
aktivitas_total = st.sidebar.slider("Aktivitas Total (menit per minggu)", 0, 1000, 200)

# Dataframe input untuk model
input_data = pd.DataFrame({
    'Usia': [usia],
    'Berat Badan': [berat],
    'Tinggi Badan': [tinggi],
    'Lingkar Pinggang': [lingkar_pinggang],
    'Tekanan Darah': [tekanan_darah],
    'IMT': [imt],
    'Aktivitas Total': [aktivitas_total]
})

# Prediksi
if st.button("🔍 Prediksi Sekarang"):
    prediction_proba = model.predict_proba(input_data)[0][1]  # probabilitas kelas 1
    prediction = model.predict(input_data)[0]

    st.subheader("📊 Hasil Prediksi")
    st.write(f"**Probabilitas Risiko Hipertensi:** {prediction_proba:.2f}")
    st.write(f"**Status:** {'Berisiko Hipertensi' if prediction == 1 else 'Tidak Berisiko'}")

    # Gauge Chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = prediction_proba*100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Risiko Hipertensi (%)", 'font': {'size': 24}},
        delta = {'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 60], 'color': "orange"},
                {'range': [60, 100], 'color': "red"}
            ],
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

    # Saran
    st.subheader("💡 Saran untuk Anda")
    if prediction == 1:
        st.error("⚠️ Anda berisiko mengalami hipertensi. Pertimbangkan hal-hal berikut:")
        st.markdown("""
        - Kurangi konsumsi garam dan makanan tinggi lemak
        - Tingkatkan aktivitas fisik secara rutin
        - Jaga berat badan ideal
        - Hindari stres dan rokok
        - Rutin cek tekanan darah ke dokter
        """)
    else:
        st.success("✅ Anda tidak berisiko hipertensi saat ini. Tetap jaga gaya hidup sehat!")
        st.markdown("""
        - Pertahankan pola makan sehat dan seimbang
        - Tetap aktif bergerak
        - Cek kesehatan secara berkala
        """)

# Footer
st.markdown("---")
st.markdown("📍 Aplikasi ini hanya untuk tujuan edukatif, bukan diagnosis medis.")
