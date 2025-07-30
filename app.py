import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

# ---------------------------
# Custom Transformer
# ---------------------------
class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, selected_features):
        self.selected_features = selected_features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.selected_features]

# ---------------------------
# Load Model
# ---------------------------
model = joblib.load("model_hipertensi_lgbm_rfe.pkl")

# ---------------------------
# Fitur Terpilih dari RFE
# ---------------------------
selected_features = [
    "Usia",
    "Berat Badan",
    "Tinggi Badan",
    "Lingkar Pinggang",
    "Tekanan Darah",
    "IMT",
    "Aktivitas Total"
]

# ---------------------------
# Judul Halaman
# ---------------------------
st.title("Prediksi Risiko Hipertensi")

# ---------------------------
# Form Input
# ---------------------------
with st.form("form_prediksi"):
    usia = st.number_input("Usia", min_value=1, max_value=100, step=1)
    berat = st.number_input("Berat Badan (kg)", min_value=20.0, max_value=200.0)
    tinggi = st.number_input("Tinggi Badan (cm)", min_value=100.0, max_value=250.0)
    lingkar_pinggang = st.number_input("Lingkar Pinggang (cm)", min_value=40.0, max_value=150.0)
    tekanan_darah = st.number_input("Tekanan Darah (mmHg)", min_value=80.0, max_value=200.0)
    imt = st.number_input("IMT", min_value=10.0, max_value=50.0)
    aktivitas_total = st.number_input("Aktivitas Total (MET/week)", min_value=0.0, max_value=10000.0)

    submitted = st.form_submit_button("Prediksi")

# ---------------------------
# Proses Prediksi
# ---------------------------
if submitted:
    input_data = pd.DataFrame([{
        "Usia": usia,
        "Berat Badan": berat,
        "Tinggi Badan": tinggi,
        "Lingkar Pinggang": lingkar_pinggang,
        "Tekanan Darah": tekanan_darah,
        "IMT": imt,
        "Aktivitas Total": aktivitas_total
    }])

    # Transformasi input berdasarkan fitur terpilih
    selector = FeatureSelector(selected_features)
    input_transformed = selector.transform(input_data)

    # Prediksi
    prediction = model.predict(input_transformed)[0]
    prob = model.predict_proba(input_transformed)[0][int(prediction)]

    # Tampilkan hasil
    st.subheader("Hasil Prediksi:")
    st.write(f"**Risiko Hipertensi:** {'Ya' if prediction == 1 else 'Tidak'}")
    st.write(f"**Probabilitas:** {prob:.2f}")
