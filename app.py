import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Definisi ulang FeatureSelector agar pickle bisa dibuka ---
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, support_mask):
        self.support_mask = support_mask
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[:, self.support_mask]

# --- Load pipeline ---
with open("pipeline_model.pkl", "rb") as f:
    pipeline = pickle.load(f)

st.set_page_config(page_title="Prediksi Risiko Hipertensi", layout="centered")
st.title("🩺 Prediksi Risiko Hipertensi")

st.markdown("Masukkan nilai-nilai berikut:")

# Urutan fitur TERPILIH:
selected_features = ['Usia', 'Berat Badan', 'Tinggi Badan',
                     'Lingkar Pinggang', 'Tekanan Darah', 'IMT', 'Aktivitas Total']

# Input user
user_input = {}
for feature in selected_features:
    user_input[feature] = st.number_input(feature, step=1.0)

# Tombol prediksi
if st.button("Prediksi"):
    # Buat dataframe dengan kolom lengkap
    default_values = pipeline.named_steps['scaler'].mean_
    all_features = pipeline.named_steps['scaler'].feature_names_in_

    input_data = {}
    for feat in all_features:
        if feat in selected_features:
            input_data[feat] = user_input[feat]
        else:
            idx = list(all_features).index(feat)
            input_data[feat] = default_values[idx]

    df_input = pd.DataFrame([input_data])

    pred = pipeline.predict(df_input)[0]
    prob = pipeline.predict_proba(df_input)[0][1]

    st.write("### Hasil Prediksi:")
    if pred == 1:
        st.success(f"⚠️ Berisiko hipertensi! Probabilitas: {prob:.4f}")
    else:
        st.info(f"✅ Tidak berisiko hipertensi. Probabilitas: {prob:.4f}")
