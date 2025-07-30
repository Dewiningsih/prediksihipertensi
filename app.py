import streamlit as st
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
import plotly.graph_objects as go

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
@st.cache_resource
def load_model():
    try:
        return joblib.load("model_hipertensi_lgbm_rfe.pkl")
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()

model = load_model()

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
# Konfigurasi Halaman
# ---------------------------
st.set_page_config(page_title="Prediksi Hipertensi", layout="wide")
st.title("🩺 Prediksi Risiko Hipertensi")
st.markdown("""
Aplikasi ini memprediksi risiko hipertensi berdasarkan data kesehatan pribadi menggunakan model machine learning.
""")

# ---------------------------
# Input Sidebar
# ---------------------------
st.sidebar.header("📝 Masukkan Data Anda")

usia = st.sidebar.number_input("Usia", min_value=1, max_value=100, step=1)
berat = st.sidebar.number_input("Berat Badan (kg)", min_value=20.0, max_value=200.0)
tinggi = st.sidebar.number_input("Tinggi Badan (cm)", min_value=100.0, max_value=250.0)
lingkar_pinggang = st.sidebar.number_input("Lingkar Pinggang (cm)", min_value=40.0, max_value=150.0)
tekanan_darah = st.sidebar.number_input("Tekanan Darah (mmHg)", min_value=80.0, max_value=200.0)
imt = st.sidebar.number_input("IMT", min_value=10.0, max_value=50.0)
aktivitas_total = st.sidebar.number_input("Aktivitas Total (MET/week)", min_value=0.0, max_value=10000.0)

if st.sidebar.button("🔍 Prediksi"):
    input_data = pd.DataFrame([{
        "Usia": usia,
        "Berat Badan": berat,
        "Tinggi Badan": tinggi,
        "Lingkar Pinggang": lingkar_pinggang,
        "Tekanan Darah": tekanan_darah,
        "IMT": imt,
        "Aktivitas Total": aktivitas_total
    }])

    try:
        selector = FeatureSelector(selected_features)
        input_transformed = selector.transform(input_data)

        prediction = model.predict(input_transformed)[0]
        prob = model.predict_proba(input_transformed)[0][1]

        st.subheader("📊 Hasil Prediksi")
        st.write(f"- **Risiko Hipertensi:** {'🟥 Ya' if prediction == 1 else '🟩 Tidak'}")
        st.write(f"- **Probabilitas:** `{prob:.2f}`")

        # Progress Bar
        st.markdown("### Visualisasi Probabilitas")
        st.progress(int(prob * 100))

        # Pie Chart
        fig = go.Figure(data=[go.Pie(
            labels=['Tidak Berisiko', 'Berisiko'],
            values=model.predict_proba(input_transformed)[0],
            hole=0.5,
            marker_colors=['green', 'red']
        )])
        fig.update_layout(title="Distribusi Probabilitas", height=400)
        st.plotly_chart(fig)

        # Saran
        st.markdown("### 💡 Saran Kesehatan")
        if prediction == 1:
            st.error("⚠️ Anda berisiko mengalami hipertensi.")
            st.markdown("""
            **Rekomendasi:**
            - Kurangi konsumsi garam dan makanan berlemak
            - Rutin berolahraga minimal 150 menit/minggu
            - Hindari stres dan cukup tidur
            - Periksa tekanan darah secara berkala
            - Hindari rokok dan alkohol
            """)
        else:
            st.success("✅ Anda tidak berisiko hipertensi saat ini.")
            st.markdown("""
            **Tetap pertahankan:**
            - Pola makan sehat & seimbang
            - Aktivitas fisik teratur
            - Monitoring kesehatan rutin
            """)
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
