import streamlit as st
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
import plotly.graph_objects as go  # <- Tambahan untuk visualisasi

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
# Judul Aplikasi
# ---------------------------
st.title("Prediksi Risiko Hipertensi")
st.markdown(
    """
    Aplikasi ini memprediksi risiko hipertensi berdasarkan data input pribadi
    menggunakan model machine learning yang telah dilatih.
    """
)

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
    # Buat dataframe dari input
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
        # Transformasi fitur
        selector = FeatureSelector(selected_features)
        input_transformed = selector.transform(input_data)

        # Prediksi
        prediction = model.predict(input_transformed)[0]
        prob = model.predict_proba(input_transformed)[0][int(prediction)]

        # Tampilkan hasil
        st.success("Prediksi Berhasil!")
        st.subheader("Hasil Prediksi:")
        st.markdown(f"- **Risiko Hipertensi:** {'🟥 Ya' if prediction == 1 else '🟩 Tidak'}")
        st.markdown(f"- **Probabilitas:** `{prob:.2f}`")

        # ---------------------------
        # Tambahan Visualisasi
        # ---------------------------
        st.markdown("### Visualisasi Probabilitas")
        st.progress(int(prob * 100))

        fig = go.Figure(data=[go.Pie(
            labels=['Tidak Berisiko', 'Berisiko'],
            values=model.predict_proba(input_transformed)[0],
            hole=0.5,
            marker_colors=['green', 'red']
        )])
        fig.update_layout(title="Distribusi Probabilitas", height=400)
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
