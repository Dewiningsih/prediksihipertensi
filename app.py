import streamlit as st
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
import plotly.graph_objects as go
import plotly.express as px

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
    "Usia", "Berat Badan", "Tinggi Badan", "Lingkar Pinggang",
    "Tekanan Darah", "IMT", "Aktivitas Total"
]

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.title("📋 Tentang Aplikasi")
st.sidebar.info(
    """
    Aplikasi ini memprediksi risiko hipertensi berdasarkan:
    
    - Usia
    - Berat & Tinggi Badan
    - Lingkar Pinggang
    - Tekanan Darah
    - IMT (Indeks Massa Tubuh)
    - Aktivitas Fisik (MET/week)

    Model yang digunakan: LightGBM + RFE
    """
)
st.sidebar.markdown("📌 Dibuat dengan ❤️ untuk edukasi dan skripsi")

# ---------------------------
# Judul Aplikasi
# ---------------------------
st.title("🩺 Prediksi Risiko Hipertensi")
st.markdown("Aplikasi ini memprediksi risiko hipertensi berdasarkan data kesehatan pribadi menggunakan model machine learning.")

# ---------------------------
# Form Input
# ---------------------------
st.header("📝 Masukkan Data Anda")
with st.form("form_prediksi"):
    usia = st.number_input("Usia", min_value=1, max_value=100, step=1)
    berat = st.number_input("Berat Badan (kg)", min_value=20.0, max_value=200.0)
    tinggi = st.number_input("Tinggi Badan (cm)", min_value=100.0, max_value=250.0)
    lingkar_pinggang = st.number_input("Lingkar Pinggang (cm)", min_value=40.0, max_value=150.0)
    tekanan_darah = st.number_input("Tekanan Darah (mmHg)", min_value=80.0, max_value=200.0)
    imt = st.number_input("IMT", min_value=10.0, max_value=50.0)
    aktivitas_total = st.number_input("Aktivitas Total (MET/week)", min_value=0.0, max_value=10000.0)

    submitted = st.form_submit_button("🔍 Prediksi")

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

        # Output hasil
        st.header("📊 Hasil Prediksi")
        st.success("Prediksi Berhasil!")

        risk_label = "🟥 Ya" if prediction == 1 else "🟩 Tidak"
        st.markdown(f"- **Risiko Hipertensi:** {risk_label}")
        st.markdown(f"- **Probabilitas:** `{prob:.2f}`")

        # ---------------------------
        # Visualisasi 1: Pie Chart
        # ---------------------------
        st.markdown("### 📈 Distribusi Probabilitas")
        st.progress(int(prob * 100))

        pie_fig = go.Figure(data=[go.Pie(
            labels=['Tidak Berisiko', 'Berisiko'],
            values=model.predict_proba(input_transformed)[0],
            hole=0.5,
            marker_colors=['green', 'red']
        )])
        pie_fig.update_layout(title="Risiko Hipertensi", height=400)
        st.plotly_chart(pie_fig, use_container_width=True)

        # ---------------------------
        # Visualisasi 2: IMT vs Kategori WHO
        # ---------------------------
        st.markdown("### 📏 Indeks Massa Tubuh Anda vs WHO")
        imt_standards = {
            "Underweight": 18.5,
            "Normal": 24.9,
            "Overweight": 29.9,
            "Obese": 50
        }
        fig_imt = px.bar(
            x=list(imt_standards.keys()),
            y=list(imt_standards.values()),
            labels={"x": "Kategori", "y": "IMT"},
            title="Perbandingan IMT Anda dengan Kategori WHO"
        )
        fig_imt.add_scatter(
            x=["Obese"],
            y=[imt],
            mode="markers",
            marker=dict(color="red", size=12),
            name="IMT Anda"
        )
        st.plotly_chart(fig_imt, use_container_width=True)

        # ---------------------------
        # Visualisasi 3: Gauge Probabilitas
        # ---------------------------
        st.markdown("### 🎯 Skor Risiko Anda")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            title={'text': "Probabilitas Risiko Hipertensi"},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': "red" if prediction == 1 else "green"},
                'steps': [
                    {'range': [0, 0.5], 'color': "lightgreen"},
                    {'range': [0.5, 0.75], 'color': "yellow"},
                    {'range': [0.75, 1], 'color': "red"}
                ]
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
