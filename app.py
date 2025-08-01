import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from lightgbm import LGBMClassifier

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
st.sidebar.info("""
Aplikasi ini memprediksi **risiko hipertensi** menggunakan model *Machine Learning* (LightGBM + RFE).

🩺 Fitur: Usia, Berat & Tinggi Badan, Lingkar Pinggang, Tekanan Darah, IMT, Aktivitas Fisik.
""")
st.sidebar.success("📌 Dibuat untuk edukasi dan kesehatan preventif.")

# ---------------------------
# Judul
# ---------------------------
st.title("🩺 Prediksi Risiko Hipertensi")
st.markdown("Masukkan data kesehatan Anda di bawah untuk mengetahui apakah Anda berisiko hipertensi.")

# ---------------------------
# Form Input
# ---------------------------
st.header("📝 Masukkan Data Anda")
with st.form("form_prediksi"):
    col1, col2 = st.columns([1, 1])
    with col1:
        usia = st.number_input("Usia", min_value=1, max_value=100)
        berat = st.number_input("Berat Badan (kg)", min_value=20.0, max_value=200.0)
        tinggi = st.number_input("Tinggi Badan (cm)", min_value=100.0, max_value=250.0)
    with col2:
        lingkar_pinggang = st.number_input("Lingkar Pinggang (cm)", min_value=40.0, max_value=150.0)
        tekanan_darah = st.number_input("Tekanan Darah (mmHg)", min_value=80.0, max_value=200.0)
        imt = st.number_input("IMT", min_value=10.0, max_value=50.0)
        aktivitas_total = st.number_input("Aktivitas Total (MET/week)", min_value=0.0, max_value=10000.0)

    submitted = st.form_submit_button("🔍 Prediksi")

# ---------------------------
# Prediksi & Visualisasi
# ---------------------------
if submitted:
    try:
        input_data = pd.DataFrame([{
            "Usia": usia,
            "Berat Badan": berat,
            "Tinggi Badan": tinggi,
            "Lingkar Pinggang": lingkar_pinggang,
            "Tekanan Darah": tekanan_darah,
            "IMT": imt,
            "Aktivitas Total": aktivitas_total
        }])

        selector = FeatureSelector(selected_features)
        input_transformed = selector.transform(input_data)

        prediction = model.predict(input_transformed)[0]
        prob = model.predict_proba(input_transformed)[0][int(prediction)]

        st.header("📊 Hasil Prediksi")
        st.success("Prediksi berhasil dilakukan!")

        risk_label = "🟥 Ya" if prediction == 1 else "🟩 Tidak"
        st.markdown(f"- **Risiko Hipertensi:** {risk_label}")
        st.markdown(f"- **Probabilitas:** `{prob:.2f}`")

        col1, col2 = st.columns([1, 1])

        # Kolom 1 - Pie dan Feature Importance
        with col1:
            st.markdown("### 🔍 Visualisasi Probabilitas")
            st.plotly_chart(go.Figure(data=[go.Pie(
                labels=["Tidak Berisiko", "Berisiko"],
                values=model.predict_proba(input_transformed)[0],
                hole=0.5,
                marker_colors=["green", "red"]
            )]), use_container_width=True)

            if hasattr(model, "feature_importances_"):
                st.markdown("### ⚙️ Pentingnya Fitur")
                feat_imp_fig = px.bar(
                    x=selected_features,
                    y=model.feature_importances_,
                    title="Feature Importance dari Model",
                    labels={"x": "Fitur", "y": "Skor"}
                )
                st.plotly_chart(feat_imp_fig, use_container_width=True)

        # Kolom 2 - IMT dan Aktivitas
        with col2:
            st.markdown("### 📏 Perbandingan IMT dengan WHO")
            imt_standards = {
                "Underweight": 18.5,
                "Normal": 24.9,
                "Overweight": 29.9,
                "Obese": 50
            }
            fig_imt = px.bar(
                x=list(imt_standards.keys()),
                y=list(imt_standards.values()),
                title="Kategori IMT Berdasarkan WHO",
                labels={"x": "Kategori", "y": "Nilai IMT"},
                color=list(imt_standards.keys())
            )
            fig_imt.add_scatter(
                x=["Obese"],
                y=[imt],
                mode="markers",
                marker=dict(size=12, color="black"),
                name="IMT Anda"
            )
            st.plotly_chart(fig_imt, use_container_width=True)

            st.markdown("### 🏃 Aktivitas Fisik Direkomendasikan")
            categories = ['Rendah', 'Sedang', 'Tinggi']
            met_values = [500, 1500, 3500]
            fig_met = px.bar(
                x=categories,
                y=met_values,
                labels={'x': 'Kategori Aktivitas', 'y': 'MET/week'},
                title='Rekomendasi Aktivitas Fisik WHO',
                color=categories,
                color_discrete_map={"Rendah": "orange", "Sedang": "green", "Tinggi": "blue"}
            )
            fig_met.add_scatter(
                x=["Tinggi"],
                y=[aktivitas_total],
                mode="markers",
                marker=dict(color="red", size=12),
                name="Aktivitas Anda"
            )
            st.plotly_chart(fig_met, use_container_width=True)

        # Edukasi
        st.markdown("### 💡 Rekomendasi Kesehatan")
        if prediction == 1:
            st.error("⚠️ Anda berisiko hipertensi.")
            st.markdown("""
            **Langkah Pencegahan:**
            - Kurangi konsumsi garam & makanan olahan
            - Olahraga rutin ≥150 menit/minggu
            - Jaga berat badan ideal
            - Kelola stres & tidur cukup
            """)
        else:
            st.success("✅ Anda tidak menunjukkan risiko hipertensi.")
            st.markdown("""
            **Tips Menjaga Kesehatan:**
            - Makan bergizi seimbang
            - Rutin aktivitas fisik
            - Hindari stres berlebihan
            """)

        # Referensi
        st.markdown("### 📚 Referensi")
        st.markdown("""
        - [WHO - Hypertension](https://www.who.int/news-room/fact-sheets/detail/hypertension)  
        - [CDC - Prevent High Blood Pressure](https://www.cdc.gov/bloodpressure/)  
        - [Kemenkes RI - P2PTM](https://p2ptm.kemkes.go.id)  
        """)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
