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
st.sidebar.title("\ud83d\udccb Tentang Aplikasi")
st.sidebar.info("""
Aplikasi ini memprediksi **risiko hipertensi** menggunakan model *Machine Learning* (LightGBM + RFE).

\ud83e\ude7a Fitur: Usia, Berat & Tinggi Badan, Lingkar Pinggang, Tekanan Darah, IMT, Aktivitas Fisik.
""")
st.sidebar.success("\ud83d\udccc Dibuat untuk edukasi dan kesehatan preventif.")

# ---------------------------
# Judul
# ---------------------------
st.title("\ud83e\ude7a Prediksi Risiko Hipertensi")
st.markdown("Masukkan data kesehatan Anda di bawah untuk mengetahui apakah Anda berisiko hipertensi.")

# ---------------------------
# Form Input
# ---------------------------
st.header("\ud83d\udcdd Masukkan Data Anda")
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

    submitted = st.form_submit_button("\ud83d\udd0d Prediksi")

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

        risk_label = "\ud83d\udd35 Tidak" if prediction == 0 else "\ud83d\udd34 Ya"
        risk_color = "green" if prediction == 0 else "red"

        st.header("\ud83d\udcca Hasil Prediksi")
        st.success("Prediksi berhasil dilakukan!")

        colA, colB = st.columns(2)
        colA.metric(label="Risiko Hipertensi", value=risk_label)
        colB.metric(label="Probabilitas", value=f"{prob:.2f}")

        st.markdown("---")
        with st.container():
            st.subheader("\ud83d\udcc8 Visualisasi Data Anda")
            col1, col2 = st.columns([1, 1])

            with col1:
                pie_chart = go.Figure(data=[go.Pie(
                    labels=["Tidak Berisiko", "Berisiko"],
                    values=model.predict_proba(input_transformed)[0],
                    hole=0.4,
                    marker_colors=["green", "red"]
                )])
                pie_chart.update_layout(title="Probabilitas Risiko", showlegend=True)
                st.plotly_chart(pie_chart, use_container_width=True)

                if hasattr(model, "feature_importances_"):
                    feat_imp_fig = px.bar(
                        x=selected_features,
                        y=model.feature_importances_,
                        title="\ud83d\udcca Pentingnya Setiap Fitur",
                        labels={"x": "Fitur", "y": "Skor"},
                        color=selected_features,
                        color_discrete_sequence=px.colors.sequential.Tealgrn
                    )
                    feat_imp_fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(feat_imp_fig, use_container_width=True)

            with col2:
                st.markdown("#### \ud83d\udccf Indeks Massa Tubuh (IMT)")
                imt_standards = {
                    "Underweight": 18.5,
                    "Normal": 24.9,
                    "Overweight": 29.9,
                    "Obese": 50
                }
                fig_imt = px.bar(
                    x=list(imt_standards.keys()),
                    y=list(imt_standards.values()),
                    color=list(imt_standards.keys()),
                    color_discrete_map={
                        "Underweight": "orange",
                        "Normal": "green",
                        "Overweight": "yellow",
                        "Obese": "red"
                    },
                    labels={"x": "Kategori", "y": "Nilai IMT"},
                    title="Kategori IMT Berdasarkan WHO"
                )
                fig_imt.add_scatter(
                    x=["Obese"],
                    y=[imt],
                    mode="markers",
                    marker=dict(size=12, color="black"),
                    name="IMT Anda"
                )
                st.plotly_chart(fig_imt, use_container_width=True)

                st.markdown("#### \ud83c\udfc3 Aktivitas Fisik vs Rekomendasi WHO")
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

        st.markdown("---")
        st.subheader("\ud83d\udca1 Rekomendasi Kesehatan")
        if prediction == 1:
            st.error("\u26a0\ufe0f Anda **berisiko hipertensi.**")
            st.markdown("""**Langkah Pencegahan:**  
            - \ud83e\uddc2 Kurangi garam & makanan olahan  
            - \ud83c\udfc3 Olahraga rutin \u2265150 menit/minggu  
            - \u2696\ufe0f Jaga berat badan ideal  
            - \ud83d\udecc Kelola stres & tidur cukup""")
        else:
            st.success("\u2705 Anda **tidak menunjukkan risiko hipertensi.**")
            st.markdown("""**Tips Menjaga Kesehatan:**  
            - \ud83e\udd57 Makan bergizi seimbang  
            - \ud83c\udfc3 Rutin aktivitas fisik  
            - \ud83d\ude0a Hindari stres berlebihan""")

        st.markdown("#### \ud83d\udcda Referensi")
        st.markdown("""
        - [WHO - Hypertension](https://www.who.int/news-room/fact-sheets/detail/hypertension)  
        - [CDC - Prevent High Blood Pressure](https://www.cdc.gov/bloodpressure/)  
        - [Kemenkes RI - P2PTM](https://p2ptm.kemkes.go.id)  
        """)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
