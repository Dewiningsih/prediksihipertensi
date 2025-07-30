import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Load model dan scaler
model = joblib.load('model_lgbm_rfe.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('selected_features.pkl')

# Sidebar
with st.sidebar:
    st.markdown("## 🩺 Prediksi Hipertensi")
    st.markdown("Gunakan aplikasi ini untuk memprediksi risiko hipertensi berdasarkan data pribadi Anda.")
    st.markdown("### 📌 Tentang Aplikasi")
    st.info("""
    Aplikasi ini memprediksi risiko hipertensi berdasarkan fitur:

    - Usia  
    - Berat & Tinggi Badan  
    - Lingkar Pinggang  
    - Tekanan Darah Sistolik & Diastolik  
    - IMT (Indeks Massa Tubuh)  
    - Aktivitas Fisik (MET/week)

    **Model**: LightGBM + Seleksi Fitur (RFE)
    """)

    st.markdown("### 🧠 Edukasi Kesehatan")
    st.success("""
    **Apa itu Hipertensi?**

    Hipertensi atau tekanan darah tinggi adalah kondisi di mana tekanan darah terhadap dinding arteri terlalu tinggi.

    **Faktor Risiko**:
    - Usia
    - Berat badan berlebih
    - Kurang aktivitas fisik
    - Pola makan tidak sehat
    - Stres

    **Pencegahan**:
    - Gaya hidup sehat
    - Olahraga rutin
    - Makanan seimbang
    - Cek tekanan darah secara berkala
    """)

    st.markdown("___")
    st.caption("🧬 Dibuat dengan ❤️ untuk edukasi dan penelitian")

# Judul
st.title("🩺 Prediksi Risiko Hipertensi")

# Input pengguna
st.header("Masukkan Data Anda")

col1, col2 = st.columns(2)

with col1:
    usia = st.number_input("Usia (tahun)", min_value=1, max_value=120, value=30)
    berat = st.number_input("Berat Badan (kg)", min_value=1, max_value=200, value=60)
    tinggi = st.number_input("Tinggi Badan (cm)", min_value=50, max_value=250, value=160)
    lingkar_pinggang = st.number_input("Lingkar Pinggang (cm)", min_value=30, max_value=200, value=80)

with col2:
    sistolik = st.number_input("Tekanan Darah Sistolik", min_value=70, max_value=250, value=120)
    diastolik = st.number_input("Tekanan Darah Diastolik", min_value=40, max_value=150, value=80)
    aktivitas_fisik = st.number_input("Aktivitas Fisik (MET/week)", min_value=0, max_value=10000, value=1500)

# Hitung IMT
tinggi_m = tinggi / 100
imt = round(berat / (tinggi_m ** 2), 2)
st.write(f"**Indeks Massa Tubuh (IMT): {imt}**")

# Prediksi
if st.button("Prediksi Risiko Hipertensi"):
    # Buat DataFrame input
    data = pd.DataFrame([{
        'Usia': usia,
        'Berat': berat,
        'Tinggi': tinggi,
        'Lingkar_Pinggang': lingkar_pinggang,
        'Sistolik': sistolik,
        'Diastolik': diastolik,
        'IMT': imt,
        'Aktivitas_Fisik': aktivitas_fisik
    }])

    # Seleksi fitur
    input_scaled = scaler.transform(data[feature_names])

    # Prediksi
    prediksi = model.predict(input_scaled)[0]

    if prediksi == 1:
        st.error("🚨 Risiko Hipertensi: **TINGGI**. Silakan konsultasikan dengan tenaga medis.")
    else:
        st.success("✅ Risiko Hipertensi: **RENDAH**. Terus jaga pola hidup sehat!")

    st.markdown("---")

    # Visualisasi: Feature Importance dan Aktivitas Fisik
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### ⚙️ Pentingnya Fitur")
        feat_imp_fig = px.bar(
            x=feature_names,
            y=model.feature_importances_,
            labels={'x': 'Fitur', 'y': 'Importance'},
            title='Feature Importance dari Model LightGBM'
        )
        st.plotly_chart(feat_imp_fig, use_container_width=True)

    with col4:
        st.markdown("### 🏃 Aktivitas Fisik Direkomendasikan")
        categories = ['Rendah', 'Sedang', 'Tinggi']
        met_values = [500, 1500, 3500]
        fig_met = px.bar(
            x=categories,
            y=met_values,
            labels={'x': 'Kategori Aktivitas', 'y': 'MET/week'},
            title='Aktivitas Fisik untuk Menurunkan Risiko Hipertensi'
        )
        st.plotly_chart(fig_met, use_container_width=True)

# Footer
st.markdown("---")
st.caption("📊 Model prediksi ini tidak menggantikan diagnosis medis.")
