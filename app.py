import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load("model_hipertensi_lgbm_rfe.pkl")

# Setup halaman
st.set_page_config(page_title="Prediksi Risiko Hipertensi", layout="wide")

# Sidebar
with st.sidebar:
    st.title("ℹ️ Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini memprediksi risiko **hipertensi** berdasarkan data kesehatan pengguna.
    
    Model: **LGBM + RFE**  
    Tujuan: **Edukasi dan Deteksi Dini**
    """)
    
    st.markdown("---")
    
    st.subheader("📌 Apa Itu MET/week?")
    st.markdown("""
    **MET (Metabolic Equivalent of Task)** adalah satuan aktivitas fisik.  
    Misalnya:  
    - Jalan cepat (4 MET) × 30 menit × 5 hari = 600 MET/week

    **Rekomendasi WHO:**  
    > Minimal 500–1000 MET/week untuk menjaga kesehatan jantung.
    """)

# Judul utama
st.title("🩺 Prediksi Risiko Hipertensi")
st.markdown("Masukkan data Anda di bawah ini untuk mengetahui risiko hipertensi:")

# Input user
col1, col2, col3 = st.columns(3)

with col1:
    usia = st.number_input("Usia", min_value=1, max_value=100, value=30)
    tinggi = st.number_input("Tinggi Badan (cm)", min_value=100, max_value=250, value=160)
    berat = st.number_input("Berat Badan (kg)", min_value=30, max_value=200, value=60)

with col2:
    lingkar_pinggang = st.number_input("Lingkar Pinggang (cm)", min_value=50, max_value=200, value=80)
    sistolik = st.number_input("Tekanan Darah Sistolik", min_value=80, max_value=200, value=120)
    diastolik = st.number_input("Tekanan Darah Diastolik", min_value=50, max_value=130, value=80)

with col3:
    imt = berat / ((tinggi / 100) ** 2)
    st.markdown(f"**IMT Anda:** {imt:.2f}")
    aktivitas = st.number_input("Aktivitas Fisik (MET/week)", min_value=0, max_value=20000, value=800)

# Prediksi
if st.button("Prediksi Risiko"):
    fitur = [[usia, berat, tinggi, lingkar_pinggang, sistolik, diastolik, imt, aktivitas]]
    hasil = model.predict(fitur)[0]
    if hasil == 1:
        st.error("⚠️ Anda berisiko mengalami hipertensi.")
    else:
        st.success("✅ Anda tidak berisiko mengalami hipertensi.")

# Visualisasi MET/week
st.markdown("---")
st.subheader("📊 Visualisasi Aktivitas Fisik Anda (MET/week)")
fig, ax = plt.subplots()
kategori = ["Anda", "Rekomendasi Minimum WHO"]
nilai = [aktivitas, 600]
warna = ["skyblue", "lightgreen"]
ax.bar(kategori, nilai, color=warna)
ax.axhline(600, color="red", linestyle="--", label="Batas Minimum WHO")
ax.set_ylabel("Nilai MET/week")
ax.set_ylim(0, max(aktivitas, 600) + 200)
ax.legend()
st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("🧠 *Model ini bersifat prediktif dan tidak menggantikan diagnosis profesional medis.*")
