import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open('best_model.pkl', 'rb'))

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Risiko Hipertensi",
    layout="centered",
    page_icon="🩺"
)

# Judul
st.title("🩺 Prediksi Risiko Hipertensi")
st.markdown("""
Selamat datang di aplikasi prediksi risiko hipertensi.  
Masukkan data kesehatan Anda dengan lengkap untuk mengetahui potensi risiko terkena hipertensi.  
""")

# Sidebar edukasi
with st.sidebar:
    st.header("ℹ️ Apa itu Hipertensi?")
    st.write("""
Hipertensi (tekanan darah tinggi) adalah kondisi di mana tekanan darah terhadap dinding arteri meningkat.  
Jika tidak dikontrol, hipertensi dapat menyebabkan penyakit jantung, stroke, dan komplikasi lainnya.
""")
    st.image("https://cdn-icons-png.flaticon.com/512/2965/2965567.png", width=120)
    st.caption("Sumber: WHO, Kemenkes")

st.markdown("---")

# Form input
st.subheader("📋 Masukkan Data Kesehatan Anda")
with st.form(key='form_hipertensi'):
    col1, col2 = st.columns(2)

    with col1:
        usia = st.number_input("Usia (tahun)", min_value=1, max_value=120)
        ldl = st.number_input("Kadar Kolesterol LDL (mg/dL)", min_value=0.0)
        trigliserida = st.number_input("Kadar Trigliserida (mg/dL)", min_value=0.0)
        tekanan_darah = st.number_input("Tekanan Darah (mmHg)", min_value=50.0)
        aktivitas_total = st.number_input("Aktivitas Total (menit/hari)", min_value=0.0)

    with col2:
        berat = st.number_input("Berat Badan (kg)", min_value=1.0)
        tinggi = st.number_input("Tinggi Badan (cm)", min_value=50.0)
        lingkar_pinggang = st.number_input("Lingkar Pinggang (cm)", min_value=10.0)
        lingkar_pinggang2 = st.number_input("Lingkar Pinggang (Ulang) (cm)", min_value=10.0)
        imt = st.number_input("Indeks Massa Tubuh (IMT)", min_value=10.0)

    submit = st.form_submit_button("🔍 Prediksi Risiko")

# Prediksi
if submit:
    input_data = np.array([[usia, ldl, trigliserida, berat, tinggi,
                            lingkar_pinggang, lingkar_pinggang2,
                            tekanan_darah, imt, aktivitas_total]])

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.markdown("---")
    st.subheader("🔎 Hasil Prediksi:")

    if prediction == 1:
        st.error(f"⚠️ **Risiko Tinggi Terkena Hipertensi**\nProbabilitas: `{prob:.2f}`")
        st.markdown("""
        **Saran:**
        - Jaga pola makan rendah garam dan kolesterol
        - Lakukan aktivitas fisik rutin
        - Hindari rokok dan alkohol
        - Periksa tekanan darah secara berkala
        """)
    else:
        st.success(f"✅ **Risiko Rendah Terkena Hipertensi**\nProbabilitas: `{prob:.2f}`")
        st.markdown("Tetap jaga gaya hidup sehat ya! 💪")

    st.markdown("---")
    st.caption("Model ini adalah alat bantu, bukan diagnosis medis. Konsultasikan dengan dokter untuk hasil pasti.")
