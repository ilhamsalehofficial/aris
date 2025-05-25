import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns

# ----------------------------
# Fungsi Naive Bayes (tanpa Laplace smoothing agar sesuai PDF)
# ----------------------------
def hitung_probabilitas_fitur(df, fitur, nilai, label_kelas, kolom_target):
    subset = df[df[kolom_target] == label_kelas]
    total = len(subset)
    cocok = len(subset[subset[fitur] == nilai])
    if total == 0:
        return 0
    return cocok / total

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🔍 Prediksi Olahraga dengan Naive Bayes")

st.markdown("""
---
### 👥 Kelompok 4 - Anggota
1. Ilham Saleh  
2. Putra Pamungkas  
3. Laras Anggi Wijayanti  
4. Sina Widianti  
5. Dila  
---
""")

# Data default (sesuai contoh di PDF)
data_default = pd.DataFrame([
    {"Cuaca": "Cerah", "Waktu": "Banyak", "Niat": "Ya", "Olahraga": "Ya"},
    {"Cuaca": "Hujan", "Waktu": "Sedikit", "Niat": "Tidak", "Olahraga": "Tidak"},
    {"Cuaca": "Cerah", "Waktu": "Sedikit", "Niat": "Ya", "Olahraga": "Ya"},
    {"Cuaca": "Mendung", "Waktu": "Banyak", "Niat": "Ya", "Olahraga": "Ya"},
    {"Cuaca": "Hujan", "Waktu": "Banyak", "Niat": "Tidak", "Olahraga": "Tidak"},
    {"Cuaca": "Cerah", "Waktu": "Banyak", "Niat": "Tidak", "Olahraga": "Ya"},
    {"Cuaca": "Mendung", "Waktu": "Sedikit", "Niat": "Ya", "Olahraga": "Tidak"},
    {"Cuaca": "Cerah", "Waktu": "Sedikit", "Niat": "Tidak", "Olahraga": "Tidak"},
])

st.sidebar.header("⚙️ Pengaturan Data")

# Upload Excel
uploaded = st.sidebar.file_uploader("📁 Upload Excel (.xlsx) (opsional)", type=["xlsx"])
expected_columns = ["Cuaca", "Waktu", "Niat", "Olahraga"]

if uploaded:
    try:
        df = pd.read_excel(uploaded)
        if all(col in df.columns for col in expected_columns):
            st.success("✅ Data berhasil diunggah dan valid.")
        else:
            st.error(f"❌ Struktur kolom tidak sesuai. Kolom yang dibutuhkan: {expected_columns}")
            df = data_default
    except Exception as e:
        st.error(f"Gagal membaca file Excel: {e}")
        df = data_default
else:
    df = data_default
    st.info("📌 Menggunakan data pelatihan bawaan.")

# Tampilkan data
with st.expander("🔍 Lihat Data Pelatihan"):
    st.dataframe(df)

# Input prediksi
st.subheader("🧪 Input Prediksi Baru")

cuaca = st.selectbox("Cuaca:", df["Cuaca"].unique())
waktu = st.selectbox("Waktu Luang:", df["Waktu"].unique())
niat = st.selectbox("Niat:", df["Niat"].unique())

input_user = {"Cuaca": cuaca, "Waktu": waktu, "Niat": niat}

if st.button("🔮 Prediksi"):
    st.subheader("📋 Langkah Perhitungan Naive Bayes")

    total_data = len(df)
    kelas_unik = df["Olahraga"].unique()
    hasil_tiap_kelas = {}

    for kelas in kelas_unik:
        subset_kelas = df[df["Olahraga"] == kelas]
        prior = len(subset_kelas) / total_data
        st.markdown(f"**P({kelas}) = {len(subset_kelas)} / {total_data} = {prior:.4f}**")

        likelihood = 1
        for fitur, nilai in input_user.items():
            prob_fitur = hitung_probabilitas_fitur(df, fitur, nilai, kelas, "Olahraga")
            st.markdown(f"- P({fitur}={nilai}|{kelas}) = {prob_fitur:.4f}")
            likelihood *= prob_fitur

        hasil_akhir = prior * likelihood
        hasil_tiap_kelas[kelas] = hasil_akhir
        st.markdown(f"➡️ P({kelas}|X) ∝ {prior:.4f} × likelihood = {hasil_akhir:.6f}")
        st.markdown("---")

    prediksi_akhir = max(hasil_tiap_kelas, key=hasil_tiap_kelas.get)

    st.success(f"🎯 Prediksi: Orang tersebut akan **olahraga? {prediksi_akhir}**")
    st.write("📊 Probabilitas Kelas:")
    st.json(hasil_tiap_kelas)

    
    fig, ax = plt.subplots()
    ax.pie(hasil_tiap_kelas.values(), labels=hasil_tiap_kelas.keys(), autopct='%1.2f%%')
    ax.set_title("Distribusi Probabilitas Prediksi")
    st.pyplot(fig)

    st.markdown("📘 **Kesimpulan:**")
    st.markdown(f"- Karena nilai probabilitas tertinggi terdapat pada kelas '**{prediksi_akhir}**', maka sistem memprediksi hasil akhir sebagai **{prediksi_akhir}**.")
