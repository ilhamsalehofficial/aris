# ----------------------------
# streamlit: untuk membangun antarmuka web.
# pandas: untuk membaca dan mengolah data.
# matplotlib: untuk membuat grafik visualisasi.
# ----------------------------
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Fungsi: Menghitung probabilitas kondisional P(Fitur = Nilai | Kelas)
# Contoh:
# Jika Cuaca = Cerah dan Olahraga = Ya, maka hitung proporsi data yang cocok
# ----------------------------
def hitung_probabilitas_fitur(df, fitur, nilai, label_kelas, kolom_target):
    subset = df[df[kolom_target] == label_kelas]
    total = len(subset)
    cocok = len(subset[subset[fitur] == nilai])
    if total == 0:
        return 0
    return cocok / total

# ----------------------------
# Menampilkan judul dan nama anggota kelompok di halaman utama.
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

# ----------------------------
# Data pelatihan bawaan jika pengguna tidak mengunggah file Excel.
# Berisi fitur Cuaca, Waktu, Niat dan target: Olahraga
# ----------------------------
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

# ----------------------------
# Upload file Excel opsional
# Jika tidak diunggah atau salah format, gunakan data bawaan
# ----------------------------
st.sidebar.header("⚙️ Pengaturan Data")
uploaded = st.sidebar.file_uploader("📁 Upload Excel (.xlsx)", type=["xlsx"])
expected_columns = ["Cuaca", "Waktu", "Niat", "Olahraga"]

if uploaded:
    try:
        df = pd.read_excel(uploaded)
        if all(col in df.columns for col in expected_columns):
            st.success("✅ Data berhasil diunggah dan valid.")
        else:
            st.error(f"❌ Kolom tidak sesuai. Diperlukan: {expected_columns}")
            df = data_default
    except Exception as e:
        st.error(f"❌ Gagal membaca file: {e}")
        df = data_default
else:
    df = data_default
    st.info("📌 Menggunakan data pelatihan bawaan.")

# ----------------------------
# Menampilkan tabel data pelatihan
# ----------------------------
with st.expander("🔍 Lihat Data Pelatihan"):
    st.dataframe(df)

# ----------------------------
# Input data baru untuk prediksi
# ----------------------------
st.subheader("🧪 Input Prediksi Baru")
cuaca = st.selectbox("Cuaca:", df["Cuaca"].unique())
waktu = st.selectbox("Waktu Luang:", df["Waktu"].unique())
niat = st.selectbox("Niat:", df["Niat"].unique())
input_user = {"Cuaca": cuaca, "Waktu": waktu, "Niat": niat}

# ----------------------------
# Proses prediksi ketika tombol ditekan
# ----------------------------
if st.button("🔮 Prediksi"):
    total_data = len(df)
    kelas_unik = df["Olahraga"].unique()
    hasil_tiap_kelas = {}

    # Iterasi untuk tiap kelas (Ya / Tidak)
    for kelas in kelas_unik:
        prior = len(df[df["Olahraga"] == kelas]) / total_data  # P(Y)
        likelihood = 1
        for fitur, nilai in input_user.items():
            prob = hitung_probabilitas_fitur(df, fitur, nilai, kelas, "Olahraga")  # P(X|Y)
            likelihood *= prob
        hasil_akhir = prior * likelihood  # P(Y|X) ~ P(Y) * P(X|Y)
        hasil_tiap_kelas[kelas] = hasil_akhir

    # Ambil kelas dengan probabilitas tertinggi sebagai hasil prediksi
    prediksi_akhir = max(hasil_tiap_kelas, key=hasil_tiap_kelas.get)

    # Tampilkan hasil prediksi
    st.success(f"🎯 Prediksi: Orang tersebut akan **olahraga? {prediksi_akhir}**")
    st.write("📊 Probabilitas Setiap Kelas:")
    st.json(hasil_tiap_kelas)

    # Pie chart visualisasi
    fig, ax = plt.subplots()
    ax.pie(hasil_tiap_kelas.values(), labels=hasil_tiap_kelas.keys(), autopct='%1.2f%%')
    ax.set_title("Distribusi Probabilitas Prediksi")
    st.pyplot(fig)

    # Kesimpulan
    st.markdown("📘 **Kesimpulan:**")
    st.markdown(f"- Karena probabilitas tertinggi terdapat pada kelas '**{prediksi_akhir}**', maka sistem memprediksi bahwa orang tersebut **{prediksi_akhir.lower()}** untuk berolahraga.")
