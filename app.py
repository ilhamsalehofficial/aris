# filename: naive_bayes_streamlit_lengkap.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Fungsi Naive Bayes Manual
# ----------------------------

def hitung_probabilitas_fitur(df, fitur, nilai, label_kelas, kolom_target):
    subset = df[df[kolom_target] == label_kelas]
    total = len(subset)
    if total == 0:
        return 0
    cocok = subset[subset[fitur] == nilai]
    return len(cocok) / total

def prediksi_naive_bayes(df, input_data, kolom_target):
    total = len(df)
    hasil_kelas = df[kolom_target].unique()
    probabilitas = {}

    for kelas in hasil_kelas:
        prior = len(df[df[kolom_target] == kelas]) / total
        likelihood = 1
        for fitur, nilai in input_data.items():
            prob = hitung_probabilitas_fitur(df, fitur, nilai, kelas, kolom_target)
            likelihood *= prob
        probabilitas[kelas] = prior * likelihood

    prediksi = max(probabilitas, key=probabilitas.get)
    return prediksi, probabilitas

# ----------------------------
# Streamlit UI
# ----------------------------

st.title("🔍 Prediksi Olahraga dengan Naive Bayes")

# Data default
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

# Upload CSV
uploaded = st.sidebar.file_uploader("📁 Upload CSV (opsional)", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.success("Data berhasil diunggah.")
else:
    df = data_default
    st.info("Menggunakan data pelatihan bawaan.")

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
    hasil, probabilitas = prediksi_naive_bayes(df, input_user, "Olahraga")
    
    st.success(f"Prediksi: Orang tersebut akan **olahraga? {hasil}**")
    
    # Tampilkan probabilitas
    st.write("📊 Probabilitas Kelas:")
    st.json(probabilitas)
    
    # Pie chart visualisasi
    fig, ax = plt.subplots()
    ax.pie(probabilitas.values(), labels=probabilitas.keys(), autopct='%1.2f%%')
    ax.set_title("Distribusi Probabilitas Prediksi")
    st.pyplot(fig)

# Evaluasi jika label tersedia
if "Olahraga" in df.columns:
    st.subheader("📈 Evaluasi Model (Akurasi)")

    benar = 0
    for idx, row in df.iterrows():
        input_data = {col: row[col] for col in ["Cuaca", "Waktu", "Niat"]}
        pred, _ = prediksi_naive_bayes(df.drop(idx), input_data, "Olahraga")  # Leave-One-Out
        if pred == row["Olahraga"]:
            benar += 1

    akurasi = benar / len(df)
    st.write(f"✅ Akurasi (Leave-One-Out): **{akurasi*100:.2f}%**")

