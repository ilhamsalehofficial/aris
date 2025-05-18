import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns

# ----------------------------
# Fungsi Naive Bayes dengan Laplace Smoothing
# ----------------------------
def hitung_probabilitas_fitur(df, fitur, nilai, label_kelas, kolom_target):
    subset = df[df[kolom_target] == label_kelas]
    total = len(subset)
    nilai_unik = df[fitur].nunique()
    cocok = len(subset[subset[fitur] == nilai])
    return (cocok + 1) / (total + nilai_unik)  # Laplace smoothing

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
    probabilitas = {}
    hasil_tiap_kelas = {}

    for kelas in kelas_unik:
        subset_kelas = df[df["Olahraga"] == kelas]
        prior = (len(subset_kelas) + 1) / (total_data + len(kelas_unik))
        st.markdown(f"**P({kelas}) = ({len(subset_kelas)} + 1) / ({total_data} + {len(kelas_unik)}) = {prior:.4f}**")

        likelihood = 1
        for fitur, nilai in input_user.items():
            nilai_unik = df[fitur].nunique()
            cocok = len(subset_kelas[subset_kelas[fitur] == nilai])
            prob_fitur = (cocok + 1) / (len(subset_kelas) + nilai_unik)
            st.markdown(f"- P({fitur}={nilai}|{kelas}) = ({cocok} + 1) / ({len(subset_kelas)} + {nilai_unik}) = {prob_fitur:.4f}")
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

# Evaluasi Model
if "Olahraga" in df.columns:
    st.subheader("📈 Evaluasi Model (Akurasi - Leave-One-Out)")
    benar, actual, predicted = 0, [], []

    for idx, row in df.iterrows():
        input_data = {col: row[col] for col in ["Cuaca", "Waktu", "Niat"]}
        target = row["Olahraga"]
        subset_df = df.drop(index=idx)

        kelas_unik = subset_df["Olahraga"].unique()
        hasil_tiap_kelas = {}

        for kelas in kelas_unik:
            subset_kelas = subset_df[subset_df["Olahraga"] == kelas]
            prior = (len(subset_kelas) + 1) / (len(subset_df) + len(kelas_unik))
            likelihood = 1
            for fitur, nilai in input_data.items():
                nilai_unik = subset_df[fitur].nunique()
                cocok = len(subset_kelas[subset_kelas[fitur] == nilai])
                prob_fitur = (cocok + 1) / (len(subset_kelas) + nilai_unik)
                likelihood *= prob_fitur
            hasil_tiap_kelas[kelas] = prior * likelihood

        pred = max(hasil_tiap_kelas, key=hasil_tiap_kelas.get)
        predicted.append(pred)
        actual.append(target)
        if pred == target:
            benar += 1

    akurasi = benar / len(df)
    st.write(f"✅ Akurasi: **{akurasi*100:.2f}%** berdasarkan {len(df)} data latih")

    cm = confusion_matrix(actual, predicted, labels=["Ya", "Tidak"])
    precision = precision_score(actual, predicted, pos_label="Ya")
    recall = recall_score(actual, predicted, pos_label="Ya")
    f1 = f1_score(actual, predicted, pos_label="Ya")

    st.write("📊 **Confusion Matrix**:")
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Ya", "Tidak"], yticklabels=["Ya", "Tidak"], ax=ax_cm)
    ax_cm.set_xlabel("Prediksi")
    ax_cm.set_ylabel("Aktual")
    st.pyplot(fig_cm)

    st.write("🔎 **Detail Evaluasi (kelas: 'Ya')**")
    st.markdown(f"- **Precision**: {precision:.2f}")
    st.markdown(f"- **Recall**: {recall:.2f}")
    st.markdown(f"- **F1-score**: {f1:.2f}")
