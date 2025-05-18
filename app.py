import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# ----------------------------
# Fungsi Naive Bayes Manual (Tanpa Laplace Smoothing)
# ----------------------------
def hitung_probabilitas_fitur(df, fitur, nilai, label_kelas, kolom_target):
    subset = df[df[kolom_target] == label_kelas]
    total = len(subset)
    cocok = len(subset[subset[fitur] == nilai])
    if total == 0:
        return 0
    return cocok / total

def naive_bayes_predict_dengan_langkah(df, input_fitur, kolom_target):
    total_data = len(df)
    kelas_unik = df[kolom_target].unique()
    hasil_tiap_kelas = {}
    langkah_perhitungan = []

    langkah_perhitungan.append("### 🔢 Langkah-langkah Perhitungan")

    # Langkah A: Prior
    langkah_perhitungan.append("#### A. Hitung Probabilitas Kelas (Prior)")
    for kelas in kelas_unik:
        prior = len(df[df[kolom_target] == kelas]) / total_data
        langkah_perhitungan.append(f"- P({kelas}) = {len(df[df[kolom_target] == kelas])} / {total_data} = {prior:.4f}")

    # Langkah B dan C
    for kelas in kelas_unik:
        prior = len(df[df[kolom_target] == kelas]) / total_data
        likelihood = 1
        langkah_perhitungan.append(f"\n#### B. Hitung Probabilitas Fitur Berdasarkan Kelas '{kelas}'")
        for fitur, nilai in input_fitur.items():
            prob = hitung_probabilitas_fitur(df, fitur, nilai, kelas, kolom_target)
            langkah_perhitungan.append(f"- P({fitur}={nilai}|{kelas}) = {prob:.4f}")
            likelihood *= prob
        posterior = prior * likelihood
        hasil_tiap_kelas[kelas] = posterior
        langkah_perhitungan.append(f"\n#### C. Hitung Nilai Akhir untuk Kelas '{kelas}'")
        langkah_perhitungan.append(f"- P({kelas}|X) ∝ {prior:.4f} × likelihood = {posterior:.6f}")
        langkah_perhitungan.append("---")

    prediksi_akhir = max(hasil_tiap_kelas, key=hasil_tiap_kelas.get)

    # Langkah D: Kesimpulan
    langkah_perhitungan.append("#### D. Kesimpulan")
    langkah_perhitungan.append(f"Karena P({prediksi_akhir}|X) memiliki nilai tertinggi, maka prediksinya adalah: **{prediksi_akhir}**")

    return prediksi_akhir, hasil_tiap_kelas, langkah_perhitungan

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

# Dataset default
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

st.subheader("🧪 Input Prediksi Baru")
cuaca = st.selectbox("Cuaca:", df["Cuaca"].unique())
waktu = st.selectbox("Waktu Luang:", df["Waktu"].unique())
niat = st.selectbox("Niat:", df["Niat"].unique())

input_user = {"Cuaca": cuaca, "Waktu": waktu, "Niat": niat}

if st.button("🔮 Prediksi"):
    prediksi_akhir, hasil_tiap_kelas, langkah_perhitungan = naive_bayes_predict_dengan_langkah(df, input_user, "Olahraga")

    for langkah in langkah_perhitungan:
        st.markdown(langkah)

    fig, ax = plt.subplots()
    ax.bar(hasil_tiap_kelas.keys(), hasil_tiap_kelas.values(), color=['skyblue', 'salmon'])
    ax.set_ylabel("Probabilitas")
    ax.set_title("Distribusi Probabilitas Prediksi")
    st.pyplot(fig)

# Evaluasi jika data berlabel tersedia
def evaluasi_model(df):
    if len(df) < 2:
        return
    fitur = ["Cuaca", "Waktu", "Niat"]
    y_true = df["Olahraga"]
    y_pred = []
    for _, row in df[fitur].iterrows():
        pred, _, _ = naive_bayes_predict_dengan_langkah(df, row.to_dict(), "Olahraga")
        y_pred.append(pred)

    st.subheader("📏 Evaluasi Model (Self-Test)")
    cm = confusion_matrix(y_true, y_pred, labels=y_true.unique())
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=y_true.unique(), yticklabels=y_true.unique())
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

    st.text("Classification Report:")
    st.text(classification_report(y_true, y_pred))

with st.expander("🧪 Lakukan Evaluasi Model (Self-Test)"):
    evaluasi_model(df)

st.markdown("""
---
📖 **Tentang Naive Bayes:**  
Naive Bayes adalah metode klasifikasi probabilistik berbasis Teorema Bayes dengan asumsi independensi antar fitur.  
Rumus dasar:  
P(H|X) ∝ P(H) × ∏ P(xi|H)  
Model ini sederhana namun efektif untuk data kategorikal dan teks.
---
""")
