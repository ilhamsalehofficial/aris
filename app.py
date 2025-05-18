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

    langkah_perhitungan.append("### 🔢 Langkah-langkah Perhitungan Naive Bayes")
    langkah_perhitungan.append("Aplikasi ini akan menghitung probabilitas apakah seseorang akan berolahraga berdasarkan tiga hal: cuaca, waktu luang, dan niat. Kami menggunakan metode Naive Bayes, yaitu menghitung peluang setiap kemungkinan lalu membandingkannya.")

    # Langkah A: Prior
    langkah_perhitungan.append("#### A. Hitung Probabilitas Tiap Kelas")
    for kelas in kelas_unik:
        prior = len(df[df[kolom_target] == kelas]) / total_data
        langkah_perhitungan.append(f"- P({kelas}) = Jumlah data dengan '{kelas}' / Total data = {len(df[df[kolom_target] == kelas])} / {total_data} = {prior:.4f}")

    # Langkah B dan C
    for kelas in kelas_unik:
        prior = len(df[df[kolom_target] == kelas]) / total_data
        likelihood = 1
        langkah_perhitungan.append(f"\n#### B. Hitung Probabilitas Tiap Fitur Jika '{kelas}'")
        for fitur, nilai in input_fitur.items():
            prob = hitung_probabilitas_fitur(df, fitur, nilai, kelas, kolom_target)
            langkah_perhitungan.append(f"- P({fitur} = {nilai} | {kelas}) = {prob:.4f}")
            likelihood *= prob
        posterior = prior * likelihood
        hasil_tiap_kelas[kelas] = posterior
        langkah_perhitungan.append(f"\n#### C. Hitung Nilai Akhir untuk '{kelas}'")
        langkah_perhitungan.append(f"- P({kelas}|X) ∝ P({kelas}) × semua fitur = {prior:.4f} × {likelihood:.4f} = {posterior:.6f}")
        langkah_perhitungan.append("---")

    prediksi_akhir = max(hasil_tiap_kelas, key=hasil_tiap_kelas.get)

    # Langkah D: Kesimpulan
    langkah_perhitungan.append("#### D. Kesimpulan")
    langkah_perhitungan.append(f"Karena nilai terbesar adalah untuk '{prediksi_akhir}', maka sistem memprediksi bahwa orang tersebut akan **{prediksi_akhir} olahraga**.")

    return prediksi_akhir, hasil_tiap_kelas, langkah_perhitungan

# ----------------------------
# Antarmuka Streamlit
# ----------------------------
st.title("🤸 Selamat datang! Aplikasi ini membantu Anda memprediksi apakah seseorang akan berolahraga")

st.markdown("""
Berdasarkan 3 hal: cuaca, waktu luang, dan niat.  
Model ini menggunakan metode Naive Bayes yang sangat cocok untuk data kategori dan mudah dimengerti.  
Silakan masukkan data, lalu tekan tombol Prediksi. Anda juga dapat melihat bagaimana proses perhitungan dilakukan secara detail.
""")

st.sidebar.header("📂 Upload Data Anda (Opsional)")
uploaded = st.sidebar.file_uploader("Unggah file Excel (.xlsx) yang berisi data Anda", type=["xlsx"])
expected_columns = ["Cuaca", "Waktu", "Niat", "Olahraga"]

# Data default jika tidak ada upload
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

if uploaded:
    try:
        df = pd.read_excel(uploaded)
        if all(col in df.columns for col in expected_columns):
            st.success("✅ Data berhasil diunggah!")
        else:
            st.error("❌ Format kolom tidak sesuai. Harus ada: Cuaca, Waktu, Niat, Olahraga.")
            df = data_default
    except:
        st.error("❌ Gagal membaca file. Pastikan itu file Excel (.xlsx).")
        df = data_default
else:
    df = data_default
    st.info("📌 Menggunakan data bawaan dari aplikasi.")

with st.expander("🔍 Lihat Data Pelatihan"):
    st.dataframe(df)

st.subheader("✏️ Masukkan Kondisi Anda")
cuaca = st.selectbox("1. Bagaimana cuacanya?", df["Cuaca"].unique())
waktu = st.selectbox("2. Apakah Anda punya waktu luang?", df["Waktu"].unique())
niat = st.selectbox("3. Apakah Anda berniat olahraga?", df["Niat"].unique())

input_user = {"Cuaca": cuaca, "Waktu": waktu, "Niat": niat}

if st.button("🔮 Lakukan Prediksi"):
    prediksi_akhir, hasil_tiap_kelas, langkah_perhitungan = naive_bayes_predict_dengan_langkah(df, input_user, "Olahraga")

    for langkah in langkah_perhitungan:
        st.markdown(langkah)

    fig, ax = plt.subplots()
    ax.bar(hasil_tiap_kelas.keys(), hasil_tiap_kelas.values(), color=['green', 'red'])
    ax.set_ylabel("Nilai Akhir (Semakin tinggi, semakin mungkin)")
    ax.set_title("Perbandingan Hasil Tiap Kelas")
    st.pyplot(fig)

# ----------------------------
# Evaluasi Model (Jika Diinginkan)
# ----------------------------
def evaluasi_model(df):
    if len(df) < 2:
        return
    fitur = ["Cuaca", "Waktu", "Niat"]
    y_true = df["Olahraga"]
    y_pred = []
    for _, row in df[fitur].iterrows():
        pred, _, _ = naive_bayes_predict_dengan_langkah(df, row.to_dict(), "Olahraga")
        y_pred.append(pred)

    st.subheader("📏 Evaluasi Model (Uji Coba ke Dirinya Sendiri)")
    cm = confusion_matrix(y_true, y_pred, labels=y_true.unique())
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap="Oranges", xticklabels=y_true.unique(), yticklabels=y_true.unique())
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Sebenarnya")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

    st.text("Laporan Evaluasi:")
    st.text(classification_report(y_true, y_pred))

with st.expander("📚 Coba Evaluasi Akurasi Model (Opsional)"):
    evaluasi_model(df)

st.markdown("""
---
🧠 **Tentang Naive Bayes:**  
Metode ini menghitung kemungkinan berdasarkan data yang ada. Misalnya, jika cuaca cerah dan Anda punya waktu serta niat, seberapa besar kemungkinan Anda akan berolahraga?  
Naive Bayes bekerja berdasarkan "pengalaman masa lalu" yang tersimpan di data, lalu menghitung peluangnya.

📘 Rumus singkat:  
P(H|X) ∝ P(H) × P(Fitur 1|H) × P(Fitur 2|H) × ...

📌 Cocok untuk pemula dan sangat efisien pada data kategori seperti ini.
""")
