# filename: naive_bayes_streamlit_final.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from io import BytesIO

st.set_page_config(page_title="Naive Bayes Lengkap", layout="centered")

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
# UI Streamlit
# ----------------------------
st.title("🔍 Aplikasi Prediksi Naive Bayes Lengkap")

st.sidebar.header("📁 Upload Data Training")
uploaded = st.sidebar.file_uploader("Unggah file CSV", type=["csv"])

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

if uploaded:
    df = pd.read_csv(uploaded)
    st.success("✅ Data berhasil diunggah.")
else:
    df = data_default
    st.info("ℹ️ Menggunakan data default.")

with st.expander("🔍 Lihat Data Pelatihan"):
    st.dataframe(df)

# Input Prediksi
st.subheader("🧪 Input Data untuk Prediksi")

fitur_cols = [col for col in df.columns if col != "Olahraga"]
input_data = {}
for col in fitur_cols:
    pilihan = df[col].dropna().unique()
    input_data[col] = st.selectbox(f"{col}:", pilihan)

metode = st.radio("🧠 Pilih Metode Klasifikasi:", ["Naive Bayes Manual", "Naive Bayes (sklearn)"])

if st.button("🔮 Prediksi"):
    if metode == "Naive Bayes Manual":
        prediksi, probabilitas = prediksi_naive_bayes(df, input_data, "Olahraga")
    else:
        # Encode data
        encoders = {col: LabelEncoder().fit(df[col]) for col in df.columns}
        df_encoded = df.copy()
        for col, encoder in encoders.items():
            df_encoded[col] = encoder.transform(df[col])

        model = CategoricalNB()
        X = df_encoded[fitur_cols]
        y = df_encoded["Olahraga"]
        model.fit(X, y)

        input_df = pd.DataFrame([input_data])
        for col in input_df.columns:
            input_df[col] = encoders[col].transform(input_df[col])
        pred_index = model.predict(input_df)[0]
        prediksi = encoders["Olahraga"].inverse_transform([pred_index])[0]

        probs = model.predict_proba(input_df)[0]
        kelas = encoders["Olahraga"].classes_
        probabilitas = {encoders["Olahraga"].inverse_transform([i])[0]: p for i, p in enumerate(probs)}

    st.success(f"Hasil Prediksi: Orang tersebut akan olahraga? **{prediksi}**")

    st.write("📊 Probabilitas:")
    st.json(probabilitas)

    fig, ax = plt.subplots()
    ax.pie(probabilitas.values(), labels=probabilitas.keys(), autopct='%1.1f%%')
    ax.set_title("Distribusi Probabilitas")
    st.pyplot(fig)

    # Simpan ke CSV
    hasil_df = pd.DataFrame([input_data])
    hasil_df["Hasil_Prediksi"] = prediksi

    csv = hasil_df.to_csv(index=False).encode()
    st.download_button("💾 Unduh Hasil Prediksi sebagai CSV", csv, "hasil_prediksi.csv", "text/csv")

# Evaluasi Model
if st.checkbox("📈 Tampilkan Evaluasi Akurasi"):
    st.subheader("Evaluasi Akurasi Model (Leave-One-Out)")

    benar = 0
    for idx, row in df.iterrows():
        input_row = {col: row[col] for col in fitur_cols}
        true_label = row["Olahraga"]
        df_train = df.drop(index=idx)

        pred, _ = prediksi_naive_bayes(df_train, input_row, "Olahraga")
        if pred == true_label:
            benar += 1

    akurasi = benar / len(df)
    st.write(f"✅ Akurasi Naive Bayes Manual: **{akurasi * 100:.2f}%**")

    if metode == "Naive Bayes (sklearn)":
        df_encoded = df.copy()
        for col in df.columns:
            df_encoded[col] = LabelEncoder().fit_transform(df[col])
        X = df_encoded[fitur_cols]
        y = df_encoded["Olahraga"]
        model = CategoricalNB()
        model.fit(X, y)
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        st.write(f"✅ Akurasi Sklearn (tanpa cross-validation): **{acc * 100:.2f}%**")
