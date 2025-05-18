import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------

# Fungsi Naive Bayes dengan Laplace Smoothing

# ----------------------------

def hitung\_probabilitas\_fitur(df, fitur, nilai, label\_kelas, kolom\_target):
subset = df\[df\[kolom\_target] == label\_kelas]
total = len(subset)
nilai\_unik = df\[fitur].nunique()
cocok = len(subset\[subset\[fitur] == nilai])
return (cocok + 1) / (total + nilai\_unik)  # Laplace smoothing

def prediksi\_naive\_bayes(df, input\_data, kolom\_target):
total = len(df)
hasil\_kelas = df\[kolom\_target].unique()
probabilitas = {}

```
for kelas in hasil_kelas:
    prior = (len(df[df[kolom_target] == kelas]) + 1) / (total + len(hasil_kelas))  # Laplace smoothing
    likelihood = 1
    for fitur, nilai in input_data.items():
        prob = hitung_probabilitas_fitur(df, fitur, nilai, kelas, kolom_target)
        likelihood *= prob
    probabilitas[kelas] = prior * likelihood

prediksi = max(probabilitas, key=probabilitas.get)
return prediksi, probabilitas
```

# ----------------------------

# Streamlit UI

# ----------------------------

st.title("🔍 Prediksi Olahraga dengan Naive Bayes")

## st.markdown("""

### 👥 Kelompok 4 - Anggota

1. Ilham Saleh
2. Putra Pamungkas
3. Laras Anggi Wijayanti
4. Sina Widianti
5. Dila

---

""")

# Data default

data\_default = pd.DataFrame(\[
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

uploaded = st.sidebar.file\_uploader("📁 Upload Excel (.xlsx) (opsional)", type=\["xlsx"])
expected\_columns = \["Cuaca", "Waktu", "Niat", "Olahraga"]

if uploaded:
try:
df = pd.read\_excel(uploaded)
if all(col in df.columns for col in expected\_columns):
st.success("✅ Data berhasil diunggah dan valid.")
else:
st.error(f"❌ Struktur kolom tidak sesuai. Kolom yang dibutuhkan: {expected\_columns}")
df = data\_default
except Exception as e:
st.error(f"Gagal membaca file Excel: {e}")
df = data\_default
else:
df = data\_default
st.info("📌 Menggunakan data pelatihan bawaan.")

# Tampilkan data

with st.expander("🔍 Lihat Data Pelatihan"):
st.dataframe(df)

# Input prediksi

st.subheader("🧪 Input Prediksi Baru")

cuaca = st.selectbox("Cuaca:", df\["Cuaca"].unique())
waktu = st.selectbox("Waktu Luang:", df\["Waktu"].unique())
niat = st.selectbox("Niat:", df\["Niat"].unique())

input\_user = {"Cuaca": cuaca, "Waktu": waktu, "Niat": niat}

if st.button("🔮 Prediksi"):
hasil, probabilitas = prediksi\_naive\_bayes(df, input\_user, "Olahraga")

```
st.success(f"🎯 Prediksi: Orang tersebut akan **olahraga? {hasil}**")

st.write("📊 Probabilitas Kelas:")
st.json(probabilitas)

fig, ax = plt.subplots()
ax.pie(probabilitas.values(), labels=probabilitas.keys(), autopct='%1.2f%%')
ax.set_title("Distribusi Probabilitas Prediksi")
st.pyplot(fig)

st.markdown("📘 **Penjelasan:**")
st.markdown(f"- Berdasarkan kombinasi fitur **{input_user}**, sistem memperkirakan probabilitas untuk setiap kelas ('Ya' atau 'Tidak').")
st.markdown(f"- Karena probabilitas tertinggi adalah untuk kelas '**{hasil}**', maka prediksi akhir mengarah ke **{hasil}**.")
```

# Evaluasi jika label tersedia

if "Olahraga" in df.columns:
st.subheader("📈 Evaluasi Model (Akurasi - Leave-One-Out)")

```
benar = 0
for idx, row in df.iterrows():
    input_data = {col: row[col] for col in ["Cuaca", "Waktu", "Niat"]}
    pred, _ = prediksi_naive_bayes(df.drop(idx), input_data, "Olahraga")
    if pred == row["Olahraga"]:
        benar += 1

akurasi = benar / len(df)
st.write(f"✅ Akurasi: **{akurasi*100:.2f}%** berdasarkan {len(df)} data latih")
```

# ----------------------------

# Tambahan: Confusion Matrix dan Metrics

# ----------------------------

from sklearn.metrics import confusion\_matrix, precision\_score, recall\_score, f1\_score
import seaborn as sns

actual = \[]
predicted = \[]

for idx, row in df.iterrows():
input\_data = {col: row\[col] for col in \["Cuaca", "Waktu", "Niat"]}
pred, \_ = prediksi\_naive\_bayes(df.drop(idx), input\_data, "Olahraga")
actual.append(row\["Olahraga"])
predicted.append(pred)

cm = confusion\_matrix(actual, predicted, labels=\["Ya", "Tidak"])
precision = precision\_score(actual, predicted, pos\_label="Ya")
recall = recall\_score(actual, predicted, pos\_label="Ya")
f1 = f1\_score(actual, predicted, pos\_label="Ya")

st.write("📊 **Confusion Matrix**:")
fig\_cm, ax\_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=\["Ya", "Tidak"], yticklabels=\["Ya", "Tidak"], ax=ax\_cm)
ax\_cm.set\_xlabel("Prediksi")
ax\_cm.set\_ylabel("Aktual")
st.pyplot(fig\_cm)

st.write("🔎 **Detail Evaluasi (kelas: 'Ya')**")
st.markdown(f"- **Precision**: {precision:.2f}")
st.markdown(f"- **Recall**: {recall:.2f}")
st.markdown(f"- **F1-score**: {f1:.2f}")
