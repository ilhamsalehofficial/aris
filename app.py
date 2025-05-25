import streamlit as st

# Data dari PDF
data = [
    {"Cuaca": "Cerah", "Waktu": "Banyak", "Niat": "Ya", "Olahraga": "Ya"},
    {"Cuaca": "Hujan", "Waktu": "Sedikit", "Niat": "Tidak", "Olahraga": "Tidak"},
    {"Cuaca": "Cerah", "Waktu": "Sedikit", "Niat": "Ya", "Olahraga": "Ya"},
    {"Cuaca": "Mendung", "Waktu": "Banyak", "Niat": "Ya", "Olahraga": "Ya"},
    {"Cuaca": "Hujan", "Waktu": "Banyak", "Niat": "Tidak", "Olahraga": "Tidak"},
    {"Cuaca": "Cerah", "Waktu": "Banyak", "Niat": "Tidak", "Olahraga": "Ya"},
    {"Cuaca": "Mendung", "Waktu": "Sedikit", "Niat": "Ya", "Olahraga": "Tidak"},
    {"Cuaca": "Cerah", "Waktu": "Sedikit", "Niat": "Tidak", "Olahraga": "Tidak"},
]

# Hitung nilai Laplace smoothing dengan jumlah nilai unik tetap 3 untuk setiap atribut (Cerah, Hujan, Mendung)
def laplace_smoothing(data, attr, value, target_class):
    subset = [d for d in data if d["Olahraga"] == target_class]
    match = [d for d in subset if d[attr] == value]

    # Tetapkan jumlah nilai unik tiap atribut secara manual
    if attr == "Cuaca":
        unique_count = 3
    elif attr == "Waktu":
        unique_count = 2
    elif attr == "Niat":
        unique_count = 2
    else:
        unique_count = 1  # default fallback

    return (len(match) + 1) / (len(subset) + unique_count)


st.title("Prediksi Olahraga (Naive Bayes sesuai PDF)")

cuaca = st.selectbox("Cuaca", ["Cerah", "Hujan", "Mendung"])
waktu = st.selectbox("Waktu Luang", ["Banyak", "Sedikit"])
niat = st.selectbox("Niat", ["Ya", "Tidak"])

if st.button("Prediksi"):
    total = len(data)
    ya_data = [d for d in data if d["Olahraga"] == "Ya"]
    tidak_data = [d for d in data if d["Olahraga"] == "Tidak"]

    p_ya = len(ya_data) / total
    p_tidak = len(tidak_data) / total

    # Hitung probabilitas dengan smoothing
    p_ya_x = p_ya * \
        laplace_smoothing(data, "Cuaca", cuaca, "Ya") * \
        laplace_smoothing(data, "Waktu", waktu, "Ya") * \
        laplace_smoothing(data, "Niat", niat, "Ya")

    p_tidak_x = p_tidak * \
        laplace_smoothing(data, "Cuaca", cuaca, "Tidak") * \
        laplace_smoothing(data, "Waktu", waktu, "Tidak") * \
        laplace_smoothing(data, "Niat", niat, "Tidak")

    # Tampilkan hasil sesuai format PDF
    st.subheader("Hasil Perhitungan:")
    st.write(f"P(Ya|X) = **{round(p_ya_x, 3)}**")
    st.write(f"P(Tidak|X) = **{round(p_tidak_x, 4)}**")

    if p_ya_x > p_tidak_x:
        st.success("✅ Prediksi: Ya — kemungkinan besar akan olahraga.")
    else:
        st.warning("❌ Prediksi: Tidak — kemungkinan besar tidak akan olahraga.")
