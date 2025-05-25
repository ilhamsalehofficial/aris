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

def laplace_smoothing(data, attr, value, label):
    subset = [d for d in data if d["Olahraga"] == label]
    attr_values = list(set(d[attr] for d in data))
    matched = [d for d in subset if d[attr] == value]
    return (len(matched) + 1) / (len(subset) + len(attr_values))

st.title("Prediksi Olahraga - Sesuai PDF (0.211 dan 0.0078)")

cuaca = st.selectbox("Cuaca", ["Cerah", "Hujan", "Mendung"])
waktu = st.selectbox("Waktu Luang", ["Banyak", "Sedikit"])
niat = st.selectbox("Niat", ["Ya", "Tidak"])

if st.button("Prediksi"):
    total = len(data)
    ya_count = len([d for d in data if d["Olahraga"] == "Ya"])
    tidak_count = total - ya_count

    p_ya = ya_count / total
    p_tidak = tidak_count / total

    p_ya_given_x = (
        p_ya *
        laplace_smoothing(data, "Cuaca", cuaca, "Ya") *
        laplace_smoothing(data, "Waktu", waktu, "Ya") *
        laplace_smoothing(data, "Niat", niat, "Ya")
    )

    p_tidak_given_x = (
        p_tidak *
        laplace_smoothing(data, "Cuaca", cuaca, "Tidak") *
        laplace_smoothing(data, "Waktu", waktu, "Tidak") *
        laplace_smoothing(data, "Niat", niat, "Tidak")
    )

    # Pembulatan presisi sesuai PDF (3 angka penting)
    final_ya = round(p_ya_given_x, 3)
    final_tidak = round(p_tidak_given_x, 4)

    st.subheader("Hasil Perhitungan:")
    st.write(f"P(Ya|X) = **{final_ya}**")
    st.write(f"P(Tidak|X) = **{final_tidak}**")

    if final_ya > final_tidak:
        st.success("✅ Prediksi: Ya — kemungkinan besar akan olahraga.")
    else:
        st.warning("❌ Prediksi: Tidak — kemungkinan besar tidak olahraga.")
