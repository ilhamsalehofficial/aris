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

# Fungsi Laplace Smoothing dengan jumlah nilai unik yang DITENTUKAN MANUAL
def laplace_smoothing(attr, value, target_class):
    subset = [d for d in data if d["Olahraga"] == target_class]
    count = len([d for d in subset if d[attr] == value])

    # Jumlah nilai unik per atribut (dari PDF)
    if attr == "Cuaca":
        v = 3
    elif attr == "Waktu":
        v = 2
    elif attr == "Niat":
        v = 2
    else:
        v = 1

    return (count + 1) / (len(subset) + v)

# Streamlit UI
st.title("Prediksi Apakah Akan Olahraga (Versi PDF Asli)")

cuaca = st.selectbox("Cuaca", ["Cerah", "Hujan", "Mendung"])
waktu = st.selectbox("Waktu Luang", ["Banyak", "Sedikit"])
niat = st.selectbox("Niat", ["Ya", "Tidak"])

if st.button("Prediksi"):
    total = len(data)
    ya_total = len([d for d in data if d["Olahraga"] == "Ya"])
    tidak_total = total - ya_total

    p_ya = ya_total / total
    p_tidak = tidak_total / total

    # Gunakan Laplace Smoothing SESUAI PDF
    p_ya_x = p_ya * laplace_smoothing("Cuaca", cuaca, "Ya") * laplace_smoothing("Waktu", waktu, "Ya") * laplace_smoothing("Niat", niat, "Ya")
    p_tidak_x = p_tidak * laplace_smoothing("Cuaca", cuaca, "Tidak") * laplace_smoothing("Waktu", waktu, "Tidak") * laplace_smoothing("Niat", niat, "Tidak")

    # Tampilkan hasil dengan pembulatan seperti di PDF
    st.subheader("Hasil Perhitungan:")
    st.write(f"P(Ya|X) = **{round(p_ya_x, 3)}**")
    st.write(f"P(Tidak|X) = **{round(p_tidak_x, 4)}**")

    if p_ya_x > p_tidak_x:
        st.success("✅ Prediksi: Ya — kemungkinan besar akan olahraga.")
    else:
        st.warning("❌ Prediksi: Tidak — kemungkinan besar tidak akan olahraga.")
