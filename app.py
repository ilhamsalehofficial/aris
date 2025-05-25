import streamlit as st

# =======================
# DATA TRAINING DARI PDF
# =======================
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

# ========================
# LAPACE SMOOTHING FIX
# ========================
def laplace_prob(data, attr, value, target):
    subset = [d for d in data if d["Olahraga"] == target]
    count_value = sum(1 for d in subset if d[attr] == value)

    # Jumlah nilai unik sesuai atribut (DITENTUKAN MANUAL SESUAI PDF)
    if attr == "Cuaca":
        v = 3  # Cerah, Hujan, Mendung
    elif attr == "Waktu":
        v = 2  # Banyak, Sedikit
    elif attr == "Niat":
        v = 2  # Ya, Tidak
    else:
        v = 1

    return (count_value + 1) / (len(subset) + v)

# ========================
# STREAMLIT UI
# ========================
st.set_page_config(page_title="Naive Bayes PDF", layout="centered")
st.title("🔍 Prediksi Apakah Akan Olahraga (Naive Bayes Sesuai PDF)")

cuaca = st.selectbox("Cuaca", ["Cerah", "Hujan", "Mendung"])
waktu = st.selectbox("Waktu Luang", ["Banyak", "Sedikit"])
niat = st.selectbox("Niat", ["Ya", "Tidak"])

if st.button("Prediksi"):
    total = len(data)
    ya_total = len([d for d in data if d["Olahraga"] == "Ya"])
    tidak_total = total - ya_total

    # Prior Probabilities
    p_ya = ya_total / total
    p_tidak = tidak_total / total

    # Posterior with Laplace Smoothing (FIX SESUAI PDF)
    p_ya_x = (
        p_ya *
        laplace_prob(data, "Cuaca", cuaca, "Ya") *
        laplace_prob(data, "Waktu", waktu, "Ya") *
        laplace_prob(data, "Niat", niat, "Ya")
    )

    p_tidak_x = (
        p_tidak *
        laplace_prob(data, "Cuaca", cuaca, "Tidak") *
        laplace_prob(data, "Waktu", waktu, "Tidak") *
        laplace_prob(data, "Niat", niat, "Tidak")
    )

    # Output
    st.subheader("📊 Hasil Perhitungan:")
    st.write(f"P(Ya|X) = **{round(p_ya_x, 3)}**")
    st.write(f"P(Tidak|X) = **{round(p_tidak_x, 4)}**")

    st.subheader("🧠 Kesimpulan:")
    if p_ya_x > p_tidak_x:
        st.success("✅ Prediksi: Ya — kemungkinan besar akan olahraga.")
    else:
        st.warning("❌ Prediksi: Tidak — kemungkinan besar tidak akan olahraga.")
