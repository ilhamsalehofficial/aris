import streamlit as st

# Dataset sesuai dengan file PDF
data = [
    {'cuaca': 'cerah', 'waktu': 'banyak', 'niat': 'ya', 'olahraga': 'ya'},
    {'cuaca': 'hujan', 'waktu': 'sedikit', 'niat': 'tidak', 'olahraga': 'tidak'},
    {'cuaca': 'cerah', 'waktu': 'sedikit', 'niat': 'ya', 'olahraga': 'ya'},
    {'cuaca': 'mendung', 'waktu': 'banyak', 'niat': 'ya', 'olahraga': 'ya'},
    {'cuaca': 'hujan', 'waktu': 'banyak', 'niat': 'tidak', 'olahraga': 'tidak'},
    {'cuaca': 'cerah', 'waktu': 'banyak', 'niat': 'tidak', 'olahraga': 'ya'},
    {'cuaca': 'mendung', 'waktu': 'sedikit', 'niat': 'ya', 'olahraga': 'tidak'},
    {'cuaca': 'cerah', 'waktu': 'sedikit', 'niat': 'tidak', 'olahraga': 'tidak'}
]

# Fungsi Naive Bayes TANPA smoothing
def naive_bayes_predict(cuaca, waktu, niat):
    total = len(data)
    kelas = ['ya', 'tidak']
    hasil = {}

    for k in kelas:
        data_k = [d for d in data if d['olahraga'] == k]
        total_k = len(data_k)

        # Prior probability
        p_k = total_k / total

        # Likelihood TANPA smoothing
        p_cuaca = len([d for d in data_k if d['cuaca'] == cuaca]) / total_k
        p_waktu = len([d for d in data_k if d['waktu'] == waktu]) / total_k
        p_niat = len([d for d in data_k if d['niat'] == niat]) / total_k

        # Posterior probability
        posterior = p_k * p_cuaca * p_waktu * p_niat
        hasil[k] = posterior

        # Tampilkan langkah perhitungan
        st.markdown(f"### Perhitungan untuk kelas '{k}':")
        st.write(f"Prior P({k}) = {p_k:.5f}")
        st.write(f"P(Cuaca={cuaca} | {k}) = {p_cuaca:.5f}")
        st.write(f"P(Waktu={waktu} | {k}) = {p_waktu:.5f}")
        st.write(f"P(Niat={niat} | {k}) = {p_niat:.5f}")
        st.success(f"P({k} | X) = {posterior:.5f}")
        st.write("---")

    return hasil

# Streamlit UI
st.title("Naive Bayes Classifier (Tanpa Smoothing) – Prediksi Olahraga")
st.markdown("Masukkan data kondisi seseorang:")

# Input pengguna
cuaca = st.selectbox("Cuaca", ["cerah", "hujan", "mendung"])
waktu = st.selectbox("Waktu Luang", ["banyak", "sedikit"])
niat = st.selectbox("Niat", ["ya", "tidak"])

# Tombol prediksi
if st.button("Prediksi Apakah Akan Olahraga"):
    hasil = naive_bayes_predict(cuaca, waktu, niat)

    st.markdown("## Hasil Akhir:")
    for k in hasil:
        st.write(f"P({k}|X) = {hasil[k]:.5f}")

    prediksi = max(hasil, key=hasil.get)
    if prediksi == "ya":
        st.success("✅ Prediksi: Orang tersebut AKAN olahraga.")
    else:
        st.error("❌ Prediksi: Orang tersebut TIDAK akan olahraga.")
