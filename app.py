import streamlit as st

# Data Training
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

# Fungsi prediksi dengan Naive Bayes
def naive_bayes_predict(cuaca, waktu, niat):
    total = len(data)
    kelas = ['ya', 'tidak']
    hasil = {}

    for k in kelas:
        data_k = [d for d in data if d['olahraga'] == k]
        total_k = len(data_k)
        p_k = total_k / total

        p_cuaca = len([d for d in data_k if d['cuaca'] == cuaca]) / total_k
        p_waktu = len([d for d in data_k if d['waktu'] == waktu]) / total_k
        p_niat = len([d for d in data_k if d['niat'] == niat]) / total_k

        hasil[k] = p_k * p_cuaca * p_waktu * p_niat

    return hasil

# Antarmuka Streamlit
st.title("Naive Bayes: Prediksi Apakah Seseorang Akan Olahraga")

cuaca = st.selectbox("Cuaca", ["cerah", "hujan", "mendung"])
waktu = st.selectbox("Waktu Luang", ["banyak", "sedikit"])
niat = st.selectbox("Niat", ["ya", "tidak"])

if st.button("Prediksi"):
    hasil = naive_bayes_predict(cuaca, waktu, niat)
    st.write("### Hasil Perhitungan:")
    for k in hasil:
        st.write(f"P({k}|X) = {hasil[k]:.5f}")
    
    prediksi = max(hasil, key=hasil.get)
    if prediksi == "ya":
        st.success("✅ Prediksi: Orang tersebut AKAN olahraga.")
    else:
        st.error("❌ Prediksi: Orang tersebut TIDAK akan olahraga.")
