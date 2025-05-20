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

# Fungsi menghitung probabilitas
def naive_bayes_predict(cuaca, waktu, niat):
    total = len(data)
    kelas = ['ya', 'tidak']
    hasil = {}

    for k in kelas:
        # Data yang sesuai dengan kelas k
        data_k = [d for d in data if d['olahraga'] == k]
        total_k = len(data_k)

        # Hitung prior probability
        p_k = total_k / total

        # Hitung likelihood
        p_cuaca = len([d for d in data_k if d['cuaca'] == cuaca]) / total_k
        p_waktu = len([d for d in data_k if d['waktu'] == waktu]) / total_k
        p_niat = len([d for d in data_k if d['niat'] == niat]) / total_k

        # Hitung posterior (tanpa P(X))
        hasil[k] = p_k * p_cuaca * p_waktu * p_niat

    # Tampilkan hasil
    for k in hasil:
        print(f"P({k}|X) =", hasil[k])

    # Tentukan prediksi
    prediksi = max(hasil, key=hasil.get)
    print("Prediksi: Orang tersebut akan olahraga." if prediksi == 'ya' else "Prediksi: Orang tersebut tidak akan olahraga.")

# Contoh prediksi
naive_bayes_predict(cuaca='cerah', waktu='banyak', niat='ya')
