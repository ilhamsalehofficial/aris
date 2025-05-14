import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
import numpy as np

# ===================== Judul Aplikasi =====================
st.title("📊 Aplikasi Prediksi Penjualan Retail")
st.write("Unggah data penjualan Anda dan pilih metode prediksi untuk melihat estimasi penjualan beberapa bulan ke depan.")

# ===================== Upload & Load Data =====================
uploaded_file = st.file_uploader("📁 Unggah file CSV (wajib memiliki kolom 'Tanggal' dan 'Penjualan')", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=['Tanggal'])
    df.set_index('Tanggal', inplace=True)
    st.success("✅ Data berhasil dimuat!")

    st.subheader("📋 Tabel Data Penjualan")
    st.dataframe(df)

    # ===================== Pilih Model =====================
    st.subheader("🔍 Pilih Metode Prediksi")
    model_type = st.selectbox(
        "Pilih salah satu metode prediksi berikut:",
        ["ARIMA", "Moving Average", "Linear Regression"],
        help="""
        ARIMA: Akurat untuk data berpola waktu.
        Moving Average: Rata-rata dari data sebelumnya.
        Linear Regression: Prediksi berdasarkan tren garis lurus.
        """
    )

    steps = st.number_input("🕒 Berapa bulan ke depan yang ingin diprediksi?", min_value=1, max_value=12, value=3)

    # ===================== Prediksi =====================
    if st.button("🔮 Jalankan Prediksi"):
        st.info(f"Model yang digunakan: **{model_type}**, jumlah bulan diprediksi: **{steps} bulan**")

        if model_type == "ARIMA":
            model = ARIMA(df['Penjualan'], order=(1, 1, 1))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=steps)
            forecast.index = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=steps, freq='MS')

        elif model_type == "Moving Average":
            rolling = df['Penjualan'].rolling(window=3).mean()
            last_avg = rolling.dropna().iloc[-1]
            forecast = pd.Series(
                [last_avg] * steps,
                index=pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=steps, freq='MS')
            )

        elif model_type == "Linear Regression":
            df['Bulan'] = np.arange(len(df))
            X = df[['Bulan']]
            y = df['Penjualan']
            model = LinearRegression().fit(X, y)
            future_months = np.arange(len(df), len(df) + steps).reshape(-1, 1)
            y_pred = model.predict(future_months)
            forecast = pd.Series(
                y_pred,
                index=pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=steps, freq='MS')
            )

        # ===================== Tampilkan Hasil Prediksi =====================
        st.subheader("📈 Grafik Prediksi Penjualan")
        fig, ax = plt.subplots()
        df['Penjualan'].plot(ax=ax, label="Data Aktual")
        forecast.plot(ax=ax, label="Hasil Prediksi", color='red')
        ax.set_xlabel("Tanggal")
        ax.set_ylabel("Jumlah Penjualan")
        plt.title("Grafik Prediksi Penjualan")
        plt.legend()
        st.pyplot(fig)

        # ===================== Tabel Hasil Prediksi =====================
        st.subheader("🗂️ Tabel Hasil Prediksi")
        forecast_df = pd.DataFrame({
            'Tanggal': forecast.index.strftime('%Y-%m'),
            'Hasil Prediksi Penjualan': forecast.values.astype(int)
        })
        st.dataframe(forecast_df)

        # ===================== Tombol Unduh =====================
        csv = forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Hasil Prediksi (CSV)",
            data=csv,
            file_name='hasil_prediksi.csv',
            mime='text/csv'
        )
