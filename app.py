import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="Prediksi Penjualan Retail", layout="wide")  # Lebar penuh layar HP

# ===================== Judul Aplikasi =====================
st.title("📊 Prediksi Penjualan Retail")
st.markdown("Unggah data penjualan Anda dan pilih metode prediksi untuk melihat estimasi penjualan beberapa bulan ke depan.")

# ===================== Upload & Load Data =====================
uploaded_file = st.file_uploader("📁 Unggah file CSV (harus ada kolom 'Tanggal' dan 'Penjualan')", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=['Tanggal'])
    df.set_index('Tanggal', inplace=True)
    st.success("✅ Data berhasil dimuat!")

    st.subheader("📋 Tabel Data Penjualan")
    st.dataframe(df, use_container_width=True)

    # ===================== Pilih Model =====================
    st.subheader("🔍 Pilih Metode Prediksi")
    model_type = st.selectbox(
        "Pilih salah satu metode prediksi:",
        ["ARIMA", "Moving Average", "Linear Regression"],
        help="""
        ARIMA: Akurat untuk data berpola waktu.
        Moving Average: Rata-rata dari data sebelumnya.
        Linear Regression: Prediksi berdasarkan tren garis lurus.
        """
    )

    steps = st.number_input("🕒 Berapa bulan ke depan ingin diprediksi?", min_value=1, max_value=12, value=3)

    # ===================== Prediksi =====================
    if st.button("🔮 Jalankan Prediksi"):
        st.info(f"Model: **{model_type}** | Bulan diprediksi: **{steps} bulan**")

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

        # ===================== Grafik =====================
        st.subheader("📈 Grafik Prediksi")
        fig, ax = plt.subplots(figsize=(10, 4))
        df['Penjualan'].plot(ax=ax, label="Data Aktual")
        forecast.plot(ax=ax, label="Hasil Prediksi", color='red')
        ax.set_xlabel("Tanggal")
        ax.set_ylabel("Penjualan")
        plt.title("Prediksi Penjualan")
        plt.legend()
        st.pyplot(fig, use_container_width=True)

        # ===================== Tabel Prediksi =====================
        st.subheader("🗂️ Tabel Hasil Prediksi")
        forecast_df = pd.DataFrame({
            'Tanggal': forecast.index.strftime('%Y-%m'),
            'Hasil Prediksi Penjualan': forecast.values.astype(int)
        })
        st.dataframe(forecast_df, use_container_width=True)

        # ===================== Tombol Unduh =====================
        csv = forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Hasil Prediksi (CSV)",
            data=csv,
            file_name='hasil_prediksi.csv',
            mime='text/csv'
        )
