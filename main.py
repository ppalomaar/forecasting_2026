import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from streamlit_option_menu import option_menu
from statsmodels.tsa.arima.model import ARIMA

# ======================
# CONFIG & FUNGSI BANTU
# ======================
st.set_page_config(page_title="ARIMAX Custom Forecaster", layout="wide")

# ======================
# SIDEBAR NAVIGATION
# ======================
with st.sidebar:
    st.title("Settings")
    # Fitur Input Data Mentah
    uploaded_file = st.file_uploader("Unggah Data Mentah (CSV)", type="csv")
    st.markdown("---")
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Data Preview", "Live Forecast ARIMAX"],
        icons=["house", "table", "magic"],
        default_index=0,
    )

# ======================
# LOGIC LOAD DATA (USER INPUT)
# ======================
def process_data(file):
    try:
        df = pd.read_csv(file)
        # Membersihkan nama kolom dari spasi
        df.columns = df.columns.str.strip()
        
        # Identifikasi kolom tanggal secara otomatis
        date_col = [col for col in df.columns if col.lower() in ['tanggal', 'date']][0]
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)
        df = df.sort_values(date_col).set_index(date_col)
        
        # Membersihkan karakter non-numerik (seperti koma di ribuan)
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace(',', '').astype(float)
        return df
    except Exception as e:
        st.error(f"Gagal memproses file: {e}. Pastikan file memiliki kolom Tanggal dan nilai angka.")
        return None

# Cek apakah file sudah diunggah
df_user = None
if uploaded_file is not None:
    df_user = process_data(uploaded_file)
    st.sidebar.success("Data Berhasil Dimuat!")
else:
    st.sidebar.warning("Silakan unggah file CSV Anda.")

# ======================
# LOGIC NAVIGATION
# ======================
if selected == "Home":
    st.markdown("""
        <h1 style='text-align: center;'>Custom ARIMAX Forecaster</h1>
        <p style='text-align: center; font-size: 18px;'>
        Unggah data mentah Anda di sidebar, pilih kolom yang ingin diramal, dan dapatkan hasil forecast tanpa batas periode.
        </p>
    """, unsafe_allow_html=True)
    if df_user is None:
        st.info("💡 **Petunjuk:** Pastikan CSV Anda memiliki satu kolom tanggal dan minimal dua kolom angka (satu untuk target, satu untuk eksogen/variabel luar).")

elif selected == "Data Preview":
    if df_user is not None:
        st.subheader("📋 Pratinjau Data Mentah")
        st.dataframe(df_user, use_container_width=True)
        st.subheader("📈 Visualisasi Cepat")
        target_view = st.selectbox("Pilih Kolom untuk Dilihat:", df_user.columns)
        st.line_chart(df_user[target_view])
    else:
        st.error("Silakan unggah data terlebih dahulu di sidebar.")

elif selected == "Live Forecast ARIMAX":
    if df_user is not None:
        st.subheader("⚙️ Konfigurasi Model & Variabel")
        
        col_target, col_exog = st.columns(2)
        target_col = col_target.selectbox("Pilih Kolom Target (Y):", df_user.columns)
        exog_col = col_exog.selectbox("Pilih Kolom Eksogen (X):", [c for c in df_user.columns if c != target_col])
        
        st.markdown("---")
        col_p, col_d, col_q = st.columns(3)
        p = col_p.number_input("Orde p (AR)", 0, 5, 1)
        d = col_d.number_input("Orde d (Diff)", 0, 2, 1)
        q = col_q.number_input("Orde q (MA)", 0, 5, 1)
        
        n_steps = st.number_input("Jumlah langkah (hari/bulan) ke depan:", min_value=1, value=30)
        
        if st.button("Hitung Peramalan"):
            with st.spinner('Menghitung...'):
                try:
                    # Model ARIMAX
                    model = ARIMA(df_user[target_col], order=(p, d, q), exog=df_user[exog_col])
                    model_fit = model.fit()
                    
                    # Eksogen masa depan (ambil rata-rata atau nilai terakhir)
                    future_exog_val = df_user[exog_col].iloc[-1]
                    exog_future = np.array([future_exog_val] * n_steps).reshape(-1, 1)
                    
                    # Forecast
                    forecast_res = model_fit.get_forecast(steps=n_steps, exog=exog_future)
                    forecast_df = forecast_res.summary_frame()
                    
                    # Index Tanggal
                    last_date = df_user.index[-1]
                    # Mendeteksi frekuensi (harian atau bulanan)
                    forecast_dates = pd.date_range(start=last_date, periods=n_steps + 1, freq='D')[1:]
                    forecast_df.index = forecast_dates
                    
                    # Plotly Chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_user.index.tail(100), y=df_user[target_col].tail(100), name='Data Asli'))
                    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['mean'], name='Ramalan', line=dict(color='red', dash='dash')))
                    
                    fig.update_layout(title="Hasil Peramalan ARIMAX", template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader("📊 Tabel Hasil")
                    st.dataframe(forecast_df[['mean', 'mean_ci_lower', 'mean_ci_upper']])
                    
                except Exception as e:
                    st.error(f"Error Model: {e}")
    else:
        st.error("Silakan unggah data terlebih dahulu di sidebar.")