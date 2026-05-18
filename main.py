import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import statsmodels.api as sm

# Force clear cache
st.cache_resource.clear()
st.cache_data.clear()

# ======================
# CONFIG & FUNGSI BANTU
# ======================
st.set_page_config(page_title="Dashboard Forecast Nilai Tukar", layout="wide")

def get_week_label(df):
    df = df.copy()
    df['Month'] = df.index.strftime('%B %Y')
    def get_week_of_month(date):
        first_day = date.replace(day=1)
        dom = date.day
        adjusted_dom = dom + first_day.weekday()
        return int((adjusted_dom - 1) / 7) + 1
    df['Week_Num'] = [get_week_of_month(d) for d in df.index]
    df['Label'] = df['Month'] + " Minggu ke-" + df['Week_Num'].astype(str)
    return df

# ======================
# LOAD & PREPROCESSING DATA HISTORIS
# ======================
@st.cache_data
def load_data():
    minyak = pd.read_csv("Brent Oil Futures Historical Data.csv")
    kurs = pd.read_csv("IDR_USD Historical Data.csv")

    kurs.columns = kurs.columns.str.strip()
    minyak.columns = minyak.columns.str.strip()

    kurs['Date'] = pd.to_datetime(kurs['Date'])
    minyak['Date'] = pd.to_datetime(minyak['Date'])

    minyak = minyak.rename(columns={'Date': 'Tanggal', 'Price': 'Harga'})
    kurs = kurs.rename(columns={'Date': 'Tanggal', 'Price': 'Harga_NilaiTukar'})

    # FIX INDENTASI: spasi berlebih dihapus
    kurs['Harga_NilaiTukar'] = kurs['Harga_NilaiTukar'].astype(str).str.replace(',', '').astype(float)
    minyak['Harga'] = minyak['Harga'].astype(str).str.replace(',', '').astype(float)

    # Membalik format IDR/USD menjadi USD/IDR agar bernilai Rp16.000-an
    kurs['Harga_NilaiTukar'] = 1 / kurs['Harga_NilaiTukar']

    kurs = kurs.drop_duplicates(subset=['Tanggal']).sort_values("Tanggal").set_index("Tanggal")
    minyak = minyak.drop_duplicates(subset=['Tanggal']).sort_values("Tanggal").set_index("Tanggal")

    df_merged = pd.merge(kurs[['Harga_NilaiTukar']], minyak[['Harga']], left_index=True, right_index=True, how='inner')
    df_merged['Year'] = df_merged.index.year
    return df_merged

data_historis = load_data()

# ======================
# TRAIN MODEL
# FIX ORDE: data sudah stasioner sebelum differencing (d=0)
# ACF/PACF menunjukkan p=2, q=0 → ARIMAX(2,0,0)
# ======================
@st.cache_resource
def train_arimax_model(df):
    endog = df['Harga_NilaiTukar']
    exog = df['Harga']
    model = sm.tsa.ARIMA(endog, order=(2, 0, 0), exog=exog)  # <-- DIUBAH dari (1,0,3) ke (2,0,0)
    model_fitted = model.fit()
    return model_fitted

# ======================
# SIDEBAR
# ======================
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Nilai Tukar Rupiah", "Harga Minyak Mentah", "Forecast Unlimited"],
        icons=["house", "currency-exchange", "fuel-pump", "graph-up-arrow"],
        default_index=0,
    )

# ======================
# HOME
# ======================
if selected == "Home":
    st.write("##")
    st.markdown("""
        <h1 style='text-align: center; font-size: 50px;'>Peramalan Nilai Tukar IDR-USD<br>Menggunakan Live ARIMAX</h1>
        <p style='text-align: center; font-size: 18px; color: #666;'>
        Dashboard ini dikembangkan untuk melakukan simulasi peramalan nilai tukar Rupiah terhadap USD 
        menggunakan model ARIMAX secara dinamis dengan variabel eksogen harga minyak mentah dunia.
        </p>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("Tentang Proyek")
    st.write("Dashboard ini membantu analisis pergerakan nilai tukar serta memberikan hasil peramalan dinamis tanpa batas periode berdasarkan input pengguna.")

# ======================
# NILAI TUKAR
# ======================
elif selected == "Nilai Tukar Rupiah":
    st.subheader("Grafik Nilai Tukar Rupiah")
    selected_year = st.selectbox("Pilih Tahun:", sorted(data_historis['Year'].unique()))
    filtered_kurs = data_historis[data_historis['Year'] == selected_year]

    fig = go.Figure(go.Scatter(
        x=filtered_kurs.index,
        y=filtered_kurs['Harga_NilaiTukar'],
        mode='lines',
        name='Nilai Tukar'
    ))
    fig.update_layout(title=f"Pergerakan Nilai Tukar Tahun {selected_year}", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# ======================
# HARGA MINYAK
# ======================
elif selected == "Harga Minyak Mentah":
    st.subheader("Grafik Harga Minyak Mentah")
    selected_year = st.selectbox("Pilih Tahun:", sorted(data_historis['Year'].unique()))
    filtered_minyak = data_historis[data_historis['Year'] == selected_year]

    fig = go.Figure(go.Scatter(
        x=filtered_minyak.index,
        y=filtered_minyak['Harga'],
        mode='lines',
        name='Harga Minyak',
        line=dict(color='orange')
    ))
    fig.update_layout(title=f"Harga Minyak Mentah Tahun {selected_year}", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# ======================
# FORECAST UNLIMITED
# ======================
elif selected == "Forecast Unlimited":
    st.subheader("Simulasi Peramalan Mandiri (Dinamis)")

    steps = st.number_input(
        "Masukkan jumlah hari ke depan yang ingin diramal:",
        min_value=1, max_value=365, value=30, step=1
    )

    model_res = train_arimax_model(data_historis)

    last_exog_value = data_historis['Harga'].iloc[-1]
    future_exog = [last_exog_value] * steps

    forecast_values = model_res.forecast(steps=steps, exog=future_exog)

    last_date = data_historis.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq='B')

    df_live_forecast = pd.DataFrame(index=future_dates)
    df_live_forecast['Forecast_ARIMAX'] = forecast_values.values
    df_live_forecast = get_week_label(df_live_forecast)

    st.markdown("---")
    st.subheader("Visualisasi Forecast Harian per Minggu")
    selected_week = st.selectbox("Pilih Periode Mingguan Hasil Forecast:", df_live_forecast['Label'].unique())
    filtered_df = df_live_forecast[df_live_forecast['Label'] == selected_week]

    fig1 = go.Figure(go.Scatter(
        x=filtered_df.index.strftime('%d %b %Y'),
        y=filtered_df['Forecast_ARIMAX'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='green')
    ))
    fig1.update_layout(title=f"Tren Forecast Hasil Simulasi: {selected_week}", template="plotly_white")
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("---")
    st.subheader("Grafik Kontinuitas: Data Historis Terakhir + Hasil Forecast Baru")

    df_history_tail = data_historis.tail(60)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df_history_tail.index,
        y=df_history_tail['Harga_NilaiTukar'],
        name='Data Historis (Aktual)',
        line=dict(color='blue')
    ))
    fig2.add_trace(go.Scatter(
        x=df_live_forecast.index,
        y=df_live_forecast['Forecast_ARIMAX'],
        name=f'Forecast ({steps} Hari Ke Depan)',
        line=dict(dash='dash', color='red')
    ))
    fig2.update_layout(
        template="plotly_white",
        title="Penyambungan Tren Data Aktual dan Hasil Proyeksi Mandiri"
    )
    st.plotly_chart(fig2, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Ringkasan Statistik Hasil Ramalan")
        st.write(f"**Nilai Rata-rata:** Rp {df_live_forecast['Forecast_ARIMAX'].mean():,.2f}")
        st.write(f"**Nilai Tertinggi:** Rp {df_live_forecast['Forecast_ARIMAX'].max():,.2f}")
        st.write(f"**Nilai Terendah:** Rp {df_live_forecast['Forecast_ARIMAX'].min():,.2f}")
    with col2:
        st.subheader("Tabel Data Eksplorasi Hasil Forecast")
        st.dataframe(df_live_forecast[['Forecast_ARIMAX']], use_container_width=True)