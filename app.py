import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import requests
from io import StringIO, BytesIO
import time

# =============================================================================
# KONFIGURASI: LINK GOOGLE DRIVE (FILE LAMA YANG BERFUNGSI)
# =============================================================================
FILE_IDS = {
    'harian': '1t_wCljhepGBqZVrvleuZKldomQKop9DY',           # Kompilasi_Data_1Tahun.csv
    'ksei': '1eTUIC120SHTCzvBk77Q87w0X56F2HkWz',             # KSEI_Shareholder_Processed.csv
    'master_5_parquet': '1tb1umgJc1giaKYyMNuQWhH7R8cH75s2X', # PARQUET LAMA
    'master_5_light': '10CS5QJU5MHafIpanEH9XU6SpCEOVd-pb'    # CSV LIGHT LAMA
}

# =============================================================================
# FUNGSI LOAD DATA DENGAN RETRY & FALLBACK
# =============================================================================
def load_csv_from_gdrive(file_id, max_retries=3):
    """Load CSV dari Google Drive dengan multiple fallback method"""
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    for attempt in range(max_retries):
        try:
            session = requests.Session()
            response = session.get(url, stream=True, timeout=30)
            
            if 'Virus scan warning' in response.text or 'Quota exceeded' in response.text:
                import re
                match = re.search(r'confirm=([0-9A-Za-z]+)', response.text)
                if match:
                    confirm_token = match.group(1)
                    url = f"https://drive.google.com/uc?export=download&confirm={confirm_token}&id={file_id}"
                    response = session.get(url, stream=True, timeout=30)
            
            response.raise_for_status()
            content = response.content.decode('utf-8')
            df = pd.read_csv(StringIO(content))
            return df
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == 2:
                try:
                    url = f"https://drive.google.com/uc?id={file_id}"
                    df = pd.read_csv(url)
                    return df
                except:
                    pass
            time.sleep(2)
    
    raise Exception(f"Gagal load file ID {file_id}")

def load_parquet_from_gdrive(file_id):
    """Load Parquet dari Google Drive (paling cepat)"""
    try:
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        buffer = BytesIO(response.content)
        df = pd.read_parquet(buffer)
        return df
    except Exception as e:
        st.warning(f"Gagal load Parquet: {e}. Mencoba format CSV...")
        return None

# =============================================================================
# CACHE DATA LOADING
# =============================================================================
@st.cache_data(ttl=3600, show_spinner="Loading data harian...")
def load_harian():
    """Load data harian (Kompilasi_Data_1Tahun)"""
    try:
        df = load_csv_from_gdrive(FILE_IDS['harian'])
        
        df['Last Trading Date'] = pd.to_datetime(df['Last Trading Date'], errors='coerce')
        df = df.dropna(subset=['Last Trading Date'])
        
        numeric_cols = ['Close', 'Volume', 'Value', 'Foreign Buy', 'Foreign Sell', 
                        'Bid Volume', 'Offer Volume', 'Avg_Order_Volume', 'MA50_AOVol',
                        'Volume Spike (x)', 'Bid/Offer Imbalance', 'Change %',
                        'Listed Shares', 'Tradeble Shares', 'Free Float', 'Net Foreign Flow']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"❌ Gagal load data harian: {e}")
        return pd.DataFrame(columns=['Stock Code', 'Last Trading Date', 'Close'])

@st.cache_data(ttl=86400, show_spinner="Loading data KSEI...")
def load_ksei():
    """Load data KSEI bulanan"""
    try:
        df = load_csv_from_gdrive(FILE_IDS['ksei'])
        
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        
        for col in df.columns:
            if 'Chg' in col or 'Vol' in col or 'Val' in col or col in ['Price', 'Avg_Price']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"❌ Gagal load data KSEI: {e}")
        return pd.DataFrame(columns=['Code', 'Date', 'Top_Buyer', 'Top_Seller'])

@st.cache_data(ttl=86400, show_spinner="Loading data kepemilikan 5% (CLEAN)...")
def load_master_5():
    """LOAD DATA MASTER 5% DENGAN PRIORITAS PARQUET (FILE LAMA)"""
    
    # 1. COBA LOAD PARQUET (FILE LAMA)
    df = load_parquet_from_gdrive(FILE_IDS['master_5_parquet'])
    if df is not None:
        if 'Tanggal_Data' in df.columns:
            df['Tanggal_Data'] = pd.to_datetime(df['Tanggal_Data'], errors='coerce')
            df = df.dropna(subset=['Tanggal_Data'])
        
        numeric_cols = ['Jumlah Saham (Prev)', 'Jumlah Saham (Curr)', 'Perubahan_Saham', 
                       'Close_Price', 'Estimasi_Nilai']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        st.success("✅ Load data 5% (Parquet) - File Lama")
        return df
    
    # 2. FALLBACK: CSV LIGHT (FILE LAMA)
    try:
        df = load_csv_from_gdrive(FILE_IDS['master_5_light'])
        df['Tanggal_Data'] = pd.to_datetime(df['Tanggal_Data'], errors='coerce')
        df = df.dropna(subset=['Tanggal_Data'])
        st.success("✅ Load data 5% (CSV Light) - File Lama")
        return df
    except Exception as e:
        st.error(f"❌ Gagal load data master 5%: {e}")
        return pd.DataFrame(columns=['Kode Efek', 'Tanggal_Data', 'UBO'])

# =============================================================================
# FORMATTER ANGKA
# =============================================================================
def format_rupiah(angka):
    if pd.isna(angka) or angka == 0:
        return "Rp 0"
    return f"Rp {angka:,.0f}".replace(",", ".")

def format_lembar(angka):
    if pd.isna(angka) or angka == 0:
        return "0"
    return f"{angka:,.0f}".replace(",", ".")

def format_persen(angka):
    if pd.isna(angka):
        return "0.00%"
    return f"{angka:.2f}%"

# =============================================================================
# KONFIGURASI HALAMAN
# =============================================================================
st.set_page_config(
    page_title="Bandar Eye IDX - Professional",
    page_icon="🐋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# SIDEBAR: FILTER GLOBAL
# =============================================================================
st.sidebar.image("https://img.icons8.com/fluency/96/whale.png", width=80)
st.sidebar.title("🐋 Bandar Eye")
st.sidebar.caption("v3.0 - 7 Tab Analisis | File Lama")

# Load semua data
with st.spinner('Memuat data harga...'):
    df_harian = load_harian()
with st.spinner('Memuat data KSEI...'):
    df_ksei = load_ksei()
with st.spinner('Memuat data kepemilikan 5%...'):
    df_master = load_master_5()

if df_harian.empty:
    st.error("⚠️ Data harian tidak tersedia. Dashboard tidak dapat berjalan.")
    st.stop()

# Debug info di sidebar
with st.sidebar.expander("🔧 System Status"):
    st.write(f"**Data Harian:** {len(df_harian):,} baris")
    st.write(f"**Data KSEI:** {len(df_ksei):,} baris")
    st.write(f"**Data 5%:** {len(df_master):,} baris")
    
    if not df_master.empty and 'UBO' in df_master.columns:
        st.write(f"**UBO Unik:** {df_master['UBO'].nunique():,}")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Adaro", df_master['Is_Adaro'].sum() if 'Is_Adaro' in df_master.columns else 0)
            st.metric("LKH", df_master['Is_LKH'].sum() if 'Is_LKH' in df_master.columns else 0)
        with col2:
            st.metric("Saratoga", df_master['Is_Saratoga'].sum() if 'Is_Saratoga' in df_master.columns else 0)
            st.metric("Nominee", df_master['Is_Nominee'].sum() if 'Is_Nominee' in df_master.columns else 0)

# Date range global
min_date = df_harian['Last Trading Date'].min()
max_date = df_harian['Last Trading Date'].max()

st.sidebar.markdown("---")
st.sidebar.subheader("📅 Filter Tanggal")
start_date = st.sidebar.date_input("Dari", min_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("Sampai", max_date, min_value=min_date, max_value=max_date)

# Filter sektor
if 'Sector' in df_harian.columns:
    sektor_list = sorted(df_harian['Sector'].dropna().unique())
    selected_sectors = st.sidebar.multiselect("🏭 Sektor", sektor_list, default=[])
else:
    selected_sectors = []

st.sidebar.markdown("---")
st.sidebar.caption("© Bandarmology IDX - Institutional Grade")

# =============================================================================
# MAIN APP: 7 TAB
# =============================================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 Momentum Bandar",
    "🏦 KSEI Big Money",
    "🕵️ Akumulasi Awal (UBO Clustered)",
    "📊 Watchlist & Konvergensi",
    "📊 Free Float & Likuiditas",
    "🌏 Foreign Flow Intelligence",
    "🔄 Free Float vs Kepemilikan 5%"
])

# =============================================================================
# TAB 1: MOMENTUM BANDAR (SESUAI SCRIPT LAMA)
# =============================================================================
with tab1:
    st.header("📈 Momentum & Anomali Bandar")
    st.caption("Deteksi akumulasi/distribusi dari data harian (Volume Spike + AOVol Anomaly)")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        min_volume_spike = st.slider("Min Volume Spike (x)", 0.0, 5.0, 1.5, 0.1)
    with col2:
        min_ao_ratio = st.slider("Min AOVol Ratio (vs MA50)", 0.0, 5.0, 2.0, 0.1)
    with col3:
        min_imbalance = st.slider("Min Bid/Offer Imbalance", -1.0, 1.0, 0.1, 0.05)
    
    df_filtered = df_harian[
        (df_harian['Last Trading Date'] >= pd.to_datetime(start_date)) &
        (df_harian['Last Trading Date'] <= pd.to_datetime(end_date))
    ].copy()
    
    if selected_sectors:
        df_filtered = df_filtered[df_filtered['Sector'].isin(selected_sectors)]
    
    filter_condition = pd.Series([True] * len(df_filtered))
    
    if 'Volume Spike (x)' in df_filtered.columns:
        filter_condition &= (df_filtered['Volume Spike (x)'] >= min_volume_spike)
    
    if 'Avg_Order_Volume' in df_filtered.columns and 'MA50_AOVol' in df_filtered.columns:
        ao_ratio = df_filtered['Avg_Order_Volume'] / df_filtered['MA50_AOVol'].replace(0, np.nan)
        filter_condition &= (ao_ratio >= min_ao_ratio)
    
    if 'Bid/Offer Imbalance' in df_filtered.columns:
        filter_condition &= (df_filtered['Bid/Offer Imbalance'] >= min_imbalance)
    
    df_anomaly = df_filtered[filter_condition].copy()
    
    df_anomaly['Potensi'] = 0
    if 'Volume Spike (x)' in df_anomaly.columns:
        df_anomaly['Potensi'] += df_anomaly['Volume Spike (x)'] * 0.3
    if 'Avg_Order_Volume' in df_anomaly.columns and 'MA50_AOVol' in df_anomaly.columns:
        ao_ratio_val = df_anomaly['Avg_Order_Volume'] / df_anomaly['MA50_AOVol'].replace(0, np.nan)
        df_anomaly['Potensi'] += ao_ratio_val.fillna(0) * 0.4
    if 'Bid/Offer Imbalance' in df_anomaly.columns:
        df_anomaly['Potensi'] += (df_anomaly['Bid/Offer Imbalance'] + 1) * 0.3
    
    # Ambil data terakhir untuk free float
    df_latest = df_harian.sort_values('Last Trading Date').groupby('Stock Code').last().reset_index()
    if 'Free Float' in df_latest.columns:
        df_anomaly = df_anomaly.merge(
            df_latest[['Stock Code', 'Free Float', 'Tradeble Shares']],
            on='Stock Code',
            how='left'
        )
        
        if 'Tradeble Shares' in df_anomaly.columns and 'Volume' in df_anomaly.columns:
            df_anomaly['Volume_vs_Tradeble'] = (df_anomaly['Volume'] / df_anomaly['Tradeble Shares']) * 100
    
    df_anomaly = df_anomaly.sort_values('Potensi', ascending=False)
    
    st.subheader(f"🎯 {len(df_anomaly)} Saham dengan Aktivitas Bandar Terdeteksi")
    
    if not df_anomaly.empty:
        if 'Free Float' in df_anomaly.columns and 'Volume_vs_Tradeble' in df_anomaly.columns:
            tight_stocks = df_anomaly[
                (df_anomaly['Free Float'] < 15) & 
                (df_anomaly['Volume_vs_Tradeble'] > 30)
            ].nlargest(5, 'Volume_vs_Tradeble')
            
            if not tight_stocks.empty:
                st.warning("⚠️ **PERHATIAN: Free Float Kecil (<15%) dengan Volume Besar!**")
                st.dataframe(
                    tight_stocks[['Stock Code', 'Free Float', 'Volume_vs_Tradeble', 'Volume Spike (x)']],
                    use_container_width=True,
                    hide_index=True
                )
        
        display_cols = ['Stock Code', 'Last Trading Date', 'Close', 'Change %', 
                       'Volume Spike (x)', 'Net Foreign Flow', 'Potensi']
        display_cols = [c for c in display_cols if c in df_anomaly.columns]
        
        st.dataframe(
            df_anomaly[display_cols].head(100),
            use_container_width=True,
            hide_index=True
        )

# =============================================================================
# TAB 2: KSEI BIG MONEY (SESUAI SCRIPT LAMA)
# =============================================================================
with tab2:
    st.header("🏦 Jejak Big Money (KSEI Bulanan)")
    st.caption("Institusi yang akumulasi/distribusi besar dalam sebulan")
    
    if df_ksei.empty:
        st.warning("Data KSEI tidak tersedia.")
    else:
        df_ksei_filtered = df_ksei[
            (df_ksei['Date'] >= pd.to_datetime(start_date)) &
            (df_ksei['Date'] <= pd.to_datetime(end_date))
        ]
        
        top_n = st.slider("Top N Buyer/Seller", 5, 30, 15)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🟢 Top Buyer (Volume)")
            if 'Top_Buyer_Vol' in df_ksei_filtered.columns:
                top_buyers = df_ksei_filtered.nlargest(top_n, 'Top_Buyer_Vol')[['Code', 'Date', 'Top_Buyer', 'Top_Buyer_Vol', 'Top_Buyer_Val']]
                st.dataframe(top_buyers, use_container_width=True, hide_index=True)
        
        with col2:
            st.subheader("🔴 Top Seller (Volume)")
            if 'Top_Seller_Vol' in df_ksei_filtered.columns:
                top_sellers = df_ksei_filtered.nsmallest(top_n, 'Top_Seller_Vol')[['Code', 'Date', 'Top_Seller', 'Top_Seller_Vol', 'Top_Seller_Val']]
                st.dataframe(top_sellers, use_container_width=True, hide_index=True)

# =============================================================================
# TAB 3: AKUMULASI AWAL (UBO CLUSTERED) - SESUAI SCRIPT LAMA
# =============================================================================
with tab3:
    st.header("🕵️ DETEKSI AKUMULASI AWAL (UBO Clustered)")
    
    if df_master.empty:
        st.warning("⚠️ Data kepemilikan 5% tidak tersedia.")
        st.stop()
    
    if 'UBO' not in df_master.columns:
        st.warning("⚠️ Kolom 'UBO' tidak ditemukan.")
        st.stop()
    
    df_master_filtered = df_master[
        (df_master['Tanggal_Data'] >= pd.to_datetime(start_date)) &
        (df_master['Tanggal_Data'] <= pd.to_datetime(end_date))
    ].copy()
    
    if df_master_filtered.empty:
        st.warning("Tidak ada data kepemilikan 5% pada periode yang dipilih.")
        st.stop()
    
    if selected_sectors and 'Sector' not in df_master_filtered.columns:
        if 'Kode Efek' in df_master_filtered.columns and 'Sector' in df_harian.columns:
            sector_map = df_harian[['Stock Code', 'Sector']].drop_duplicates('Stock Code')
            df_master_filtered = df_master_filtered.merge(sector_map, left_on='Kode Efek', right_on='Stock Code', how='left')
    
    st.subheader("⚙️ Parameter Akumulasi Awal")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        min_beli = st.number_input("Minimal Beli (lembar)", min_value=1000, value=100000, step=10000, format="%d")
    with col2:
        lookback_days = st.number_input("Lookback (hari)", min_value=7, max_value=90, value=30)
    with col3:
        min_freq = st.number_input("Minimal Frekuensi", min_value=1, max_value=20, value=2)
    
    cutoff_date = pd.to_datetime(end_date) - timedelta(days=lookback_days)
    
    if 'Aksi' in df_master_filtered.columns and 'Perubahan_Saham' in df_master_filtered.columns:
        df_beli = df_master_filtered[
            (df_master_filtered['Aksi'] == 'Beli') &
            (df_master_filtered['Perubahan_Saham'] >= min_beli) &
            (df_master_filtered['Tanggal_Data'] >= cutoff_date)
        ].copy()
        
        if not df_beli.empty:
            df_akumulasi = df_beli.groupby(['UBO', 'Kode Efek']).agg({
                'Perubahan_Saham': 'sum',
                'Estimasi_Nilai': 'sum',
                'Tanggal_Data': ['first', 'last', 'count']
            }).reset_index()
            
            df_akumulasi.columns = [
                'UBO', 'Kode Efek',
                'Total_Beli_Lembar', 'Total_Nilai_Rp',
                'Tgl_Pertama', 'Tgl_Terakhir', 'Frekuensi_Transaksi'
            ]
            
            df_akumulasi['Skor_Akumulasi'] = (
                np.log1p(df_akumulasi['Total_Beli_Lembar']) * 0.5 +
                np.log1p(df_akumulasi['Frekuensi_Transaksi']) * 0.3 +
                (1 - (df_akumulasi['Tgl_Terakhir'] - df_akumulasi['Tgl_Pertama']).dt.days / lookback_days) * 0.2
            )
            
            df_akumulasi = df_akumulasi[df_akumulasi['Frekuensi_Transaksi'] >= min_freq]
            df_akumulasi = df_akumulasi.sort_values('Skor_Akumulasi', ascending=False)
            
            st.subheader(f"🎯 {len(df_akumulasi)} Indikasi Akumulasi Awal")
            
            if not df_akumulasi.empty:
                df_display = df_akumulasi.head(50).copy()
                df_display['Total_Beli_Lembar'] = df_display['Total_Beli_Lembar'].apply(format_lembar)
                df_display['Total_Nilai_Rp'] = df_display['Total_Nilai_Rp'].apply(format_rupiah)
                
                st.dataframe(
                    df_display[['UBO', 'Kode Efek', 'Total_Beli_Lembar', 'Total_Nilai_Rp', 'Frekuensi_Transaksi']],
                    use_container_width=True,
                    hide_index=True
                )

# =============================================================================
# TAB 4: WATCHLIST (SESUAI SCRIPT LAMA)
# =============================================================================
with tab4:
    st.header("📋 Watchlist")
    
    all_stocks = sorted(df_harian['Stock Code'].unique())
    default_stocks = [s for s in ["BBCA", "BBRI", "ADRO", "TLKM"] if s in all_stocks]
    watchlist = st.multiselect("Pilih Saham:", all_stocks, default=default_stocks)
    
    if watchlist:
        df_watch = df_harian[df_harian['Stock Code'].isin(watchlist)].copy()
        
        st.subheader("🚦 Status Watchlist")
        
        summary_data = []
        for stock in watchlist:
            stock_data = df_watch[df_watch['Stock Code'] == stock].sort_values('Last Trading Date').iloc[-1:] if not df_watch[df_watch['Stock Code'] == stock].empty else None
            
            if stock_data is not None and not stock_data.empty:
                summary_data.append({
                    'Stock': stock,
                    'Harga': stock_data['Close'].iloc[0],
                    'Change %': stock_data['Change %'].iloc[0] if 'Change %' in stock_data.columns else 0,
                    'Volume Spike': stock_data.get('Volume Spike (x)', pd.Series([0])).iloc[0]
                })
        
        if summary_data:
            df_summary = pd.DataFrame(summary_data)
            df_summary['Harga'] = df_summary['Harga'].apply(format_rupiah)
            df_summary['Change %'] = df_summary['Change %'].apply(lambda x: f"{x:.2f}%")
            df_summary['Volume Spike'] = df_summary['Volume Spike'].apply(lambda x: f"{x:.1f}x")
            
            st.dataframe(df_summary, use_container_width=True, hide_index=True)

# =============================================================================
# TAB 5: FREE FLOAT & LIKUIDITAS (BARU)
# =============================================================================
with tab5:
    st.header("📊 Free Float & Likuiditas Analyzer")
    st.caption("Analisis struktur kepemilikan saham yang beredar di publik")
    
    df_latest = df_harian.sort_values('Last Trading Date').groupby('Stock Code').last().reset_index()
    
    if selected_sectors:
        df_latest = df_latest[df_latest['Sector'].isin(selected_sectors)]
    
    if 'Free Float' in df_latest.columns and 'Tradeble Shares' in df_latest.columns:
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            min_free_float = st.slider("Min Free Float (%)", 0.0, 100.0, 20.0, 5.0)
        with col_f2:
            min_volume = st.number_input("Min Volume (juta)", 0, 1000, 100) * 1e6
        
        df_latest['Turnover_Ratio'] = (df_latest['Volume'] / df_latest['Tradeble Shares']) * 100
        
        df_filtered = df_latest[
            (df_latest['Free Float'] >= min_free_float) &
            (df_latest['Volume'] >= min_volume)
        ].sort_values('Turnover_Ratio', ascending=False)
        
        st.subheader(f"🎯 {len(df_filtered)} Saham dengan Likuiditas Tinggi")
        
        if not df_filtered.empty:
            df_display = df_filtered[['Stock Code', 'Sector', 'Close', 'Free Float', 'Turnover_Ratio', 'Volume']].head(50).copy()
            df_display['Close'] = df_display['Close'].apply(format_rupiah)
            df_display['Free Float'] = df_display['Free Float'].apply(lambda x: f"{x:.1f}%")
            df_display['Turnover_Ratio'] = df_display['Turnover_Ratio'].apply(lambda x: f"{x:.2f}%")
            df_display['Volume'] = df_display['Volume'].apply(format_lembar)
            
            st.dataframe(df_display, use_container_width=True, hide_index=True)
    else:
        st.warning("⚠️ Data Free Float atau Tradeble Shares tidak tersedia")

# =============================================================================
# TAB 6: FOREIGN FLOW INTELLIGENCE (BARU)
# =============================================================================
with tab6:
    st.header("🌏 Foreign Flow Intelligence")
    st.caption("Analisis mendalam arus modal asing")
    
    if 'Net Foreign Flow' not in df_harian.columns:
        st.warning("⚠️ Kolom Foreign Flow tidak tersedia")
    else:
        df_foreign = df_harian[
            (df_harian['Last Trading Date'] >= pd.to_datetime(start_date)) &
            (df_harian['Last Trading Date'] <= pd.to_datetime(end_date))
        ].copy()
        
        if selected_sectors:
            df_foreign = df_foreign[df_foreign['Sector'].isin(selected_sectors)]
        
        df_foreign_agg = df_foreign.groupby('Stock Code').agg({
            'Net Foreign Flow': 'sum',
            'Close': 'last',
            'Sector': 'first'
        }).reset_index()
        
        st.subheader("🟢 Top 10 Net Foreign Buy")
        top_buy = df_foreign_agg.nlargest(10, 'Net Foreign Flow')
        if not top_buy.empty:
            fig_buy = px.bar(
                top_buy,
                x='Stock Code',
                y='Net Foreign Flow',
                color='Sector',
                title="Saham dengan Akumulasi Asing Terbesar"
            )
            fig_buy.update_layout(height=400)
            st.plotly_chart(fig_buy, use_container_width=True)
        
        st.subheader("🔴 Top 10 Net Foreign Sell")
        top_sell = df_foreign_agg.nsmallest(10, 'Net Foreign Flow')
        if not top_sell.empty:
            fig_sell = px.bar(
                top_sell,
                x='Stock Code',
                y='Net Foreign Flow',
                color='Sector',
                title="Saham dengan Distribusi Asing Terbesar"
            )
            fig_sell.update_layout(height=400)
            st.plotly_chart(fig_sell, use_container_width=True)

# =============================================================================
# TAB 7: FREE FLOAT VS KEPEMILIKAN 5% (BARU)
# =============================================================================
with tab7:
    st.header("🔄 Free Float vs Kepemilikan 5%")
    
    if df_master.empty:
        st.warning("Data kepemilikan 5% tidak tersedia")
    else:
        df_master_latest = df_master.sort_values('Tanggal_Data').groupby('Kode Efek').last().reset_index()
        
        df_own_agg = df_master_latest.groupby('Kode Efek').agg({
            'Jumlah Saham (Curr)': 'sum',
            'UBO': 'nunique'
        }).reset_index()
        
        df_own_agg.columns = ['Kode Efek', 'Total_Kepemilikan_5persen', 'Jumlah_UBO']
        
        df_latest = df_harian.sort_values('Last Trading Date').groupby('Stock Code').last().reset_index()
        
        df_analysis = pd.merge(
            df_latest[['Stock Code', 'Listed Shares', 'Free Float']],
            df_own_agg,
            left_on='Stock Code',
            right_on='Kode Efek',
            how='inner'
        )
        
        if not df_analysis.empty and 'Free Float' in df_analysis.columns:
            df_analysis['Kepemilikan_Persen'] = (df_analysis['Total_Kepemilikan_5persen'] / df_analysis['Listed Shares']) * 100
            
            st.subheader("📊 Korelasi Free Float vs Kepemilikan")
            
            fig = px.scatter(
                df_analysis,
                x='Free Float',
                y='Kepemilikan_Persen',
                hover_name='Stock Code',
                title="Free Float vs Kepemilikan 5%",
                labels={'Free Float': 'Free Float (%)', 'Kepemilikan_Persen': 'Kepemilikan 5% (%)'}
            )
            fig.add_hline(y=50, line_dash="dash", line_color="gray")
            fig.add_vline(x=30, line_dash="dash", line_color="gray")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.caption(f"🐋 Bandar Eye IDX - v3.0 (7 Tab) | Last Update: {datetime.now().strftime('%d-%m-%Y %H:%M')} | Data 5%: {len(df_master):,} records")
