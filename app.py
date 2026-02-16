import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import requests
from io import StringIO
import time

# =============================================================================
# KONFIGURASI: LINK GOOGLE DRIVE (HANYA CSV LIGHT)
# =============================================================================
FILE_IDS = {
    'harian': '1t_wCljhepGBqZVrvleuZKldomQKop9DY',           # Kompilasi_Data_1Tahun.csv
    'ksei': '1eTUIC120SHTCzvBk77Q87w0X56F2HkWz',             # KSEI_Shareholder_Processed.csv
    'master_5_light': '13Fj_EUhFuDI5LKZDBA1wmvGJPfVDWo6k'    # CSV Light ANDA (PASTIKAN ID INI BENAR)
}

# =============================================================================
# FUNGSI LOAD CSV DARI GOOGLE DRIVE
# =============================================================================
@st.cache_data(ttl=3600, show_spinner="Loading data...")
def load_csv_from_gdrive(file_id, max_retries=3):
    """Load CSV dari Google Drive dengan multiple fallback method"""
    
    # Method 1: Standard download
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    for attempt in range(max_retries):
        try:
            session = requests.Session()
            response = session.get(url, stream=True, timeout=30)
            
            # Handle virus warning
            if 'Virus scan warning' in response.text or 'Quota exceeded' in response.text:
                import re
                match = re.search(r'confirm=([0-9A-Za-z]+)', response.text)
                if match:
                    confirm_token = match.group(1)
                    url = f"https://drive.google.com/uc?export=download&confirm={confirm_token}&id={file_id}"
                    response = session.get(url, stream=True, timeout=30)
            
            response.raise_for_status()
            
            # Baca CSV
            content = response.content.decode('utf-8')
            df = pd.read_csv(StringIO(content))
            return df
            
        except Exception as e:
            st.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt == 2:  # Last attempt
                try:
                    # Method 2: Direct download
                    url = f"https://drive.google.com/uc?id={file_id}"
                    df = pd.read_csv(url)
                    return df
                except:
                    pass
            time.sleep(2)
    
    st.error(f"❌ Gagal load file ID {file_id} setelah {max_retries} percobaan")
    return None

# =============================================================================
# FUNGSI LOAD DATA HARIAN
# =============================================================================
@st.cache_data(ttl=3600, show_spinner="Loading data harian...")
def load_harian():
    """Load data harian dari Google Drive"""
    try:
        df = load_csv_from_gdrive(FILE_IDS['harian'])
        
        if df is None or df.empty:
            st.error("❌ Data harian tidak tersedia")
            return pd.DataFrame()
        
        # Parsing tanggal
        if 'Last Trading Date' in df.columns:
            df['Last Trading Date'] = pd.to_datetime(df['Last Trading Date'], errors='coerce')
            df = df.dropna(subset=['Last Trading Date'])
        
        # Konversi numerik untuk kolom penting
        numeric_cols = [
            'Close', 'Volume', 'Value', 'Change %', 'Volume Spike (x)',
            'Avg_Order_Volume', 'MA50_AOVol', 'Bid/Offer Imbalance',
            'Foreign Buy', 'Foreign Sell', 'Net Foreign Flow',
            'Listed Shares', 'Tradeble Shares', 'Free Float'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        st.sidebar.success(f"✅ Data Harian: {len(df):,} baris")
        return df
        
    except Exception as e:
        st.error(f"❌ Gagal load data harian: {e}")
        return pd.DataFrame()

# =============================================================================
# FUNGSI LOAD DATA KSEI
# =============================================================================
@st.cache_data(ttl=86400, show_spinner="Loading data KSEI...")
def load_ksei():
    """Load data KSEI bulanan"""
    try:
        df = load_csv_from_gdrive(FILE_IDS['ksei'])
        
        if df is None or df.empty:
            return pd.DataFrame()
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date'])
        
        st.sidebar.success(f"✅ Data KSEI: {len(df):,} baris")
        return df
        
    except Exception as e:
        st.warning(f"⚠️ Data KSEI tidak tersedia: {e}")
        return pd.DataFrame()

# =============================================================================
# FUNGSI LOAD DATA MASTER 5% (CSV LIGHT)
# =============================================================================
@st.cache_data(ttl=86400, show_spinner="Loading data kepemilikan 5%...")
def load_master_5():
    """Load data master 5% dari CSV Light"""
    try:
        df = load_csv_from_gdrive(FILE_IDS['master_5_light'])
        
        if df is None or df.empty:
            st.error("❌ Data kepemilikan 5% tidak tersedia")
            return pd.DataFrame()
        
        # Parsing tanggal
        if 'Tanggal_Data' in df.columns:
            df['Tanggal_Data'] = pd.to_datetime(df['Tanggal_Data'], errors='coerce')
            df = df.dropna(subset=['Tanggal_Data'])
        
        # Konversi numerik
        numeric_cols = ['Jumlah Saham (Curr)', 'Perubahan_Saham', 'Close_Price', 'Estimasi_Nilai']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Pastikan kolom boolean ada
        bool_cols = ['Is_Adaro', 'Is_LKH', 'Is_Saratoga', 'Is_Nominee', 'Is_Foreign']
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].astype(bool)
        
        st.sidebar.success(f"✅ Data 5%: {len(df):,} baris")
        return df
        
    except Exception as e:
        st.error(f"❌ Gagal load data 5%: {e}")
        return pd.DataFrame()

# =============================================================================
# FORMATTER ANGKA
# =============================================================================
def format_rupiah(angka):
    """Format angka ke Rupiah dengan separator titik"""
    if pd.isna(angka) or angka == 0:
        return "Rp 0"
    return f"Rp {angka:,.0f}".replace(",", ".")

def format_lembar(angka):
    """Format lembar saham dengan separator titik"""
    if pd.isna(angka) or angka == 0:
        return "0"
    return f"{angka:,.0f}".replace(",", ".")

def format_persen(angka):
    """Format persentase dengan 2 desimal"""
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
st.sidebar.caption("v3.0 - CSV Light Version | 20 Tahun Cycle Experience")

# Load semua data
with st.spinner('Memuat data harga...'):
    df_harian = load_harian()
with st.spinner('Memuat data KSEI...'):
    df_ksei = load_ksei()
with st.spinner('Memuat data kepemilikan 5%...'):
    df_master = load_master_5()

# Cek apakah data berhasil di-load
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
    st.sidebar.warning("Kolom 'Sector' tidak ditemukan")

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
# TAB 1: MOMENTUM BANDAR
# =============================================================================
with tab1:
    st.header("📈 Momentum & Anomali Bandar")
    st.caption("Deteksi akumulasi/distribusi dari data harian")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_volume_spike = st.slider("Min Volume Spike (x)", 0.0, 5.0, 1.5, 0.1)
    with col2:
        min_ao_ratio = st.slider("Min AOVol Ratio (vs MA50)", 0.0, 5.0, 2.0, 0.1)
    with col3:
        min_imbalance = st.slider("Min Bid/Offer Imbalance", -1.0, 1.0, 0.1, 0.05)
    
    # Filter data
    df_filtered = df_harian[
        (df_harian['Last Trading Date'] >= pd.to_datetime(start_date)) &
        (df_harian['Last Trading Date'] <= pd.to_datetime(end_date))
    ].copy()
    
    if selected_sectors:
        df_filtered = df_filtered[df_filtered['Sector'].isin(selected_sectors)]
    
    # Filter kondisi
    filter_condition = pd.Series([True] * len(df_filtered))
    
    if 'Volume Spike (x)' in df_filtered.columns:
        filter_condition &= (df_filtered['Volume Spike (x)'] >= min_volume_spike)
    
    if 'Avg_Order_Volume' in df_filtered.columns and 'MA50_AOVol' in df_filtered.columns:
        ao_ratio = df_filtered['Avg_Order_Volume'] / df_filtered['MA50_AOVol'].replace(0, np.nan)
        filter_condition &= (ao_ratio >= min_ao_ratio)
    
    if 'Bid/Offer Imbalance' in df_filtered.columns:
        filter_condition &= (df_filtered['Bid/Offer Imbalance'] >= min_imbalance)
    
    df_anomaly = df_filtered[filter_condition].copy()
    
    # Hitung potensi
    df_anomaly['Potensi'] = 0
    if 'Volume Spike (x)' in df_anomaly.columns:
        df_anomaly['Potensi'] += df_anomaly['Volume Spike (x)'] * 0.3
    if 'Avg_Order_Volume' in df_anomaly.columns and 'MA50_AOVol' in df_anomaly.columns:
        ao_ratio_val = df_anomaly['Avg_Order_Volume'] / df_anomaly['MA50_AOVol'].replace(0, np.nan)
        df_anomaly['Potensi'] += ao_ratio_val.fillna(0) * 0.4
    if 'Bid/Offer Imbalance' in df_anomaly.columns:
        df_anomaly['Potensi'] += (df_anomaly['Bid/Offer Imbalance'] + 1) * 0.3
    
    # Ambil data terakhir untuk merge free float
    df_latest = df_harian.sort_values('Last Trading Date').groupby('Stock Code').last().reset_index()
    df_anomaly = df_anomaly.merge(
        df_latest[['Stock Code', 'Free Float', 'Tradeble Shares']],
        on='Stock Code',
        how='left'
    )
    
    # Hitung volume vs tradeble
    if 'Tradeble Shares' in df_anomaly.columns and 'Volume' in df_anomaly.columns:
        df_anomaly['Volume_vs_Tradeble'] = (df_anomaly['Volume'] / df_anomaly['Tradeble Shares']) * 100
    
    df_anomaly = df_anomaly.sort_values('Potensi', ascending=False)
    
    st.subheader(f"🎯 {len(df_anomaly)} Saham dengan Aktivitas Bandar Terdeteksi")
    
    if not df_anomaly.empty:
        # PERINGATAN untuk free float kecil
        if 'Free Float' in df_anomaly.columns and 'Volume_vs_Tradeble' in df_anomaly.columns:
            tight_stocks = df_anomaly[
                (df_anomaly['Free Float'] < 15) & 
                (df_anomaly['Volume_vs_Tradeble'] > 30)
            ].nlargest(5, 'Volume_vs_Tradeble')
            
            if not tight_stocks.empty:
                st.warning("⚠️ **PERHATIAN: Free Float Kecil (<15%) dengan Volume Besar!**")
                
                tight_display = tight_stocks[['Stock Code', 'Free Float', 'Volume_vs_Tradeble', 'Volume Spike (x)']].copy()
                tight_display['Free Float'] = tight_display['Free Float'].apply(lambda x: f"{x:.1f}%")
                tight_display['Volume_vs_Tradeble'] = tight_display['Volume_vs_Tradeble'].apply(lambda x: f"{x:.1f}%")
                st.dataframe(tight_display, use_container_width=True, hide_index=True)
        
        # Kolom display
        display_cols = ['Stock Code', 'Last Trading Date']
        optional_cols = ['Close', 'Change %', 'Volume Spike (x)', 'Avg_Order_Volume', 
                        'MA50_AOVol', 'Bid/Offer Imbalance', 'Net Foreign Flow',
                        'Free Float', 'Volume_vs_Tradeble', 'Potensi']
        
        for col in optional_cols:
            if col in df_anomaly.columns:
                display_cols.append(col)
        
        df_display = df_anomaly[display_cols].head(100).copy()
        
        # Format kolom
        if 'Close' in df_display.columns:
            df_display['Close'] = df_display['Close'].apply(format_rupiah)
        if 'Free Float' in df_display.columns:
            df_display['Free Float'] = df_display['Free Float'].apply(lambda x: f"{x:.1f}%")
        if 'Volume_vs_Tradeble' in df_display.columns:
            df_display['Volume_vs_Tradeble'] = df_display['Volume_vs_Tradeble'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(df_display, use_container_width=True, hide_index=True)
    else:
        st.info("Tidak ada saham dengan kriteria tersebut. Coba turunkan threshold.")

# =============================================================================
# TAB 2: KSEI BIG MONEY TRACKER
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
                top_buyers['Top_Buyer_Vol'] = top_buyers['Top_Buyer_Vol'].apply(format_lembar)
                top_buyers['Top_Buyer_Val'] = top_buyers['Top_Buyer_Val'].apply(format_rupiah)
                st.dataframe(top_buyers, use_container_width=True, hide_index=True)
        
        with col2:
            st.subheader("🔴 Top Seller (Volume)")
            if 'Top_Seller_Vol' in df_ksei_filtered.columns:
                top_sellers = df_ksei_filtered.nsmallest(top_n, 'Top_Seller_Vol')[['Code', 'Date', 'Top_Seller', 'Top_Seller_Vol', 'Top_Seller_Val']]
                top_sellers['Top_Seller_Vol'] = top_sellers['Top_Seller_Vol'].apply(format_lembar)
                top_sellers['Top_Seller_Val'] = top_sellers['Top_Seller_Val'].apply(format_rupiah)
                st.dataframe(top_sellers, use_container_width=True, hide_index=True)

# =============================================================================
# TAB 3: AKUMULASI AWAL (UBO CLUSTERED)
# =============================================================================
with tab3:
    st.header("🕵️ DETEKSI AKUMULASI AWAL (UBO Clustered)")
    
    if df_master.empty:
        st.warning("⚠️ Data kepemilikan 5% tidak tersedia.")
        st.stop()
    
    if 'UBO' not in df_master.columns:
        st.warning("⚠️ Kolom 'UBO' tidak ditemukan.")
        st.stop()
    
    # Filter tanggal
    df_master_filtered = df_master[
        (df_master['Tanggal_Data'] >= pd.to_datetime(start_date)) &
        (df_master['Tanggal_Data'] <= pd.to_datetime(end_date))
    ].copy()
    
    if df_master_filtered.empty:
        st.warning("Tidak ada data kepemilikan 5% pada periode yang dipilih.")
        st.stop()
    
    # PARAMETER DETEKSI
    st.subheader("⚙️ Parameter Akumulasi Awal")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        min_beli = st.number_input("Minimal Beli (lembar)", min_value=1000, value=100000, step=10000, format="%d")
    with col2:
        lookback_days = st.number_input("Lookback (hari)", min_value=7, max_value=90, value=30)
    with col3:
        min_freq = st.number_input("Minimal Frekuensi", min_value=1, max_value=20, value=2)
    
    cutoff_date = pd.to_datetime(end_date) - timedelta(days=lookback_days)
    
    # DETEKSI AKUMULASI
    if 'Aksi' in df_master_filtered.columns and 'Perubahan_Saham' in df_master_filtered.columns:
        df_beli = df_master_filtered[
            (df_master_filtered['Aksi'] == 'Beli') &
            (df_master_filtered['Perubahan_Saham'] >= min_beli) &
            (df_master_filtered['Tanggal_Data'] >= cutoff_date)
        ].copy()
        
        if not df_beli.empty:
            # Agregasi per UBO dan saham
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
            
            # Skor akumulasi
            df_akumulasi['Skor_Akumulasi'] = (
                np.log1p(df_akumulasi['Total_Beli_Lembar']) * 0.5 +
                np.log1p(df_akumulasi['Frekuensi_Transaksi']) * 0.3 +
                (1 - (df_akumulasi['Tgl_Terakhir'] - df_akumulasi['Tgl_Pertama']).dt.days / lookback_days) * 0.2
            )
            
            df_akumulasi = df_akumulasi[df_akumulasi['Frekuensi_Transaksi'] >= min_freq]
            df_akumulasi = df_akumulasi.sort_values('Skor_Akumulasi', ascending=False)
            
            st.subheader(f"🎯 {len(df_akumulasi)} Indikasi Akumulasi Awal")
            
            if not df_akumulasi.empty:
                # Format display
                df_display = df_akumulasi.head(50).copy()
                df_display['Total_Beli_Lembar'] = df_display['Total_Beli_Lembar'].apply(format_lembar)
                df_display['Total_Nilai_Rp'] = df_display['Total_Nilai_Rp'].apply(format_rupiah)
                df_display['Skor_Akumulasi'] = df_display['Skor_Akumulasi'].apply(lambda x: f"{x:.2f}")
                
                display_columns = ['UBO', 'Kode Efek', 'Total_Beli_Lembar', 'Total_Nilai_Rp', 
                                  'Frekuensi_Transaksi', 'Tgl_Pertama', 'Tgl_Terakhir', 'Skor_Akumulasi']
                
                st.dataframe(
                    df_display[[c for c in display_columns if c in df_display.columns]],
                    use_container_width=True,
                    hide_index=True
                )

# =============================================================================
# TAB 4: WATCHLIST
# =============================================================================
with tab4:
    st.header("📋 Watchlist")
    
    all_stocks = sorted(df_harian['Stock Code'].unique())
    default_stocks = [s for s in ["BBCA", "BBRI", "ADRO", "TLKM"] if s in all_stocks]
    watchlist = st.multiselect("Pilih Saham untuk Watchlist:", all_stocks, default=default_stocks)
    
    if watchlist:
        df_watch = df_harian[df_harian['Stock Code'].isin(watchlist)].copy()
        df_watch_latest = df_watch.sort_values('Last Trading Date').groupby('Stock Code').last().reset_index()
        
        st.subheader("🚦 Status Watchlist")
        
        summary_data = []
        for stock in watchlist:
            stock_data = df_watch_latest[df_watch_latest['Stock Code'] == stock]
            if stock_data.empty:
                continue
                
            last_close = stock_data['Close'].iloc[0]
            last_vol_spike = stock_data.get('Volume Spike (x)', pd.Series([0])).iloc[0]
            
            summary_data.append({
                'Stock': stock,
                'Harga': last_close,
                'Volume Spike': last_vol_spike
            })
        
        if summary_data:
            df_summary = pd.DataFrame(summary_data)
            df_summary['Harga'] = df_summary['Harga'].apply(format_rupiah)
            df_summary['Volume Spike'] = df_summary['Volume Spike'].apply(lambda x: f"{x:.1f}x")
            
            st.dataframe(df_summary, use_container_width=True, hide_index=True)

# =============================================================================
# TAB 5: FREE FLOAT ANALYZER
# =============================================================================
with tab5:
    st.header("📊 Free Float Analyzer")
    
    df_latest = df_harian.sort_values('Last Trading Date').groupby('Stock Code').last().reset_index()
    
    if selected_sectors:
        df_latest = df_latest[df_latest['Sector'].isin(selected_sectors)]
    
    if 'Free Float' in df_latest.columns:
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            min_free_float = st.slider("Min Free Float (%)", 0.0, 100.0, 20.0, 5.0)
        with col_f2:
            min_volume = st.number_input("Min Volume (juta)", 0, 1000, 100) * 1e6
        
        df_filtered = df_latest[
            (df_latest['Free Float'] >= min_free_float) &
            (df_latest['Volume'] >= min_volume)
        ].sort_values('Free Float', ascending=False)
        
        st.subheader(f"🎯 {len(df_filtered)} Saham")
        
        if not df_filtered.empty:
            display_cols = ['Stock Code', 'Sector', 'Close', 'Free Float', 'Volume']
            df_display = df_filtered[display_cols].copy()
            df_display['Close'] = df_display['Close'].apply(format_rupiah)
            df_display['Free Float'] = df_display['Free Float'].apply(lambda x: f"{x:.1f}%")
            df_display['Volume'] = df_display['Volume'].apply(format_lembar)
            
            st.dataframe(df_display, use_container_width=True, hide_index=True)

# =============================================================================
# TAB 6: FOREIGN FLOW
# =============================================================================
with tab6:
    st.header("🌏 Foreign Flow")
    
    if 'Net Foreign Flow' not in df_harian.columns:
        st.warning("Kolom Foreign Flow tidak tersedia")
    else:
        df_foreign = df_harian[
            (df_harian['Last Trading Date'] >= pd.to_datetime(start_date)) &
            (df_harian['Last Trading Date'] <= pd.to_datetime(end_date))
        ].copy()
        
        if selected_sectors:
            df_foreign = df_foreign[df_foreign['Sector'].isin(selected_sectors)]
        
        df_foreign_agg = df_foreign.groupby('Stock Code').agg({
            'Net Foreign Flow': 'sum',
            'Close': 'last'
        }).reset_index()
        
        top_buy = df_foreign_agg.nlargest(10, 'Net Foreign Flow')
        st.subheader("🟢 Top 10 Net Foreign Buy")
        st.dataframe(top_buy, use_container_width=True, hide_index=True)

# =============================================================================
# TAB 7: FREE FLOAT VS KEPEMILIKAN
# =============================================================================
with tab7:
    st.header("🔄 Free Float vs Kepemilikan 5%")
    
    if df_master.empty:
        st.warning("Data kepemilikan 5% tidak tersedia")
    else:
        # Hitung kepemilikan agregat
        df_master_latest = df_master.sort_values('Tanggal_Data').groupby('Kode Efek').last().reset_index()
        
        df_own_agg = df_master_latest.groupby('Kode Efek').agg({
            'Jumlah Saham (Curr)': 'sum',
            'UBO': 'nunique'
        }).reset_index()
        
        df_own_agg.columns = ['Kode Efek', 'Total_Kepemilikan_5persen', 'Jumlah_UBO']
        
        # Merge dengan data harga
        df_latest = df_harian.sort_values('Last Trading Date').groupby('Stock Code').last().reset_index()
        
        df_analysis = pd.merge(
            df_latest[['Stock Code', 'Listed Shares', 'Free Float', 'Volume']],
            df_own_agg,
            left_on='Stock Code',
            right_on='Kode Efek',
            how='inner'
        )
        
        if not df_analysis.empty:
            df_analysis['Kepemilikan_Persen'] = (df_analysis['Total_Kepemilikan_5persen'] / df_analysis['Listed Shares']) * 100
            
            st.subheader("📊 Korelasi Free Float vs Kepemilikan")
            
            fig = px.scatter(
                df_analysis,
                x='Free Float',
                y='Kepemilikan_Persen',
                hover_name='Stock Code',
                title="Free Float vs Kepemilikan 5%"
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.caption(f"🐋 Bandar Eye IDX - CSV Light Version | Last Update: {datetime.now().strftime('%d-%m-%Y %H:%M')}")
