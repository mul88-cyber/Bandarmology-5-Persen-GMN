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
# KONFIGURASI: LINK GOOGLE DRIVE (FILE SUDAH PRE-PROCESSED DI COLAB)
# =============================================================================
FILE_IDS = {
    'harian': '1t_wCljhepGBqZVrvleuZKldomQKop9DY',           # Kompilasi_Data_1Tahun.csv
    'ksei': '1eTUIC120SHTCzvBk77Q87w0X56F2HkWz',             # KSEI_Shareholder_Processed.csv
    'master_5_parquet': '1Y39bJ20mBBbmcZCcXdVCf9AwAAChbzK5', # PARQUET ANDA
    'master_5_light': '13Fj_EUhFuDI5LKZDBA1wmvGJPfVDWo6k'    # CSV LIGHT ANDA
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
        
        with requests.Session() as session:
            response = session.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            if 'Virus scan warning' in response.text:
                import re
                match = re.search(r'confirm=([0-9A-Za-z]+)', response.text)
                if match:
                    confirm_token = match.group(1)
                    url = f"https://drive.google.com/uc?export=download&confirm={confirm_token}&id={file_id}"
                    response = session.get(url, stream=True, timeout=60)
            
            content = BytesIO()
            for chunk in response.iter_content(chunk_size=8192):
                content.write(chunk)
            content.seek(0)
            
            df = pd.read_parquet(content)
            return df
    except Exception as e:
        st.warning(f"⚠️ Gagal load Parquet: {e}")
        return None

# =============================================================================
# CACHE DATA LOADING (OPTIMUM UNTUK STREAMLIT CLOUD)
# =============================================================================
@st.cache_data(ttl=3600, show_spinner="Loading data harian...")
def load_harian():
    """Load data harian dengan semua kolom"""
    try:
        df = load_parquet_from_gdrive(FILE_IDS['harian'])
        if df is None:
            df = load_csv_from_gdrive(FILE_IDS['harian'])
        
        # Parsing tanggal
        df['Last Trading Date'] = pd.to_datetime(df['Last Trading Date'], errors='coerce')
        df = df.dropna(subset=['Last Trading Date'])
        
        # Konversi numerik untuk semua kolom
        numeric_cols = [
            'Open Price', 'High', 'Low', 'Close', 'Typical Price',
            'Volume', 'Value', 'Frequency', 'Volume Spike (x)',
            'Avg_Order_Volume', 'MA50_AOVol', 'MA20_vol',
            'Bid Volume', 'Offer Volume', 'Bid/Offer Imbalance',
            'Foreign Buy', 'Foreign Sell', 'Net Foreign Flow',
            'Listed Shares', 'Tradeble Shares', 'Free Float',
            'Change %', 'Non Regular Volume', 'Non Regular Value'
        ]
        
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

@st.cache_data(ttl=86400, show_spinner="Loading data kepemilikan 5%...")
def load_master_5():
    """Load data master 5% - HANYA CSV LIGHT"""
    
    # LANGSUNG PAKAI CSV LIGHT
    try:
        df = load_csv_from_gdrive(FILE_IDS['master_5_light'])
        df['Tanggal_Data'] = pd.to_datetime(df['Tanggal_Data'], errors='coerce')
        df = df.dropna(subset=['Tanggal_Data'])
        
        # Konversi numerik
        numeric_cols = ['Jumlah Saham (Curr)', 'Perubahan_Saham', 'Close_Price', 'Estimasi_Nilai']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        st.sidebar.info("📄 Load data 5% (CSV Light)")
        return df
    except Exception as e:
        st.error(f"❌ Gagal load CSV Light: {e}")
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
st.sidebar.caption("v3.0 - Free Float Intelligence | 20 Tahun Cycle Experience")

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
    
    if 'UBO' in df_master.columns:
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
# TAB 1: MOMENTUM BANDAR (Harian)
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
    df_anomaly['Volume_vs_Tradeble'] = (df_anomaly['Volume'] / df_anomaly['Tradeble Shares']) * 100
    
    df_anomaly = df_anomaly.sort_values('Potensi', ascending=False)
    
    st.subheader(f"🎯 {len(df_anomaly)} Saham dengan Aktivitas Bandar Terdeteksi")
    
    if not df_anomaly.empty:
        # PERINGATAN untuk free float kecil
        tight_stocks = df_anomaly[
            (df_anomaly['Free Float'] < 15) & 
            (df_anomaly['Volume_vs_Tradeble'] > 30)
        ].nlargest(5, 'Volume_vs_Tradeble')
        
        if not tight_stocks.empty:
            st.warning("⚠️ **PERHATIAN: Free Float Kecil (<15%) dengan Volume Besar!**")
            st.warning("Indikasi transaksi antar pemilik >5% atau potensi manipulasi")
            
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
        
        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True
        )
        
        # Scatter plot
        st.subheader("📊 Volume Spike vs AOVol Ratio")
        
        if ('Volume Spike (x)' in df_anomaly.columns and 
            'Avg_Order_Volume' in df_anomaly.columns):
            
            df_scatter = df_anomaly.head(50).copy()
            df_scatter['AOVol_Ratio'] = df_scatter['Avg_Order_Volume'] / df_scatter['MA50_AOVol'].replace(0, np.nan)
            df_scatter = df_scatter.dropna(subset=['Volume Spike (x)', 'AOVol_Ratio'])
            
            if not df_scatter.empty:
                fig = px.scatter(
                    df_scatter,
                    x='Volume Spike (x)',
                    y='AOVol_Ratio',
                    color='Free Float' if 'Free Float' in df_scatter.columns else None,
                    size='Potensi',
                    hover_data=['Stock Code', 'Close'],
                    title="Warna = Free Float (%) - Merah = Free Float Kecil"
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Tidak ada saham dengan kriteria tersebut. Coba turunkan threshold.")

# =============================================================================
# TAB 2: KSEI BIG MONEY TRACKER (Bulanan)
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
        
        st.subheader("🔍 Detail Saham")
        kode_saham = st.text_input("Masukkan Kode Saham (contoh: AADI, BBCA)", "").upper()
        
        if kode_saham:
            df_detail = df_ksei_filtered[df_ksei_filtered['Code'] == kode_saham].sort_values('Date', ascending=False)
            if not df_detail.empty:
                cols_detail = ['Date', 'Price', 'Total_Local', 'Total_Foreign', 
                              'Top_Buyer', 'Top_Buyer_Vol', 'Top_Seller', 'Top_Seller_Vol']
                cols_detail = [c for c in cols_detail if c in df_detail.columns]
                
                df_detail_display = df_detail[cols_detail].head(12).copy()
                if 'Top_Buyer_Vol' in df_detail_display.columns:
                    df_detail_display['Top_Buyer_Vol'] = df_detail_display['Top_Buyer_Vol'].apply(format_lembar)
                if 'Top_Seller_Vol' in df_detail_display.columns:
                    df_detail_display['Top_Seller_Vol'] = df_detail_display['Top_Seller_Vol'].apply(format_lembar)
                if 'Price' in df_detail_display.columns:
                    df_detail_display['Price'] = df_detail_display['Price'].apply(format_rupiah)
                
                st.dataframe(df_detail_display, use_container_width=True, hide_index=True)
            else:
                st.warning(f"Data tidak ditemukan untuk {kode_saham}")

# =============================================================================
# TAB 3: AKUMULASI AWAL DARI DATA 5% (UBO CLUSTERED)
# =============================================================================
with tab3:
    st.header("🕵️ DETEKSI AKUMULASI AWAL (UBO Clustered)")
    st.markdown("""
    > **Bandarmology Intelligence**: Data sudah di-cluster per **Ultimate Beneficial Owner (UBO)**.
    > *PT ADARO STRATEGIC INVESTMENTS*, *U20B2S3 A90G4...*, *ADARO STRATEGIC INVESTMENTS PT* → **1 entitas: ADARO** ✅
    """)
    
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
    
    # Filter sektor
    if selected_sectors and 'Sector' not in df_master_filtered.columns:
        if 'Kode Efek' in df_master_filtered.columns and 'Sector' in df_harian.columns:
            sector_map = df_harian[['Stock Code', 'Sector']].drop_duplicates('Stock Code')
            df_master_filtered = df_master_filtered.merge(sector_map, left_on='Kode Efek', right_on='Stock Code', how='left')
    
    if selected_sectors and 'Sector' in df_master_filtered.columns:
        df_master_filtered = df_master_filtered[df_master_filtered['Sector'].isin(selected_sectors)]
    
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
        
        # Tambah jumlah rekening
        if 'Rekening_Bersih' in df_beli.columns:
            rekening_list = df_beli.groupby(['UBO', 'Kode Efek'])['Rekening_Bersih'].nunique().reset_index()
            rekening_list.columns = ['UBO', 'Kode Efek', 'Jumlah_Rekening']
            df_akumulasi = df_akumulasi.merge(rekening_list, on=['UBO', 'Kode Efek'], how='left')
        
        # Skor akumulasi
        df_akumulasi['Skor_Akumulasi'] = (
            np.log1p(df_akumulasi['Total_Beli_Lembar']) * 0.5 +
            np.log1p(df_akumulasi['Frekuensi_Transaksi']) * 0.3 +
            (1 - (df_akumulasi['Tgl_Terakhir'] - df_akumulasi['Tgl_Pertama']).dt.days / lookback_days) * 0.2
        )
        
        df_akumulasi = df_akumulasi[df_akumulasi['Frekuensi_Transaksi'] >= min_freq]
        df_akumulasi = df_akumulasi.sort_values('Skor_Akumulasi', ascending=False)
        
        # Merge dengan sektor
        if 'Sector' not in df_akumulasi.columns:
            sector_map = df_harian[['Stock Code', 'Sector']].drop_duplicates('Stock Code')
            df_akumulasi = df_akumulasi.merge(sector_map, left_on='Kode Efek', right_on='Stock Code', how='left')
        
        st.subheader(f"🎯 {len(df_akumulasi)} Indikasi Akumulasi Awal (per UBO)")
        
        if not df_akumulasi.empty:
            # Filter cepat
            col_filter1, col_filter2, col_filter3 = st.columns(3)
            with col_filter1:
                show_adaro = st.checkbox("🎯 Hanya Adaro Group", value=False)
            with col_filter2:
                show_lkh = st.checkbox("👑 Hanya Lo Kheng Hong", value=False)
            with col_filter3:
                show_saratoga = st.checkbox("📊 Hanya Saratoga", value=False)
            
            df_filtered_display = df_akumulasi.copy()
            
            # Apply filters
            if show_adaro:
                adaro_ubos = df_master[df_master['Is_Adaro'] == True]['UBO'].unique()
                df_filtered_display = df_filtered_display[df_filtered_display['UBO'].isin(adaro_ubos)]
            if show_lkh:
                lkh_ubos = df_master[df_master['Is_LKH'] == True]['UBO'].unique()
                df_filtered_display = df_filtered_display[df_filtered_display['UBO'].isin(lkh_ubos)]
            if show_saratoga:
                saratoga_ubos = df_master[df_master['Is_Saratoga'] == True]['UBO'].unique()
                df_filtered_display = df_filtered_display[df_filtered_display['UBO'].isin(saratoga_ubos)]
            
            # Format display
            df_display = df_filtered_display.head(50).copy()
            df_display['Total_Beli_Lembar'] = df_display['Total_Beli_Lembar'].apply(format_lembar)
            df_display['Total_Nilai_Rp'] = df_display['Total_Nilai_Rp'].apply(format_rupiah)
            df_display['Skor_Akumulasi'] = df_display['Skor_Akumulasi'].apply(lambda x: f"{x:.2f}")
            
            display_columns = ['UBO', 'Kode Efek', 'Sector', 'Total_Beli_Lembar', 
                              'Total_Nilai_Rp', 'Frekuensi_Transaksi', 'Jumlah_Rekening', 
                              'Tgl_Pertama', 'Tgl_Terakhir', 'Skor_Akumulasi']
            
            st.dataframe(
                df_display[[c for c in display_columns if c in df_display.columns]],
                use_container_width=True,
                hide_index=True
            )
            
            # TOP 10 AKUMULATOR
            st.subheader("💰 Top 10 UBO Akumulator Terbesar")
            top10 = df_akumulasi.nlargest(10, 'Total_Nilai_Rp').copy()
            top10['Label'] = top10['UBO'].apply(lambda x: x[:30] + '...' if len(x) > 30 else x)
            
            fig = px.bar(
                top10,
                x='Total_Nilai_Rp',
                y='Label',
                color='Sector',
                orientation='h',
                title="Total Nilai Pembelian (Estimasi) per UBO"
            )
            fig.update_layout(height=500)
            fig.update_xaxes(tickformat=",.0f")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Tidak ada data akumulasi dengan kriteria tersebut.")

# =============================================================================
# TAB 4: WATCHLIST & KONVERGENSI
# =============================================================================
with tab4:
    st.header("📋 Watchlist & Konvergensi Sinyal")
    
    all_stocks = sorted(df_harian['Stock Code'].unique())
    default_stocks = [s for s in ["BBCA", "BBRI", "ADRO", "TLKM"] if s in all_stocks]
    watchlist = st.multiselect("Pilih Saham untuk Watchlist:", all_stocks, default=default_stocks)
    
    if watchlist:
        # Data untuk watchlist
        df_watch = df_harian[df_harian['Stock Code'].isin(watchlist)].copy()
        df_watch_latest = df_watch.sort_values('Last Trading Date').groupby('Stock Code').last().reset_index()
        
        # Data kepemilikan 5%
        df_watch_5 = df_master[df_master['Kode Efek'].isin(watchlist)] if not df_master.empty else pd.DataFrame()
        
        # Tabel ringkasan
        st.subheader("🚦 Status Watchlist")
        
        summary_data = []
        for stock in watchlist:
            # Data harga
            stock_data = df_watch_latest[df_watch_latest['Stock Code'] == stock]
            if stock_data.empty:
                continue
                
            last_close = stock_data['Close'].iloc[0]
            last_vol = stock_data['Volume'].iloc[0]
            last_vol_spike = stock_data.get('Volume Spike (x)', pd.Series([0])).iloc[0]
            last_foreign = stock_data.get('Net Foreign Flow', pd.Series([0])).iloc[0]
            
            # Data kepemilikan 5%
            aksi_5 = "-"
            ubo_aktif = "-"
            if not df_watch_5.empty:
                df_stock_5 = df_watch_5[df_watch_5['Kode Efek'] == stock].sort_values('Tanggal_Data', ascending=False)
                if not df_stock_5.empty:
                    last_aksi = df_stock_5.iloc[0]['Aksi']
                    last_ubo = df_stock_5.iloc[0]['UBO']
                    aksi_5 = f"{last_aksi} ({last_ubo[:15]}...)" if len(last_ubo) > 15 else f"{last_aksi} ({last_ubo})"
                    
                    # Hitung UBO aktif 7 hari terakhir
                    cutoff_7 = pd.to_datetime(end_date) - timedelta(days=7)
                    ubo_aktif = df_stock_5[df_stock_5['Tanggal_Data'] >= cutoff_7]['UBO'].nunique()
            
            summary_data.append({
                'Stock': stock,
                'Harga': last_close,
                'Change %': stock_data['Change %'].iloc[0] if 'Change %' in stock_data.columns else 0,
                'Volume Spike': last_vol_spike,
                'Foreign Flow': last_foreign,
                'Aksi 5%': aksi_5,
                'UBO Aktif (7d)': ubo_aktif
            })
        
        if summary_data:
            df_summary = pd.DataFrame(summary_data)
            
            # Format
            df_summary['Harga'] = df_summary['Harga'].apply(format_rupiah)
            df_summary['Change %'] = df_summary['Change %'].apply(lambda x: f"{x:.2f}%")
            df_summary['Volume Spike'] = df_summary['Volume Spike'].apply(lambda x: f"{x:.1f}x")
            df_summary['Foreign Flow'] = df_summary['Foreign Flow'].apply(format_rupiah)
            
            # Color coding untuk aksi
            def color_aksi(val):
                if 'Beli' in str(val):
                    return 'background-color: #90EE90'
                elif 'Jual' in str(val):
                    return 'background-color: #FFB6C6'
                return ''
            
            st.dataframe(
                df_summary.style.applymap(color_aksi, subset=['Aksi 5%']),
                use_container_width=True,
                hide_index=True
            )
        
        # Detail per saham
        st.markdown("---")
        for stock in watchlist:
            with st.expander(f"🔍 Detail {stock}"):
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.write("**📈 Data Harian (10 Hari Terakhir)**")
                    df_detail_h = df_watch[df_watch['Stock Code'] == stock].tail(10).copy()
                    if not df_detail_h.empty:
                        df_display_h = df_detail_h[['Last Trading Date', 'Close', 'Change %', 'Volume', 'Volume Spike (x)']].copy()
                        df_display_h['Close'] = df_display_h['Close'].apply(format_rupiah)
                        df_display_h['Volume'] = df_display_h['Volume'].apply(format_lembar)
                        st.dataframe(df_display_h, use_container_width=True, hide_index=True)
                
                with col_b:
                    st.write("**👥 Aktivitas 5% Terakhir**")
                    if not df_watch_5.empty:
                        df_detail_5 = df_watch_5[df_watch_5['Kode Efek'] == stock].tail(10).copy()
                        if not df_detail_5.empty:
                            df_display_5 = df_detail_5[['Tanggal_Data', 'UBO', 'Aksi', 'Jumlah Saham (Curr)', 'Perubahan_Saham']].copy()
                            df_display_5['Jumlah Saham (Curr)'] = df_display_5['Jumlah Saham (Curr)'].apply(format_lembar)
                            df_display_5['Perubahan_Saham'] = df_display_5['Perubahan_Saham'].apply(format_lembar)
                            st.dataframe(df_display_5, use_container_width=True, hide_index=True)
    else:
        st.info("Silakan pilih saham untuk watchlist.")

# =============================================================================
# TAB 5: FREE FLOAT & LIQUIDITY ANALYZER
# =============================================================================
with tab5:
    st.header("📊 Free Float & Likuiditas Analyzer")
    st.caption("Analisis struktur kepemilikan saham yang beredar di publik")
    
    # Ambil data terakhir untuk setiap saham
    df_latest = df_harian.sort_values('Last Trading Date').groupby('Stock Code').last().reset_index()
    
    if selected_sectors:
        df_latest = df_latest[df_latest['Sector'].isin(selected_sectors)]
    
    # =====================================================================
    # METRIK FREE FLOAT
    # =====================================================================
    st.subheader("📈 Statistik Free Float")
    
    col_ff1, col_ff2, col_ff3, col_ff4 = st.columns(4)
    
    with col_ff1:
        avg_free_float = df_latest['Free Float'].mean()
        st.metric("Rata-rata Free Float", f"{avg_free_float:.1f}%")
    
    with col_ff2:
        total_listed = df_latest['Listed Shares'].sum() / 1e9
        st.metric("Total Listed Shares", f"{total_listed:.2f} Miliar")
    
    with col_ff3:
        total_tradeble = df_latest['Tradeble Shares'].sum() / 1e9
        st.metric("Total Tradeble", f"{total_tradeble:.2f} Miliar")
    
    with col_ff4:
        # Rata-rata turnover (volume / tradeble shares)
        df_latest['Turnover'] = (df_latest['Volume'] / df_latest['Tradeble Shares']) * 100
        avg_turnover = df_latest['Turnover'].mean()
        st.metric("Rata-rata Turnover", f"{avg_turnover:.2f}%")
    
    # =====================================================================
    # FILTER PARAMETER
    # =====================================================================
    st.markdown("---")
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        min_free_float = st.slider("Min Free Float (%)", 0.0, 100.0, 20.0, 5.0, key="ff_min")
    
    with col_f2:
        min_volume = st.number_input("Min Volume (juta)", 0, 1000, 100, key="ff_vol") * 1e6
    
    with col_f3:
        min_turnover = st.slider("Min Turnover Ratio (%)", 0.0, 10.0, 1.0, 0.1, key="ff_turn")
    
    # Hitung turnover
    df_latest['Turnover_Ratio'] = (df_latest['Volume'] / df_latest['Tradeble Shares']) * 100
    df_latest['Nilai_Transaksi_M'] = df_latest['Value'] / 1e6  # Dalam juta
    df_latest['Free_Float_Kategori'] = pd.cut(
        df_latest['Free Float'],
        bins=[0, 10, 25, 50, 100],
        labels=['Very Tight (<10%)', 'Tight (10-25%)', 'Moderate (25-50%)', 'Loose (>50%)']
    )
    
    # Filter
    df_filtered = df_latest[
        (df_latest['Free Float'] >= min_free_float) &
        (df_latest['Volume'] >= min_volume) &
        (df_latest['Turnover_Ratio'] >= min_turnover)
    ].sort_values('Turnover_Ratio', ascending=False)
    
    st.subheader(f"🎯 {len(df_filtered)} Saham dengan Likuiditas Tinggi")
    
    if not df_filtered.empty:
        # Display columns
        display_cols = [
            'Stock Code', 'Company Name', 'Sector', 'Close',
            'Free Float', 'Tradeble Shares', 'Volume', 'Turnover_Ratio',
            'Nilai_Transaksi_M', 'Free_Float_Kategori'
        ]
        
        # Format display
        df_display = df_filtered[display_cols].copy()
        df_display['Close'] = df_display['Close'].apply(format_rupiah)
        df_display['Tradeble Shares'] = df_display['Tradeble Shares'].apply(format_lembar)
        df_display['Volume'] = df_display['Volume'].apply(format_lembar)
        df_display['Nilai_Transaksi_M'] = df_display['Nilai_Transaksi_M'].apply(lambda x: f"Rp {x:.0f} Jt")
        df_display['Turnover_Ratio'] = df_display['Turnover_Ratio'].apply(lambda x: f"{x:.2f}%")
        df_display['Free Float'] = df_display['Free Float'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(df_display, use_container_width=True, hide_index=True)
        
        # =================================================================
        # VISUALISASI FREE FLOAT DISTRIBUTION
        # =================================================================
        st.subheader("📊 Distribusi Free Float per Sektor")
        
        fig_ff = px.box(
            df_latest,
            x='Sector',
            y='Free Float',
            title="Distribusi Free Float per Sektor",
            points="all"
        )
        fig_ff.update_layout(height=500, xaxis_tickangle=-45)
        st.plotly_chart(fig_ff, use_container_width=True)
        
        # =================================================================
        # SCATTER PLOT: FREE FLOAT vs TURNOVER
        # =================================================================
        st.subheader("🔄 Korelasi Free Float vs Turnover")
        
        fig_scatter = px.scatter(
            df_latest,
            x='Free Float',
            y='Turnover_Ratio',
            color='Sector',
            size='Volume',
            hover_name='Stock Code',
            title="Semakin ke kanan atas = Semakin likuid",
            labels={'Free Float': 'Free Float (%)', 'Turnover_Ratio': 'Turnover Ratio (%)'}
        )
        fig_scatter.update_layout(height=500)
        st.plotly_chart(fig_scatter, use_container_width=True)

# =============================================================================
# TAB 6: FOREIGN FLOW INTELLIGENCE
# =============================================================================
with tab6:
    st.header("🌏 Foreign Flow Intelligence")
    st.caption("Analisis mendalam arus modal asing")
    
    if 'Net Foreign Flow' not in df_harian.columns:
        st.warning("Kolom Foreign Flow tidak tersedia")
    else:
        df_foreign = df_harian[
            (df_harian['Last Trading Date'] >= pd.to_datetime(start_date)) &
            (df_harian['Last Trading Date'] <= pd.to_datetime(end_date))
        ].copy()
        
        if selected_sectors:
            df_foreign = df_foreign[df_foreign['Sector'].isin(selected_sectors)]
        
        # Aggregate per saham
        df_foreign_agg = df_foreign.groupby('Stock Code').agg({
            'Net Foreign Flow': 'sum',
            'Foreign Buy': 'sum',
            'Foreign Sell': 'sum',
            'Volume': 'sum',
            'Value': 'sum',
            'Close': 'last',
            'Sector': 'first',
            'Free Float': 'first'
        }).reset_index()
        
        # Hitung rasio
        df_foreign_agg['Foreign_Ratio'] = (df_foreign_agg['Net Foreign Flow'] / df_foreign_agg['Value'].abs()) * 100
        df_foreign_agg['Buy_Sell_Ratio'] = df_foreign_agg['Foreign Buy'] / df_foreign_agg['Foreign Sell'].replace(0, 1)
        
        # Top 10 Net Buy
        st.subheader("🟢 Top 10 Net Foreign Buy")
        top_buy = df_foreign_agg.nlargest(10, 'Net Foreign Flow')
        fig_buy = px.bar(
            top_buy,
            x='Stock Code',
            y='Net Foreign Flow',
            color='Sector',
            title="Saham dengan Akumulasi Asing Terbesar",
            labels={'Net Foreign Flow': 'Net Foreign Flow (Rp)'}
        )
        fig_buy.update_layout(height=400)
        fig_buy.update_yaxes(tickformat=",.0f")
        st.plotly_chart(fig_buy, use_container_width=True)
        
        # Top 10 Net Sell
        st.subheader("🔴 Top 10 Net Foreign Sell")
        top_sell = df_foreign_agg.nsmallest(10, 'Net Foreign Flow')
        fig_sell = px.bar(
            top_sell,
            x='Stock Code',
            y='Net Foreign Flow',
            color='Sector',
            title="Saham dengan Distribusi Asing Terbesar",
            labels={'Net Foreign Flow': 'Net Foreign Flow (Rp)'}
        )
        fig_sell.update_layout(height=400)
        fig_sell.update_yaxes(tickformat=",.0f")
        st.plotly_chart(fig_sell, use_container_width=True)
        
        # Foreign Flow vs Free Float
        st.subheader("📊 Korelasi Foreign Flow vs Free Float")
        fig_ff_foreign = px.scatter(
            df_foreign_agg,
            x='Free Float',
            y='Foreign_Ratio',
            color='Sector',
            size='Value',
            hover_name='Stock Code',
            title="Asing lebih suka free float besar?",
            labels={'Free Float': 'Free Float (%)', 'Foreign_Ratio': 'Foreign Flow % terhadap Value'}
        )
        fig_ff_foreign.update_layout(height=500)
        st.plotly_chart(fig_ff_foreign, use_container_width=True)
        
        # Tabel detail
        st.subheader("📋 Detail Foreign Flow per Saham")
        
        min_foreign_val = st.number_input("Min Nilai Foreign Flow (Rp Miliar)", 0.0, 1000.0, 10.0) * 1e9
        df_detail_foreign = df_foreign_agg[abs(df_foreign_agg['Net Foreign Flow']) >= min_foreign_val].sort_values('Net Foreign Flow', ascending=False)
        
        if not df_detail_foreign.empty:
            df_display = df_detail_foreign[['Stock Code', 'Sector', 'Close', 'Net Foreign Flow', 
                                           'Foreign_Ratio', 'Free Float', 'Value']].copy()
            df_display['Close'] = df_display['Close'].apply(format_rupiah)
            df_display['Net Foreign Flow'] = df_display['Net Foreign Flow'].apply(format_rupiah)
            df_display['Value'] = df_display['Value'].apply(format_rupiah)
            df_display['Foreign_Ratio'] = df_display['Foreign_Ratio'].apply(lambda x: f"{x:.2f}%")
            df_display['Free Float'] = df_display['Free Float'].apply(lambda x: f"{x:.1f}%")
            
            st.dataframe(df_display, use_container_width=True, hide_index=True)

# =============================================================================
# TAB 7: FREE FLOAT VS KEPEMILIKAN 5%
# =============================================================================
with tab7:
    st.header("🔄 Korelasi Free Float vs Kepemilikan 5%")
    st.caption("Analisis struktur kepemilikan dan dampaknya pada pergerakan saham")
    
    if df_master.empty:
        st.warning("Data kepemilikan 5% tidak tersedia")
    else:
        # =====================================================================
        # 1. HITUNG KEPEMILIKAN 5% AGGREGATE PER SAHAM
        # =====================================================================
        # Gunakan data terakhir untuk setiap saham
        df_master_latest = df_master.sort_values('Tanggal_Data').groupby('Kode Efek').last().reset_index()
        
        df_own_agg = df_master_latest.groupby('Kode Efek').agg({
            'Jumlah Saham (Curr)': 'sum',
            'UBO': 'nunique',
        }).reset_index()
        
        df_own_agg.columns = ['Kode Efek', 'Total_Kepemilikan_5persen', 'Jumlah_UBO']
        
        # Tambah flag UBO
        if 'Is_Adaro' in df_master_latest.columns:
            adaro_count = df_master_latest[df_master_latest['Is_Adaro'] == True].groupby('Kode Efek').size().reset_index(name='Adaro_Count')
            df_own_agg = df_own_agg.merge(adaro_count, on='Kode Efek', how='left')
        
        if 'Is_LKH' in df_master_latest.columns:
            lkh_count = df_master_latest[df_master_latest['Is_LKH'] == True].groupby('Kode Efek').size().reset_index(name='LKH_Count')
            df_own_agg = df_own_agg.merge(lkh_count, on='Kode Efek', how='left')
        
        if 'Is_Saratoga' in df_master_latest.columns:
            saratoga_count = df_master_latest[df_master_latest['Is_Saratoga'] == True].groupby('Kode Efek').size().reset_index(name='Saratoga_Count')
            df_own_agg = df_own_agg.merge(saratoga_count, on='Kode Efek', how='left')
        
        # =====================================================================
        # 2. MERGE DENGAN DATA HARGA
        # =====================================================================
        df_latest = df_harian.sort_values('Last Trading Date').groupby('Stock Code').last().reset_index()
        
        df_analysis = pd.merge(
            df_latest[['Stock Code', 'Company Name', 'Sector', 'Listed Shares', 
                      'Tradeble Shares', 'Free Float', 'Close', 'Volume']],
            df_own_agg,
            left_on='Stock Code',
            right_on='Kode Efek',
            how='inner'
        )
        
        if df_analysis.empty:
            st.warning("Tidak ada data yang cocok antara kepemilikan 5% dan harga")
            st.stop()
        
        # =====================================================================
        # 3. HITUNG METRIK PENTING
        # =====================================================================
        df_analysis['Kepemilikan_Persen'] = (df_analysis['Total_Kepemilikan_5persen'] / df_analysis['Listed Shares']) * 100
        df_analysis['Sisa_Tradeble'] = df_analysis['Tradeble Shares'] - df_analysis['Total_Kepemilikan_5persen']
        df_analysis['Sisa_Tradeble_Persen'] = (df_analysis['Sisa_Tradeble'] / df_analysis['Tradeble Shares']) * 100
        df_analysis['Volume_vs_Tradeble'] = (df_analysis['Volume'] / df_analysis['Tradeble Shares']) * 100
        
        # Handle infinite/NaN
        df_analysis = df_analysis.replace([np.inf, -np.inf], np.nan)
        df_analysis = df_analysis.dropna(subset=['Kepemilikan_Persen', 'Free Float'])
        
        # =====================================================================
        # 4. KLASIFIKASI STRUKTUR KEPEMILIKAN
        # =====================================================================
        conditions = [
            (df_analysis['Kepemilikan_Persen'] > 80),
            (df_analysis['Kepemilikan_Persen'] > 60),
            (df_analysis['Kepemilikan_Persen'] > 40),
            (df_analysis['Kepemilikan_Persen'] <= 40)
        ]
        choices = ['Sangat Terkonsentrasi (>80%)', 'Terkonsentrasi (60-80%)', 
                   'Moderat (40-60%)', 'Tersebar (<40%)']
        df_analysis['Struktur_Kepemilikan'] = np.select(conditions, choices, default='Tidak Diketahui')
        
        # =====================================================================
        # 5. FILTER PARAMETER
        # =====================================================================
        st.subheader("⚙️ Filter Analisis")
        
        col_a1, col_a2, col_a3 = st.columns(3)
        with col_a1:
            min_kepemilikan = st.slider("Min Kepemilikan 5% (%)", 0, 100, 50, key="kepemilikan")
        with col_a2:
            max_free_float = st.slider("Max Free Float (%)", 0, 100, 30, key="ff_max")  # Max karena kita cari yang kecil
        with col_a3:
            min_volume_ratio = st.slider("Min Volume/Tradeble (%)", 0, 100, 20, key="vol_ratio")
        
        df_filtered = df_analysis[
            (df_analysis['Kepemilikan_Persen'] >= min_kepemilikan) &
            (df_analysis['Free Float'] <= max_free_float) &
            (df_analysis['Volume_vs_Tradeble'] >= min_volume_ratio)
        ].sort_values('Volume_vs_Tradeble', ascending=False)
        
        # =====================================================================
        # 6. TAMPILKAN HASIL
        # =====================================================================
        st.subheader(f"🎯 {len(df_filtered)} Saham dengan Potensi Akumulasi Kuat")
        st.caption("Kepemilikan 5% besar, Free Float kecil, tapi volume besar = indikasi transaksi antar bandar")
        
        if not df_filtered.empty:
            # PERINGATAN untuk yang ekstrem
            extreme_stocks = df_filtered[
                (df_filtered['Kepemilikan_Persen'] > 85) & 
                (df_filtered['Free Float'] < 10) &
                (df_filtered['Volume_vs_Tradeble'] > 50)
            ]
            
            if not extreme_stocks.empty:
                st.error("🚨 **PERINGATAN EKSTREM: Saham dengan Potensi Manipulasi Tinggi!**")
                st.error("Kepemilikan >85%, Free Float <10%, Volume >50% Tradeble Shares")
                
                extreme_display = extreme_stocks[['Stock Code', 'Company Name', 'Kepemilikan_Persen', 
                                                  'Free Float', 'Volume_vs_Tradeble']].copy()
                extreme_display['Kepemilikan_Persen'] = extreme_display['Kepemilikan_Persen'].apply(lambda x: f"{x:.1f}%")
                extreme_display['Free Float'] = extreme_display['Free Float'].apply(lambda x: f"{x:.1f}%")
                extreme_display['Volume_vs_Tradeble'] = extreme_display['Volume_vs_Tradeble'].apply(lambda x: f"{x:.1f}%")
                st.dataframe(extreme_display, use_container_width=True, hide_index=True)
            
            # Format display
            df_display = df_filtered[[
                'Stock Code', 'Company Name', 'Sector', 'Close',
                'Kepemilikan_Persen', 'Free Float', 'Volume_vs_Tradeble',
                'Jumlah_UBO', 'Struktur_Kepemilikan'
            ]].copy()
            
            df_display['Close'] = df_display['Close'].apply(format_rupiah)
            df_display['Kepemilikan_Persen'] = df_display['Kepemilikan_Persen'].apply(lambda x: f"{x:.1f}%")
            df_display['Free Float'] = df_display['Free Float'].apply(lambda x: f"{x:.1f}%")
            df_display['Volume_vs_Tradeble'] = df_display['Volume_vs_Tradeble'].apply(lambda x: f"{x:.1f}%")
            
            st.dataframe(df_display, use_container_width=True, hide_index=True)
            
            # =================================================================
            # VISUALISASI 1: FREE FLOAT VS KEPEMILIKAN
            # =================================================================
            st.subheader("📊 Peta Struktur Kepemilikan")
            
            fig1 = px.scatter(
                df_analysis,
                x='Free Float',
                y='Kepemilikan_Persen',
                color='Struktur_Kepemilikan',
                size='Volume_vs_Tradeble',
                hover_name='Stock Code',
                text='Stock Code',
                title="Kuadran: Kanan Bawah = Free Float Besar, Kepemilikan Kecil | Kiri Atas = Free Float Kecil, Kepemilikan Besar",
                labels={'Free Float': 'Free Float (%)', 'Kepemilikan_Persen': 'Kepemilikan 5% (%)'}
            )
            fig1.update_traces(textposition='top center')
            fig1.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5)
            fig1.add_vline(x=30, line_dash="dash", line_color="gray", opacity=0.5)
            fig1.update_layout(height=600)
            st.plotly_chart(fig1, use_container_width=True)
            
            # =================================================================
            # VISUALISASI 2: TOP 10 VOLUME vs TRADEBLE
            # =================================================================
            st.subheader("🔥 Top 10 Saham dengan Volume Tertinggi vs Tradeble Shares")
            
            top10 = df_filtered.nlargest(10, 'Volume_vs_Tradeble')
            fig2 = px.bar(
                top10,
                x='Stock Code',
                y='Volume_vs_Tradeble',
                color='Sector',
                text='Volume_vs_Tradeble',
                title="Volume hari ini vs Total Tradeble Shares (%)",
                labels={'Volume_vs_Tradeble': 'Volume / Tradeble (%)'}
            )
            fig2.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
            
            # =================================================================
            # ANALISIS PER UBO
            # =================================================================
            st.subheader("👥 Detail Kepemilikan per UBO")
            
            selected_stock = st.selectbox(
                "Pilih saham untuk detail kepemilikan UBO:",
                df_filtered['Stock Code'].tolist(),
                key="ubo_detail"
            )
            
            if selected_stock:
                # Data kepemilikan dari master (data terbaru)
                df_ubo_detail = df_master[
                    (df_master['Kode Efek'] == selected_stock) &
                    (df_master['Tanggal_Data'] == df_master[df_master['Kode Efek'] == selected_stock]['Tanggal_Data'].max())
                ].copy()
                
                if not df_ubo_detail.empty:
                    df_ubo_detail = df_ubo_detail.groupby('UBO').agg({
                        'Jumlah Saham (Curr)': 'sum',
                        'Perubahan_Saham': 'sum',
                        'Rekening_Bersih': 'nunique'
                    }).reset_index()
                    
                    df_ubo_detail.columns = ['UBO', 'Jumlah_Saham', 'Perubahan_Hari_Ini', 'Jumlah_Rekening']
                    df_ubo_detail['Persentase'] = (df_ubo_detail['Jumlah_Saham'] / 
                                                   df_analysis[df_analysis['Stock Code']==selected_stock]['Listed Shares'].iloc[0]) * 100
                    
                    df_ubo_detail = df_ubo_detail.sort_values('Jumlah_Saham', ascending=False)
                    
                    # Format display
                    df_ubo_detail['Jumlah_Saham'] = df_ubo_detail['Jumlah_Saham'].apply(format_lembar)
                    df_ubo_detail['Perubahan_Hari_Ini'] = df_ubo_detail['Perubahan_Hari_Ini'].apply(format_lembar)
                    df_ubo_detail['Persentase'] = df_ubo_detail['Persentase'].apply(lambda x: f"{x:.2f}%")
                    
                    st.dataframe(df_ubo_detail, use_container_width=True, hide_index=True)
                    
                    # Pie chart komposisi kepemilikan
                    fig_pie = px.pie(
                        df_ubo_detail,
                        values='Jumlah_Saham',
                        names='UBO',
                        title=f"Komposisi Kepemilikan 5% - {selected_stock}"
                    )
                    fig_pie.update_layout(height=500)
                    st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Tidak ada saham dengan kriteria tersebut. Coba turunkan threshold.")

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    st.caption("🐋 Bandar Eye IDX - Institutional Grade")
with col_f2:
    st.caption(f"Data Update: {datetime.now().strftime('%d-%m-%Y %H:%M')}")
with col_f3:
    st.caption(f"Total Records: {len(df_master):,} (5% ownership)")

# Debug info di footer (hidden by default)
with st.expander("🔧 Technical Info"):
    st.json({
        "data_harian": len(df_harian),
        "data_ksei": len(df_ksei),
        "data_master": len(df_master),
        "date_range": f"{min_date} to {max_date}",
        "sectors": len(sektor_list) if sektor_list else 0
    })
