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
# File ID untuk dataset yang sudah di-clean (format PARQUET recommended)
FILE_IDS = {
    'harian': '1t_wCljhepGBqZVrvleuZKldomQKop9DY',           # Kompilasi_Data_1Tahun.csv
    'ksei': '1eTUIC120SHTCzvBk77Q87w0X56F2HkWz',             # KSEI_Shareholder_Processed.csv
    'master_5_parquet': '1tb1umgJc1giaKYyMNuQWhH7R8cH75s2X', # GANTI DENGAN FILE ID PARQUET ANDA!
    'master_5_light': '10CS5QJU5MHafIpanEH9XU6SpCEOVd-pb'    # GANTI DENGAN FILE ID CSV LIGHT ANDA!
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
        
        # Simpan ke BytesIO dan baca dengan pd.read_parquet
        buffer = BytesIO(response.content)
        df = pd.read_parquet(buffer)
        return df
    except Exception as e:
        st.warning(f"Gagal load Parquet: {e}. Mencoba format CSV...")
        return None

# =============================================================================
# CACHE DATA LOADING (OPTIMUM UNTUK STREAMLIT CLOUD)
# =============================================================================
@st.cache_data(ttl=3600, show_spinner="Loading data harian...")
def load_harian():
    """Load data harian (Kompilasi_Data_1Tahun)"""
    try:
        df = load_csv_from_gdrive(FILE_IDS['harian'])
        
        # Parsing tanggal
        df['Last Trading Date'] = pd.to_datetime(df['Last Trading Date'], errors='coerce')
        df = df.dropna(subset=['Last Trading Date'])
        
        # Konversi numerik
        numeric_cols = ['Close', 'Volume', 'Value', 'Foreign Buy', 'Foreign Sell', 
                        'Bid Volume', 'Offer Volume', 'Avg_Order_Volume', 'MA50_AOVol',
                        'Volume Spike (x)', 'Bid/Offer Imbalance', 'Change %']
        
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
        
        # Parsing tanggal
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        
        # Konversi numerik
        for col in df.columns:
            if 'Chg' in col or 'Vol' in col or 'Val' in col or col in ['Price', 'Avg_Price']:
                df[col] = pd.to_datetime(df[col], errors='coerce') if 'Date' in col else pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"❌ Gagal load data KSEI: {e}")
        return pd.DataFrame(columns=['Code', 'Date', 'Top_Buyer', 'Top_Seller'])

@st.cache_data(ttl=86400, show_spinner="Loading data kepemilikan 5% (CLEAN)...")
def load_master_5():
    """
    LOAD DATA MASTER 5% YANG SUDAH DI-CLEAN DI COLAB
    PRIORITAS: Parquet -> CSV Light -> Original + Clean On The Fly
    """
    
    # 1. COBA LOAD PARQUET (PALING CEPAT)
    if 'master_5_parquet' in FILE_IDS:
        df = load_parquet_from_gdrive(FILE_IDS['master_5_parquet'])
        if df is not None:
            # Pastikan kolom tanggal dalam format datetime
            if 'Tanggal_Data' in df.columns:
                df['Tanggal_Data'] = pd.to_datetime(df['Tanggal_Data'], errors='coerce')
                df = df.dropna(subset=['Tanggal_Data'])
            
            # Pastikan kolom numerik
            numeric_cols = ['Jumlah Saham (Prev)', 'Jumlah Saham (Curr)', 'Perubahan_Saham', 
                           'Close_Price', 'Estimasi_Nilai']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            st.success("✅ Load data 5% (Parquet) - SUPER CEPAT!")
            return df
    
    # 2. FALLBACK: LOAD CSV LIGHT
    if 'master_5_light' in FILE_IDS:
        try:
            df = load_csv_from_gdrive(FILE_IDS['master_5_light'])
            df['Tanggal_Data'] = pd.to_datetime(df['Tanggal_Data'], errors='coerce')
            df = df.dropna(subset=['Tanggal_Data'])
            st.success("✅ Load data 5% (CSV Light)")
            return df
        except:
            pass
    
    # 3. FALLBACK TERAKHIR: LOAD ORIGINAL + CLEAN SEDERHANA
    st.warning("⚠️ Load data original. Pastikan Anda sudah menjalankan script cleaning di Colab!")
    try:
        df = load_csv_from_gdrive(FILE_IDS['master_5'])
        df['Tanggal_Data'] = pd.to_datetime(df['Tanggal_Data'], errors='coerce')
        df = df.dropna(subset=['Tanggal_Data'])
        
        # Hitung kolom dasar
        df['Perubahan_Saham'] = df['Jumlah Saham (Curr)'] - df['Jumlah Saham (Prev)']
        df['Aksi'] = 'Tahan'
        df.loc[df['Perubahan_Saham'] > 0, 'Aksi'] = 'Beli'
        df.loc[df['Perubahan_Saham'] < 0, 'Aksi'] = 'Jual'
        df['Estimasi_Nilai'] = df['Perubahan_Saham'] * df['Close_Price']
        
        return df
    except Exception as e:
        st.error(f"❌ Gagal load data master 5%: {e}")
        return pd.DataFrame(columns=['Kode Efek', 'Tanggal_Data', 'UBO'])

# =============================================================================
# FORMATTER ANGKA (SUPAYA ENAK DIBACA)
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
st.sidebar.caption("v2.0 - Clean Architecture | 20 Tahun Cycle Experience")

# Load semua data
with st.spinner('Memuat data harga...'):
    df_harian = load_harian()
with st.spinner('Memuat data KSEI...'):
    df_ksei = load_ksei()
with st.spinner('Memuat data kepemilikan 5% (Clean)...'):
    df_master = load_master_5()

# Cek apakah data berhasil di-load
if df_harian.empty:
    st.error("⚠️ Data harian tidak tersedia. Dashboard tidak dapat berjalan.")
    st.stop()

if df_master.empty:
    st.warning("⚠️ Data kepemilikan 5% tidak tersedia. Tab 3 tidak akan berfungsi penuh.")

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
# MAIN APP: 4 TAB
# =============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Momentum Bandar", 
    "🏦 KSEI Big Money", 
    "🕵️ Akumulasi Awal (UBO Clustered)", 
    "📊 Watchlist & Konvergensi"
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
    
    df_anomaly = df_anomaly.sort_values('Potensi', ascending=False)
    
    st.subheader(f"🎯 {len(df_anomaly)} Saham dengan Aktivitas Bandar Terdeteksi")
    
    if not df_anomaly.empty:
        # Kolom display
        display_cols = ['Stock Code', 'Last Trading Date']
        optional_cols = ['Close', 'Change %', 'Volume Spike (x)', 'Avg_Order_Volume', 
                        'MA50_AOVol', 'Bid/Offer Imbalance', 'Net Foreign Flow', 
                        'Final Signal', 'Potensi']
        
        for col in optional_cols:
            if col in df_anomaly.columns:
                display_cols.append(col)
        
        st.dataframe(
            df_anomaly[display_cols].head(100),
            use_container_width=True,
            hide_index=True,
            column_config={
                'Close': st.column_config.NumberColumn(format="Rp %d"),
                'Change %': st.column_config.NumberColumn(format="%.2f%%"),
                'Volume Spike (x)': st.column_config.NumberColumn(format="%.1fx"),
                'Avg_Order_Volume': st.column_config.NumberColumn(format="%.0f"),
                'Bid/Offer Imbalance': st.column_config.NumberColumn(format="%.2f"),
                'Net Foreign Flow': st.column_config.NumberColumn(format="Rp %d"),
                'Potensi': st.column_config.NumberColumn(format="%.2f")
            }
        )
        
        # Scatter plot
        st.subheader("📊 Volume Spike vs AOVol Ratio")
        
        if ('Volume Spike (x)' in df_anomaly.columns and 
            'Avg_Order_Volume' in df_anomaly.columns and 
            'MA50_AOVol' in df_anomaly.columns and
            'Final Signal' in df_anomaly.columns):
            
            df_scatter = df_anomaly.head(50).copy()
            df_scatter['AOVol_Ratio'] = df_scatter['Avg_Order_Volume'] / df_scatter['MA50_AOVol'].replace(0, np.nan)
            df_scatter = df_scatter.dropna(subset=['Volume Spike (x)', 'AOVol_Ratio', 'Final Signal'])
            
            if not df_scatter.empty:
                fig = px.scatter(
                    df_scatter,
                    x='Volume Spike (x)',
                    y='AOVol_Ratio',
                    color='Final Signal',
                    size='Potensi' if 'Potensi' in df_scatter.columns else None,
                    hover_data=['Stock Code', 'Close'] if 'Close' in df_scatter.columns else ['Stock Code'],
                    title="Semakin ke kanan atas = Semakin kuat akumulasi"
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
                st.dataframe(
                    top_buyers,
                    column_config={
                        'Top_Buyer_Vol': st.column_config.NumberColumn(format="%d lot"),
                        'Top_Buyer_Val': st.column_config.NumberColumn(format="Rp %d")
                    },
                    use_container_width=True,
                    hide_index=True
                )
        
        with col2:
            st.subheader("🔴 Top Seller (Volume)")
            if 'Top_Seller_Vol' in df_ksei_filtered.columns:
                top_sellers = df_ksei_filtered.nsmallest(top_n, 'Top_Seller_Vol')[['Code', 'Date', 'Top_Seller', 'Top_Seller_Vol', 'Top_Seller_Val']]
                st.dataframe(
                    top_sellers,
                    column_config={
                        'Top_Seller_Vol': st.column_config.NumberColumn(format="%d lot"),
                        'Top_Seller_Val': st.column_config.NumberColumn(format="Rp %d")
                    },
                    use_container_width=True,
                    hide_index=True
                )
        
        st.subheader("🔍 Detail Saham")
        kode_saham = st.text_input("Masukkan Kode Saham (contoh: AADI, BBCA)", "").upper()
        
        if kode_saham:
            df_detail = df_ksei_filtered[df_ksei_filtered['Code'] == kode_saham].sort_values('Date', ascending=False)
            if not df_detail.empty:
                cols_detail = ['Date', 'Price', 'Total_Local', 'Total_Foreign', 
                              'Top_Buyer', 'Top_Buyer_Vol', 'Top_Seller', 'Top_Seller_Vol']
                cols_detail = [c for c in cols_detail if c in df_detail.columns]
                st.dataframe(
                    df_detail[cols_detail].head(12),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.warning(f"Data tidak ditemukan untuk {kode_saham}")

# =============================================================================
# TAB 3: AKUMULASI AWAL DARI DATA 5% (UBO CLUSTERED - CLEAN VERSION)
# =============================================================================
with tab3:
    st.header("🕵️ DETEKSI AKUMULASI AWAL (UBO Clustered)")
    st.markdown("""
    > **Bandarmology Intelligence**: Data sudah di-cluster per **Ultimate Beneficial Owner (UBO)**.
    > *PT ADARO STRATEGIC INVESTMENTS*, *U20B2S3 A90G4...*, *ADARO STRATEGIC INVESTMENTS PT* → **1 entitas: ADARO** ✅
    """)
    
    if df_master.empty:
        st.warning("⚠️ Data kepemilikan 5% tidak tersedia. Jalankan script cleaning di Colab terlebih dahulu.")
        st.stop()
    
    # Pastikan kolom UBO ada
    if 'UBO' not in df_master.columns:
        st.warning("⚠️ Kolom 'UBO' tidak ditemukan. Pastikan Anda sudah menjalankan script cleaning di Colab.")
        st.info("Mencoba menggunakan kolom 'Nama Pemegang Saham' sebagai fallback...")
        df_master['UBO'] = df_master['Nama Pemegang Saham']
        df_master['Rekening_Bersih'] = df_master['Nama Rekening Efek']
    
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
    
    # =========================================================================
    # PARAMETER DETEKSI AKUMULASI AWAL
    # =========================================================================
    st.subheader("⚙️ Parameter Akumulasi Awal")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        min_beli = st.number_input("Minimal Beli (lembar)", min_value=1000, value=100000, step=10000, 
                                   help="Filter pembelian minimal", format="%d")
    with col2:
        lookback_days = st.number_input("Lookback (hari)", min_value=7, max_value=90, value=30, 
                                       help="Dalam X hari terakhir")
    with col3:
        min_freq = st.number_input("Minimal Frekuensi", min_value=1, max_value=20, value=2,
                                  help="Minimal berapa kali transaksi")
    
    cutoff_date = pd.to_datetime(end_date) - timedelta(days=lookback_days)
    
    # =========================================================================
    # DETEKSI AKUMULASI AWAL PER UBO - VERSI FIX (ANTI ERROR)
    # =========================================================================
    df_beli = df_master_filtered[
        (df_master_filtered['Aksi'] == 'Beli') &
        (df_master_filtered['Perubahan_Saham'] >= min_beli) &
        (df_master_filtered['Tanggal_Data'] >= cutoff_date)
    ].copy()
    
    if not df_beli.empty:
        # --- METHOD 1: AGGRESSI DENGAN NAMA KOLOM EKSPLISIT (LEBIH AMAN) ---
        
        # 1. Groupby dan hitung agregasi SATU PER SATU
        df_akumulasi = df_beli.groupby(['UBO', 'Kode Efek']).agg({
            'Perubahan_Saham': 'sum',
            'Estimasi_Nilai': 'sum',
            'Tanggal_Data': ['first', 'last', 'count']
        }).reset_index()
        
        # 2. Flatten column names dengan cara AMAN
        # Kolom hasil groupby akan seperti: ('Perubahan_Saham', 'sum'), dll
        df_akumulasi.columns = [
            'UBO', 'Kode Efek',
            'Total_Beli_Lembar', 'Total_Nilai_Rp',
            'Tgl_Pertama', 'Tgl_Terakhir', 'Frekuensi_Transaksi'
        ]
        
        # 3. Tambahkan kolom tambahan secara TERPISAH (jika ada)
        if 'Rekening_Bersih' in df_beli.columns:
            rekening_list = df_beli.groupby(['UBO', 'Kode Efek'])['Rekening_Bersih'].apply(lambda x: list(x.unique())).reset_index()
            rekening_list.columns = ['UBO', 'Kode Efek', 'Daftar_Rekening']
            df_akumulasi = df_akumulasi.merge(rekening_list, on=['UBO', 'Kode Efek'], how='left')
            df_akumulasi['Jumlah_Rekening'] = df_akumulasi['Daftar_Rekening'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        
        if 'Pemegang_Bersih' in df_beli.columns:
            pemegang_first = df_beli.groupby(['UBO', 'Kode Efek'])['Pemegang_Bersih'].first().reset_index()
            pemegang_first.columns = ['UBO', 'Kode Efek', 'Pemegang_Saham']
            df_akumulasi = df_akumulasi.merge(pemegang_first, on=['UBO', 'Kode Efek'], how='left')
        
        if 'Is_Adaro' in df_beli.columns:
            adaro_flag = df_beli.groupby(['UBO', 'Kode Efek'])['Is_Adaro'].first().reset_index()
            adaro_flag.columns = ['UBO', 'Kode Efek', 'Is_Adaro']
            df_akumulasi = df_akumulasi.merge(adaro_flag, on=['UBO', 'Kode Efek'], how='left')
        
        if 'Is_LKH' in df_beli.columns:
            lkh_flag = df_beli.groupby(['UBO', 'Kode Efek'])['Is_LKH'].first().reset_index()
            lkh_flag.columns = ['UBO', 'Kode Efek', 'Is_LKH']
            df_akumulasi = df_akumulasi.merge(lkh_flag, on=['UBO', 'Kode Efek'], how='left')
        
        if 'Is_Saratoga' in df_beli.columns:
            saratoga_flag = df_beli.groupby(['UBO', 'Kode Efek'])['Is_Saratoga'].first().reset_index()
            saratoga_flag.columns = ['UBO', 'Kode Efek', 'Is_Saratoga']
            df_akumulasi = df_akumulasi.merge(saratoga_flag, on=['UBO', 'Kode Efek'], how='left')
        
        # 4. Skor Akumulasi Awal
        df_akumulasi['Skor_Akumulasi'] = (
            np.log1p(df_akumulasi['Total_Beli_Lembar']) * 0.5 +
            np.log1p(df_akumulasi['Frekuensi_Transaksi']) * 0.3 +
            (1 - (df_akumulasi['Tgl_Terakhir'] - df_akumulasi['Tgl_Pertama']).dt.days / lookback_days) * 0.2
        )
        
        # 5. Filter frekuensi minimal
        df_akumulasi = df_akumulasi[df_akumulasi['Frekuensi_Transaksi'] >= min_freq]
        df_akumulasi = df_akumulasi.sort_values('Skor_Akumulasi', ascending=False)
        
        # 6. Merge dengan sektor
        if 'Sector' not in df_akumulasi.columns and 'Kode Efek' in df_akumulasi.columns:
            sector_map = df_harian[['Stock Code', 'Sector']].drop_duplicates('Stock Code')
            df_akumulasi = df_akumulasi.merge(sector_map, left_on='Kode Efek', right_on='Stock Code', how='left')
        
        # =========================================================================
        # TAMPILKAN HASIL AKUMULASI AWAL
        # =========================================================================
        st.subheader(f"🎯 {len(df_akumulasi)} Indikasi Akumulasi Awal (per UBO)")
        st.caption("Semakin tinggi Skor = Semakin agresif akumulasi dalam waktu singkat")
        
        if not df_akumulasi.empty:
            # Display format
            df_display = df_akumulasi.copy()
            df_display['Total_Beli_Lembar_Display'] = df_display['Total_Beli_Lembar'].apply(format_lembar)
            df_display['Total_Nilai_Rp_Display'] = df_display['Total_Nilai_Rp'].apply(format_rupiah)
            df_display['Skor_Display'] = df_display['Skor_Akumulasi'].apply(lambda x: f"{x:.2f}")
            
            # Kolom display
            display_columns = ['UBO', 'Kode Efek']
            if 'Sector' in df_display.columns:
                display_columns.append('Sector')
            display_columns.extend(['Total_Beli_Lembar_Display', 'Total_Nilai_Rp_Display', 
                                   'Frekuensi_Transaksi', 'Tgl_Pertama', 'Tgl_Terakhir', 'Skor_Display'])
            
            if 'Jumlah_Rekening' in df_display.columns:
                display_columns.insert(6, 'Jumlah_Rekening')
            
            # Filter cepat
            col_filter1, col_filter2, col_filter3 = st.columns(3)
            with col_filter1:
                show_adaro = st.checkbox("🎯 Hanya Adaro Group", value=False)
            with col_filter2:
                show_lkh = st.checkbox("👑 Hanya Lo Kheng Hong", value=False)
            with col_filter3:
                show_saratoga = st.checkbox("📊 Hanya Saratoga", value=False)
            
            df_filtered_display = df_display.copy()
            if show_adaro and 'Is_Adaro' in df_display.columns:
                df_filtered_display = df_filtered_display[df_filtered_display['Is_Adaro'] == True]
            if show_lkh and 'Is_LKH' in df_display.columns:
                df_filtered_display = df_filtered_display[df_filtered_display['Is_LKH'] == True]
            if show_saratoga and 'Is_Saratoga' in df_display.columns:
                df_filtered_display = df_filtered_display[df_filtered_display['Is_Saratoga'] == True]
            
            st.dataframe(
                df_filtered_display[display_columns].head(50),
                column_config={
                    'UBO': st.column_config.TextColumn("Ultimate Beneficial Owner", width="large"),
                    'Total_Beli_Lembar_Display': st.column_config.TextColumn("Total Beli (Lembar)"),
                    'Total_Nilai_Rp_Display': st.column_config.TextColumn("Estimasi Nilai"),
                    'Frekuensi_Transaksi': st.column_config.NumberColumn("Frekuensi", format="%d"),
                    'Jumlah_Rekening': st.column_config.NumberColumn("Jml Rekening"),
                    'Tgl_Pertama': st.column_config.DateColumn(format="DD-MM-YY"),
                    'Tgl_Terakhir': st.column_config.DateColumn(format="DD-MM-YY"),
                    'Skor_Display': st.column_config.TextColumn("Skor")
                },
                use_container_width=True,
                hide_index=True
            )
            
            # =========================================================================
            # TOP 10 AKUMULATOR
            # =========================================================================
            st.subheader("💰 Top 10 UBO Akumulator Terbesar (Nilai Rp)")
            top10 = df_akumulasi.nlargest(10, 'Total_Nilai_Rp').copy()
            top10['Label'] = top10['UBO'].apply(lambda x: x[:30] + '...' if len(x) > 30 else x)
            top10['Nilai_Display'] = top10['Total_Nilai_Rp'].apply(format_rupiah)
            
            fig = px.bar(
                top10,
                x='Total_Nilai_Rp',
                y='Label',
                color='Sector' if 'Sector' in top10.columns else None,
                orientation='h',
                title="Total Nilai Pembelian (Estimasi) per UBO",
                labels={'Total_Nilai_Rp': 'Nilai (Rp)', 'Label': 'Ultimate Beneficial Owner'},
                hover_data={'Total_Nilai_Rp': False, 'Nilai_Display': True, 'Kode Efek': True}
            )
            fig.update_layout(height=500)
            fig.update_xaxes(tickformat=",.0f")
            st.plotly_chart(fig, use_container_width=True)
            
            # =============================================================================
            # DEEP DIVE: MULTI-LINE CHART PER UBO + HARGA (SECONDARY AXIS)
            # =============================================================================
            st.subheader("📈 DEEP DIVE: Perbandingan Akumulasi per UBO + Harga")
            st.caption("**Multi-line chart**: Setiap Ultimate Beneficial Owner (UBO) adalah 1 garis. Garis merah putus-putus = Harga (secondary axis)")
            
            col_a, col_b, col_c = st.columns([2, 1, 1])
            
            with col_a:
                stock_options = sorted(df_master_filtered['Kode Efek'].unique())
                selected_stock_dd = st.selectbox(
                    "Pilih Kode Efek untuk Deep Dive",
                    stock_options,
                    index=0 if len(stock_options) > 0 else None
                )
            
            with col_b:
                ubo_filter = st.selectbox(
                    "Filter UBO",
                    ["Semua UBO", "Hanya Adaro Group", "Hanya LKH", "Hanya Saratoga", "Non-Nominee"],
                    index=0
                )
            
            with col_c:
                weekly_option = st.selectbox(
                    "Interval",
                    ["Weekly", "Bi-Weekly", "Monthly"],
                    index=0
                )
            
            # Frekuensi resample
            freq_map = {"Weekly": "W", "Bi-Weekly": "2W", "Monthly": "ME"}
            freq = freq_map[weekly_option]
            
            if selected_stock_dd:
                # ============= DATA KEPEMILIKAN =============
                df_all_ubo = df_master_filtered[
                    df_master_filtered['Kode Efek'] == selected_stock_dd
                ].copy()
                
                # ============= DATA HARGA =============
                df_harga = df_harian[
                    df_harian['Stock Code'] == selected_stock_dd
                ].sort_values('Last Trading Date').copy()
                
                if not df_all_ubo.empty and not df_harga.empty:
                    # Filter UBO
                    all_ubos = df_all_ubo['UBO'].unique()
                    
                    if ubo_filter == "Hanya Adaro Group":
                        all_ubos = [u for u in all_ubos if 'ADARO' in str(u).upper()]
                    elif ubo_filter == "Hanya LKH":
                        all_ubos = [u for u in all_ubos if 'LO KHENG HONG' in str(u).upper()]
                    elif ubo_filter == "Hanya Saratoga":
                        all_ubos = [u for u in all_ubos if 'SARATOGA' in str(u).upper()]
                    elif ubo_filter == "Non-Nominee":
                        all_ubos = [u for u in all_ubos if 'NOMINEE' not in str(u).upper()]
                    
                    if len(all_ubos) == 0:
                        st.warning("Tidak ada UBO dengan filter tersebut.")
                    else:
                        st.info(f"Menampilkan {len(all_ubos)} Ultimate Beneficial Owner")
                        
                        # ============= BUAT FIGURE DUAL AXIS =============
                        fig_deep = go.Figure()
                        
                        # --- SET 1: LINE CHART KEPEMILIKAN (Primary Axis - Kiri) ---
                        colors = px.colors.qualitative.Plotly + px.colors.qualitative.Alphabet * 5
                        
                        for idx, ubo in enumerate(all_ubos[:10]):  # Maks 10 UBO agar tidak terlalu padat
                            df_ubo = df_all_ubo[df_all_ubo['UBO'] == ubo].sort_values('Tanggal_Data')
                            
                            if len(df_ubo) >= 2:
                                # Resample ke weekly
                                df_ubo.set_index('Tanggal_Data', inplace=True)
                                df_ubo_weekly = df_ubo.resample(freq).last().dropna(subset=['Jumlah Saham (Curr)']).reset_index()
                                df_ubo.reset_index(inplace=True)
                                
                                label = ubo[:20] + '...' if len(ubo) > 20 else ubo
                                
                                fig_deep.add_trace(go.Scatter(
                                    x=df_ubo_weekly['Tanggal_Data'],
                                    y=df_ubo_weekly['Jumlah Saham (Curr)'],
                                    mode='lines+markers',
                                    name=label,
                                    line=dict(color=colors[idx % len(colors)], width=2.5),
                                    marker=dict(size=6),
                                    yaxis='y',  # Primary axis (kiri)
                                    hovertemplate='<b>%{x|%d-%m-%Y}</b><br>' +
                                                 f'<span style="color:{colors[idx % len(colors)]}">●</span> {ubo}<br>' +
                                                 'Kepemilikan: %{customdata}<extra></extra>',
                                    customdata=df_ubo_weekly['Jumlah Saham (Curr)'].apply(format_lembar)
                                ))
                        
                        # --- SET 2: LINE CHART HARGA (Secondary Axis - Kanan) ---
                        df_harga.set_index('Last Trading Date', inplace=True)
                        df_harga_weekly = df_harga.resample(freq).last().dropna(subset=['Close']).reset_index()
                        df_harga.reset_index(inplace=True)
                        
                        fig_deep.add_trace(go.Scatter(
                            x=df_harga_weekly['Last Trading Date'],
                            y=df_harga_weekly['Close'],
                            mode='lines+markers',
                            name='💲 HARGA CLOSE',
                            line=dict(color='#E74C3C', width=3, dash='dot'),
                            marker=dict(size=8, symbol='diamond', color='#E74C3C'),
                            yaxis='y2',  # Secondary axis (kanan)
                            hovertemplate='<b>%{x|%d-%m-%Y}</b><br>' +
                                         '<span style="color:#E74C3C">💲</span> Harga: %{customdata}<extra></extra>',
                            customdata=df_harga_weekly['Close'].apply(format_rupiah)
                        ))
                        
                        # --- SET 3: VOLUME SPIKE (Marker di Harga) ---
                        if 'Volume Spike (x)' in df_harga_weekly.columns:
                            df_harga_weekly['Volume Spike (x)'] = pd.to_numeric(df_harga_weekly['Volume Spike (x)'], errors='coerce')
                            spike = df_harga_weekly[df_harga_weekly['Volume Spike (x)'] > 1.5]
                            if not spike.empty:
                                fig_deep.add_trace(go.Scatter(
                                    x=spike['Last Trading Date'],
                                    y=spike['Close'],
                                    mode='markers',
                                    name='⚡ Volume Spike',
                                    marker=dict(color='#F39C12', size=12, symbol='star'),
                                    yaxis='y2',
                                    hovertemplate='<b>%{x|%d-%m-%Y}</b><br>' +
                                                 '<span style="color:#F39C12">⚡</span> Volume Spike: %{customdata:.1f}x<extra></extra>',
                                    customdata=spike['Volume Spike (x)']
                                ))
                        
                        # ============= UPDATE LAYOUT DUAL AXIS =============
                        fig_deep.update_layout(
                            title=f"<b>AKUMULASI vs HARGA</b> - {selected_stock_dd}",
                            height=600,
                            hovermode='x unified',
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=-0.35,
                                xanchor="center",
                                x=0.5,
                                font=dict(size=11),
                                itemsizing='constant'
                            ),
                            # Primary Axis (Kiri) - Kepemilikan
                            yaxis=dict(
                                title=dict(
                                    text="Jumlah Saham (Lembar)",
                                    font=dict(color='#2C3E50', size=13)
                                ),
                                tickformat=",.0f",
                                gridcolor='lightgray',
                                showgrid=True,
                                color='#2C3E50'
                            ),
                            # Secondary Axis (Kanan) - Harga
                            yaxis2=dict(
                                title=dict(
                                    text="Harga (Rp)",
                                    font=dict(color='#E74C3C', size=13)
                                ),
                                tickformat=",.0f",
                                overlaying='y',
                                side='right',
                                showgrid=False,
                                color='#E74C3C'
                            ),
                            xaxis=dict(
                                title="Tanggal",
                                tickformat="%d-%m-%Y",
                                gridcolor='lightgray'
                            ),
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            margin=dict(l=80, r=80, t=80, b=120)
                        )
                        
                        # Tambahkan range slider untuk zoom temporal
                        fig_deep.update_xaxes(rangeslider_visible=True, rangeslider_thickness=0.08)
                        
                        st.plotly_chart(fig_deep, use_container_width=True)
                        
                        # ============= METRIK RINGKASAN =============
                        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                        
                        with col_m1:
                            # Harga terakhir
                            last_price = df_harga['Close'].iloc[-1] if not df_harga.empty else 0
                            prev_price = df_harga['Close'].iloc[-2] if len(df_harga) > 1 else last_price
                            delta_price = ((last_price - prev_price) / prev_price * 100) if prev_price > 0 else 0
                            st.metric(
                                "Harga Terkini",
                                format_rupiah(last_price),
                                delta=f"{delta_price:.1f}%",
                                delta_color="normal"
                            )
                        
                        with col_m2:
                            # Total UBO aktif
                            st.metric("UBO Aktif", len(all_ubos))
                        
                        with col_m3:
                            # Total volume spike (30 hari terakhir)
                            if 'Volume Spike (x)' in df_harga.columns:
                                spike_30d = df_harga[
                                    (df_harga['Last Trading Date'] >= pd.to_datetime(end_date) - timedelta(days=30)) &
                                    (df_harga['Volume Spike (x)'] > 1.5)
                                ].shape[0]
                                st.metric("Volume Spike (30d)", spike_30d)
                            else:
                                st.metric("Volume Spike (30d)", 0)
                        
                        with col_m4:
                            # Rata-rata harga 20 hari
                            ma20 = df_harga['Close'].tail(20).mean()
                            st.metric("MA20", format_rupiah(ma20))
                        
                        # ============= TABEL RINGKASAN PER UBO =============
                        with st.expander("📋 Lihat Ringkasan Aktivitas per Ultimate Beneficial Owner"):
                            summary_data = []
                            for ubo in all_ubos[:10]:  # Top 10 UBO
                                df_ubo_sum = df_all_ubo[df_all_ubo['UBO'] == ubo].sort_values('Tanggal_Data')
                                
                                if not df_ubo_sum.empty:
                                    first_date = df_ubo_sum['Tanggal_Data'].iloc[0]
                                    last_date = df_ubo_sum['Tanggal_Data'].iloc[-1]
                                    first_hold = df_ubo_sum['Jumlah Saham (Curr)'].iloc[0]
                                    last_hold = df_ubo_sum['Jumlah Saham (Curr)'].iloc[-1]
                                    change = last_hold - first_hold
                                    change_pct = (change / first_hold * 100) if first_hold > 0 else 0
                                    
                                    total_beli = df_ubo_sum[df_ubo_sum['Perubahan_Saham'] > 0]['Perubahan_Saham'].sum()
                                    total_jual = abs(df_ubo_sum[df_ubo_sum['Perubahan_Saham'] < 0]['Perubahan_Saham'].sum())
                                    frekuensi = len(df_ubo_sum)
                                    jml_rekening = df_ubo_sum['Rekening_Bersih'].nunique() if 'Rekening_Bersih' in df_ubo_sum.columns else 1
                                    
                                    summary_data.append({
                                        'UBO': ubo[:30] + '...' if len(ubo) > 30 else ubo,
                                        'Jml Rekening': jml_rekening,
                                        'Periode': f"{first_date.strftime('%d/%m/%y')} - {last_date.strftime('%d/%m/%y')}",
                                        'Kepemilikan Awal': format_lembar(first_hold),
                                        'Kepemilikan Akhir': format_lembar(last_hold),
                                        'Perubahan': format_lembar(change),
                                        'Δ %': f"{change_pct:+.1f}%",
                                        'Total Beli': format_lembar(total_beli),
                                        'Total Jual': format_lembar(total_jual),
                                        'Frekuensi': frekuensi
                                    })
                            
                            df_summary = pd.DataFrame(summary_data)
                            st.dataframe(df_summary, use_container_width=True, hide_index=True)
                else:
                    st.warning(f"Tidak ada data kepemilikan atau harga untuk {selected_stock_dd}")

# =============================================================================
# TAB 4: SCANNER SINYAL & DIVERGENSI (FIX FORMAT ANGKA)
# =============================================================================
with tab4:
    st.header("⚡ Scanner Sinyal & Divergensi")
    st.caption("Deteksi anomali antara Pergerakan Harga vs Pergerakan Barang (5% Owner & Foreign)")

    # Pilihan Sub-Menu
    mode_scan = st.radio(
        "Pilih Mode Analisa:",
        ["💎 Hunter: Divergensi (Harga Turun, Akumulasi Naik)", 
         "🗺️ Map: Foreign vs Local Flow", 
         "📋 Watchlist Personal"],
        horizontal=True
    )

    st.markdown("---")

    # -------------------------------------------------------------------------
    # MODE 1: DIVERGENCE HUNTER
    # -------------------------------------------------------------------------
    if mode_scan == "💎 Hunter: Divergensi (Harga Turun, Akumulasi Naik)":
        st.subheader("💎 Deteksi Saham 'Salah Harga'")
        st.info("Mencari saham yang harganya JATUH, tapi kepemilikan investor 5% (Big Player) malah BERTAMBAH.")
        
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            lookback_div = st.slider("Periode Pantauan (Hari)", 5, 60, 20)
        with col_p2:
            min_price_drop = st.slider("Minimal Harga TURUN (%)", 0, 50, 5)
        
        if st.button("🔍 Scan Divergensi Sekarang"):
            with st.spinner("Menganalisa korelasi Harga vs Akumulasi..."):
                # 1. Siapkan Rentang Tanggal
                end_date_div = df_harian['Last Trading Date'].max()
                start_date_div = end_date_div - timedelta(days=lookback_div)
                
                # 2. Hitung Perubahan Harga
                df_period = df_harian[
                    (df_harian['Last Trading Date'] >= start_date_div) & 
                    (df_harian['Last Trading Date'] <= end_date_div)
                ].copy()
                
                price_start = df_period.sort_values('Last Trading Date').groupby('Stock Code')['Close'].first()
                price_end = df_period.sort_values('Last Trading Date').groupby('Stock Code')['Close'].last()
                
                df_div = pd.DataFrame({'Price_Start': price_start, 'Price_End': price_end})
                df_div['Price_Chg_Pct'] = (df_div['Price_End'] - df_div['Price_Start']) / df_div['Price_Start'] * 100
                
                df_div = df_div[df_div['Price_Chg_Pct'] <= -min_price_drop]
                
                # 3. Hitung Perubahan Kepemilikan 5%
                if not df_master.empty:
                    df_master_div = df_master[
                        (df_master['Tanggal_Data'] >= start_date_div) & 
                        (df_master['Tanggal_Data'] <= end_date_div)
                    ].copy()
                    
                    grouped = df_master_div.groupby('Kode Efek')
                    results = []
                    
                    for stock, group in grouped:
                        group = group.sort_values('Tanggal_Data')
                        if len(group['Tanggal_Data'].unique()) > 1:
                            tgl_awal = group['Tanggal_Data'].min()
                            total_awal = group[group['Tanggal_Data'] == tgl_awal]['Jumlah Saham (Curr)'].sum()
                            
                            tgl_akhir = group['Tanggal_Data'].max()
                            total_akhir = group[group['Tanggal_Data'] == tgl_akhir]['Jumlah Saham (Curr)'].sum()
                            
                            if total_awal > 0:
                                chg_pct = (total_akhir - total_awal) / total_awal * 100
                                results.append({
                                    'Stock Code': stock,
                                    'Own_Chg_Pct': chg_pct,
                                    'Own_Start': total_awal,
                                    'Own_End': total_akhir
                                })
                    
                    df_own_chg = pd.DataFrame(results)
                    
                    if not df_own_chg.empty:
                        # 4. Gabungkan Data
                        df_final = df_div.merge(df_own_chg, on='Stock Code', how='inner')
                        df_final = df_final[df_final['Own_Chg_Pct'] > 0.1]
                        df_final = df_final.sort_values('Own_Chg_Pct', ascending=False)
                        
                        st.success(f"Ditemukan {len(df_final)} saham mengalami Divergensi Bullish!")
                        
                        if not df_final.empty:
                            # --- FORMATTING DISPLAY DENGAN SEPARATOR ---
                            df_display = df_final.copy()
                            df_display['Harga Awal'] = df_display['Price_Start'].apply(format_rupiah)
                            df_display['Harga Akhir'] = df_display['Price_End'].apply(format_rupiah)
                            df_display['Drop (%)'] = df_display['Price_Chg_Pct'].apply(lambda x: f"{x:.2f}%")
                            
                            df_display['Lembar Awal'] = df_display['Own_Start'].apply(format_lembar)
                            df_display['Lembar Akhir'] = df_display['Own_End'].apply(format_lembar)
                            df_display['Akumulasi (%)'] = df_display['Own_Chg_Pct'].apply(lambda x: f"+{x:.2f}%")

                            st.dataframe(
                                df_display[['Stock Code', 'Harga Awal', 'Harga Akhir', 'Drop (%)', 'Lembar Awal', 'Lembar Akhir', 'Akumulasi (%)']],
                                hide_index=True,
                                use_container_width=True
                            )
                            
                            # Visualisasi Scatter
                            fig_div = px.scatter(
                                df_final,
                                x='Price_Chg_Pct',
                                y='Own_Chg_Pct',
                                text='Stock Code',
                                color='Own_Chg_Pct',
                                title=f"Peta Divergensi (Lookback: {lookback_div} hari)",
                                labels={'Price_Chg_Pct': 'Penurunan Harga (%)', 'Own_Chg_Pct': 'Kenaikan Kepemilikan 5% (%)'}
                            )
                            fig_div.update_traces(textposition='top center')
                            fig_div.add_vline(x=-10, line_dash="dash", line_color="red")
                            st.plotly_chart(fig_div, use_container_width=True)
                    else:
                        st.warning("Tidak ada data perubahan kepemilikan yang signifikan.")
                else:
                    st.error("Data Master 5% kosong.")

    # -------------------------------------------------------------------------
    # MODE 2: FOREIGN VS LOCAL FLOW MAP (Tidak perlu ubah format, krn grafik)
    # -------------------------------------------------------------------------
    elif mode_scan == "🗺️ Map: Foreign vs Local Flow":
        st.subheader("🗺️ Peta Kekuatan: Asing vs Lokal")
        
        days_avg = st.slider("Rata-rata Data (Hari Terakhir)", 1, 5, 1)
        date_cutoff = df_harian['Last Trading Date'].max() - timedelta(days=days_avg*2)
        df_flow = df_harian[df_harian['Last Trading Date'] >= date_cutoff].copy()
        
        df_flow_agg = df_flow.groupby('Stock Code').agg({
            'Change %': 'mean',
            'Net Foreign Flow': 'mean',
            'Close': 'last',
            'Value': 'mean'
        }).reset_index()
        
        min_val = st.number_input("Min Value Transaksi Harian (Rp Miliar)", 0.5, 50.0, 1.0) * 1e9
        df_flow_agg = df_flow_agg[df_flow_agg['Value'] >= min_val]
        
        def classify_flow(row):
            if row['Change %'] > 0 and row['Net Foreign Flow'] > 0: return "Foreign Driven Up"
            elif row['Change %'] > 0 and row['Net Foreign Flow'] < 0: return "Local Markup"
            elif row['Change %'] < 0 and row['Net Foreign Flow'] > 0: return "Foreign Dip Buy"
            elif row['Change %'] < 0 and row['Net Foreign Flow'] < 0: return "Distribution"
            return "Neutral"

        df_flow_agg['Status'] = df_flow_agg.apply(classify_flow, axis=1)
        
        fig_flow = px.scatter(
            df_flow_agg,
            x='Net Foreign Flow',
            y='Change %',
            color='Status',
            size='Value',
            hover_name='Stock Code',
            title=f"Market Structure Map (Avg {days_avg} Hari)",
            color_discrete_map={
                "Foreign Driven Up": "green", "Local Markup": "orange",
                "Foreign Dip Buy": "blue", "Distribution": "red"
            }
        )
        fig_flow.add_vline(x=0, line_width=1, line_color="black")
        fig_flow.add_hline(y=0, line_width=1, line_color="black")
        st.plotly_chart(fig_flow, use_container_width=True)

    # -------------------------------------------------------------------------
    # MODE 3: WATCHLIST PERSONAL (FIX FORMAT)
    # -------------------------------------------------------------------------
    else:
        st.subheader("📋 Watchlist Personal")
        
        all_stocks = sorted(df_harian['Stock Code'].unique())
        default_stocks = [s for s in ["BBCA", "BBRI", "ADRO", "TLKM"] if s in all_stocks]
        watchlist = st.multiselect("Pilih Saham:", all_stocks, default=default_stocks)
        
        if watchlist:
            df_watch_harian = df_harian[df_harian['Stock Code'].isin(watchlist)]
            df_watch_5 = df_master[df_master['Kode Efek'].isin(watchlist)] if not df_master.empty else pd.DataFrame()
            
            st.markdown("### 🚦 Status Watchlist")
            summary_watch = []
            
            for stock in watchlist:
                last_row = df_watch_harian[df_watch_harian['Stock Code'] == stock].sort_values('Last Trading Date').iloc[-1] if not df_watch_harian[df_watch_harian['Stock Code'] == stock].empty else None
                
                act_5 = "-"
                if not df_watch_5.empty:
                    df_s = df_watch_5[df_watch_5['Kode Efek'] == stock].sort_values('Tanggal_Data', ascending=False).head(5)
                    if not df_s.empty:
                        last_act = df_s.iloc[0]['Aksi']
                        act_5 = f"{last_act} ({df_s.iloc[0]['Tanggal_Data'].strftime('%d/%m')})"
                
                if last_row is not None:
                    summary_watch.append({
                        "Code": stock,
                        "Close_Raw": last_row['Close'], # Simpan raw utk sorting jk perlu
                        "Chg_Raw": last_row['Change %'],
                        "Vol_Raw": last_row.get('Volume Spike (x)', 0),
                        "Flow_Raw": last_row.get('Net Foreign Flow', 0),
                        "5% Activity": act_5
                    })
            
            df_sum_w = pd.DataFrame(summary_watch)
            
            # --- APPLY FORMATTING ---
            if not df_sum_w.empty:
                df_sum_w['Harga'] = df_sum_w['Close_Raw'].apply(format_rupiah)
                df_sum_w['Chg %'] = df_sum_w['Chg_Raw'].apply(lambda x: f"{x:.2f}%")
                df_sum_w['Vol Spike'] = df_sum_w['Vol_Raw'].apply(lambda x: f"{x:.1f}x")
                df_sum_w['Foreign Flow'] = df_sum_w['Flow_Raw'].apply(format_rupiah)
                
                st.dataframe(
                    df_sum_w[['Code', 'Harga', 'Chg %', 'Vol Spike', 'Foreign Flow', '5% Activity']],
                    hide_index=True,
                    use_container_width=True
                )
            
            st.markdown("---")
            for stock in watchlist:
                with st.expander(f"🔍 Detail {stock}"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.write("**Data Harian (5 Hari)**")
                        # Format ulang data harian untuk detail
                        d_h = df_watch_harian[df_watch_harian['Stock Code'] == stock].tail(5).copy()
                        d_h['Close'] = d_h['Close'].apply(format_rupiah)
                        d_h['Volume'] = d_h['Volume'].apply(format_lembar)
                        d_h['Value'] = d_h['Value'].apply(format_rupiah)
                        st.dataframe(d_h[['Last Trading Date', 'Close', 'Change %', 'Volume', 'Volume Spike (x)']], use_container_width=True, hide_index=True)
                    with col_b:
                        st.write("**Aktivitas 5% Terakhir**")
                        if not df_watch_5.empty:
                             d_5 = df_watch_5[df_watch_5['Kode Efek'] == stock].tail(5).copy()
                             d_5['Jumlah Saham (Curr)'] = d_5['Jumlah Saham (Curr)'].apply(format_lembar)
                             st.dataframe(d_5[['Tanggal_Data', 'UBO', 'Aksi', 'Jumlah Saham (Curr)']], use_container_width=True, hide_index=True)
        else:
            st.info("Silakan pilih saham.")

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.caption("🐋 Bandar Eye IDX - Institutional Grade | Data diproses di Colab, dashboard di Streamlit")
st.caption(f"Last Updated: {datetime.now().strftime('%d-%m-%Y %H:%M')} | {len(df_master) if not df_master.empty else 0:,} records 5% ownership")
