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
# KONFIGURASI: LINK GOOGLE DRIVE (PAKAI FORMAT EXPORT)
# =============================================================================
# Cara setting: File di GDrive -> Share -> "Anyone with link" -> Copy FILE_ID
# Gunakan link export CSV untuk hindari virus scan

FILE_IDS = {
    'harian': '1t_wCljhepGBqZVrvleuZKldomQKop9DY',      # Kompilasi_Data_1Tahun
    'ksei': '1eTUIC120SHTCzvBk77Q87w0X56F2HkWz',        # KSEI_Shareholder_Processed
    'master_5': '1mS7Xp_PMqFnLTikU7giDZ42mqcbsiYvx'     # MASTER_DATABASE_5persen
}

# =============================================================================
# FUNGSI LOAD DATA DENGAN RETRY & FALLBACK
# =============================================================================
def load_csv_from_gdrive(file_id, max_retries=3):
    """
    Load CSV dari Google Drive dengan multiple fallback method.
    Mengatasi error 403, 500, dan virus scan warning.
    """
    
    # METHOD 1: Export link format (paling reliable untuk public file)
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    for attempt in range(max_retries):
        try:
            # Gunakan session untuk handle cookies
            session = requests.Session()
            
            # Download dengan stream untuk handle large file
            response = session.get(url, stream=True, timeout=30)
            
            # Handle Google Drive virus scan warning
            if 'Virus scan warning' in response.text or 'Quota exceeded' in response.text:
                # Extract confirmation token
                import re
                match = re.search(r'confirm=([0-9A-Za-z]+)', response.text)
                if match:
                    confirm_token = match.group(1)
                    url = f"https://drive.google.com/uc?export=download&confirm={confirm_token}&id={file_id}"
                    response = session.get(url, stream=True, timeout=30)
            
            response.raise_for_status()
            
            # Baca CSV dari response content
            content = response.content.decode('utf-8')
            df = pd.read_csv(StringIO(content))
            
            print(f"✅ Success loading file ID: {file_id}")
            return df
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for ID {file_id}: {e}")
            
            # METHOD 2: Direct download link (alternative)
            if attempt == 1:
                try:
                    url = f"https://drive.google.com/uc?id={file_id}&export=download"
                    response = requests.get(url, timeout=30)
                    content = response.content.decode('utf-8')
                    df = pd.read_csv(StringIO(content))
                    return df
                except:
                    pass
            
            # METHOD 3: Menggunakan pandas read_csv dengan URL (fallback terakhir)
            if attempt == 2:
                try:
                    url = f"https://drive.google.com/uc?id={file_id}"
                    df = pd.read_csv(url)
                    return df
                except:
                    pass
            
            time.sleep(2)  # Backoff sebelum retry
    
    raise Exception(f"Gagal load file ID {file_id} setelah {max_retries} percobaan")

# =============================================================================
# CACHE DATA LOADING DENGAN ERROR HANDLING
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
        # Return empty dataframe dengan struktur minimal
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
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"❌ Gagal load data KSEI: {e}")
        return pd.DataFrame(columns=['Code', 'Date', 'Top_Buyer', 'Top_Seller'])

@st.cache_data(ttl=3600, show_spinner="Loading data kepemilikan 5%...")
def load_master_5():
    """Load data MASTER_DATABASE_5persen"""
    try:
        df = load_csv_from_gdrive(FILE_IDS['master_5'])
        
        # Parsing tanggal
        df['Tanggal_Data'] = pd.to_datetime(df['Tanggal_Data'], errors='coerce')
        df = df.dropna(subset=['Tanggal_Data'])
        
        # Konversi numerik
        df['Jumlah Saham (Prev)'] = pd.to_numeric(df['Jumlah Saham (Prev)'], errors='coerce')
        df['Jumlah Saham (Curr)'] = pd.to_numeric(df['Jumlah Saham (Curr)'], errors='coerce')
        df['Close_Price'] = pd.to_numeric(df['Close_Price'], errors='coerce')
        
        # Hitung perubahan
        df['Perubahan_Saham'] = df['Jumlah Saham (Curr)'] - df['Jumlah Saham (Prev)']
        df['Perubahan_Persen'] = (df['Perubahan_Saham'] / df['Jumlah Saham (Prev)'].replace(0, np.nan)) * 100
        
        # Klasifikasi aksi
        df['Aksi'] = 'Tahan'
        df.loc[df['Perubahan_Saham'] > 0, 'Aksi'] = 'Beli'
        df.loc[df['Perubahan_Saham'] < 0, 'Aksi'] = 'Jual'
        
        # Estimasi nilai transaksi
        df['Estimasi_Nilai'] = df['Perubahan_Saham'] * df['Close_Price']
        
        return df
    except Exception as e:
        st.error(f"❌ Gagal load data master 5%: {e}")
        return pd.DataFrame(columns=['Kode Efek', 'Tanggal_Data', 'Nama Pemegang Saham'])

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
# KONFIGURASI HALAMAN (WAJIB PALING ATAS)
# =============================================================================
st.set_page_config(
    page_title="Bandar Eye IDX",
    page_icon="🐋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# SIDEBAR: FILTER GLOBAL
# =============================================================================
st.sidebar.image("https://img.icons8.com/fluency/96/whale.png", width=80)
st.sidebar.title("🐋 Bandar Eye")
st.sidebar.caption("v1.0 - 20 Tahun Cycle Experience")

# Load semua data di awal
with st.spinner('Memuat data harga...'):
    df_harian = load_harian()
with st.spinner('Memuat data KSEI...'):
    df_ksei = load_ksei()
with st.spinner('Memuat data kepemilikan 5%...'):
    df_master = load_master_5()

with st.sidebar.expander("🔧 Debug Info"):
    st.write("**Kolom di df_harian:**")
    st.write(list(df_harian.columns))
    st.write("**Sample data:**")
    st.dataframe(df_harian.head(2))
    
# Date range global
min_date = df_harian['Last Trading Date'].min()
max_date = df_harian['Last Trading Date'].max()

st.sidebar.markdown("---")
st.sidebar.subheader("📅 Filter Tanggal")
start_date = st.sidebar.date_input("Dari", min_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("Sampai", max_date, min_value=min_date, max_value=max_date)

# Filter untuk sektor
sektor_list = sorted(df_harian['Sector'].dropna().unique())
selected_sectors = st.sidebar.multiselect("🏭 Sektor", sektor_list, default=[])

st.sidebar.markdown("---")
st.sidebar.caption("© Bandarmology IDX")

# =============================================================================
# MAIN APP: 4 TAB
# =============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Momentum Bandar", 
    "🏦 KSEI Big Money", 
    "🕵️ Akumulasi Awal (5%)", 
    "📊 Watchlist"
])

# =============================================================================
# TAB 1: MOMENTUM BANDAR (Harian) - VERSI FIX
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
    
    # --- CEK KOLOM YANG TERSEDIA ---
    required_cols = {
        'Volume Spike (x)': min_volume_spike,
        'Avg_Order_Volume': min_ao_ratio,
        'MA50_AOVol': None,
        'Bid/Offer Imbalance': min_imbalance
    }
    
    # Filter hanya kolom yang ada
    filter_condition = pd.Series([True] * len(df_filtered))
    
    if 'Volume Spike (x)' in df_filtered.columns:
        filter_condition &= (df_filtered['Volume Spike (x)'] >= min_volume_spike)
    else:
        st.warning("⚠️ Kolom 'Volume Spike (x)' tidak ditemukan. Volume spike tidak difilter.")
    
    if 'Avg_Order_Volume' in df_filtered.columns and 'MA50_AOVol' in df_filtered.columns:
        ao_ratio = df_filtered['Avg_Order_Volume'] / df_filtered['MA50_AOVol'].replace(0, np.nan)
        filter_condition &= (ao_ratio >= min_ao_ratio)
    else:
        st.warning("⚠️ Kolom AOVol tidak lengkap. Anomali order volume tidak difilter.")
    
    if 'Bid/Offer Imbalance' in df_filtered.columns:
        filter_condition &= (df_filtered['Bid/Offer Imbalance'] >= min_imbalance)
    else:
        st.warning("⚠️ Kolom 'Bid/Offer Imbalance' tidak ditemukan.")
    
    # Terapkan filter
    df_anomaly = df_filtered[filter_condition].copy()
    
    # Hitung potensi hanya jika kolom tersedia
    df_anomaly['Potensi'] = 0  # default
    
    if 'Volume Spike (x)' in df_anomaly.columns:
        df_anomaly['Potensi'] += df_anomaly['Volume Spike (x)'] * 0.3
    
    if 'Avg_Order_Volume' in df_anomaly.columns and 'MA50_AOVol' in df_anomaly.columns:
        ao_ratio_val = df_anomaly['Avg_Order_Volume'] / df_anomaly['MA50_AOVol'].replace(0, np.nan)
        df_anomaly['Potensi'] += ao_ratio_val.fillna(0) * 0.4
    
    if 'Bid/Offer Imbalance' in df_anomaly.columns:
        df_anomaly['Potensi'] += (df_anomaly['Bid/Offer Imbalance'] + 1) * 0.3
    
    df_anomaly = df_anomaly.sort_values('Potensi', ascending=False)
    
    # Tampilkan metrik
    st.subheader(f"🎯 {len(df_anomaly)} Saham dengan Aktivitas Bandar Terdeteksi")
    
    if not df_anomaly.empty:
        # Siapkan kolom display yang pasti ada
        display_cols = ['Stock Code', 'Last Trading Date']
        
        # Tambahkan kolom opsional jika ada
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
        
        # --- VISUALISASI SCATTER (FIX) ---
        st.subheader("📊 Volume Spike vs AOVol Ratio")
        
        # Pastikan kolom untuk scatter tersedia
        if ('Volume Spike (x)' in df_anomaly.columns and 
            'Avg_Order_Volume' in df_anomaly.columns and 
            'MA50_AOVol' in df_anomaly.columns and
            'Final Signal' in df_anomaly.columns):
            
            # Hitung AOVol Ratio
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
                st.info("Data tidak cukup untuk scatter plot. Coba turunkan threshold.")
        else:
            st.warning("Kolom yang diperlukan untuk scatter plot tidak tersedia.")
        
    else:
        st.info("Tidak ada saham dengan kriteria tersebut. Coba turunkan threshold.")

# =============================================================================
# TAB 2: KSEI BIG MONEY TRACKER (Bulanan)
# =============================================================================
with tab2:
    st.header("🏦 Jejak Big Money (KSEI Bulanan)")
    st.caption("Institusi yang akumulasi/distribusi besar dalam sebulan")
    
    # Filter periode
    df_ksei_filtered = df_ksei[
        (df_ksei['Date'] >= pd.to_datetime(start_date)) &
        (df_ksei['Date'] <= pd.to_datetime(end_date))
    ]
    
    # Pilih institusi teratas
    top_n = st.slider("Top N Buyer/Seller", 5, 30, 15)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🟢 Top Buyer (Volume)")
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
    
    # Filter Stock Specific
    st.subheader("🔍 Detail Saham")
    kode_saham = st.text_input("Masukkan Kode Saham (contoh: AADI, BBCA)", "").upper()
    
    if kode_saham:
        df_detail = df_ksei_filtered[df_ksei_filtered['Code'] == kode_saham].sort_values('Date', ascending=False)
        if not df_detail.empty:
            cols_detail = ['Date', 'Price', 'Total_Local', 'Total_Foreign', 
                          'Top_Buyer', 'Top_Buyer_Vol', 'Top_Seller', 'Top_Seller_Vol']
            st.dataframe(
                df_detail[cols_detail].head(12),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning(f"Data tidak ditemukan untuk {kode_saham}")

# =============================================================================
# TAB 3: AKUMULASI AWAL DARI DATA 5% (DENGAN LINE CHART DEEP DIVE)
# =============================================================================
with tab3:
    st.header("🕵️ DETEKSI AKUMULASI AWAL (Data 5%)")
    st.markdown("""
    > **Filosofi Bandarmology**: Sebelum volume spike, sebelum AOVol anomaly, sebelum berita keluar, 
    > **bandar sejati mulai masuk melalui entitas >5%**. Ini adalah jejak paling awal.
    """)
    
    # Filter data master 5%
    df_master_filtered = df_master[
        (df_master['Tanggal_Data'] >= pd.to_datetime(start_date)) &
        (df_master['Tanggal_Data'] <= pd.to_datetime(end_date))
    ].copy()
    
    if selected_sectors:
        # Join dengan data harian untuk ambil sektor
        sector_map = df_harian[['Stock Code', 'Sector']].drop_duplicates('Stock Code')
        df_master_filtered = df_master_filtered.merge(sector_map, left_on='Kode Efek', right_on='Stock Code', how='left')
        df_master_filtered = df_master_filtered[df_master_filtered['Sector'].isin(selected_sectors)]
    
    # PARAMETER DETEKSI AKUMULASI AWAL
    st.subheader("⚙️ Parameter Akumulasi Awal")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        min_beli = st.number_input("Minimal Beli (lembar)", min_value=1000, value=100000, step=10000, 
                                   help="Filter pembelian minimal", format="%d")
    with col2:
        max_saham_beredar = st.number_input("Maksimal Kepemilikan (%)", min_value=0.1, max_value=100.0, value=10.0, step=0.5, 
                                           help="Batas atas % kepemilikan (bandar awal < 10%)")
    with col3:
        lookback_days = st.number_input("Lookback (hari)", min_value=7, max_value=90, value=30, 
                                       help="Dalam X hari terakhir")
    
    # 1. Filter: Aksi BELI dan jumlah beli >= minimal
    df_beli = df_master_filtered[
        (df_master_filtered['Aksi'] == 'Beli') &
        (df_master_filtered['Perubahan_Saham'] >= min_beli)
    ].copy()
    
    # 2. Hitung tanggal pertama kali beli dalam lookback
    cutoff_date = pd.to_datetime(end_date) - timedelta(days=lookback_days)
    df_beli = df_beli[df_beli['Tanggal_Data'] >= cutoff_date]
    
    # 3. Kelompokkan berdasarkan (Kode Efek, Nama Pemegang Saham) untuk lihat akumulasi total
    df_akumulasi = df_beli.groupby(
        ['Kode Efek', 'Nama Pemegang Saham', 'Nama Pemegang Rekening Efek']
    ).agg({
        'Perubahan_Saham': 'sum',
        'Estimasi_Nilai': 'sum',
        'Tanggal_Data': ['first', 'last', 'count']
    }).reset_index()
    
    # Flatten column names
    df_akumulasi.columns = ['Kode Efek', 'Nama Pemegang Saham', 'Nama Rekening', 
                            'Total_Beli_Lembar', 'Total_Nilai_Rp', 
                            'Tgl_Pertama', 'Tgl_Terakhir', 'Frekuensi_Transaksi']
    
    # 4. Skor Akumulasi Awal
    df_akumulasi['Skor_Akumulasi'] = (
        np.log1p(df_akumulasi['Total_Beli_Lembar']) * 0.5 +
        np.log1p(df_akumulasi['Frekuensi_Transaksi']) * 0.3 +
        (1 - (df_akumulasi['Tgl_Terakhir'] - df_akumulasi['Tgl_Pertama']).dt.days / lookback_days) * 0.2
    )
    
    df_akumulasi = df_akumulasi.sort_values('Skor_Akumulasi', ascending=False)
    
    # TAMPILKAN HASIL DENGAN FORMAT RIBUAN
    st.subheader(f"🎯 {len(df_akumulasi)} Entitas dengan Indikasi Akumulasi Awal")
    st.caption("Semakin tinggi Skor = Semakin agresif akumulasi dalam waktu singkat")
    
    if not df_akumulasi.empty:
        # Merge dengan sektor
        df_akumulasi = df_akumulasi.merge(
            df_harian[['Stock Code', 'Sector']].drop_duplicates('Stock Code'),
            left_on='Kode Efek',
            right_on='Stock Code',
            how='left'
        )
        
        # Buat versi display dengan format yang sudah diformat
        df_display = df_akumulasi.copy()
        df_display['Total_Beli_Lembar_Display'] = df_display['Total_Beli_Lembar'].apply(format_lembar)
        df_display['Total_Nilai_Rp_Display'] = df_display['Total_Nilai_Rp'].apply(format_rupiah)
        df_display['Skor_Display'] = df_display['Skor_Akumulasi'].apply(lambda x: f"{x:.2f}")
        
        st.dataframe(
            df_display[['Kode Efek', 'Nama Pemegang Saham', 'Sector',
                       'Total_Beli_Lembar_Display', 'Total_Nilai_Rp_Display', 
                       'Frekuensi_Transaksi', 'Tgl_Pertama', 'Tgl_Terakhir', 'Skor_Display']].head(50),
            column_config={
                'Total_Beli_Lembar_Display': st.column_config.TextColumn("Total Beli (Lembar)"),
                'Total_Nilai_Rp_Display': st.column_config.TextColumn("Estimasi Nilai"),
                'Frekuensi_Transaksi': st.column_config.NumberColumn(format="%d"),
                'Tgl_Pertama': st.column_config.DateColumn(format="DD-MM-YY"),
                'Tgl_Terakhir': st.column_config.DateColumn(format="DD-MM-YY"),
                'Skor_Display': st.column_config.TextColumn("Skor")
            },
            use_container_width=True,
            hide_index=True
        )
        
        # Chart: Top 10 akumulator
        st.subheader("💰 Top 10 Entitas Akumulator Terbesar (Nilai Rp)")
        top10 = df_akumulasi.nlargest(10, 'Total_Nilai_Rp').copy()
        top10['Label'] = top10['Nama Pemegang Saham'].str[:30] + '...'
        top10['Nilai_Display'] = top10['Total_Nilai_Rp'].apply(format_rupiah)
        
        fig = px.bar(
            top10,
            x='Total_Nilai_Rp',
            y='Label',
            color='Sector',
            orientation='h',
            title="Total Nilai Pembelian (Estimasi)",
            labels={'Total_Nilai_Rp': 'Nilai (Rp)', 'Label': 'Pemegang Saham'},
            hover_data={'Total_Nilai_Rp': False, 'Nilai_Display': True, 'Kode Efek': True}
        )
        fig.update_layout(height=500)
        fig.update_xaxes(tickformat=",.0f")  # Format sumbu x dengan separator
        st.plotly_chart(fig, use_container_width=True)
        
# ============= DEEP DIVE LINE CHART - BY WEEKLY =============
st.subheader("📈 DEEP DIVE: Timeline Akumulasi per Rekening Efek")
st.caption("Pergerakan kepemilikan dan harga secara weekly")

# Input untuk deep dive
col_a, col_b, col_c = st.columns([2, 1, 1])

with col_a:
    # Pilihan rekening efek dari hasil akumulasi
    rekening_options = df_akumulasi['Nama Rekening'].unique()
    selected_rekening = st.selectbox(
        "Pilih Nama Pemegang Rekening Efek",
        rekening_options,
        index=0 if len(rekening_options) > 0 else None
    )

with col_b:
    # Pilihan saham spesifik dari rekening tersebut
    if selected_rekening:
        stocks_of_rekening = df_akumulasi[
            df_akumulasi['Nama Rekening'] == selected_rekening
        ]['Kode Efek'].unique()
        selected_stock = st.selectbox(
            "Kode Efek",
            stocks_of_rekening,
            index=0 if len(stocks_of_rekening) > 0 else None
        )

with col_c:
    # Pilih interval weekly
    weekly_option = st.selectbox(
        "Interval",
        ["Weekly", "Bi-Weekly", "Monthly"],
        index=0
    )

if selected_rekening and selected_stock:
    # Filter data untuk rekening dan saham terpilih
    df_timeline = df_master_filtered[
        (df_master_filtered['Nama Rekening Efek'] == selected_rekening) &
        (df_master_filtered['Kode Efek'] == selected_stock)
    ].sort_values('Tanggal_Data').copy()
    
    if not df_timeline.empty:
        # --- RESAMPLE KE WEEKLY ---
        df_timeline.set_index('Tanggal_Data', inplace=True)
        
        # Tentukan frekuensi resample
        if weekly_option == "Weekly":
            freq = 'W'  # Weekly
        elif weekly_option == "Bi-Weekly":
            freq = '2W'  # 2 minggu
        else:
            freq = 'ME'  # Month End
            
        # Resample: ambil data terakhir di setiap periode
        df_weekly = df_timeline.resample(freq).last().dropna(subset=['Kode Efek']).reset_index()
        
        # Kembalikan index ke kolom biasa
        df_timeline.reset_index(inplace=True)
        
        st.success(f"Menampilkan {len(df_weekly)} periode {weekly_option.lower()} dari {len(df_timeline)} transaksi harian")
        
        # ============= CHART 1: KEPEMILIKAN =============
        st.subheader(f"📊 Kepemilikan: {selected_rekening}")
        
        fig_ownership = go.Figure()
        
        # Line chart kepemilikan (weekly)
        fig_ownership.add_trace(go.Scatter(
            x=df_weekly['Tanggal_Data'],
            y=df_weekly['Jumlah Saham (Curr)'],
            mode='lines+markers',
            name='Jumlah Kepemilikan',
            line=dict(color='#2E86AB', width=3),
            marker=dict(size=10, symbol='circle'),
            hovertemplate='<b>%{x|%d-%m-%Y}</b><br>' +
                         'Kepemilikan: %{customdata}<extra></extra>',
            customdata=df_weekly['Jumlah Saham (Curr)'].apply(lambda x: format_lembar(x))
        ))
        
        # Tambahkan anotasi beli/jual di weekly chart
        beli_weekly = df_weekly[df_weekly['Perubahan_Saham'] > 0]
        jual_weekly = df_weekly[df_weekly['Perubahan_Saham'] < 0]
        
        fig_ownership.add_trace(go.Scatter(
            x=beli_weekly['Tanggal_Data'],
            y=beli_weekly['Jumlah Saham (Curr)'],
            mode='markers',
            name='Ada Pembelian',
            marker=dict(color='#2ECC71', size=14, symbol='triangle-up'),
            hovertemplate='<b>%{x|%d-%m-%Y}</b><br>Beli: %{customdata}<extra></extra>',
            customdata=beli_weekly['Perubahan_Saham'].apply(lambda x: format_lembar(x))
        ))
        
        fig_ownership.add_trace(go.Scatter(
            x=jual_weekly['Tanggal_Data'],
            y=jual_weekly['Jumlah Saham (Curr)'],
            mode='markers',
            name='Ada Penjualan',
            marker=dict(color='#E74C3C', size=14, symbol='triangle-down'),
            hovertemplate='<b>%{x|%d-%m-%Y}</b><br>Jual: %{customdata}<extra></extra>',
            customdata=jual_weekly['Perubahan_Saham'].apply(lambda x: format_lembar(abs(x)))
        ))
        
        fig_ownership.update_layout(
            title=f"Pergerakan Kepemilikan - {selected_stock}",
            height=450,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            yaxis=dict(
                title="Jumlah Saham (Lembar)",
                tickformat=",.0f",
                gridcolor='lightgray'
            ),
            xaxis=dict(
                title="Tanggal (Weekly)",
                tickformat="%d-%m-%Y"
            )
        )
        
        st.plotly_chart(fig_ownership, use_container_width=True)
        
        # ============= CHART 2: HARGA CLOSE =============
        st.subheader(f"📈 Pergerakan Harga - {selected_stock}")
        
        # Ambil data harga dari df_harian
        df_harga = df_harian[
            df_harian['Stock Code'] == selected_stock
        ].sort_values('Last Trading Date').copy()
        
        if not df_harga.empty:
            # Resample harga ke weekly juga
            df_harga.set_index('Last Trading Date', inplace=True)
            df_harga_weekly = df_harga.resample(freq).last().dropna(subset=['Close']).reset_index()
            df_harga.reset_index(inplace=True)
            
            fig_price = go.Figure()
            
            # Line chart harga
            fig_price.add_trace(go.Scatter(
                x=df_harga_weekly['Last Trading Date'],
                y=df_harga_weekly['Close'],
                mode='lines+markers',
                name='Harga Close',
                line=dict(color='#E88873', width=3),
                marker=dict(size=8, color='#E88873'),
                hovertemplate='<b>%{x|%d-%m-%Y}</b><br>' +
                             'Harga: %{customdata}<extra></extra>',
                customdata=df_harga_weekly['Close'].apply(lambda x: format_rupiah(x))
            ))
            
            # Tambahkan volume spike sebagai scatter
            if 'Volume Spike (x)' in df_harga_weekly.columns:
                spike = df_harga_weekly[df_harga_weekly['Volume Spike (x)'] > 1.5]
                if not spike.empty:
                    fig_price.add_trace(go.Scatter(
                        x=spike['Last Trading Date'],
                        y=spike['Close'],
                        mode='markers',
                        name='Volume Spike',
                        marker=dict(color='#F39C12', size=12, symbol='star'),
                        hovertemplate='<b>%{x|%d-%m-%Y}</b><br>' +
                                     'Volume Spike: %{customdata:.1f}x<extra></extra>',
                        customdata=spike['Volume Spike (x)']
                    ))
            
            fig_price.update_layout(
                title=f"Pergerakan Harga - {selected_stock}",
                height=400,
                hovermode='x unified',
                yaxis=dict(
                    title="Harga (Rp)",
                    tickformat=",.0f",
                    gridcolor='lightgray'
                ),
                xaxis=dict(
                    title="Tanggal (Weekly)",
                    tickformat="%d-%m-%Y"
                )
            )
            
            st.plotly_chart(fig_price, use_container_width=True)
        else:
            st.warning(f"Data harga tidak ditemukan untuk {selected_stock}")
        
        # ============= METRIK RINGKASAN =============
        st.subheader("📋 Ringkasan Aktivitas")
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        with col_m1:
            awal = df_timeline['Jumlah Saham (Curr)'].iloc[0]
            akhir = df_timeline['Jumlah Saham (Curr)'].iloc[-1]
            perubahan = akhir - awal
            st.metric(
                "Perubahan Kepemilikan",
                format_lembar(perubahan),
                delta=f"{((perubahan/awal)*100):.1f}%" if awal > 0 else "N/A"
            )
        
        with col_m2:
            total_beli = df_timeline[df_timeline['Perubahan_Saham'] > 0]['Perubahan_Saham'].sum()
            st.metric("Total Pembelian", format_lembar(total_beli))
        
        with col_m3:
            total_jual = df_timeline[df_timeline['Perubahan_Saham'] < 0]['Perubahan_Saham'].sum()
            st.metric("Total Penjualan", format_lembar(abs(total_jual)))
        
        with col_m4:
            # Harga terakhir
            last_price = df_harian[
                df_harian['Stock Code'] == selected_stock
            ]['Close'].iloc[-1] if not df_harian[df_harian['Stock Code'] == selected_stock].empty else 0
            st.metric("Harga Terkini", format_rupiah(last_price))
        
        # ============= DETAIL TRANSAKSI =============
        with st.expander("📋 Lihat Detail Transaksi Harian"):
            df_detail = df_timeline[[
                'Tanggal_Data', 'Aksi', 'Perubahan_Saham', 
                'Jumlah Saham (Curr)', 'Close_Price', 'Estimasi_Nilai'
            ]].copy()
            
            df_detail['Tanggal_Data'] = df_detail['Tanggal_Data'].dt.strftime('%d-%m-%Y')
            df_detail['Perubahan_Saham'] = df_detail['Perubahan_Saham'].apply(
                lambda x: f"{format_lembar(x)} ({x:+,.0f})".replace(",", ".")
            )
            df_detail['Jumlah Saham (Curr)'] = df_detail['Jumlah Saham (Curr)'].apply(format_lembar)
            df_detail['Close_Price'] = df_detail['Close_Price'].apply(format_rupiah)
            df_detail['Estimasi_Nilai'] = df_detail['Estimasi_Nilai'].apply(format_rupiah)
            
            st.dataframe(
                df_detail.sort_values('Tanggal_Data', ascending=False),
                use_container_width=True,
                hide_index=True
            )
            
            # Download button
            csv = df_detail.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Detail Transaksi (CSV)",
                data=csv,
                file_name=f"{selected_stock}_{selected_rekening}_transaksi.csv",
                mime="text/csv"
            )
            
    else:
        st.info(f"Tidak ada data historis untuk rekening {selected_rekening} pada saham {selected_stock}")
else:
    st.info("Pilih Nama Pemegang Rekening Efek dan Kode Efek untuk melihat timeline")
# =============================================================================
# TAB 4: WATCHLIST & KONVERGENSI SINYAL
# =============================================================================
with tab4:
    st.header("📊 Watchlist & Konvergensi Sinyal")
    st.caption("Gabungan sinyal dari 3 sumber data")
    
    # Input watchlist
    watchlist_input = st.text_area("✏️ Daftar Saham (pisahkan dengan koma)", value="AADI, BBCA, TLKM, ASII")
    watchlist = [s.strip().upper() for s in watchlist_input.split(",") if s.strip()]
    
    if watchlist:
        # Filter data untuk watchlist
        df_watch_harian = df_harian[df_harian['Stock Code'].isin(watchlist)]
        df_watch_ksei = df_ksei[df_ksei['Code'].isin(watchlist)]
        df_watch_5 = df_master[df_master['Kode Efek'].isin(watchlist)]
        
        # Tampilkan konvergensi
        st.subheader("🎯 Status Terkini")
        
        # Ambil data terakhir per saham
        last_data = df_watch_harian.sort_values('Last Trading Date').groupby('Stock Code').last().reset_index()
        
        # Merge dengan sinyal dari KSEI dan 5%
        for stock in watchlist:
            with st.expander(f"{stock} - {last_data[last_data['Stock Code'] == stock]['Close'].values[0] if stock in last_data['Stock Code'].values else 'N/A'}"):
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown("**📈 Data Harian**")
                    stock_harian = df_watch_harian[df_watch_harian['Stock Code'] == stock].sort_values('Last Trading Date', ascending=False).head(5)
                    if not stock_harian.empty:
                        st.dataframe(
                            stock_harian[['Last Trading Date', 'Close', 'Volume Spike (x)', 'Big_Player_Anomaly', 'Final Signal']],
                            hide_index=True,
                            use_container_width=True
                        )
                
                with col_b:
                    st.markdown("**🏦 KSEI Bulanan**")
                    stock_ksei = df_watch_ksei[df_watch_ksei['Code'] == stock].sort_values('Date', ascending=False).head(3)
                    if not stock_ksei.empty:
                        st.dataframe(
                            stock_ksei[['Date', 'Top_Buyer', 'Top_Buyer_Vol', 'Top_Seller', 'Top_Seller_Vol']],
                            hide_index=True,
                            use_container_width=True
                        )
                
                st.markdown("**🕵️ Aktivitas 5%**")
                stock_5 = df_watch_5[df_watch_5['Kode Efek'] == stock].sort_values('Tanggal_Data', ascending=False).head(10)
                if not stock_5.empty:
                    stock_5_beli = stock_5[stock_5['Aksi'] == 'Beli']
                    if not stock_5_beli.empty:
                        st.success(f"Ada aktivitas BELI: {len(stock_5_beli)} transaksi")
                        st.dataframe(
                            stock_5_beli[['Tanggal_Data', 'Nama Pemegang Saham', 'Perubahan_Saham', 'Close_Price']],
                            hide_index=True,
                            use_container_width=True
                        )
                    else:
                        st.info("Tidak ada pembelian signifikan (>5%) dalam periode ini")
    else:
        st.warning("Masukkan minimal 1 kode saham")

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.caption("🐋 Bandar Eye IDX - Dikembangkan oleh Trader 20 Tahun | Data diperbarui via Google Colab pipeline")
