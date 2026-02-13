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
        
        # ============= DEEP DIVE LINE CHART =============
        st.subheader("📈 DEEP DIVE: Timeline Akumulasi per Entitas")
        st.caption("Pilih entitas untuk melihat pergerakan kepemilikan dari waktu ke waktu")
        
        # Input untuk deep dive
        col_a, col_b = st.columns([2, 1])
        with col_a:
            # Pilihan entitas dari hasil akumulasi
            entity_options = df_akumulasi['Nama Pemegang Saham'].unique()
            selected_entity = st.selectbox(
                "Pilih Nama Pemegang Saham",
                entity_options,
                index=0 if len(entity_options) > 0 else None
            )
        
        with col_b:
            # Pilihan saham spesifik dari entitas tersebut
            if selected_entity:
                stocks_of_entity = df_akumulasi[df_akumulasi['Nama Pemegang Saham'] == selected_entity]['Kode Efek'].unique()
                selected_stock = st.selectbox(
                    "Pilih Kode Efek",
                    stocks_of_entity,
                    index=0 if len(stocks_of_entity) > 0 else None
                )
        
        if selected_entity and selected_stock:
            # Filter data untuk entitas dan saham terpilih
            df_timeline = df_master_filtered[
                (df_master_filtered['Nama Pemegang Saham'] == selected_entity) &
                (df_master_filtered['Kode Efek'] == selected_stock)
            ].sort_values('Tanggal_Data').copy()
            
            if not df_timeline.empty:
                # Hitung kumulatif kepemilikan
                df_timeline['Kepemilikan_Kumulatif'] = df_timeline['Jumlah Saham (Curr)'].cumsum()  # atau pakai Curr langsung?
                # Lebih tepat: Karena ini sudah snapshot, kita pakai Jumlah Saham (Curr) per tanggal
                df_timeline['Kepemilikan'] = df_timeline['Jumlah Saham (Curr)']
                
                # Hitung perubahan bersih dari awal periode
                df_timeline['Perubahan_Dari_Awal'] = df_timeline['Kepemilikan'] - df_timeline['Kepemilikan'].iloc[0]
                
                # Buat line chart dengan dual axis
                fig_deep = go.Figure()
                
                # Line chart: Kepemilikan
                fig_deep.add_trace(go.Scatter(
                    x=df_timeline['Tanggal_Data'],
                    y=df_timeline['Kepemilikan'],
                    mode='lines+markers',
                    name='Jumlah Kepemilikan',
                    line=dict(color='#2E86AB', width=3),
                    marker=dict(size=8),
                    yaxis='y',
                    hovertemplate='Tanggal: %{x|%d-%m-%Y}<br>Kepemilikan: %{customdata}<extra></extra>',
                    customdata=df_timeline['Kepemilikan'].apply(lambda x: format_lembar(x))
                ))
                
                # Line chart: Harga Close (secondary axis)
                if 'Close_Price' in df_timeline.columns:
                    fig_deep.add_trace(go.Scatter(
                        x=df_timeline['Tanggal_Data'],
                        y=df_timeline['Close_Price'],
                        mode='lines+markers',
                        name='Harga Close',
                        line=dict(color='#E88873', width=2, dash='dot'),
                        marker=dict(size=6),
                        yaxis='y2',
                        hovertemplate='Tanggal: %{x|%d-%m-%Y}<br>Harga: %{customdata}<extra></extra>',
                        customdata=df_timeline['Close_Price'].apply(lambda x: format_rupiah(x))
                    ))
                
                # Bar chart: Volume Beli/Jual
                colors = ['#2ECC71' if x > 0 else '#E74C3C' for x in df_timeline['Perubahan_Saham']]
                fig_deep.add_trace(go.Bar(
                    x=df_timeline['Tanggal_Data'],
                    y=df_timeline['Perubahan_Saham'],
                    name='Transaksi',
                    marker_color=colors,
                    opacity=0.5,
                    yaxis='y3',
                    hovertemplate='Tanggal: %{x|%d-%m-%Y}<br>Transaksi: %{customdata}<extra></extra>',
                    customdata=df_timeline['Perubahan_Saham'].apply(lambda x: format_lembar(x) + (' (Beli)' if x > 0 else ' (Jual)'))
                ))
                
                # Update layout untuk dual axis + bar
                fig_deep.update_layout(
                    title=f"{selected_entity} pada {selected_stock}",
                    height=600,
                    hovermode='x unified',
                    xaxis=dict(
                        title="Tanggal",
                        tickformat="%d-%m-%Y"
                    ),
                    yaxis=dict(
                        title="Jumlah Kepemilikan (Lembar)",
                        titlefont=dict(color='#2E86AB'),
                        tickfont=dict(color='#2E86AB'),
                        tickformat=",.0f"
                    ),
                    yaxis2=dict(
                        title="Harga (Rp)",
                        titlefont=dict(color='#E88873'),
                        tickfont=dict(color='#E88873'),
                        tickformat=",.0f",
                        overlaying='y',
                        side='right',
                        position=0.95
                    ),
                    yaxis3=dict(
                        title="Volume Transaksi",
                        titlefont=dict(color='#7F8C8D'),
                        tickfont=dict(color='#7F8C8D'),
                        tickformat=",.0f",
                        anchor="free",
                        overlaying="y",
                        side="right",
                        position=1.0
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="center",
                        x=0.5
                    ),
                    margin=dict(r=150)  # Memberi ruang untuk axis kanan
                )
                
                st.plotly_chart(fig_deep, use_container_width=True)
                
                # Tampilkan ringkasan
                col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
                
                with col_metric1:
                    awal = df_timeline['Kepemilikan'].iloc[0]
                    akhir = df_timeline['Kepemilikan'].iloc[-1]
                    perubahan = akhir - awal
                    st.metric(
                        "Perubahan Kepemilikan",
                        format_lembar(perubahan),
                        delta=f"{((perubahan/awal)*100):.1f}%" if awal > 0 else "N/A"
                    )
                
                with col_metric2:
                    total_beli = df_timeline[df_timeline['Perubahan_Saham'] > 0]['Perubahan_Saham'].sum()
                    st.metric("Total Pembelian", format_lembar(total_beli))
                
                with col_metric3:
                    total_jual = df_timeline[df_timeline['Perubahan_Saham'] < 0]['Perubahan_Saham'].sum()
                    st.metric("Total Penjualan", format_lembar(abs(total_jual)))
                
                with col_metric4:
                    avg_price = df_timeline['Close_Price'].mean()
                    st.metric("Rata-rata Harga", format_rupiah(avg_price))
                
                # Tampilkan data detail
                with st.expander("📋 Lihat Detail Transaksi"):
                    df_detail_timeline = df_timeline[['Tanggal_Data', 'Aksi', 'Perubahan_Saham', 
                                                      'Jumlah Saham (Curr)', 'Close_Price', 'Estimasi_Nilai']].copy()
                    df_detail_timeline['Perubahan_Saham'] = df_detail_timeline['Perubahan_Saham'].apply(format_lembar)
                    df_detail_timeline['Jumlah Saham (Curr)'] = df_detail_timeline['Jumlah Saham (Curr)'].apply(format_lembar)
                    df_detail_timeline['Close_Price'] = df_detail_timeline['Close_Price'].apply(format_rupiah)
                    df_detail_timeline['Estimasi_Nilai'] = df_detail_timeline['Estimasi_Nilai'].apply(format_rupiah)
                    
                    st.dataframe(
                        df_detail_timeline.sort_values('Tanggal_Data', ascending=False),
                        use_container_width=True,
                        hide_index=True
                    )
            else:
                st.info(f"Tidak ada data historis untuk {selected_entity} pada {selected_stock}")
    else:
        st.info("Belum ditemukan akumulasi awal dengan parameter ini. Coba turunkan threshold minimal beli.")

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
