import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# =============================================================================
# KONFIGURASI: LINK GOOGLE DRIVE (PUBLIC SHARING)
# =============================================================================
# Cara setting: File di GDrive -> Share -> "Anyone with link" -> Copy link ID
# https://drive.google.com/uc?id=FILE_ID

FILE_IDS = {
    'harian': '1t_wCljhepGBqZVrvleuZKldomQKop9DY',      # Kompilasi_Data_1Tahun
    'ksei': '1eTUIC120SHTCzvBk77Q87w0X56F2HkWz',        # KSEI_Shareholder_Processed
    'master_5': '1mS7Xp_PMqFnLTikU7giDZ42mqcbsiYvx'     # MASTER_DATABASE_5persen
}

# =============================================================================
# CACHE DATA LOADING (KINERJA + HEMAT BANDWIDTH)
# =============================================================================
@st.cache_data(ttl=3600)  # Refresh setiap 1 jam
def load_harian():
    """Load data harian (Kompilasi_Data_1Tahun)"""
    url = f"https://drive.google.com/uc?id={FILE_IDS['harian']}"
    df = pd.read_csv(url)
    
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

@st.cache_data(ttl=86400)  # Refresh 24 jam (KSEI bulanan)
def load_ksei():
    """Load data KSEI bulanan (sudah diproses)"""
    url = f"https://drive.google.com/uc?id={FILE_IDS['ksei']}"
    df = pd.read_csv(url)
    
    # Parsing tanggal
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    
    # Konversi numerik untuk kolom perubahan
    for col in df.columns:
        if 'Chg' in col or 'Vol' in col or 'Val' in col or col in ['Price', 'Avg_Price']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

@st.cache_data(ttl=3600)
def load_master_5():
    """LOAD MASTER DATABASE 5% - SENJATA UTAMA DETEKSI AKUMULASI AWAL"""
    url = f"https://drive.google.com/uc?id={FILE_IDS['master_5']}"
    df = pd.read_csv(url)
    
    # Parsing tanggal
    df['Tanggal_Data'] = pd.to_datetime(df['Tanggal_Data'], errors='coerce')
    df = df.dropna(subset=['Tanggal_Data'])
    
    # Konversi numerik
    df['Jumlah Saham (Prev)'] = pd.to_numeric(df['Jumlah Saham (Prev)'], errors='coerce')
    df['Jumlah Saham (Curr)'] = pd.to_numeric(df['Jumlah Saham (Curr)'], errors='coerce')
    df['Close_Price'] = pd.to_numeric(df['Close_Price'], errors='coerce')
    
    # Hitung perubahan saham
    df['Perubahan_Saham'] = df['Jumlah Saham (Curr)'] - df['Jumlah Saham (Prev)']
    df['Perubahan_Persen'] = (df['Perubahan_Saham'] / df['Jumlah Saham (Prev)']) * 100
    
    # Klasifikasi aksi
    df['Aksi'] = 'Tahan'
    df.loc[df['Perubahan_Saham'] > 0, 'Aksi'] = 'Beli'
    df.loc[df['Perubahan_Saham'] < 0, 'Aksi'] = 'Jual'
    
    # Estimate nilai transaksi
    df['Estimasi_Nilai'] = df['Perubahan_Saham'] * df['Close_Price']
    
    return df

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
# TAB 3: AKUMULASI AWAL DARI DATA 5% (SENJATA ANDALAN)
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
        min_beli = st.number_input("Minimal Beli (lembar)", min_value=1000, value=100000, step=10000, help="Filter pembelian minimal")
    with col2:
        max_saham_beredar = st.number_input("Maksimal Kepemilikan (%)", min_value=0.1, max_value=100.0, value=10.0, step=0.5, help="Batas atas % kepemilikan (bandar awal < 10%)")
    with col3:
        lookback_days = st.number_input("Lookback (hari)", min_value=7, max_value=90, value=30, help="Dalam X hari terakhir")
    
    # Hitung total saham beredar per kode
    # Asumsi: Total saham bisa diambil dari data KSEI atau dari harga * free float? 
    # Untuk sederhana, kita pakai estimasi: Close_Price * something.
    # TAPI untuk deteksi akumulasi awal, kita fokus ke PEMBELI BARU atau PENAMBAHAN SIGNIFIKAN.
    
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
    
    # Filter: maksimal kepemilikan (estimasi)
    # Kita tidak punya data total saham beredar, jadi kita abaikan atau bisa merge dari data lain
    # Sementara skip filter max_saham_beredar, fokus ke akumulasi besar
    
    # 4. Skor Akumulasi Awal
    df_akumulasi['Skor_Akumulasi'] = (
        np.log1p(df_akumulasi['Total_Beli_Lembar']) * 0.5 +
        np.log1p(df_akumulasi['Frekuensi_Transaksi']) * 0.3 +
        (1 - (df_akumulasi['Tgl_Terakhir'] - df_akumulasi['Tgl_Pertama']).dt.days / lookback_days) * 0.2
    )
    
    df_akumulasi = df_akumulasi.sort_values('Skor_Akumulasi', ascending=False)
    
    # TAMPILKAN HASIL
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
        
        st.dataframe(
            df_akumulasi[['Kode Efek', 'Nama Pemegang Saham', 'Nama Rekening', 'Sector',
                         'Total_Beli_Lembar', 'Total_Nilai_Rp', 'Frekuensi_Transaksi',
                         'Tgl_Pertama', 'Tgl_Terakhir', 'Skor_Akumulasi']].head(50),
            column_config={
                'Total_Beli_Lembar': st.column_config.NumberColumn(format="%d", help="Total lembar dibeli"),
                'Total_Nilai_Rp': st.column_config.NumberColumn(format="Rp %d"),
                'Tgl_Pertama': st.column_config.DateColumn(format="DD-MM-YY"),
                'Tgl_Terakhir': st.column_config.DateColumn(format="DD-MM-YY"),
                'Skor_Akumulasi': st.column_config.NumberColumn(format="%.2f")
            },
            use_container_width=True,
            hide_index=True
        )
        
        # Chart: Top 10 akumulator
        st.subheader("💰 Top 10 Entitas Akumulator Terbesar (Nilai Rp)")
        top10 = df_akumulasi.nlargest(10, 'Total_Nilai_Rp').copy()
        top10['Label'] = top10['Nama Pemegang Saham'].str[:30] + '...'
        
        fig = px.bar(
            top10,
            x='Total_Nilai_Rp',
            y='Label',
            color='Sector',
            orientation='h',
            title="Total Nilai Pembelian (Estimasi)",
            labels={'Total_Nilai_Rp': 'Nilai (Rp)', 'Label': 'Pemegang Saham'}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Analisis per saham
        st.subheader("🔍 Detail Saham Terakumulasi")
        kode_5 = st.selectbox("Pilih Kode Efek", df_akumulasi['Kode Efek'].unique())
        
        if kode_5:
            df_detail_5 = df_master_filtered[
                (df_master_filtered['Kode Efek'] == kode_5) &
                (df_master_filtered['Aksi'] == 'Beli')
            ].sort_values('Tanggal_Data', ascending=False)
            
            st.dataframe(
                df_detail_5[['Tanggal_Data', 'Nama Pemegang Saham', 'Nama Pemegang Rekening Efek',
                           'Perubahan_Saham', 'Close_Price', 'Estimasi_Nilai']],
                column_config={
                    'Perubahan_Saham': st.column_config.NumberColumn(format="%d"),
                    'Close_Price': st.column_config.NumberColumn(format="Rp %d"),
                    'Estimasi_Nilai': st.column_config.NumberColumn(format="Rp %d")
                },
                use_container_width=True,
                hide_index=True
            )
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
