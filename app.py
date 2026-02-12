"""
================================================================================
🦅 BANDARMOLOGI ULTIMATE - X-RAY EDITION
================================================================================
Engine: v3.0 (Forensic Regex + Logic Priority Fix)
Features:
- Deep Unmasking (Detects Real Owner behind Nominees)
- Pledge/Repo Hunter (Detects collateral accounts)
- Smart Grouping (Consolidates ownership)
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
import io
import warnings
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. KONFIGURASI HALAMAN
# ==============================================================================
st.set_page_config(
    layout="wide",
    page_title="Bandarmologi X-Ray",
    page_icon="🦅",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Look
st.markdown("""
<style>
    .metric-card {background-color: #1E1E1E; padding: 15px; border-radius: 10px; border: 1px solid #333;}
    div[data-testid="stMetricValue"] {font-size: 1.4rem !important;}
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {height: 40px; white-space: pre-wrap; background-color: #0E1117; border-radius: 5px;}
    .stTabs [aria-selected="true"] {background-color: #262730; color: #00CC96; border-bottom: 2px solid #00CC96;}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. INTELLIGENCE ENGINE (OTAK FORENSIK)
# ==============================================================================

class BandarXRay:
    """Mesin Forensik untuk Membedah Data KSEI"""

    # --- A. DEFINISI POLA REGEX BANK (PRIORITAS TINGGI) ---
    PATTERNS_NOMINEE = [
        # 1. HSBC Variants
        (r'(?:HSBC|HPTS|HSTSRBACT|HSSTRBACT|HASDBACR|HDSIVBICSI|HINSVBECS).*?(?:PRIVATE BANKING|FUND SVS|CLIENT|A/C)\s+(?:DIVISION-?)?\s*(?:PT\.?)?\s*(.+)', 'HSBC'),
        
        # 2. UBS Variants
        (r'(?:UBS AG|USBTRS|U20B9S1|U20B2S3|UINBVSE|DINBVSE|UINBVS).*?(?:S/A|A/C|BRANCH|TR AC CL|SEPNOTRSE)\s*(?:PT\.?)?\s*(.+)', 'UBS AG'),
        
        # 3. Deutsche Bank Variants
        (r'(?:DB AG|DEUTSCHE BANK|D21B4|D20B4|D22B5S9).*?(?:A/C|S/A|CLT)\s*(?:PT\.?)?\s*(.+)', 'Deutsche Bank'),
        
        # 4. Citibank Variants
        (r'(?:CITIBANK|CITI).*?(?:S/A|CBHK|PBGSG)\s*(?:PT\.?)?\s*(.+)', 'Citibank'),
        
        # 5. Standard Chartered Variants
        (r'(?:STANDARD CHARTERED|SCB).*?(?:S/A|A/C|CUSTODY)\s*(?:PT\.?)?\s*(.+)', 'Standard Chartered'),
        
        # 6. Bank of Singapore / JPMorgan
        (r'(?:BOS LTD|BANK OF SINGAPORE|BINOVSE).*?(?:S/A|A/C)\s*(?:PT\.?)?\s*(.+)', 'Bank of Singapore'),
        (r'(?:JPMCB|JPMORGAN|JINPVMECSBT).*?(?:RE-|NA RE-)\s*(?:PT\.?)?\s*(.+)', 'JPMorgan'),
        
        # 7. General Pattern (S/A, QQ, OBO) - Catch All
        (r'.*?(?:S/A|QQ|OBO|BENEFICIARY)\s+(?:PT\.?)?\s*(.+)', 'Nominee General'),
        (r'.*?(?:A/C CLIENT|CLIENT A/C|CLIENT)\s+(?:PT\.?)?\s*(.+)', 'Nominee General')
    ]

    # --- B. KEYWORDS ---
    PLEDGE_KEYWORDS = ['PLEDGE', 'REPO', 'JAMINAN', 'AGUNAN', 'COLLATERAL', 'LOCKED', 'LENDING', 'MARGIN']
    
    DIRECT_INDICATORS = [
        ' PT', 'PT ', 'PT.', ' TBK', 'TBK ', 'LTD', 'INC', 'CORP', 'CO.', 'COMPANY',
        'DRS.', 'DR.', 'IR.', 'H.', 'HJ.', 'YAYASAN', 'DANA PENSIUN'
    ]

    @staticmethod
    def clean_name(text):
        """Membersihkan sampah referensi angka/kode di belakang nama"""
        if pd.isna(text): return "-"
        text = str(text).strip()
        
        # Buang referensi angka di belakang (e.g., "- 2091145195" atau "(ID1234)")
        text = re.sub(r'\s*[-–—]\s*\d+.*$', '', text) 
        text = re.sub(r'\s*\([A-Z0-9\s]+\)$', '', text)
        
        # Buang gelar umum untuk standarisasi (Optional, hati-hati over-clean)
        # text = re.sub(r'\b(PT|TBK|LTD)\b', '', text, flags=re.IGNORECASE)
        
        return text.strip().upper()

    @staticmethod
    def classify_account(row):
        """
        Logika Utama: Menerima baris data, mengembalikan (REAL_OWNER, TYPE, STATUS)
        """
        holder = str(row['Nama Pemegang Saham']).upper()
        # Handle account kosong
        account = str(row['Nama Rekening Efek']).upper() if pd.notna(row.get('Nama Rekening Efek')) else ""
        
        status = "NORMAL"
        real_owner = holder # Default: Pemilik adalah nama pemegang
        holding_type = "DIRECT"
        
        # ---------------------------------------------------------
        # 1. DETEKSI STATUS (PLEDGE/REPO) - Check First!
        # ---------------------------------------------------------
        if any(k in account for k in BandarXRay.PLEDGE_KEYWORDS):
            status = "⚠️ PLEDGE / REPO"
            # Jika pledge, biasanya nama asli ada di belakang kata kunci OBO/QQ
            # Tapi kita biarkan logic unmasking di bawah yang menangani ekstraksi namanya
        
        # ---------------------------------------------------------
        # 2. LOGIKA UNMASKING (URUTAN KRUSIAL!)
        # ---------------------------------------------------------
        
        # STEP A: Cek Pola Bank/Nominee Spesifik (Regex)
        # Ini harus duluan sebelum cek "Direct". 
        # Contoh: "CITIBANK S/A PT ADARO". Ada "PT", tapi ini NOMINEE.
        
        found_nominee = False
        if account and len(account) > 3:
            for pattern, source_type in BandarXRay.PATTERNS_NOMINEE:
                match = re.search(pattern, account)
                if match:
                    extracted = match.group(1)
                    real_owner = BandarXRay.clean_name(extracted)
                    holding_type = f"NOMINEE ({source_type})"
                    found_nominee = True
                    break
        
        # STEP B: Jika Tidak Ketemu Pola Nominee, Baru Cek Direct
        if not found_nominee:
            # Jika nama rekening mirip dengan nama pemegang -> Direct
            # Atau jika nama rekening mengandung indikator perusahaan/perorangan tanpa keyword bank
            is_direct_indicator = any(k in account for k in BandarXRay.DIRECT_INDICATORS)
            
            # Simple similarity check
            if holder in account or account in holder:
                holding_type = "DIRECT"
                real_owner = BandarXRay.clean_name(holder)
            elif is_direct_indicator:
                # Kemungkinan nama variasi
                holding_type = "DIRECT (VARIANT)"
                real_owner = BandarXRay.clean_name(account)
            else:
                holding_type = "DIRECT / UNKNOWN"
                real_owner = BandarXRay.clean_name(holder)

        # ---------------------------------------------------------
        # 3. KOREKSI KHUSUS (OVERRIDE)
        # ---------------------------------------------------------
        # Jika status Pledge, tandai tipe holdingnya
        if status != "NORMAL":
            holding_type = "INDIRECT (COLLATERAL)"
            
        return pd.Series([real_owner, holding_type, status])

# ==============================================================================
# 3. DATA LOADER & PROCESSING
# ==============================================================================

@st.cache_resource
def get_gdrive_service():
    try:
        if "gdrive_creds" in st.secrets:
            creds = service_account.Credentials.from_service_account_info(
                st.secrets["gdrive_creds"],
                scopes=['https://www.googleapis.com/auth/drive.readonly']
            )
            return build('drive', 'v3', credentials=creds)
        return None
    except Exception as e:
        st.error(f"Auth Error: {e}")
        return None

@st.cache_data(ttl=3600)
def load_data():
    service = get_gdrive_service()
    if not service: return pd.DataFrame()
    
    try:
        FOLDER_ID = st.secrets["gdrive"]["folder_id"]
        FILENAME = "MASTER_DATABASE_5persen.csv"
        
        results = service.files().list(q=f"name = '{FILENAME}' and '{FOLDER_ID}' in parents and trashed = false", fields="files(id)").execute()
        items = results.get('files', [])
        if not items: return pd.DataFrame()
        
        request = service.files().get_media(fileId=items[0]['id'])
        file_stream = io.BytesIO()
        downloader = MediaIoBaseDownload(file_stream, request)
        done = False
        while done is False: status, done = downloader.next_chunk()
        file_stream.seek(0)
        
        df = pd.read_csv(file_stream, dtype={'Kode Efek': str})
        df['Tanggal_Data'] = pd.to_datetime(df['Tanggal_Data'])
        
        # Cleaning Angka
        for col in ['Jumlah Saham (Prev)', 'Jumlah Saham (Curr)']:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            
        df['Net_Flow'] = df['Jumlah Saham (Curr)'] - df['Jumlah Saham (Prev)']
        
        # Handle kolom Nama Rekening Efek (Format lama mungkin tidak ada)
        if 'Nama Rekening Efek' not in df.columns:
            df['Nama Rekening Efek'] = '-'
        df['Nama Rekening Efek'] = df['Nama Rekening Efek'].fillna('-')
        
        return df
    except Exception as e:
        st.error(f"Load Error: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def process_data_forensics(df):
    """Menjalankan Logic BandarXRay pada data"""
    if df.empty: return df
    
    # Optimasi: Apply hanya pada kombinasi unik (Holder + Account)
    # Ini mempercepat proses 100x lipat dibanding apply per baris di jutaan data
    unique_pairs = df[['Nama Pemegang Saham', 'Nama Rekening Efek']].drop_duplicates()
    
    # Jalankan Klasifikasi
    unique_pairs[['REAL_OWNER', 'HOLDING_TYPE', 'ACCOUNT_STATUS']] = unique_pairs.apply(BandarXRay.classify_account, axis=1)
    
    # Merge kembali ke data utama
    df_merged = pd.merge(df, unique_pairs, on=['Nama Pemegang Saham', 'Nama Rekening Efek'], how='left')
    
    return df_merged

# ==============================================================================
# 4. DASHBOARD UI
# ==============================================================================

# --- LOAD & PROCESS ---
with st.spinner('Menghubungkan ke Database KSEI...'):
    df_raw = load_data()

if df_raw.empty:
    st.warning("Data tidak ditemukan atau GDrive belum dikonfigurasi.")
    st.stop()

with st.spinner('Menjalankan Analisa Forensik (Unmasking Nominees)...'):
    df = process_data_forensics(df_raw)

# --- SIDEBAR ---
with st.sidebar:
    st.title("🦅 X-RAY CONTROL")
    
    # Filter Saham
    all_stocks = sorted(df['Kode Efek'].unique())
    sel_stock = st.multiselect("Kode Saham", all_stocks, default=all_stocks[:1] if all_stocks else None)
    
    # Filter Tanggal
    min_d, max_d = df['Tanggal_Data'].min().date(), df['Tanggal_Data'].max().date()
    sel_date = st.date_input("Periode", [min_d, max_d])
    
    st.divider()
    
    # Filter Khusus
    st.subheader("Filter Lanjutan")
    show_pledge = st.checkbox("Tampilkan HANYA REPO/GADAI ⚠️")
    show_top_only = st.checkbox("Hanya Top 20 Holder")

# --- FILTERING DATA ---
df_view = df.copy()

if sel_stock:
    df_view = df_view[df_view['Kode Efek'].isin(sel_stock)]

if len(sel_date) == 2:
    df_view = df_view[
        (df_view['Tanggal_Data'].dt.date >= sel_date[0]) & 
        (df_view['Tanggal_Data'].dt.date <= sel_date[1])
    ]

if show_pledge:
    df_view = df_view[df_view['ACCOUNT_STATUS'].str.contains("PLEDGE|REPO")]

# ==============================================================================
# 5. TABS VISUALIZATION
# ==============================================================================

if not df_view.empty:
    st.title(f"Bandarmologi X-Ray: {', '.join(sel_stock) if sel_stock else 'ALL MARKET'}")
    
    # Hitung data tanggal terakhir untuk snapshot
    last_date = df_view['Tanggal_Data'].max()
    df_last = df_view[df_view['Tanggal_Data'] == last_date]

    tab1, tab2, tab3, tab4 = st.tabs([
        "👑 ULTIMATE HOLDER", 
        "⚠️ REPO MONITOR", 
        "🌊 FLOW ANALYSIS",
        "🕵️ NOMINEE MAPPING"
    ])

    # --- TAB 1: ULTIMATE HOLDER (THE GODZILLA VIEW) ---
    with tab1:
        st.markdown(f"### Peta Kepemilikan Asli (Per {last_date.date()})")
        st.caption("Data ini menggabungkan kepemilikan satu orang yang tersebar di banyak akun (Direct + Nominee).")
        
        # Group by REAL_OWNER
        uh_group = df_last.groupby('REAL_OWNER').agg({
            'Jumlah Saham (Curr)': 'sum',
            'Nama Pemegang Saham': 'nunique', # Jumlah Akun
            'HOLDING_TYPE': lambda x: ', '.join(x.unique())[:50] + '...' if len(str(x.unique())) > 50 else ', '.join(x.unique()),
            'ACCOUNT_STATUS': lambda x: '⚠️ ADA REPO' if any('PLEDGE' in s for s in x) else 'Clean'
        }).sort_values('Jumlah Saham (Curr)', ascending=False)
        
        if show_top_only:
            uh_group = uh_group.head(20)
            
        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.dataframe(
                uh_group.style.format({'Jumlah Saham (Curr)': '{:,.0f}'}),
                use_container_width=True,
                height=600,
                column_config={
                    "REAL_OWNER": "Beneficial Owner (Pemilik Asli)",
                    "Nama Pemegang Saham": "Jml Akun",
                    "HOLDING_TYPE": "Tipe Penyimpanan",
                    "ACCOUNT_STATUS": "Status Risiko"
                }
            )
        
        with c2:
            # Pie Chart Komposisi Top 5
            top5 = uh_group.head(5).reset_index()
            fig_pie = px.pie(top5, values='Jumlah Saham (Curr)', names='REAL_OWNER', title="Top 5 Penguasa Saham", hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Metric Total
            total_share = df_last['Jumlah Saham (Curr)'].sum()
            st.metric("Total Lembar Saham (>5%)", f"{total_share:,.0f}")

    # --- TAB 2: REPO MONITOR (RISK VIEW) ---
    with tab2:
        st.markdown("### ⚠️ Radar Saham Digadaikan (Forced Sell Risk)")
        st.caption("Mendeteksi akun dengan kata kunci: PLEDGE, REPO, COLLATERAL, JAMINAN, MARGIN.")
        
        # Ambil semua data pledge di periode ini (bukan cuma last date, biar kelihatan history)
        df_repo = df_view[df_view['ACCOUNT_STATUS'].str.contains("PLEDGE|REPO")]
        
        if not df_repo.empty:
            repo_last = df_repo[df_repo['Tanggal_Data'] == last_date]
            
            col_a, col_b = st.columns(2)
            col_a.metric("Total Saham Tergadai (Saat Ini)", f"{repo_last['Jumlah Saham (Curr)'].sum():,.0f}")
            col_b.metric("Jumlah Pihak Terlibat Repo", f"{repo_last['REAL_OWNER'].nunique()}")
            
            st.subheader("Daftar Akun Terindikasi Repo")
            st.dataframe(
                repo_last[['REAL_OWNER', 'Nama Pemegang Saham', 'Nama Rekening Efek', 'Jumlah Saham (Curr)']]
                .sort_values('Jumlah Saham (Curr)', ascending=False)
                .style.format({'Jumlah Saham (Curr)': '{:,.0f}'}),
                use_container_width=True
            )
            
            st.subheader("Historical Repo Trend")
            repo_trend = df_repo.groupby('Tanggal_Data')['Jumlah Saham (Curr)'].sum().reset_index()
            fig_repo = px.area(repo_trend, x='Tanggal_Data', y='Jumlah Saham (Curr)', title="Tren Volume Saham Digadaikan", color_discrete_sequence=['#FF4B4B'])
            st.plotly_chart(fig_repo, use_container_width=True)
            
        else:
            st.success("✅ AMAN! Tidak ditemukan indikasi Repo/Gadai pada data yang difilter.")

    # --- TAB 3: FLOW ANALYSIS (SMART MONEY) ---
    with tab3:
        st.markdown("### 🌊 Analisa Aliran Barang (Accumulation vs Distribution)")
        
        # Hitung Net Flow per REAL OWNER selama periode
        flow_stats = df_view.groupby('REAL_OWNER')['Net_Flow'].sum().reset_index()
        flow_stats = flow_stats[flow_stats['Net_Flow'] != 0].sort_values('Net_Flow', ascending=False)
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🟢 Top Accumulator (Buying)")
            st.dataframe(flow_stats.head(10).style.format({'Net_Flow': '{:,.0f}'}), use_container_width=True, hide_index=True)
        with c2:
            st.subheader("🔴 Top Distributor (Selling)")
            st.dataframe(flow_stats.tail(10).sort_values('Net_Flow', ascending=True).style.format({'Net_Flow': '{:,.0f}'}), use_container_width=True, hide_index=True)
            
        st.divider()
        st.subheader("📈 Lacak Pergerakan Pemain")
        players = sorted(df_view['REAL_OWNER'].unique())
        target = st.selectbox("Pilih Real Owner:", players)
        
        if target:
            track_df = df_view[df_view['REAL_OWNER'] == target]
            fig_track = px.line(track_df, x='Tanggal_Data', y='Jumlah Saham (Curr)', color='Kode Efek', markers=True, title=f"Trend Kepemilikan: {target}")
            st.plotly_chart(fig_track, use_container_width=True)

    # --- TAB 4: NOMINEE MAPPING (X-RAY) ---
    with tab4:
        st.markdown("### 🕵️ Bedah Nominee")
        st.caption("Melihat detail: Siapa memakai Bank apa?")
        
        # Filter hanya yang Nominee
        df_nom = df_last[df_last['HOLDING_TYPE'].str.contains("NOMINEE")]
        
        if not df_nom.empty:
            mapping = df_nom.groupby(['REAL_OWNER', 'Nama Pemegang Saham']).agg({
                'Nama Rekening Efek': 'first',
                'Jumlah Saham (Curr)': 'sum'
            }).reset_index().sort_values('REAL_OWNER')
            
            st.dataframe(mapping, use_container_width=True, column_config={
                "Nama Pemegang Saham": "Bank/Kustodian",
                "Nama Rekening Efek": "Detail Rekening (Bukti)",
                "REAL_OWNER": "Pemilik Asli"
            })
        else:
            st.info("Tidak ada kepemilikan Nominee di saham ini.")

else:
    st.info("Mohon pilih saham di Sidebar.")

# --- FOOTER ---
st.divider()
st.caption(f"Bandarmologi X-Ray Engine v3.0 | Data Processed: {len(df):,} Rows")
