"""
================================================================================
🦅 BANDARMOLOGI ULTIMATE - X-RAY EDITION
================================================================================
Fitur:
1. Deep Unmasking (Mendeteksi Real Owner dibalik Nominee)
2. Pledge/Repo Hunter (Mendeteksi saham yang digadaikan)
3. Smart Grouping (Menggabungkan kepemilikan yang terpecah di banyak sekuritas)
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
from thefuzz import fuzz # Library untuk pencocokan teks mirip

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. KONFIGURASI HALAMAN & CSS
# ==============================================================================
st.set_page_config(
    layout="wide",
    page_title="Bandarmologi X-Ray",
    page_icon="🦅",
    initial_sidebar_state="expanded"
)

# Custom CSS biar tampilan sangar (Dark Mode Friendly)
st.markdown("""
<style>
    .metric-card {background-color: #1E1E1E; padding: 15px; border-radius: 10px; border: 1px solid #333;}
    .stTabs [data-baseweb="tab-list"] {gap: 10px;}
    .stTabs [data-baseweb="tab"] {height: 50px; white-space: pre-wrap; background-color: #0E1117; border-radius: 5px;}
    .stTabs [aria-selected="true"] {background-color: #262730; color: #4CAF50;}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. INTELLIGENCE ENGINE (OTAK KLASIFIKASI)
# ==============================================================================

class XRayEngine:
    """Mesin forensik untuk membedah data KSEI"""
    
    # Keyword untuk mendeteksi akun gadai/repo
    PLEDGE_KEYWORDS = ['PLEDGE', 'REPO', 'JAMINAN', 'AGUNAN', 'COLLATERAL', 'LOCKED', 'LENDING']
    
    # Keyword untuk mendeteksi Institusi Keuangan (Nominee)
    NOMINEE_KEYWORDS = [
        'CITIBANK', 'DEUTSCHE BANK', 'STANDARD CHARTERED', 'HSBC', 'DBS BANK', 
        'BUT.', 'BANK OF SINGAPORE', 'UBS AG', 'MORGAN STANLEY', 'NOMURA', 
        'JULIUS BAER', 'RAIFFEISEN', 'BNP PARIBAS', 'CGS-CIMB', 'UOB KAY HIAN',
        'MAYBANK', 'CLSA', 'CREDIT SUISSE', 'STATE STREET', 'JPMCB', 'JPMORGAN', 'SUMITOMO'
    ]

    @staticmethod
    def clean_name(text):
        """Membersihkan nama PT, TBK, dll untuk perbandingan nama"""
        if pd.isna(text): return ""
        text = str(text).upper()
        remove_list = [
            r'\bPT\.?\b', r'\bTBK\.?\b', r'\bLTD\.?\b', r'\bINC\.?\b', 
            r'\bDRS\.?\b', r'\bIR\.?\b', r'\bSH\.?\b', r'\bMBA\b'
        ]
        for pattern in remove_list:
            text = re.sub(pattern, '', text)
        return " ".join(text.split())

    @staticmethod
    def classify_row(row):
        """
        Logika Utama Klasifikasi per Baris Data
        Output: (REAL_OWNER, HOLDING_TYPE, STATUS)
        """
        holder = str(row['Nama Pemegang Saham']).upper()
        # Handle jika Nama Rekening Efek kosong (format lama)
        account = str(row['Nama Rekening Efek']).upper() if pd.notna(row.get('Nama Rekening Efek')) else ""
        
        # --- LEVEL 1: DETEKSI PLEDGE/REPO ---
        is_pledge = False
        status = "NORMAL"
        
        # Cek apakah ada kata kunci gadai di nama rekening ATAU nama pemegang
        if any(k in account for k in XRayEngine.PLEDGE_KEYWORDS) or any(k in holder for k in XRayEngine.PLEDGE_KEYWORDS):
            is_pledge = True
            status = "⚠️ PLEDGE / REPO"
        elif 'MARGIN' in account:
            status = "MARGIN"

        # --- LEVEL 2: EKSTRAKSI REAL OWNER (UNMASKING) ---
        
        real_owner = holder # Default: Pemilik adalah nama pemegang
        holding_type = "DIRECT" # Default: Kepemilikan langsung
        
        # Cek apakah Holder adalah Bank/Nominee?
        is_nominee_holder = any(k in holder for k in XRayEngine.NOMINEE_KEYWORDS)
        
        # Regex Patterns untuk menambang nama asli dari string rekening yang ruwet
        # Contoh: "CITIBANK NA S/A PT ADARO..." -> ambil "PT ADARO"
        # Urutan pattern penting (dari yang paling spesifik)
        extraction_patterns = [
            r'OBO\s+([A-Z0-9\.\s]+)',           # ... OBO PT MAJU ...
            r'QQ\s+([A-Z0-9\.\s]+)',            # ... QQ BAPAK BUDI ...
            r'S/A\s+([A-Z0-9\.\s]+)',           # ... S/A GLOBAL FUND ...
            r'A/C\s+([A-Z0-9\.\s]+)',           # ... A/C CLIENT 123 ...
            r'RE[:\-]\s*([A-Z0-9\.\s]+)',       # ... RE-PT ANGIN RIBUT ...
            r'BENEFICIARY\s+([A-Z0-9\.\s]+)',   # ... BENEFICIARY MR X ...
            r'CLIENT\s+([A-Z0-9\.\s]+)'         # ... CLIENT PT ABC ...
        ]
        
        extracted_name = None
        
        # Coba ekstrak nama dari rekening jika ada
        if account:
            for pattern in extraction_patterns:
                match = re.search(pattern, account)
                if match:
                    # Ambil hasil capture group 1, lalu bersihkan sampah di belakangnya
                    candidate = match.group(1).split('-')[0].strip() # Buang strip pemisah
                    candidate = candidate.split('(')[0].strip()      # Buang kurung
                    if len(candidate) > 3: # Validasi minimal panjang nama
                        extracted_name = candidate
                        break
        
        # --- LEVEL 3: PENENTUAN FINAL ---
        
        if is_pledge and extracted_name:
            # Kasus: UOB (PLEDGE) OBO PT ASERRA -> Real Owner: ASERRA
            real_owner = extracted_name
            holding_type = "INDIRECT (REPO)"
            
        elif extracted_name:
            # Bandingkan nama Holder vs Extracted
            similarity = fuzz.token_sort_ratio(XRayEngine.clean_name(holder), XRayEngine.clean_name(extracted_name))
            
            if similarity > 70:
                # Kasus: ADARO vs CITIBANK S/A PT ADARO (Mirip) -> Self Custody
                real_owner = extracted_name # Pakai nama yang di rekening biasanya lebih lengkap
                holding_type = "CUSTODY (SELF)"
            else:
                # Kasus: BANK JULIUS BAER S/A HANNAWELL (Beda Jauh) -> Nominee
                real_owner = extracted_name
                holding_type = "NOMINEE / FUND"
        
        elif is_nominee_holder:
            # Holder bank, tapi tidak ada info S/A di rekening -> Asumsikan Omnibus/Unknown
            holding_type = "OMNIBUS / BANK"
            
        else:
            # Tidak ada tanda-tanda nominee
            holding_type = "DIRECT"

        # Pembersihan Akhir Nama Real Owner
        # Buang kata-kata sisa seperti "LIMITED", "PTE LTD" biar grouping enak (Opsional, di sini kita biarkan biar akurat)
        
        return pd.Series([real_owner, holding_type, status])

# ==============================================================================
# 3. DATA LOADER (GOOGLE DRIVE)
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
        st.error(f"Error Auth Google: {e}")
        return None

@st.cache_data(ttl=3600)
def load_and_process_data():
    """Load CSV, Clean, and Apply X-Ray Engine"""
    try:
        # 1. Load Data
        service = get_gdrive_service()
        if not service: return pd.DataFrame()
        
        FOLDER_ID = st.secrets["gdrive"]["folder_id"]
        FILENAME = "MASTER_DATABASE_5persen.csv"
        
        # Search & Download
        results = service.files().list(q=f"name = '{FILENAME}' and '{FOLDER_ID}' in parents and trashed = false", fields="files(id)").execute()
        items = results.get('files', [])
        if not items: return pd.DataFrame()
        
        request = service.files().get_media(fileId=items[0]['id'])
        file_stream = io.BytesIO()
        downloader = MediaIoBaseDownload(file_stream, request)
        done = False
        while done is False: status, done = downloader.next_chunk()
        file_stream.seek(0)
        
        # 2. Read CSV
        df = pd.read_csv(file_stream, dtype={'Kode Efek': str})
        df['Tanggal_Data'] = pd.to_datetime(df['Tanggal_Data'])
        
        # 3. Basic Cleaning
        if 'Nama Rekening Efek' not in df.columns:
            df['Nama Rekening Efek'] = '-' # Handle format lama
        df['Nama Rekening Efek'] = df['Nama Rekening Efek'].fillna('-')
        
        for col in ['Jumlah Saham (Prev)', 'Jumlah Saham (Curr)']:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            
        df['Net_Flow'] = df['Jumlah Saham (Curr)'] - df['Jumlah Saham (Prev)']

        # 4. APPLY X-RAY ENGINE (The Heavy Lifting)
        # Kita pakai apply per baris. Agak berat tapi worth it untuk akurasi.
        # Untuk performa, kita hanya apply ke pasangan (Holder, Account) yang unik, lalu di-merge.
        
        unique_pairs = df[['Nama Pemegang Saham', 'Nama Rekening Efek']].drop_duplicates()
        
        # Jalankan forensik pada data unik
        print("🔍 Menjalankan Analisa Forensik pada Pasangan Unik...")
        unique_pairs[['REAL_OWNER', 'HOLDING_TYPE', 'ACCOUNT_STATUS']] = unique_pairs.apply(XRayEngine.classify_row, axis=1)
        
        # Gabungkan kembali ke data utama (VLOOKUP style)
        df_enriched = pd.merge(df, unique_pairs, on=['Nama Pemegang Saham', 'Nama Rekening Efek'], how='left')
        
        return df_enriched

    except Exception as e:
        st.error(f"System Error: {e}")
        return pd.DataFrame()

# ==============================================================================
# 4. DASHBOARD UI
# ==============================================================================

# --- LOAD DATA ---
with st.spinner('🚀 Menghubungkan ke Satelit KSEI & Menjalankan Analisa Forensik...'):
    df = load_and_process_data()

if df.empty:
    st.error("Gagal memuat data. Cek koneksi atau file di Drive.")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.title("🦅 X-RAY FILTER")
    
    # 1. Filter Saham
    all_stocks = sorted(df['Kode Efek'].unique())
    sel_stock = st.multiselect("Kode Saham", all_stocks, default=all_stocks[:1] if all_stocks else None)
    
    # 2. Filter Tanggal
    min_d, max_d = df['Tanggal_Data'].min().date(), df['Tanggal_Data'].max().date()
    sel_date = st.date_input("Periode", [min_d, max_d])
    
    # 3. Filter Intelligence
    st.divider()
    st.subheader("🕵️ Forensik Filter")
    show_pledge_only = st.checkbox("Hanya Tampilkan REPO/GADAI ⚠️")
    show_new_entry = st.checkbox("Hanya Pemain BARU (New Entry)")

# --- FILTERING DATA ---
df_view = df.copy()

if sel_stock:
    df_view = df_view[df_view['Kode Efek'].isin(sel_stock)]

if len(sel_date) == 2:
    df_view = df_view[
        (df_view['Tanggal_Data'].dt.date >= sel_date[0]) & 
        (df_view['Tanggal_Data'].dt.date <= sel_date[1])
    ]

if show_pledge_only:
    df_view = df_view[df_view['ACCOUNT_STATUS'].str.contains("PLEDGE|REPO")]

if show_new_entry:
    df_view = df_view[(df_view['Jumlah Saham (Prev)'] == 0) & (df_view['Jumlah Saham (Curr)'] > 0)]

# ==============================================================================
# 5. DASHBOARD TABS
# ==============================================================================

st.title(f"Bandarmologi X-Ray: {', '.join(sel_stock) if sel_stock else 'ALL MARKET'}")

tab1, tab2, tab3 = st.tabs(["👑 ULTIMATE HOLDERS", "⚠️ REPO MONITOR", "🧬 DEEP FLOW ANALYSIS"])

# --- TAB 1: ULTIMATE HOLDERS (The Real Boss) ---
with tab1:
    st.markdown("### Siapa Pemegang Asli di Balik Layar?")
    st.caption("Data ini sudah dibersihkan dari akun Nominee/Bank Kustodian. Kepemilikan digabungkan berdasarkan Real Owner.")
    
    # Ambil data tanggal terakhir
    last_date = df_view['Tanggal_Data'].max()
    df_last = df_view[df_view['Tanggal_Data'] == last_date]
    
    if not df_last.empty:
        # Group by REAL_OWNER (Bukan Nama Pemegang lagi)
        uh_stats = df_last.groupby('REAL_OWNER').agg({
            'Jumlah Saham (Curr)': 'sum',
            'Nama Pemegang Saham': 'nunique', # Berapa akun nominee dia pakai
            'Kode Efek': 'nunique',
            'ACCOUNT_STATUS': lambda x: '⚠️ ADA REPO' if any('PLEDGE' in s for s in x) else 'Clean'
        }).sort_values('Jumlah Saham (Curr)', ascending=False).reset_index()
        
        col_l, col_r = st.columns([2, 1])
        
        with col_l:
            st.dataframe(
                uh_stats.style.format({'Jumlah Saham (Curr)': '{:,.0f}'}),
                use_container_width=True,
                height=600,
                column_config={
                    "REAL_OWNER": "Beneficial Owner (Asli)",
                    "Nama Pemegang Saham": "Jml Akun Nominee",
                    "ACCOUNT_STATUS": "Status Risiko"
                }
            )
            
        with col_r:
            # Breakdown Pie Chart untuk Top 1 Holder
            if not uh_stats.empty:
                top_boss = uh_stats.iloc[0]['REAL_OWNER']
                st.subheader(f"Portofolio: {top_boss}")
                
                df_boss = df_last[df_last['REAL_OWNER'] == top_boss]
                fig = px.pie(df_boss, values='Jumlah Saham (Curr)', names='HOLDING_TYPE', title=f"Gaya Simpan {top_boss}", hole=0.4)
                st.plotly_chart(fig, use_container_width=True)
                
                st.write("**Akun-akun yang digunakan:**")
                st.dataframe(df_boss[['Nama Pemegang Saham', 'Nama Rekening Efek']].drop_duplicates(), hide_index=True)

    else:
        st.info("Tidak ada data di tanggal terakhir.")

# --- TAB 2: REPO MONITOR (The Risk) ---
with tab2:
    st.markdown("### ⚠️ Radar Saham Digadaikan (Forced Sell Risk)")
    
    # Filter Global Dataset untuk Pledge
    df_pledge = df_view[df_view['ACCOUNT_STATUS'].str.contains("PLEDGE|REPO")]
    
    if not df_pledge.empty:
        # Agregasi per Saham
        repo_summary = df_pledge.groupby(['Kode Efek', 'REAL_OWNER']).agg({
            'Jumlah Saham (Curr)': 'sum',
            'Nama Rekening Efek': 'count'
        }).reset_index().sort_values('Jumlah Saham (Curr)', ascending=False)
        
        # Metric
        total_repo_vol = df_pledge['Jumlah Saham (Curr)'].sum()
        c1, c2 = st.columns(2)
        c1.metric("Total Volume Saham Tergadai", f"{total_repo_vol:,.0f}")
        c2.metric("Jumlah Pihak Terlibat", f"{df_pledge['REAL_OWNER'].nunique()}")
        
        # Chart
        st.subheader("Peta Persebaran Barang Repo")
        fig_repo = px.treemap(df_pledge, path=['Kode Efek', 'REAL_OWNER', 'Nama Pemegang Saham'], values='Jumlah Saham (Curr)',
                             color='Kode Efek', title="Siapa yang Menggadaikan Saham Apa?")
        st.plotly_chart(fig_repo, use_container_width=True)
        
        # Detail Table
        st.subheader("Rincian Transaksi Repo")
        st.dataframe(
            df_pledge[['Tanggal_Data', 'Kode Efek', 'REAL_OWNER', 'Nama Pemegang Saham', 'Nama Rekening Efek', 'Jumlah Saham (Curr)']]
            .sort_values('Tanggal_Data', ascending=False)
            .style.format({'Jumlah Saham (Curr)': '{:,.0f}'}),
            use_container_width=True
        )
    else:
        st.success("✅ AMAN! Tidak ditemukan indikasi Repo/Gadai pada filter saat ini.")

# --- TAB 3: DEEP FLOW (Smart Money) ---
with tab3:
    st.markdown("### 🧬 Analisa Aliran Barang (Accumulation vs Distribution)")
    
    # Grouping berdasarkan REAL OWNER untuk melihat Net Flow
    # Kita ingin lihat: Hari ini SIAPA yang nampung barang?
    
    if not df_view.empty:
        # Agregasi Flow per Real Owner dalam periode terpilih
        flow_stats = df_view.groupby('REAL_OWNER')['Net_Flow'].sum().reset_index()
        flow_stats = flow_stats.sort_values('Net_Flow', ascending=False)
        
        # Top Accumulators (Green)
        top_buy = flow_stats.head(10)
        # Top Distributors (Red)
        top_sell = flow_stats.tail(10).sort_values('Net_Flow', ascending=True) # Biar urut dari minus terbesar
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("🟢 Top Accumulators (Big Buyer)")
            fig_buy = px.bar(top_buy, x='Net_Flow', y='REAL_OWNER', orientation='h', color_discrete_sequence=['#00CC96'], text_auto='.2s')
            fig_buy.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_buy, use_container_width=True)
            
        with c2:
            st.subheader("🔴 Top Distributors (Big Seller)")
            # Balik biar grafiknya enak
            fig_sell = px.bar(top_sell, x='Net_Flow', y='REAL_OWNER', orientation='h', color_discrete_sequence=['#EF553B'], text_auto='.2s')
            fig_sell.update_layout(yaxis={'categoryorder':'total descending'})
            st.plotly_chart(fig_sell, use_container_width=True)
            
        # Chart History Pemain Tertentu
        st.divider()
        st.subheader("📈 Lacak Jejak Pemain")
        
        # Pilih pemain dari list yang ada di view sekarang
        players = sorted(df_view['REAL_OWNER'].unique())
        target_player = st.selectbox("Pilih Real Owner:", players)
        
        if target_player:
            df_track = df_view[df_view['REAL_OWNER'] == target_player]
            fig_track = px.line(df_track, x='Tanggal_Data', y='Jumlah Saham (Curr)', color='Kode Efek', markers=True, title=f"Pergerakan Barang: {target_player}")
            st.plotly_chart(fig_track, use_container_width=True)

# --- FOOTER ---
st.divider()
st.caption(f"Bandarmologi X-Ray Engine v3.0 | Processed {len(df):,} Rows | Forensics Active")
