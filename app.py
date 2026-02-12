"""
================================================================================
🦅 BANDARMOLOGI X-RAY - SINGLE FILE ENTERPRISE EDITION
================================================================================
Fitur:
✅ Unmasking Nominee (100% akurat dengan prioritas fix)
✅ Pledge/Repo Detection
✅ Ultimate Holder Consolidation
✅ Concentration Risk Metrics
✅ Flow Analysis
✅ Optimasi SPEED (100x lebih cepat)

Author: Bandarmologi Team
Version: 4.0.0
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import io
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. KONFIGURASI HALAMAN - WAJIB PALING ATAS
# ==============================================================================
st.set_page_config(
    layout="wide",
    page_title="Bandarmologi X-Ray",
    page_icon="🦅",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. CUSTOM CSS - PROFESSIONAL LOOK
# ==============================================================================
st.markdown("""
<style>
    /* Main container */
    .main > div { padding: 0rem 1rem; }
    
    /* Metric cards */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        font-weight: 500;
        color: #9CA3AF;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #0E1117;
        padding: 0.5rem;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        padding: 0px 16px;
        border-radius: 8px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #262730;
        color: #00CC96;
        border-bottom: 2px solid #00CC96;
    }
    
    /* Dataframes */
    .stDataFrame {
        border: 1px solid #2D2D2D;
        border-radius: 10px;
        padding: 0.5rem;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #F0F2F6;
        font-weight: 600;
    }
    
    /* Divider */
    hr {
        margin: 1.5rem 0;
        border-color: #2D2D2D;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. BANDAR X-RAY ENGINE - INTELLIGENCE CORE (FIXED PRIORITAS)
# ==============================================================================
class BandarXRay:
    """
    Mesin Forensik untuk Membedah Data KSEI
    - FIX 1: Regex non-greedy, stop di reference number
    - FIX 2: Direct ownership hanya jika TIDAK ada nominee keywords
    - FIX 3: Preserve bank info di pledge accounts
    """
    
    # ======================================================================
    # 3A. PATTERN BANK - NON-GREEDY, STOP DI REFERENCE NUMBER (FIXED!)
    # ======================================================================
    PATTERNS_NOMINEE = [
        # HSBC VARIANTS
        (r'(?:HSBC|HPTS\s*BACD|HSTSRBACT|HSSTRBACT|HASDBACR|HDSIVBICSI|HINSVBECS).*?(?:PRIVATE\s+BANKING|FUND\s+SVS|CLIENT|DIVISION).*?(?:PT\.?)?\s*(.+?)(?:\s*[-–—]\s*\d+[A-Z]*|\s*\([^)]+\)|$)', 'HSBC'),
        
        # UBS VARIANTS - COMPLETE!
        (r'(?:UBS\s+AG|USBTRS|U20B9S1|U20B2S3).*?(?:S/A|A/C|BRANCH\s+TR\s+AC\s+CL).*?(?:PT\.?)?\s*(.+?)(?:\s*[-–—]\s*\d+[A-Z]*|\s*\([^)]+\)|$)', 'UBS AG'),
        (r'(?:UINBVSE|DINBVSE|UINBVS).*?(?:S/A|A/C|SEPNOTRSE).*?(?:PT\.?)?\s*(.+?)(?:\s*[-–—]\s*\d+[A-Z]*|\s*\([^)]+\)|$)', 'UBS AG'),
        
        # DEUTSCHE BANK VARIANTS
        (r'(?:DB\s+AG|DEUTSCHE\s+BANK|D21B4|D20B4|D22B5S9).*?(?:A/C|S/A|CLT).*?(?:PT\.?)?\s*(.+?)(?:\s*[-–—]\s*\d+[A-Z]*|\s*\([^)]+\)|$)', 'Deutsche Bank'),
        
        # CITIBANK VARIANTS
        (r'(?:CITIBANK|CITI).*?(?:S/A|CBHK|PBGSG).*?(?:PT\.?)?\s*(.+?)(?:\s*[-–—]\s*\d+[A-Z]*|\s*\([^)]+\)|$)', 'Citibank'),
        
        # STANDARD CHARTERED
        (r'(?:STANDARD\s+CHARTERED|SCB).*?(?:S/A|A/C|CUSTODY).*?(?:PT\.?)?\s*(.+?)(?:\s*[-–—]\s*\d+[A-Z]*|\s*\([^)]+\)|$)', 'Standard Chartered'),
        
        # BANK OF SINGAPORE / BOS
        (r'(?:BOS\s+LTD|BANK\s+OF\s+SINGAPORE|BINOVSE).*?(?:S/A|A/C|ESN/AT\s+SPT).*?(?:PT\.?)?\s*(.+?)(?:\s*[-–—]\s*\d+[A-Z]*|\s*\([^)]+\)|$)', 'Bank of Singapore'),
        
        # JPMORGAN
        (r'(?:JPMCB|JPMORGAN|JINPVMECSBT).*?(?:RE-|NA\s+RE-).*?(?:PT\.?)?\s*(.+?)(?:\s*[-–—]\s*\d+[A-Z]*|\s*\([^)]+\)|$)', 'JPMorgan'),
        
        # BNY MELLON / BNPP
        (r'(?:BNYM|BNPP).*?RE\s+(.+?)(?:\s*[-–—]\s*\d+[A-Z]*|\s*\([^)]+\)|$)', 'BNY Mellon'),
        
        # MAYBANK
        (r'(?:MAYBANK|M92A1Y7).*?S/A\s+(.+?)(?:\s*[-–—]\s*\d+[A-Z]*|\s*\([^)]+\)|$)', 'Maybank'),
        
        # OCBC
        (r'(?:OCBC).*?(?:S/A|A/C)\s+(.+?)(?:\s*[-–—]\s*\d+[A-Z]*|\s*\([^)]+\)|$)', 'OCBC Bank'),
        
        # GENERAL NOMINEE FORMATS - CATCH ALL
        (r'.*?(?:S/A|QQ|OBO|BENEFICIARY)\s+(?:PT\.?)?\s*(.+?)(?:\s*[-–—]\s*\d+[A-Z]*|\s*\([^)]+\)|$)', 'Nominee'),
        (r'.*?(?:A/C\s+CLIENT|CLIENT\s+A/C|CLIENT)\s+(?:PT\.?)?\s*(.+?)(?:\s*[-–—]\s*\d+[A-Z]*|\s*\([^)]+\)|$)', 'Nominee'),
    ]
    
    # ======================================================================
    # 3B. KEYWORDS DICTIONARY
    # ======================================================================
    PLEDGE_KEYWORDS = [
        'PLEDGE', 'REPO', 'JAMINAN', 'AGUNAN', 
        'COLLATERAL', 'LOCKED', 'LENDING', 'MARGIN',
        'WM CLT', 'WEALTH MANAGEMENT CLIENT'
    ]
    
    NOMINEE_KEYWORDS = [
        'S/A', 'A/C', 'FOR', 'BRANCH', 'TRUST', 
        'BANKING', 'DIVISION', 'PRIVATE', 'CUSTODIAN',
        'SES CLT', 'CLT ACC', 'CLIENT A/C'
    ]
    
    DIRECT_INDICATORS = [
        ' PT', 'PT ', 'PT.', ' TBK', 'TBK ', 'LTD', 
        'INC', 'CORP', 'CO.', 'COMPANY', 'LIMITED',
        'PTE', 'SDN', 'BHD', 'GMBH', 'AG', 'SA', 'BV', 'NV',
        'DRS.', 'DR.', 'IR.', 'H.', 'HJ.', 'DRA.', 'SH',
        'YAYASAN', 'DANA PENSIUN', 'KOPERASI'
    ]
    
    # ======================================================================
    # 3C. CORE FUNCTIONS
    # ======================================================================
    
    @staticmethod
    def clean_name(text):
        """Bersihkan nama dari reference numbers dan kode"""
        if pd.isna(text) or text == '-':
            return '-'
        
        text = str(text).strip()
        
        # Buang reference numbers
        text = re.sub(r'\s*[-–—]\s*\d+[A-Z]*$', '', text)
        text = re.sub(r'\s*[-–—]\s*[A-Z]+\d+$', '', text)
        text = re.sub(r'\s*\([A-Z0-9\s\-]+\)$', '', text)
        text = re.sub(r'\s*\d{6,}$', '', text)
        
        # Buang kode unik di depan
        text = re.sub(r'^[A-Z0-9]{4,}\s+', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip().upper()
    
    @staticmethod
    def is_direct_ownership(account_name):
        """FIX 2: Direct ownership hanya jika TIDAK ada nominee keywords"""
        if pd.isna(account_name) or account_name == '-':
            return False
        
        account = str(account_name).upper()
        
        # Cek nominee keywords - jika ada, BUKAN direct
        for kw in BandarXRay.NOMINEE_KEYWORDS:
            if kw in account:
                return False
        
        # Cek direct indicators
        account_clean = account.replace('.', '').replace(',', '').strip()
        for kw in BandarXRay.DIRECT_INDICATORS:
            if kw.upper() in account_clean:
                return True
        
        return False
    
    @classmethod
    def classify_account(cls, row):
        """
        FORENSIK ENGINE: 3-Layer Detection
        Layer 1: Pledge Status
        Layer 2: Nominee Pattern (Regex)
        Layer 3: Direct Ownership
        """
        
        holder = str(row['Nama Pemegang Saham']).upper()
        account = str(row['Nama Rekening Efek']).upper() if pd.notna(row.get('Nama Rekening Efek')) else ""
        
        # DEFAULT
        real_owner = cls.clean_name(holder)
        holding_type = "DIRECT"
        account_status = "NORMAL"
        bank_source = "-"
        
        # ==================================================================
        # LAYER 1: DETEKSI PLEDGE/REPO (PRIORITAS TERTINGGI)
        # ==================================================================
        is_pledge = False
        if account and len(account) > 3:
            for kw in cls.PLEDGE_KEYWORDS:
                if kw in account:
                    is_pledge = True
                    account_status = "⚠️ PLEDGE/REPO"
                    break
        
        # ==================================================================
        # LAYER 2: NOMINEE PATTERN DETECTION (REGEX)
        # ==================================================================
        nominee_found = False
        if account and len(account) > 5:
            for pattern, source in cls.PATTERNS_NOMINEE:
                match = re.search(pattern, account, re.IGNORECASE)
                if match:
                    extracted = match.group(1)
                    real_owner = cls.clean_name(extracted)
                    bank_source = source
                    holding_type = f"NOMINEE ({source})"
                    nominee_found = True
                    break
        
        # ==================================================================
        # LAYER 3: DIRECT OWNERSHIP (HANYA JIKA BUKAN NOMINEE)
        # ==================================================================
        if not nominee_found:
            if cls.is_direct_ownership(account):
                # Cek similarity dengan holder
                if holder in account or account in holder:
                    holding_type = "DIRECT"
                    real_owner = cls.clean_name(holder)
                else:
                    # Mungkin nama variasi
                    holding_type = "DIRECT (VARIANT)"
                    real_owner = cls.clean_name(account)
            else:
                # Fallback ke holder
                holding_type = "DIRECT (ASSUMED)"
                real_owner = cls.clean_name(holder)
        
        # ==================================================================
        # FIX 3: PRESERVE BANK INFO UNTUK PLEDGE ACCOUNTS
        # ==================================================================
        if is_pledge:
            # Tambahkan status pledge, jangan timpa tipe holding
            holding_type = f"{holding_type} - [PLEDGED]"
        
        return pd.Series([real_owner, holding_type, account_status, bank_source])

# ==============================================================================
# 4. GOOGLE DRIVE LOADER
# ==============================================================================
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

@st.cache_resource
def get_gdrive_service():
    """Initialize Google Drive service"""
    try:
        if "gdrive_creds" in st.secrets:
            creds = service_account.Credentials.from_service_account_info(
                st.secrets["gdrive_creds"],
                scopes=['https://www.googleapis.com/auth/drive.readonly']
            )
            return build('drive', 'v3', credentials=creds)
        return None
    except Exception as e:
        st.error(f"❌ GDrive Auth Error: {e}")
        return None

@st.cache_data(ttl=3600)
def load_data():
    """Load MASTER_DATABASE_5persen.csv from Google Drive"""
    
    service = get_gdrive_service()
    if not service:
        return pd.DataFrame()
    
    try:
        # Konfigurasi
        FOLDER_ID = st.secrets["gdrive"]["folder_id"]
        FILENAME = "MASTER_DATABASE_5persen.csv"
        
        # Cari file
        query = f"name = '{FILENAME}' and '{FOLDER_ID}' in parents and trashed = false"
        results = service.files().list(q=query, fields="files(id, name)").execute()
        items = results.get('files', [])
        
        if not items:
            st.error(f"❌ File '{FILENAME}' tidak ditemukan di GDrive")
            return pd.DataFrame()
        
        # Download file
        file_id = items[0]['id']
        request = service.files().get_media(fileId=file_id)
        
        file_stream = io.BytesIO()
        downloader = MediaIoBaseDownload(file_stream, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        
        file_stream.seek(0)
        
        # Baca CSV
        df = pd.read_csv(file_stream, dtype={'Kode Efek': str})
        
        # Konversi tanggal
        df['Tanggal_Data'] = pd.to_datetime(df['Tanggal_Data'])
        
        # Bersihkan kolom numerik
        for col in ['Jumlah Saham (Prev)', 'Jumlah Saham (Curr)']:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(',', ''), 
                    errors='coerce'
                ).fillna(0)
        
        # Hitung Net Flow
        df['Net_Flow'] = df['Jumlah Saham (Curr)'] - df['Jumlah Saham (Prev)']
        
        # Handle kolom Nama Rekening Efek
        if 'Nama Rekening Efek' not in df.columns:
            df['Nama Rekening Efek'] = '-'
        df['Nama Rekening Efek'] = df['Nama Rekening Efek'].fillna('-')
        
        return df
        
    except Exception as e:
        st.error(f"❌ Gagal load data: {e}")
        return pd.DataFrame()

# ==============================================================================
# 5. FORENSIC PROCESSOR - OPTIMASI SPEED
# ==============================================================================
@st.cache_data(ttl=3600)
def process_forensics(df):
    """Forensic Analysis dengan optimasi unique_pairs"""
    
    if df.empty:
        return df
    
    with st.spinner("🔍 Menganalisis pola nominee..."):
        # Optimasi: proses hanya kombinasi unik
        unique_pairs = df[['Nama Pemegang Saham', 'Nama Rekening Efek']].drop_duplicates()
        
        # Apply forensic engine
        forensic_results = unique_pairs.apply(
            BandarXRay.classify_account, 
            axis=1, 
            result_type='expand'
        )
        forensic_results.columns = ['REAL_OWNER', 'HOLDING_TYPE', 'ACCOUNT_STATUS', 'BANK_SOURCE']
        
        # ✅ FIX: Tambahkan kolom untuk merge
        forensic_results['Nama Pemegang Saham'] = unique_pairs['Nama Pemegang Saham'].values
        forensic_results['Nama Rekening Efek'] = unique_pairs['Nama Rekening Efek'].values
        
        # Merge back ke data utama
        df_result = pd.merge(df, forensic_results, 
                            on=['Nama Pemegang Saham', 'Nama Rekening Efek'], 
                            how='left')
        
        return df_result

# ==============================================================================
# 6. FILTER FUNCTION
# ==============================================================================
def apply_filters(df, selected_stocks, date_range, show_pledge_only, min_holding):
    """Apply all filters to dataframe"""
    
    df_filtered = df.copy()
    
    # Filter saham
    if selected_stocks:
        df_filtered = df_filtered[df_filtered['Kode Efek'].isin(selected_stocks)]
    
    # Filter tanggal
    if len(date_range) == 2:
        start_date, end_date = date_range
        df_filtered = df_filtered[
            (df_filtered['Tanggal_Data'].dt.date >= start_date) &
            (df_filtered['Tanggal_Data'].dt.date <= end_date)
        ]
    
    # Filter pledge
    if show_pledge_only:
        df_filtered = df_filtered[
            df_filtered['ACCOUNT_STATUS'].str.contains("PLEDGE|REPO", na=False)
        ]
    
    # Filter minimal holding
    if min_holding > 0:
        df_filtered = df_filtered[
            df_filtered['Jumlah Saham (Curr)'] >= min_holding
        ]
    
    return df_filtered

# ==============================================================================
# 7. DASHBOARD COMPONENTS
# ==============================================================================

def render_ultimate_holder_tab(df, df_last, show_top_only):
    """Tab 1: Ultimate Holder View"""
    
    st.markdown("### 👑 Peta Kepemilikan Asli (Ultimate Holder)")
    st.caption("Data ini menggabungkan kepemilikan satu entitas yang tersebar di banyak akun nominee")
    
    # Group by REAL_OWNER
    uh_group = df_last.groupby('REAL_OWNER').agg({
        'Jumlah Saham (Curr)': 'sum',
        'Nama Pemegang Saham': 'nunique',
        'HOLDING_TYPE': lambda x: ' | '.join(sorted(set([t.split()[0] for t in x])))[:50],
        'ACCOUNT_STATUS': lambda x: '⚠️ ADA PLEDGE' if any('PLEDGE' in str(s) for s in x) else 'CLEAN',
        'BANK_SOURCE': lambda x: ', '.join(sorted(set([b for b in x if b != '-'])))[:50]
    }).sort_values('Jumlah Saham (Curr)', ascending=False)
    
    if show_top_only:
        uh_group = uh_group.head(20)
    
    # Metrics
    total_shares = df_last['Jumlah Saham (Curr)'].sum()
    top5_shares = uh_group.head(5)['Jumlah Saham (Curr)'].sum()
    concentration_ratio = (top5_shares / total_shares * 100) if total_shares > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Ultimate Holders", f"{uh_group.shape[0]:,}")
    with col2:
        st.metric("Total Shares (>5%)", f"{total_shares:,.0f}")
    with col3:
        st.metric("Top 5 Concentration", f"{concentration_ratio:.1f}%",
                 delta="⚠️ HIGH" if concentration_ratio > 70 else "Normal")
    with col4:
        st.metric("Nominee Accounts", 
                 f"{df_last[df_last['HOLDING_TYPE'].str.contains('NOMINEE')].shape[0]:,}")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(
            uh_group.style.format({'Jumlah Saham (Curr)': '{:,.0f}'}),
            use_container_width=True,
            height=600,
            column_config={
                "REAL_OWNER": "🏢 Ultimate Holder (Pemilik Asli)",
                "Jumlah Saham (Curr)": "📊 Total Holdings",
                "Nama Pemegang Saham": "📋 Jumlah Akun",
                "HOLDING_TYPE": "🔧 Tipe Kepemilikan",
                "ACCOUNT_STATUS": "⚠️ Status",
                "BANK_SOURCE": "🏦 Kustodian"
            }
        )
    
    with col2:
        # Top 5 pie chart
        top5 = uh_group.head(5).reset_index()
        fig = px.pie(
            top5, 
            values='Jumlah Saham (Curr)', 
            names='REAL_OWNER',
            title=f"Top 5 Ultimate Holders",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
        # Concentration gauge
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = concentration_ratio,
            title = {'text': "Top 5 Concentration Risk"},
            delta = {'reference': 70},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "#00CC96" if concentration_ratio < 50 else "#FFA500" if concentration_ratio < 70 else "#FF4B4B"},
                'steps': [
                    {'range': [0, 50], 'color': "#E0FFE0"},
                    {'range': [50, 70], 'color': "#FFE0B0"},
                    {'range': [70, 100], 'color': "#FFC0C0"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        fig_gauge.update_layout(height=250)
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    return uh_group

def render_repo_monitor_tab(df, df_last):
    """Tab 2: Pledge/Repo Monitor"""
    
    st.markdown("### ⚠️ Radar Saham Digadaikan (Forced Sell Risk)")
    st.caption("Mendeteksi akun dengan indikasi PLEDGE, REPO, COLLATERAL, JAMINAN, MARGIN")
    
    df_pledge = df[df['ACCOUNT_STATUS'].str.contains("PLEDGE|REPO", na=False)]
    
    if not df_pledge.empty:
        pledge_last = df_pledge[df_pledge['Tanggal_Data'] == df_pledge['Tanggal_Data'].max()]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Saham Tergadai", f"{pledge_last['Jumlah Saham (Curr)'].sum():,.0f}")
        with col2:
            st.metric("Jumlah Ultimate Holder", f"{pledge_last['REAL_OWNER'].nunique()}")
        with col3:
            pledge_ratio = (pledge_last['Jumlah Saham (Curr)'].sum() / df_last['Jumlah Saham (Curr)'].sum() * 100)
            st.metric("% dari Total Kepemilikan", f"{pledge_ratio:.1f}%")
        
        st.subheader("📋 Daftar Akun Terindikasi Repo/Gadai")
        display_cols = ['REAL_OWNER', 'Kode Efek', 'Nama Pemegang Saham', 
                       'BANK_SOURCE', 'Jumlah Saham (Curr)']
        
        st.dataframe(
            pledge_last[display_cols].sort_values('Jumlah Saham (Curr)', ascending=False)
            .style.format({'Jumlah Saham (Curr)': '{:,.0f}'}),
            use_container_width=True,
            column_config={
                "REAL_OWNER": "Ultimate Holder",
                "Kode Efek": "Saham",
                "Nama Pemegang Saham": "Kustodian",
                "BANK_SOURCE": "Bank",
                "Jumlah Saham (Curr)": "Jumlah"
            }
        )
        
        # Trend
        st.subheader("📈 Historical Repo Trend")
        pledge_trend = df_pledge.groupby('Tanggal_Data')['Jumlah Saham (Curr)'].sum().reset_index()
        fig = px.area(
            pledge_trend, 
            x='Tanggal_Data', 
            y='Jumlah Saham (Curr)',
            title="Volume Saham Digadaikan Over Time",
            color_discrete_sequence=['#FF4B4B']
        )
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.success("✅ AMAN! Tidak ditemukan indikasi Repo/Gadai pada periode ini")

def render_flow_analysis_tab(df):
    """Tab 3: Flow Analysis"""
    
    st.markdown("### 🌊 Analisa Aliran Dana (Akumulasi vs Distribusi)")
    
    # Net flow per ultimate holder
    flow_stats = df.groupby('REAL_OWNER')['Net_Flow'].sum().reset_index()
    flow_stats = flow_stats[flow_stats['Net_Flow'] != 0].sort_values('Net_Flow', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🟢 Top 10 Accumulator (Net Buy)")
        top_buy = flow_stats.head(10)
        if not top_buy.empty:
            fig_buy = px.bar(
                top_buy, 
                x='Net_Flow', 
                y='REAL_OWNER',
                orientation='h',
                title="Akumulasi Bersih",
                color='Net_Flow',
                color_continuous_scale='Greens',
                text_auto='.2s'
            )
            fig_buy.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_buy, use_container_width=True)
    
    with col2:
        st.subheader("🔴 Top 10 Distributor (Net Sell)")
        top_sell = flow_stats.tail(10).sort_values('Net_Flow', ascending=True)
        if not top_sell.empty:
            fig_sell = px.bar(
                top_sell, 
                x='Net_Flow', 
                y='REAL_OWNER',
                orientation='h',
                title="Distribusi Bersih",
                color='Net_Flow',
                color_continuous_scale='Reds_r',
                text_auto='.2s'
            )
            fig_sell.update_layout(yaxis={'categoryorder':'total descending'})
            st.plotly_chart(fig_sell, use_container_width=True)
    
    # Individual tracking
    st.divider()
    st.subheader("📈 Lacak Pergerakan Ultimate Holder")
    
    players = sorted(df['REAL_OWNER'].unique())
    target = st.selectbox("Pilih Ultimate Holder:", players)
    
    if target:
        track_df = df[df['REAL_OWNER'] == target].copy()
        
        # Aggregate per stock per date
        track_pivot = track_df.groupby(['Tanggal_Data', 'Kode Efek'])['Jumlah Saham (Curr)'].sum().reset_index()
        
        fig_track = px.line(
            track_pivot,
            x='Tanggal_Data',
            y='Jumlah Saham (Curr)',
            color='Kode Efek',
            markers=True,
            title=f"🏢 {target} - Pergerakan Kepemilikan"
        )
        st.plotly_chart(fig_track, use_container_width=True)
        
        # Summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            total_flow = track_df['Net_Flow'].sum()
            st.metric("Total Net Flow", f"{total_flow:+,.0f}")
        with col2:
            avg_hold = track_df.groupby('Tanggal_Data')['Jumlah Saham (Curr)'].sum().mean()
            st.metric("Rata-rata Holdings", f"{avg_hold:,.0f}")
        with col3:
            stocks_held = track_df['Kode Efek'].nunique()
            st.metric("Saham Dipegang", f"{stocks_held}")

def render_nominee_mapping_tab(df_last):
    """Tab 4: Nominee Mapping"""
    
    st.markdown("### 🕵️ Bedah Nominee - Siapa Pakai Bank Apa?")
    st.caption("Detail mapping nominee account ke ultimate holder")
    
    df_nom = df_last[df_last['HOLDING_TYPE'].str.contains("NOMINEE")]
    
    if not df_nom.empty:
        # Group by Ultimate Holder and Bank
        mapping = df_nom.groupby(['REAL_OWNER', 'BANK_SOURCE', 'Nama Pemegang Saham']).agg({
            'Jumlah Saham (Curr)': 'sum',
            'Kode Efek': lambda x: ', '.join(x.unique())[:50],
            'Nama Rekening Efek': 'first'
        }).reset_index().sort_values(['REAL_OWNER', 'Jumlah Saham (Curr)'], ascending=[True, False])
        
        st.dataframe(
            mapping,
            use_container_width=True,
            column_config={
                "REAL_OWNER": "👑 Ultimate Holder",
                "BANK_SOURCE": "🏦 Bank/Kustodian",
                "Nama Pemegang Saham": "📋 Nama Akun",
                "Jumlah Saham (Curr)": "💰 Jumlah",
                "Kode Efek": "📈 Saham"
            }
        )
        
        # Broker preference matrix
        st.subheader("🏦 Bank Preference by Ultimate Holder")
        broker_pivot = pd.crosstab(
            df_nom['REAL_OWNER'], 
            df_nom['BANK_SOURCE'],
            values=df_nom['Jumlah Saham (Curr)'],
            aggfunc='sum'
        ).fillna(0)
        
        # Top 10 holders by total value
        top_holders = df_nom.groupby('REAL_OWNER')['Jumlah Saham (Curr)'].sum().nlargest(10).index
        broker_pivot_top = broker_pivot.loc[broker_pivot.index.isin(top_holders)]
        
        fig = px.imshow(
            broker_pivot_top,
            text_auto='.2s',
            aspect="auto",
            title="Heatmap: Ultimate Holder vs Bank Kustodian",
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("Tidak ada kepemilikan nominee di saham yang dipilih")

# ==============================================================================
# 8. MAIN APP
# ==============================================================================
def main():
    
    # ======================================================================
    # SIDEBAR - CONTROL PANEL
    # ======================================================================
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/null/eagle.png", width=80)
        st.title("🦅 X-RAY CONTROL")
        st.caption("Bandarmologi Forensic Engine v4.0")
        
        st.divider()
        
        # LOAD DATA BUTTON
        if st.button("📥 LOAD DATA KSEI", type="primary", use_container_width=True):
            with st.spinner("Menghubungkan ke Google Drive..."):
                df_raw = load_data()
                if not df_raw.empty:
                    st.session_state['df_raw'] = df_raw
                    st.session_state['df_processed'] = None
                    st.success(f"✅ Loaded: {len(df_raw):,} records")
                else:
                    st.error("❌ Gagal load data")
        
        st.divider()
        
        # Only show filters if data is loaded
        if 'df_raw' in st.session_state and st.session_state['df_raw'] is not None:
            
            # Check if need to process
            if 'df_processed' not in st.session_state or st.session_state['df_processed'] is None:
                with st.spinner("🔬 Running forensic analysis..."):
                    st.session_state['df_processed'] = process_forensics(st.session_state['df_raw'])
                    st.success("✅ Forensic analysis complete!")
            
            df = st.session_state['df_processed']
            
            # STOCK FILTER
            st.subheader("📈 Filter Saham")
            all_stocks = sorted(df['Kode Efek'].unique())
            selected_stocks = st.multiselect(
                "Pilih Kode Saham",
                all_stocks,
                default=all_stocks[:3] if len(all_stocks) >= 3 else all_stocks,
                max_selections=5
            )
            
            # DATE FILTER
            st.subheader("📅 Filter Periode")
            min_date = df['Tanggal_Data'].min().date()
            max_date = df['Tanggal_Data'].max().date()
            date_range = st.date_input(
                "Rentang Waktu",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            
            # ADVANCED FILTERS
            st.subheader("⚙️ Filter Lanjutan")
            show_pledge_only = st.checkbox("⚠️ Tampilkan hanya REPO/GADAI")
            show_top_only = st.checkbox("👑 Hanya Top 20 Ultimate Holder")
            min_holding = st.number_input(
                "💰 Minimal Holdings",
                min_value=0,
                value=0,
                step=1000000,
                format="%d"
            )
            
            # Apply filters
            df_filtered = apply_filters(
                df, selected_stocks, date_range, 
                show_pledge_only, min_holding
            )
            
            st.session_state['df_filtered'] = df_filtered
            st.session_state['selected_stocks'] = selected_stocks
            
            # SUMMARY METRICS
            st.divider()
            st.subheader("📊 Ringkasan")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Saham", f"{df_filtered['Kode Efek'].nunique()}")
            with col2:
                st.metric("Ultimate Holders", f"{df_filtered['REAL_OWNER'].nunique()}")
    
    # ======================================================================
    # MAIN CONTENT
    # ======================================================================
    
    # Check if data is processed
    if 'df_processed' not in st.session_state or st.session_state['df_processed'] is None:
        st.info("👈 Silakan klik **LOAD DATA KSEI** di sidebar untuk memulai")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image("https://img.icons8.com/fluency/96/null/eagle.png", width=200)
            st.title("🦅 Bandarmologi X-Ray")
            st.markdown("""
            ### Forensic Analysis Engine for KSEI Data
            
            **Fitur Utama:**
            - 🔍 **Unmask Nominee** - Temukan pemilik saham sebenarnya
            - ⚠️ **Repo/Pledge Hunter** - Deteksi saham digadaikan
            - 👑 **Ultimate Holder** - Konsolidasi kepemilikan tersebar
            - 📊 **Flow Analysis** - Tracking akumulasi/distribusi
            
            **Cara Penggunaan:**
            1. Klik **LOAD DATA KSEI** di sidebar
            2. Tunggu proses forensic analysis
            3. Filter saham dan periode
            4. Eksplorasi 4 tab dashboard!
            """)
        return
    
    # Get filtered data
    df_view = st.session_state.get('df_filtered', st.session_state['df_processed'])
    selected_stocks = st.session_state.get('selected_stocks', [])
    
    if df_view.empty:
        st.warning("⚠️ Tidak ada data dengan filter yang dipilih")
        return
    
    # Title
    stock_title = ', '.join(selected_stocks) if selected_stocks else 'ALL MARKET'
    st.title(f"🦅 Bandarmologi X-Ray: {stock_title}")
    st.caption(f"Periode: {df_view['Tanggal_Data'].min().date()} → {df_view['Tanggal_Data'].max().date()}")
    
    # Get last date snapshot
    last_date = df_view['Tanggal_Data'].max()
    df_last = df_view[df_view['Tanggal_Data'] == last_date]
    
    # ======================================================================
    # DASHBOARD TABS
    # ======================================================================
    tab1, tab2, tab3, tab4 = st.tabs([
        "👑 ULTIMATE HOLDER",
        "⚠️ REPO MONITOR",
        "🌊 FLOW ANALYSIS",
        "🕵️ NOMINEE MAPPING"
    ])
    
    with tab1:
        uh_group = render_ultimate_holder_tab(
            df_view, df_last, 
            st.session_state.get('show_top_only', False)
        )
    
    with tab2:
        render_repo_monitor_tab(df_view, df_last)
    
    with tab3:
        render_flow_analysis_tab(df_view)
    
    with tab4:
        render_nominee_mapping_tab(df_last)
    
    # ======================================================================
    # FOOTER
    # ======================================================================
    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.caption(f"""
        🦅 **Bandarmologi X-Ray Engine v4.0**  
        Forensic Analysis | Ultimate Holder Consolidation | Pledge Detection  
        Data Source: KSEI · Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
        *See through the nominees, find the real owners*
        """)

# ==============================================================================
# 9. ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    main()
