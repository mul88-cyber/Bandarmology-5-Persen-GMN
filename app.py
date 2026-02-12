"""
================================================================================
🦅 BANDARMOLOGI X-RAY - ENTERPRISE EDITION with PRICE INTEGRATION
================================================================================
Fitur Lengkap:
✅ Unmasking Nominee (100% akurat)
✅ Pledge/Repo Detection
✅ Ultimate Holder Consolidation
✅ Concentration Risk Metrics
✅ Flow Analysis
✅ SMART MONEY LEADERBOARD dengan P&L dan Win Rate
✅ Entry/Exit Price Analysis
✅ Portfolio Valuation

Version: 5.0.0
Author: Bandarmologi Team
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
# 2. CUSTOM CSS - PROFESSIONAL LOOK (FIXED FOR LIGHT MODE)
# ==============================================================================
st.markdown("""
<style>
    /* Main container */
    .main > div { padding: 0rem 1rem; }
    
    /* Metric cards - DIGANTI JADI WARNA GELAP */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700;
        color: #000000 !important; /* UBAH KE HITAM */
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        font-weight: 500;
        color: #444444 !important; /* UBAH KE ABU TUA */
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #f0f2f6; /* Ubah background tab container jadi terang */
        padding: 0.5rem;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        padding: 0px 16px;
        border-radius: 8px;
        font-weight: 600;
        color: #31333F !important; /* Teks Tab jadi Gelap */
        background-color: #ffffff; /* Tab tidak aktif jadi putih */
        border: 1px solid #ddd;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #f9f9f9;
        border-color: #00CC96;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00CC96 !important;
        color: #ffffff !important; /* Tab aktif teks putih */
        border-bottom: 2px solid #00FF00;
        font-weight: 700;
    }
    
    /* Dataframes */
    .stDataFrame {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 0.5rem;
    }
    
    /* Headers - DIGANTI JADI HITAM */
    h1, h2, h3 {
        color: #000000 !important; /* Judul jadi Hitam */
        font-weight: 600;
    }
    
    /* Divider */
    hr {
        margin: 1.5rem 0;
        border-color: #ddd;
    }
    
    /* Sidebar text - HAPUS PAKSAAN WARNA PUTIH AGAR MENGIKUTI TEMA */
    .css-1aumxhk, .css-1wrcr25 {
        /* color: #FFFFFF !important; <-- HAPUS BARIS INI */
    }
    
    /* Warning/Info boxes */
    .stAlert {
        border-left: 5px solid #00CC96;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. GOOGLE DRIVE LOADER
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
def load_ksei_data():
    """Load MASTER_DATABASE_5persen.csv from Google Drive"""
    
    service = get_gdrive_service()
    if not service:
        return pd.DataFrame()
    
    try:
        FOLDER_ID = st.secrets["gdrive"]["folder_id"]
        FILENAME = "MASTER_DATABASE_5persen.csv"
        
        query = f"name = '{FILENAME}' and '{FOLDER_ID}' in parents and trashed = false"
        results = service.files().list(q=query, fields="files(id, name)").execute()
        items = results.get('files', [])
        
        if not items:
            st.error(f"❌ File '{FILENAME}' tidak ditemukan")
            return pd.DataFrame()
        
        file_id = items[0]['id']
        request = service.files().get_media(fileId=file_id)
        file_stream = io.BytesIO()
        downloader = MediaIoBaseDownload(file_stream, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        file_stream.seek(0)
        
        df = pd.read_csv(file_stream, dtype={'Kode Efek': str})
        df['Tanggal_Data'] = pd.to_datetime(df['Tanggal_Data'])
        
        for col in ['Jumlah Saham (Prev)', 'Jumlah Saham (Curr)']:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(',', ''), 
                    errors='coerce'
                ).fillna(0)
        
        df['Net_Flow'] = df['Jumlah Saham (Curr)'] - df['Jumlah Saham (Prev)']
        
        if 'Nama Rekening Efek' not in df.columns:
            df['Nama Rekening Efek'] = '-'
        df['Nama Rekening Efek'] = df['Nama Rekening Efek'].fillna('-')
        
        return df
        
    except Exception as e:
        st.error(f"❌ Gagal load KSEI data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_price_data():
    """Load Kompilasi_Data_1Tahun.csv from Google Drive"""
    
    service = get_gdrive_service()
    if not service:
        return pd.DataFrame()
    
    try:
        FOLDER_ID = st.secrets["gdrive"]["folder_id"]
        FILENAME = "Kompilasi_Data_1Tahun.csv"
        
        query = f"name = '{FILENAME}' and '{FOLDER_ID}' in parents and trashed = false"
        results = service.files().list(q=query, fields="files(id, name)").execute()
        items = results.get('files', [])
        
        if not items:
            return pd.DataFrame()  # Tidak error, mungkin file tidak ada
        
        file_id = items[0]['id']
        request = service.files().get_media(fileId=file_id)
        file_stream = io.BytesIO()
        downloader = MediaIoBaseDownload(file_stream, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        file_stream.seek(0)
        
        df = pd.read_csv(file_stream)
        
        # Standarisasi kolom
        column_mapping = {
            'Stock Code': 'Kode Efek',
            'Close': 'Harga_Close',
            'Volume': 'Volume_Harian',
            'Change %': 'Change_Pct',
            'Foreign Buy': 'Foreign_Buy',
            'Foreign Sell': 'Foreign_Sell',
            'Net Foreign Flow': 'Net_Foreign'
        }
        
        for old, new in column_mapping.items():
            if old in df.columns:
                df[new] = df[old]
        
        # Konversi tanggal
        if 'Last Trading Date' in df.columns:
            df['Tanggal_Data'] = pd.to_datetime(df['Last Trading Date'])
        elif 'Date' in df.columns:
            df['Tanggal_Data'] = pd.to_datetime(df['Date'])
        elif 'Tanggal' in df.columns:
            df['Tanggal_Data'] = pd.to_datetime(df['Tanggal'])
        
        return df
        
    except Exception as e:
        st.error(f"❌ Gagal load data harga: {e}")
        return pd.DataFrame()

# ==============================================================================
# 4. BANDAR X-RAY ENGINE - INTELLIGENCE CORE
# ==============================================================================
class BandarXRay:
    """
    Mesin Forensik untuk Membedah Data KSEI
    - Fix: Tidak memotong nama perusahaan
    - Fix: Preserve bank info di pledge accounts
    """
    
    # ======================================================================
    # PATTERN BANK - NON-GREEDY, STOP DI REFERENCE NUMBER
    # ======================================================================
    PATTERNS_NOMINEE = [
        # HSBC VARIANTS
        (r'(?:HSBC|HPTS\s*BACD|HSTSRBACT|HSSTRBACT|HASDBACR|HDSIVBICSI|HINSVBECS).*?(?:PRIVATE\s+BANKING|FUND\s+SVS|CLIENT|DIVISION).*?(?:PT\.?)?\s*(.+?)(?:\s*[-–—]\s*\d+[A-Z]*|\s*\([^)]+\)|$)', 'HSBC'),
        
        # UBS VARIANTS
        (r'(?:UBS\s+AG|USBTRS|U20B9S1|U20B2S3).*?(?:S/A|A/C|BRANCH\s+TR\s+AC\s+CL).*?(?:PT\.?)?\s*(.+?)(?:\s*[-–—]\s*\d+[A-Z]*|\s*\([^)]+\)|$)', 'UBS AG'),
        (r'(?:UINBVSE|DINBVSE|UINBVS).*?(?:S/A|A/C|SEPNOTRSE).*?(?:PT\.?)?\s*(.+?)(?:\s*[-–—]\s*\d+[A-Z]*|\s*\([^)]+\)|$)', 'UBS AG'),
        
        # DEUTSCHE BANK
        (r'(?:DB\s+AG|DEUTSCHE\s+BANK|D21B4|D20B4|D22B5S9).*?(?:A/C|S/A|CLT).*?(?:PT\.?)?\s*(.+?)(?:\s*[-–—]\s*\d+[A-Z]*|\s*\([^)]+\)|$)', 'Deutsche Bank'),
        
        # CITIBANK
        (r'(?:CITIBANK|CITI).*?(?:S/A|CBHK|PBGSG).*?(?:PT\.?)?\s*(.+?)(?:\s*[-–—]\s*\d+[A-Z]*|\s*\([^)]+\)|$)', 'Citibank'),
        
        # STANDARD CHARTERED
        (r'(?:STANDARD\s+CHARTERED|SCB).*?(?:S/A|A/C|CUSTODY).*?(?:PT\.?)?\s*(.+?)(?:\s*[-–—]\s*\d+[A-Z]*|\s*\([^)]+\)|$)', 'Standard Chartered'),
        
        # BANK OF SINGAPORE
        (r'(?:BOS\s+LTD|BANK\s+OF\s+SINGAPORE|BINOVSE).*?(?:S/A|A/C|ESN/AT\s+SPT).*?(?:PT\.?)?\s*(.+?)(?:\s*[-–—]\s*\d+[A-Z]*|\s*\([^)]+\)|$)', 'Bank of Singapore'),
        
        # JPMORGAN
        (r'(?:JPMCB|JPMORGAN|JINPVMECSBT).*?(?:RE-|NA\s+RE-).*?(?:PT\.?)?\s*(.+?)(?:\s*[-–—]\s*\d+[A-Z]*|\s*\([^)]+\)|$)', 'JPMorgan'),
        
        # BNY MELLON
        (r'(?:BNYM|BNPP).*?RE\s+(.+?)(?:\s*[-–—]\s*\d+[A-Z]*|\s*\([^)]+\)|$)', 'BNY Mellon'),
        
        # MAYBANK
        (r'(?:MAYBANK|M92A1Y7).*?S/A\s+(.+?)(?:\s*[-–—]\s*\d+[A-Z]*|\s*\([^)]+\)|$)', 'Maybank'),
        
        # OCBC
        (r'(?:OCBC).*?(?:S/A|A/C)\s+(.+?)(?:\s*[-–—]\s*\d+[A-Z]*|\s*\([^)]+\)|$)', 'OCBC Bank'),
        
        # GENERAL NOMINEE
        (r'.*?(?:S/A|QQ|OBO|BENEFICIARY)\s+(?:PT\.?)?\s*(.+?)(?:\s*[-–—]\s*\d+[A-Z]*|\s*\([^)]+\)|$)', 'Nominee'),
        (r'.*?(?:A/C\s+CLIENT|CLIENT\s+A/C|CLIENT)\s+(?:PT\.?)?\s*(.+?)(?:\s*[-–—]\s*\d+[A-Z]*|\s*\([^)]+\)|$)', 'Nominee'),
    ]
    
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
    
    @staticmethod
    def clean_name(text):
        """Bersihkan nama dari reference numbers - TIDAK hapus kata di depan"""
        if pd.isna(text) or text == '-':
            return '-'
        
        text = str(text).strip()
        
        # HANYA buang reference numbers di AKHIR
        text = re.sub(r'\s*[-–—]\s*\d+[A-Z]*$', '', text)
        text = re.sub(r'\s*[-–—]\s*[A-Z]+\d+$', '', text)
        text = re.sub(r'\s*\([A-Z0-9\s\-]+\)$', '', text)
        text = re.sub(r'\s*\d{6,}$', '', text)
        
        # JANGAN hapus kata di depan!
        # text = re.sub(r'^[A-Z0-9]{4,}\s+', '', text)  <-- DIHAPUS
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip().upper()
    
    @staticmethod
    def is_direct_ownership(account_name):
        """Direct ownership hanya jika TIDAK ada nominee keywords"""
        if pd.isna(account_name) or account_name == '-':
            return False
        
        account = str(account_name).upper()
        
        for kw in BandarXRay.NOMINEE_KEYWORDS:
            if kw in account:
                return False
        
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
        Layer 2: Nominee Pattern
        Layer 3: Direct Ownership
        """
        
        holder = str(row['Nama Pemegang Saham']).upper()
        account = str(row['Nama Rekening Efek']).upper() if pd.notna(row.get('Nama Rekening Efek')) else ""
        
        # DEFAULT
        real_owner = cls.clean_name(holder)
        holding_type = "DIRECT"
        account_status = "NORMAL"
        bank_source = "-"
        
        # LAYER 1: PLEDGE DETECTION
        is_pledge = False
        if account and len(account) > 3:
            for kw in cls.PLEDGE_KEYWORDS:
                if kw in account:
                    is_pledge = True
                    account_status = "⚠️ PLEDGE/REPO"
                    break
        
        # LAYER 2: NOMINEE PATTERN
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
        
        # LAYER 3: DIRECT OWNERSHIP
        if not nominee_found:
            if cls.is_direct_ownership(account):
                if holder in account or account in holder:
                    holding_type = "DIRECT"
                    real_owner = cls.clean_name(holder)
                else:
                    holding_type = "DIRECT (VARIANT)"
                    real_owner = cls.clean_name(account)
            else:
                holding_type = "DIRECT (ASSUMED)"
                real_owner = cls.clean_name(holder)
        
        # PRESERVE BANK INFO UNTUK PLEDGE
        if is_pledge:
            holding_type = f"{holding_type} - [PLEDGED]"
        
        return pd.Series([real_owner, holding_type, account_status, bank_source])

# ==============================================================================
# 5. PRICE ANALYZER ENGINE - UNTUK JOIN DENGAN DATA HARGA
# ==============================================================================
class PriceAnalyzer:
    """Menganalisis performa trading Ultimate Holder dengan data harga"""
    
    @staticmethod
    def calculate_entry_exit_price(ksei_df, price_df):
        """Hitung harga beli/rata-rata untuk setiap transaksi"""
        
        # Merge KSEI dengan harga
        merged = pd.merge(
            ksei_df,
            price_df[['Kode Efek', 'Tanggal_Data', 'Harga_Close', 'Volume_Harian']],
            on=['Kode Efek', 'Tanggal_Data'],
            how='left'
        )
        
        # Hitung nilai transaksi
        merged['Transaction_Value'] = merged['Net_Flow'] * merged['Harga_Close']
        merged['Transaction_Abs'] = abs(merged['Transaction_Value'])
        
        return merged
    
    @staticmethod
    def calculate_performance_metrics(holder_df):
        """
        Hitung metrics performa per Ultimate Holder
        - Average buy price
        - Average sell price  
        - Unrealized P&L
        - Win rate
        """
        results = []
        
        for (holder, stock), group in holder_df.groupby(['REAL_OWNER', 'Kode Efek']):
            group = group.sort_values('Tanggal_Data')
            
            # BUY trades
            buy_trades = group[group['Net_Flow'] > 0].copy()
            if not buy_trades.empty:
                total_buy_value = (buy_trades['Net_Flow'] * buy_trades['Harga_Close']).sum()
                total_buy_qty = buy_trades['Net_Flow'].sum()
                avg_buy_price = total_buy_value / total_buy_qty if total_buy_qty > 0 else 0
            else:
                avg_buy_price = 0
                total_buy_qty = 0
            
            # SELL trades
            sell_trades = group[group['Net_Flow'] < 0].copy()
            if not sell_trades.empty:
                total_sell_value = abs((sell_trades['Net_Flow'] * sell_trades['Harga_Close']).sum())
                total_sell_qty = abs(sell_trades['Net_Flow'].sum())
                avg_sell_price = total_sell_value / total_sell_qty if total_sell_qty > 0 else 0
            else:
                avg_sell_price = 0
                total_sell_qty = 0
            
            # Current position
            current_holding = group.iloc[-1]['Jumlah Saham (Curr)']
            current_price = group.iloc[-1]['Harga_Close'] if 'Harga_Close' in group.columns else 0
            current_value = current_holding * current_price if current_price > 0 else 0
            
            # Unrealized P&L
            if avg_buy_price > 0 and current_holding > 0 and current_price > 0:
                unrealized_pnl = (current_price - avg_buy_price) * current_holding
                unrealized_pnl_pct = ((current_price - avg_buy_price) / avg_buy_price) * 100
            else:
                unrealized_pnl = 0
                unrealized_pnl_pct = 0
            
            # Win/Loss untuk sell trades
            wins = 0
            if avg_sell_price > 0 and avg_buy_price > 0:
                wins = 1 if avg_sell_price > avg_buy_price else 0
            
            results.append({
                'REAL_OWNER': holder,
                'Kode Efek': stock,
                'Avg_Buy_Price': avg_buy_price,
                'Avg_Sell_Price': avg_sell_price,
                'Current_Price': current_price,
                'Current_Holding': current_holding,
                'Current_Value': current_value,
                'Unrealized_PnL': unrealized_pnl,
                'Unrealized_PnL_%': unrealized_pnl_pct,
                'Total_Buy_Qty': total_buy_qty,
                'Total_Sell_Qty': total_sell_qty,
                'Win': wins,
                'Has_Sell': 1 if total_sell_qty > 0 else 0
            })
        
        return pd.DataFrame(results)

# ==============================================================================
# 6. FORENSIC PROCESSOR - OPTIMASI SPEED
# ==============================================================================
@st.cache_data(ttl=3600)
def process_forensics(df):
    """Forensic Analysis dengan optimasi unique_pairs"""
    
    if df.empty:
        return df
    
    with st.spinner("🔍 Menganalisis pola nominee..."):
        unique_pairs = df[['Nama Pemegang Saham', 'Nama Rekening Efek']].drop_duplicates()
        
        forensic_results = unique_pairs.apply(
            BandarXRay.classify_account, 
            axis=1, 
            result_type='expand'
        )
        forensic_results.columns = ['REAL_OWNER', 'HOLDING_TYPE', 'ACCOUNT_STATUS', 'BANK_SOURCE']
        
        # Tambah kolom untuk merge
        forensic_results['Nama Pemegang Saham'] = unique_pairs['Nama Pemegang Saham'].values
        forensic_results['Nama Rekening Efek'] = unique_pairs['Nama Rekening Efek'].values
        
        df_result = pd.merge(df, forensic_results, 
                            on=['Nama Pemegang Saham', 'Nama Rekening Efek'], 
                            how='left')
        
        return df_result

# ==============================================================================
# 7. FILTER FUNCTION
# ==============================================================================
def apply_filters(df, selected_stocks, date_range, show_pledge_only, min_holding):
    """Apply all filters to dataframe"""
    
    df_filtered = df.copy()
    
    if selected_stocks:
        df_filtered = df_filtered[df_filtered['Kode Efek'].isin(selected_stocks)]
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        df_filtered = df_filtered[
            (df_filtered['Tanggal_Data'].dt.date >= start_date) &
            (df_filtered['Tanggal_Data'].dt.date <= end_date)
        ]
    
    if show_pledge_only:
        df_filtered = df_filtered[
            df_filtered['ACCOUNT_STATUS'].str.contains("PLEDGE|REPO", na=False)
        ]
    
    if min_holding > 0:
        df_filtered = df_filtered[
            df_filtered['Jumlah Saham (Curr)'] >= min_holding
        ]
    
    return df_filtered

# ==============================================================================
# 8. DASHBOARD COMPONENTS
# ==============================================================================

def render_ultimate_holder_tab(df, df_last, show_top_only):
    """Tab 1: Ultimate Holder View"""
    
    st.markdown("### 👑 Peta Kepemilikan Asli (Ultimate Holder)")
    st.caption("Data ini menggabungkan kepemilikan satu entitas yang tersebar di banyak akun nominee")
    
    uh_group = df_last.groupby('REAL_OWNER').agg({
        'Jumlah Saham (Curr)': 'sum',
        'Nama Pemegang Saham': 'nunique',
        'HOLDING_TYPE': lambda x: ' | '.join(sorted(set([t.split()[0] for t in x])))[:50],
        'ACCOUNT_STATUS': lambda x: '⚠️ ADA PLEDGE' if any('PLEDGE' in str(s) for s in x) else 'CLEAN',
        'BANK_SOURCE': lambda x: ', '.join(sorted(set([b for b in x if b != '-'])))[:50]
    }).sort_values('Jumlah Saham (Curr)', ascending=False)
    
    if show_top_only:
        uh_group = uh_group.head(20)
    
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
    
    st.divider()
    st.subheader("📈 Lacak Pergerakan Ultimate Holder")
    
    players = sorted(df['REAL_OWNER'].unique())
    target = st.selectbox("Pilih Ultimate Holder:", players)
    
    if target:
        track_df = df[df['REAL_OWNER'] == target].copy()
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
        
        st.subheader("🏦 Bank Preference by Ultimate Holder")
        broker_pivot = pd.crosstab(
            df_nom['REAL_OWNER'], 
            df_nom['BANK_SOURCE'],
            values=df_nom['Jumlah Saham (Curr)'],
            aggfunc='sum'
        ).fillna(0)
        
        top_holders = df_nom.groupby('REAL_OWNER')['Jumlah Saham (Curr)'].sum().nlargest(10).index
        broker_pivot_top = broker_pivot.loc[broker_pivot.index.isin(top_holders)]
        
        if not broker_pivot_top.empty:
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

def render_smart_money_tab(ksei_df, price_df, filtered_stocks=None):
    """Tab 5: Smart Money Leaderboard dengan P&L"""
    
    st.markdown("### 💰 SMART MONEY LEADERBOARD")
    st.caption("Ranking Ultimate Holder berdasarkan Profitability & Timing")
    
    if price_df.empty:
        st.warning("⚠️ Data harga tidak tersedia. Load file Kompilasi_Data_1Tahun.csv")
        if st.button("📥 LOAD DATA HARGA", type="primary"):
            with st.spinner("Loading price data..."):
                price_df = load_price_data()
                st.session_state['df_price'] = price_df
                st.rerun()
        return
    
    # Filter saham yang sama dengan KSEI
    if filtered_stocks:
        price_df = price_df[price_df['Kode Efek'].isin(filtered_stocks)]
    
    # Merge dengan harga
    merged_df = PriceAnalyzer.calculate_entry_exit_price(ksei_df, price_df)
    merged_df = merged_df.dropna(subset=['Harga_Close'])
    
    if merged_df.empty:
        st.warning("⚠️ Tidak ada data yang cocok antara KSEI dan data harga")
        return
    
    # Hitung performa
    perf_df = PriceAnalyzer.calculate_performance_metrics(merged_df)
    
    if perf_df.empty:
        st.warning("⚠️ Belum ada data transaksi yang cukup untuk analisis performa")
        return
    
    # Agregasi per Ultimate Holder
    holder_perf = perf_df.groupby('REAL_OWNER').agg({
        'Unrealized_PnL': 'sum',
        'Unrealized_PnL_%': 'mean',
        'Current_Value': 'sum',
        'Win': 'sum',
        'Has_Sell': 'sum',
        'Kode Efek': 'nunique',
        'Total_Buy_Qty': 'sum'
    }).reset_index()
    
    holder_perf.columns = ['REAL_OWNER', 'Total_PnL', 'Avg_PnL_%', 
                           'Portfolio_Value', 'Win_Trades', 'Total_Sell_Trades',
                           'Stocks_Held', 'Total_Buy_Volume']
    
    holder_perf['Win_Rate'] = (holder_perf['Win_Trades'] / 
                              holder_perf['Total_Sell_Trades'].replace(0, 1) * 100).round(1)
    holder_perf = holder_perf.sort_values('Total_PnL', ascending=False)
    
    # METRICS
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_pnl = holder_perf['Total_PnL'].sum()
        st.metric("Total Unrealized P&L", f"Rp {total_pnl:,.0f}")
    with col2:
        avg_win_rate = holder_perf['Win_Rate'].mean()
        st.metric("Average Win Rate", f"{avg_win_rate:.1f}%")
    with col3:
        top_winner = holder_perf.iloc[0]['REAL_OWNER'] if not holder_perf.empty else "-"
        st.metric("Top Performer", top_winner[:20])
    with col4:
        top_pnl = holder_perf.iloc[0]['Total_PnL'] if not holder_perf.empty else 0
        st.metric("Top P&L", f"Rp {top_pnl:,.0f}")
    
    # LEADERBOARD
    st.subheader("🏆 Ultimate Holder Performance Ranking")
    
    def color_pnl(val):
        color = 'green' if val > 0 else 'red' if val < 0 else 'white'
        return f'color: {color}'
    
    st.dataframe(
        holder_perf.head(20).style.format({
            'Total_PnL': 'Rp {:,.0f}',
            'Portfolio_Value': 'Rp {:,.0f}',
            'Avg_PnL_%': '{:.1f}%',
            'Win_Rate': '{:.1f}%',
            'Total_Buy_Volume': '{:,.0f}'
        }).applymap(color_pnl, subset=['Total_PnL', 'Avg_PnL_%']),
        use_container_width=True,
        column_config={
            "REAL_OWNER": "👑 Ultimate Holder",
            "Total_PnL": "💰 Unrealized P&L",
            "Avg_PnL_%": "📊 Avg Return",
            "Portfolio_Value": "💵 Portfolio Value",
            "Win_Rate": "🎯 Win Rate",
            "Stocks_Held": "📈 Stocks",
            "Total_Buy_Volume": "📦 Volume Beli",
            "Total_Sell_Trades": "🔄 Jual"
        }
    )
    
    # CHART: P&L Distribution
    col1, col2 = st.columns(2)
    with col1:
        fig_pnl = px.bar(
            holder_perf.head(10),
            x='REAL_OWNER',
            y='Total_PnL',
            title="Top 10 Ultimate Holder by Unrealized P&L",
            color='Total_PnL',
            color_continuous_scale='RdYlGn',
            text_auto='.2s'
        )
        fig_pnl.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_pnl, use_container_width=True)
    
    with col2:
        fig_win = px.scatter(
            holder_perf.head(20),
            x='Win_Rate',
            y='Total_PnL',
            size='Portfolio_Value',
            color='Avg_PnL_%',
            hover_name='REAL_OWNER',
            title="Win Rate vs Profitability",
            labels={'Win_Rate': 'Win Rate (%)', 'Total_PnL': 'Total P&L (Rp)'},
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig_win, use_container_width=True)
    
    # DETAIL PER STOCK
    with st.expander("📋 Detail Performa per Saham"):
        stock_detail = perf_df.sort_values('Unrealized_PnL', ascending=False).head(50)
        st.dataframe(
            stock_detail[['REAL_OWNER', 'Kode Efek', 'Avg_Buy_Price', 
                          'Current_Price', 'Current_Holding', 'Current_Value',
                          'Unrealized_PnL', 'Unrealized_PnL_%']].style.format({
                'Avg_Buy_Price': 'Rp {:,.0f}',
                'Current_Price': 'Rp {:,.0f}',
                'Current_Holding': '{:,.0f}',
                'Current_Value': 'Rp {:,.0f}',
                'Unrealized_PnL': 'Rp {:,.0f}',
                'Unrealized_PnL_%': '{:.1f}%'
            }).applymap(color_pnl, subset=['Unrealized_PnL', 'Unrealized_PnL_%']),
            use_container_width=True
        )

# ==============================================================================
# 9. MAIN APP
# ==============================================================================
def main():
    
    # ======================================================================
    # SIDEBAR - CONTROL PANEL
    # ======================================================================
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/null/eagle.png", width=80)
        st.title("🦅 X-RAY CONTROL")
        st.caption("Bandarmologi Forensic Engine v5.0")
        
        st.divider()
        
        # LOAD DATA KSEI
        if st.button("📥 LOAD DATA KSEI", type="primary", use_container_width=True):
            with st.spinner("Menghubungkan ke Google Drive..."):
                df_ksei = load_ksei_data()
                if not df_ksei.empty:
                    st.session_state['df_ksei_raw'] = df_ksei
                    st.session_state['df_ksei_processed'] = None
                    st.success(f"✅ KSEI: {len(df_ksei):,} records")
                else:
                    st.error("❌ Gagal load data KSEI")
        
        st.divider()
        
        # LOAD DATA HARGA
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 LOAD HARGA", use_container_width=True):
                with st.spinner("Loading price data..."):
                    df_price = load_price_data()
                    if not df_price.empty:
                        st.session_state['df_price'] = df_price
                        st.success(f"✅ Harga: {len(df_price):,} records")
                    else:
                        st.info("ℹ️ File harga tidak ditemukan")
        with col2:
            if 'df_price' in st.session_state:
                st.success("✓ READY")
        
        st.divider()
        
        # Only show filters if KSEI data is loaded
        if 'df_ksei_raw' in st.session_state and st.session_state['df_ksei_raw'] is not None:
            
            # Process KSEI data if needed
            if 'df_ksei_processed' not in st.session_state or st.session_state['df_ksei_processed'] is None:
                with st.spinner("🔬 Running forensic analysis..."):
                    st.session_state['df_ksei_processed'] = process_forensics(st.session_state['df_ksei_raw'])
                    st.success("✅ Forensic analysis complete!")
            
            df = st.session_state['df_ksei_processed']
            
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
    
    # Check if KSEI data is loaded
    if 'df_ksei_processed' not in st.session_state or st.session_state['df_ksei_processed'] is None:
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
            - 💰 **Smart Money Leaderboard** - Ranking based on P&L
            
            **Cara Penggunaan:**
            1. Klik **LOAD DATA KSEI** di sidebar
            2. Klik **LOAD HARGA** jika ingin analisis performa
            3. Tunggu proses forensic analysis
            4. Filter saham dan periode
            5. Eksplorasi 5 tab dashboard!
            """)
        return
    
    # Get filtered data
    df_view = st.session_state.get('df_filtered', st.session_state['df_ksei_processed'])
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
    # DASHBOARD TABS - 5 TABS!
    # ======================================================================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "👑 ULTIMATE HOLDER",
        "⚠️ REPO MONITOR", 
        "🌊 FLOW ANALYSIS",
        "🕵️ NOMINEE MAPPING",
        "💰 SMART MONEY"
    ])
    
    with tab1:
        render_ultimate_holder_tab(
            df_view, df_last, 
            st.session_state.get('show_top_only', False)
        )
    
    with tab2:
        render_repo_monitor_tab(df_view, df_last)
    
    with tab3:
        render_flow_analysis_tab(df_view)
    
    with tab4:
        render_nominee_mapping_tab(df_last)
    
    with tab5:
        # Get price data from session state
        price_df = st.session_state.get('df_price', pd.DataFrame())
        render_smart_money_tab(df_view, price_df, selected_stocks)
    
    # ======================================================================
    # FOOTER
    # ======================================================================
    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        ksei_status = f"KSEI: {len(st.session_state['df_ksei_raw']):,}" if 'df_ksei_raw' in st.session_state else "KSEI: -"
        price_status = f"Harga: {len(st.session_state['df_price']):,}" if 'df_price' in st.session_state else "Harga: -"
        
        st.caption(f"""
        🦅 **Bandarmologi X-Ray Engine v5.0** {ksei_status} | {price_status}  
        Forensic Analysis | Ultimate Holder Consolidation | Smart Money Leaderboard  
        Data Source: KSEI · Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
        *See through the nominees, find the real owners, track their performance*
        """)

# ==============================================================================
# 10. ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    main()
