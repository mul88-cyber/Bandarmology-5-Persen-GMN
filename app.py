"""
================================================================================
🦅 BANDARMOLOGI X-RAY - ENTERPRISE EDITION v6.1
================================================================================
✅ STOCK SCREENER: Deteksi saham akumulasi/distribusi otomatis
✅ DEEP DIVE: Analisa satu saham (Chart Harga vs Flow Bandar)
✅ UNMASKING ENGINE: Tetap aktif (Logic Prioritas Bank)
✅ SMART MONEY: Valuasi portfolio akurat (snapshot tanggal terakhir)
✅ FIXED: Screener double counting, Deep Dive filter mismatch, Smart Money valuation
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
# 1. KONFIGURASI HALAMAN
# ==============================================================================
st.set_page_config(layout="wide", page_title="Bandarmologi X-Ray", page_icon="🦅")

# Custom CSS - Professional Look
st.markdown("""
<style>
    .main > div { padding: 0rem 1rem; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem !important; font-weight: 600; color: #000000 !important; }
    div[data-testid="stMetricLabel"] { font-size: 0.9rem !important; color: #444444 !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 2px; background-color: #f0f2f6; padding: 0.5rem; border-radius: 10px; }
    .stTabs [data-baseweb="tab"] { 
        height: 40px; padding: 0px 16px; border-radius: 8px; 
        background-color: #ffffff; color: #31333F !important; border: 1px solid #ddd;
    }
    .stTabs [aria-selected="true"] { 
        background-color: #00CC96 !important; color: #ffffff !important; 
        border-bottom: 2px solid #00FF00; font-weight: 600;
    }
    .stDataFrame { border: 1px solid #e0e0e0; border-radius: 10px; padding: 0.5rem; }
    h1, h2, h3 { color: #1E1E1E !important; font-weight: 600; }
    .stAlert { background-color: #f8f9fa; border-left: 5px solid #00CC96; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. BANDAR X-RAY ENGINE (FORENSIK) - OPTIMIZED
# ==============================================================================
class BandarXRay:
    """Forensic Engine for Unmasking Nominee Accounts"""
    
    PATTERNS_NOMINEE = [
        (r'(?:HSBC|HPTS|HSTSRBACT|HSSTRBACT|HASDBACR|HDSIVBICSI|HINSVBECS).*?(?:PRIVATE|FUND|CLIENT|DIVISION).*?(?:PT\.?)?\s*(.+?)(?:\s*[-–—]\s*\d+[A-Z]*|\s*\([^)]+\)|$)', 'HSBC'),
        (r'(?:UBS\s+AG|USBTRS|U20B9S1|U20B2S3|UINBVSE|DINBVSE|UINBVS).*?(?:S/A|A/C|BRANCH|TR|SEPNOTRSE).*?(?:PT\.?)?\s*(.+?)(?:\s*[-–—]\s*\d+[A-Z]*|\s*\([^)]+\)|$)', 'UBS AG'),
        (r'(?:DB\s+AG|DEUTSCHE\s+BANK|D21B4|D20B4|D22B5S9).*?(?:A/C|S/A|CLT).*?(?:PT\.?)?\s*(.+?)(?:\s*[-–—]\s*\d+[A-Z]*|\s*\([^)]+\)|$)', 'Deutsche Bank'),
        (r'(?:CITIBANK|CITI).*?(?:S/A|CBHK|PBGSG).*?(?:PT\.?)?\s*(.+?)(?:\s*[-–—]\s*\d+[A-Z]*|\s*\([^)]+\)|$)', 'Citibank'),
        (r'(?:STANDARD\s+CHARTERED|SCB).*?(?:S/A|A/C|CUSTODY).*?(?:PT\.?)?\s*(.+?)(?:\s*[-–—]\s*\d+[A-Z]*|\s*\([^)]+\)|$)', 'Standard Chartered'),
        (r'(?:BOS\s+LTD|BANK\s+OF\s+SINGAPORE|BINOVSE).*?(?:S/A|A/C).*?(?:PT\.?)?\s*(.+?)(?:\s*[-–—]\s*\d+[A-Z]*|\s*\([^)]+\)|$)', 'Bank of Singapore'),
        (r'(?:JPMCB|JPMORGAN|JINPVMECSBT).*?(?:RE-|NA\s+RE-).*?(?:PT\.?)?\s*(.+?)(?:\s*[-–—]\s*\d+[A-Z]*|\s*\([^)]+\)|$)', 'JPMorgan'),
        (r'(?:BNYM|BNPP).*?RE\s+(.+?)(?:\s*[-–—]\s*\d+[A-Z]*|\s*\([^)]+\)|$)', 'BNY Mellon'),
        (r'.*?(?:S/A|QQ|OBO|BENEFICIARY)\s+(?:PT\.?)?\s*(.+?)(?:\s*[-–—]\s*\d+[A-Z]*|\s*\([^)]+\)|$)', 'Nominee General'),
        (r'.*?(?:A/C\s+CLIENT|CLIENT\s+A/C|CLIENT)\s+(?:PT\.?)?\s*(.+?)(?:\s*[-–—]\s*\d+[A-Z]*|\s*\([^)]+\)|$)', 'Nominee General'),
    ]
    
    PLEDGE_KEYWORDS = ['PLEDGE', 'REPO', 'JAMINAN', 'AGUNAN', 'COLLATERAL', 'LOCKED', 'MARGIN', 'WM CLT']
    NOMINEE_KEYWORDS = ['S/A', 'A/C', 'FOR', 'BRANCH', 'TRUST', 'CUSTODIAN', 'BANKING', 'DIVISION']
    DIRECT_INDICATORS = [' PT', 'PT ', 'PT.', ' TBK', 'TBK ', ' LTD', ' INC', ' CORP', ' CO.', ' COMPANY', 
                        'DRS.', 'DR.', 'IR.', 'H.', 'HJ.', 'YAYASAN', 'DANA PENSIUN', 'KOPERASI']

    @staticmethod
    def clean_name(text):
        """Clean name from reference numbers - preserve company names"""
        if pd.isna(text) or text == '-': 
            return '-'
        text = str(text).strip()
        # Only remove trailing references, never remove from beginning
        text = re.sub(r'\s*[-–—]\s*\d+[A-Z]*$', '', text)
        text = re.sub(r'\s*[-–—]\s*[A-Z]+\d+$', '', text)
        text = re.sub(r'\s*\([A-Z0-9\s\-]+\)$', '', text)
        text = re.sub(r'\s*\d{6,}$', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip().upper()

    @staticmethod
    def is_direct(name):
        """Check if account is direct ownership (no nominee keywords)"""
        name = str(name).upper()
        # If contains nominee keywords, NOT direct
        if any(k in name for k in BandarXRay.NOMINEE_KEYWORDS):
            return False
        # Check for direct indicators
        name_clean = name.replace('.', '').replace(',', '')
        return any(k in name_clean for k in BandarXRay.DIRECT_INDICATORS)

    @classmethod
    def classify_account(cls, row):
        """3-Layer Forensic Classification"""
        holder = str(row['Nama Pemegang Saham']).upper()
        account = str(row['Nama Rekening Efek']).upper() if pd.notna(row.get('Nama Rekening Efek')) else ""
        
        # Default values
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
        
        # LAYER 2: NOMINEE PATTERN DETECTION
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
        
        # LAYER 3: DIRECT OWNERSHIP (ONLY IF NOT NOMINEE)
        if not nominee_found:
            if cls.is_direct(account):
                if holder in account or account in holder:
                    holding_type = "DIRECT"
                    real_owner = cls.clean_name(holder)
                else:
                    holding_type = "DIRECT (VARIANT)"
                    real_owner = cls.clean_name(account)
            else:
                holding_type = "DIRECT (ASSUMED)"
                real_owner = cls.clean_name(holder)
        
        # Append pledge status
        if is_pledge:
            holding_type += " [REPO]"
            
        return pd.Series([real_owner, holding_type, account_status, bank_source])

# ==============================================================================
# 3. DATA LOADERS - OPTIMIZED
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
    except Exception:
        pass
    return None

@st.cache_data(ttl=3600)
def load_data_complete():
    """Load KSEI and Price Data from Google Drive"""
    service = get_gdrive_service()
    if not service: 
        return pd.DataFrame(), pd.DataFrame()
    
    df_ksei = pd.DataFrame()
    df_price = pd.DataFrame()
    
    # ========== LOAD KSEI DATA ==========
    try:
        folder_id = st.secrets['gdrive']['folder_id']
        query = f"name = 'MASTER_DATABASE_5persen.csv' and '{folder_id}' in parents and trashed = false"
        results = service.files().list(q=query, fields="files(id)").execute()
        files = results.get('files', [])
        
        if files:
            file_id = files[0]['id']
            request = service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
            fh.seek(0)
            
            df_ksei = pd.read_csv(fh, dtype={'Kode Efek': str})
            df_ksei['Tanggal_Data'] = pd.to_datetime(df_ksei['Tanggal_Data'])
            
            # Clean numeric columns
            for col in ['Jumlah Saham (Prev)', 'Jumlah Saham (Curr)']:
                if col in df_ksei.columns:
                    df_ksei[col] = pd.to_numeric(
                        df_ksei[col].astype(str).str.replace(',', ''), 
                        errors='coerce'
                    ).fillna(0)
            
            df_ksei['Net_Flow'] = df_ksei['Jumlah Saham (Curr)'] - df_ksei['Jumlah Saham (Prev)']
            
            if 'Nama Rekening Efek' not in df_ksei.columns:
                df_ksei['Nama Rekening Efek'] = '-'
            df_ksei['Nama Rekening Efek'] = df_ksei['Nama Rekening Efek'].fillna('-')
            
    except Exception as e:
        st.error(f"❌ Error loading KSEI data: {e}")
    
    # ========== LOAD PRICE DATA ==========
    try:
        folder_id = st.secrets['gdrive']['folder_id']
        query = f"name = 'Kompilasi_Data_1Tahun.csv' and '{folder_id}' in parents and trashed = false"
        results = service.files().list(q=query, fields="files(id)").execute()
        files = results.get('files', [])
        
        if files:
            file_id = files[0]['id']
            request = service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
            fh.seek(0)
            
            df_price = pd.read_csv(fh)
            
            # Standardize column names
            column_mapping = {
                'Stock Code': 'Kode Efek',
                'Close': 'Harga_Close',
                'Volume': 'Volume_Harian',
                'Last Trading Date': 'Tanggal_Data',
                'Date': 'Tanggal_Data',
                'Tanggal': 'Tanggal_Data'
            }
            
            for old, new in column_mapping.items():
                if old in df_price.columns:
                    df_price[new] = df_price[old]
            
            if 'Tanggal_Data' in df_price.columns:
                df_price['Tanggal_Data'] = pd.to_datetime(df_price['Tanggal_Data'])
            
    except Exception as e:
        st.error(f"❌ Error loading Price data: {e}")
    
    return df_ksei, df_price

@st.cache_data(ttl=3600)
def process_forensics(df):
    """Apply forensic analysis with unique pairs optimization"""
    if df.empty:
        return df
    
    # Process only unique combinations (100x faster)
    unique_pairs = df[['Nama Pemegang Saham', 'Nama Rekening Efek']].drop_duplicates()
    
    # Apply classification
    forensic_results = unique_pairs.apply(
        BandarXRay.classify_account, 
        axis=1, 
        result_type='expand'
    )
    forensic_results.columns = ['REAL_OWNER', 'HOLDING_TYPE', 'ACCOUNT_STATUS', 'BANK_SOURCE']
    
    # Add back for merge
    unique_pairs = unique_pairs.reset_index(drop=True)
    forensic_results = forensic_results.reset_index(drop=True)
    unique_pairs = pd.concat([unique_pairs, forensic_results], axis=1)
    
    # Merge back to original dataframe
    df_result = pd.merge(
        df, 
        unique_pairs, 
        on=['Nama Pemegang Saham', 'Nama Rekening Efek'], 
        how='left'
    )
    
    return df_result

# ==============================================================================
# 4. MAIN APP - LOAD & PROCESS
# ==============================================================================
with st.spinner('🔄 Menghubungkan ke Google Drive & Memproses Data...'):
    df_ksei_raw, df_price = load_data_complete()
    
    if not df_ksei_raw.empty:
        df = process_forensics(df_ksei_raw)
    else:
        st.error("❌ GAGAL: Data KSEI tidak dapat dimuat. Periksa koneksi dan secrets.")
        st.stop()

# ==============================================================================
# 5. SIDEBAR - GLOBAL FILTERS
# ==============================================================================
with st.sidebar:
    st.title("🦅 X-RAY CONTROL")
    st.caption("Enterprise Edition v6.1")
    st.divider()
    
    # Stock Filter
    all_stocks = sorted(df['Kode Efek'].unique())
    sel_stock = st.multiselect(
        "📈 Filter Saham (Global)", 
        all_stocks,
        help="Filter ini berlaku untuk Tab 3, 4, 5. Tab Screener & Deep Dive memiliki filter sendiri."
    )
    
    # Date Filter
    min_date = df['Tanggal_Data'].min().date()
    max_date = df['Tanggal_Data'].max().date()
    sel_date = st.date_input(
        "📅 Periode Analisis",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    st.divider()
    
    # Advanced Filter - Minimum Flow untuk Screener
    min_flow_value = st.number_input(
        "💰 Minimal Net Flow (Rp)",
        min_value=0,
        value=1_000_000_000,  # 1 Milyar
        step=1_000_000_000,
        format="%d",
        help="Tampilkan hanya saham dengan estimasi flow > nilai ini di Stock Screener"
    )
    
    # Data Summary
    st.divider()
    st.subheader("📊 Data Summary")
    st.metric("Total Records", f"{len(df):,}")
    st.metric("Unique Stocks", f"{df['Kode Efek'].nunique()}")
    st.metric("Unique Holders", f"{df['REAL_OWNER'].nunique()}")
    
    if not df_price.empty:
        st.success("✅ Data Harga Tersedia")
    else:
        st.warning("⚠️ Data Harga Tidak Tersedia")

# ==============================================================================
# 6. APPLY GLOBAL FILTERS (UNTUK TAB 3,4,5)
# ==============================================================================
df_view = df.copy()
if sel_stock:
    df_view = df_view[df_view['Kode Efek'].isin(sel_stock)]
if len(sel_date) == 2:
    start_date, end_date = sel_date
    df_view = df_view[
        (df_view['Tanggal_Data'].dt.date >= start_date) & 
        (df_view['Tanggal_Data'].dt.date <= end_date)
    ]

# ==============================================================================
# 7. DASHBOARD TABS - 5 FEATURES
# ==============================================================================
st.title("🦅 Bandarmologi X-Ray: Enterprise Edition")
st.caption(f"Periode Global: {min_date} → {max_date} | Data: {len(df):,} records")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 STOCK SCREENER", 
    "🔬 DEEP DIVE", 
    "👑 ULTIMATE HOLDER",
    "⚠️ REPO MONITOR",
    "💰 SMART MONEY"
])

# ==============================================================================
# TAB 1: STOCK SCREENER - RADAR AKUMULASI (FIXED)
# ==============================================================================
with tab1:
    st.header("🎯 Stock Screener: Radar Akumulasi")
    st.markdown("Mendeteksi saham yang sedang diakumulasi/distribusi oleh Pemegang >5%")
    
    # Use unfiltered df for screener (but with date filter)
    df_screener = df.copy()
    if len(sel_date) == 2:
        df_screener = df_screener[
            (df_screener['Tanggal_Data'].dt.date >= sel_date[0]) & 
            (df_screener['Tanggal_Data'].dt.date <= sel_date[1])
        ]
    
    if not df_screener.empty:
        # Group by stock - FIXED: reset_index() to avoid duplicate indices
        screener = df_screener.groupby('Kode Efek').agg({
            'Net_Flow': 'sum',
            'Jumlah Saham (Curr)': 'sum',
            'REAL_OWNER': 'nunique'
        }).reset_index()  # ← FIXED: Prevents duplicate stock names
        
        # Merge with latest price
        if not df_price.empty:
            # Get latest price for each stock
            latest_prices = df_price.sort_values('Tanggal_Data').groupby('Kode Efek').last().reset_index()
            latest_prices = latest_prices[['Kode Efek', 'Harga_Close']]
            screener = pd.merge(screener, latest_prices, on='Kode Efek', how='left')
        else:
            screener['Harga_Close'] = 0
        
        # Calculate estimated value flow
        screener['Value_Flow'] = screener['Net_Flow'] * screener['Harga_Close']
        
        # Apply minimum flow filter
        screener = screener[abs(screener['Value_Flow']) >= min_flow_value]
        
        # Classify status
        screener['Status'] = screener['Net_Flow'].apply(
            lambda x: "AKUMULASI 🟢" if x > 0 else "DISTRIBUSI 🔴" if x < 0 else "NEUTRAL ⚪"
        )
        
        # Sort and display
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🟢 Top 10 Akumulasi (Big Money In)")
            top_accum = screener[screener['Net_Flow'] > 0].sort_values('Value_Flow', ascending=False).head(10)
            if not top_accum.empty:
                st.dataframe(
                    top_accum[['Kode Efek', 'Net_Flow', 'Value_Flow', 'Harga_Close', 'REAL_OWNER']]
                    .style.format({
                        'Net_Flow': '{:,.0f}',
                        'Value_Flow': 'Rp {:,.0f}',
                        'Harga_Close': 'Rp {:,.0f}'
                    }),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Kode Efek": "Saham",
                        "Net_Flow": "Volume Beli",
                        "Value_Flow": "Estimasi Nilai",
                        "Harga_Close": "Harga",
                        "REAL_OWNER": "Jumlah Bandar"
                    }
                )
            else:
                st.info("Tidak ada saham dengan akumulasi signifikan")
        
        with col2:
            st.subheader("🔴 Top 10 Distribusi (Big Money Out)")
            top_dist = screener[screener['Net_Flow'] < 0].sort_values('Value_Flow', ascending=True).head(10)
            if not top_dist.empty:
                st.dataframe(
                    top_dist[['Kode Efek', 'Net_Flow', 'Value_Flow', 'Harga_Close', 'REAL_OWNER']]
                    .style.format({
                        'Net_Flow': '{:,.0f}',
                        'Value_Flow': 'Rp {:,.0f}',
                        'Harga_Close': 'Rp {:,.0f}'
                    }),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Kode Efek": "Saham",
                        "Net_Flow": "Volume Jual",
                        "Value_Flow": "Estimasi Nilai",
                        "Harga_Close": "Harga",
                        "REAL_OWNER": "Jumlah Bandar"
                    }
                )
            else:
                st.info("Tidak ada saham dengan distribusi signifikan")
        
        # Summary metrics
        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            total_accum = screener[screener['Net_Flow'] > 0]['Value_Flow'].sum()
            st.metric("Total Akumulasi", f"Rp {total_accum:,.0f}")
        with col2:
            total_dist = abs(screener[screener['Net_Flow'] < 0]['Value_Flow'].sum())
            st.metric("Total Distribusi", f"Rp {total_dist:,.0f}")
        with col3:
            net_flow = total_accum - total_dist
            st.metric("Net Flow", f"Rp {net_flow:+,.0f}", 
                     delta="Positif" if net_flow > 0 else "Negatif")

# ==============================================================================
# TAB 2: DEEP DIVE - SINGLE STOCK ANALYSIS (FIXED)
# ==============================================================================
with tab2:
    st.header("🔬 Deep Dive Analysis")
    st.markdown("Bedah tuntas satu saham: Korelasi Harga vs Pergerakan Bandar")
    
    # Stock selector - independent of global filter
    target_stock = st.selectbox("🔍 Pilih Saham untuk Deep Dive:", all_stocks)
    
    if target_stock:
        # Filter data for selected stock (no global filter)
        df_deep = df[df['Kode Efek'] == target_stock].copy()
        df_deep = df_deep.sort_values('Tanggal_Data')
        
        if not df_deep.empty:
            # ========== CHART 1: PRICE VS BANDAR HOLDINGS ==========
            st.subheader("📈 Korelasi: Harga Saham vs Kepemilikan Bandar")
            
            # Calculate daily total holdings
            daily_holdings = df_deep.groupby('Tanggal_Data')['Jumlah Saham (Curr)'].sum().reset_index()
            
            # Get price data
            price_deep = df_price[df_price['Kode Efek'] == target_stock].sort_values('Tanggal_Data') if not df_price.empty else pd.DataFrame()
            
            # Create dual-axis chart
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Bandar holdings (area chart)
            fig.add_trace(
                go.Scatter(
                    x=daily_holdings['Tanggal_Data'], 
                    y=daily_holdings['Jumlah Saham (Curr)'], 
                    name="Kepemilikan Bandar",
                    fill='tozeroy',
                    line=dict(color='#00CC96', width=2),
                    hovertemplate='Tanggal: %{x}<br>Bandar Holdings: %{y:,.0f}<extra></extra>'
                ),
                secondary_y=False
            )
            
            # Price line
            if not price_deep.empty:
                # Filter price dates to match KSEI data range
                min_date_deep = daily_holdings['Tanggal_Data'].min()
                max_date_deep = daily_holdings['Tanggal_Data'].max()
                price_filtered = price_deep[
                    (price_deep['Tanggal_Data'] >= min_date_deep) & 
                    (price_deep['Tanggal_Data'] <= max_date_deep)
                ]
                
                fig.add_trace(
                    go.Scatter(
                        x=price_filtered['Tanggal_Data'], 
                        y=price_filtered['Harga_Close'],
                        name="Harga Saham",
                        line=dict(color='#FFA500', width=2.5),
                        hovertemplate='Tanggal: %{x}<br>Harga: Rp %{y:,.0f}<extra></extra>'
                    ),
                    secondary_y=True
                )
            
            fig.update_layout(
                title=f"{target_stock} - Korelasi Harga vs Akumulasi Bandar",
                hovermode="x unified",
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
            )
            fig.update_yaxes(title_text="Jumlah Saham (Lembar)", secondary_y=False, gridcolor='lightgray')
            fig.update_yaxes(title_text="Harga (Rp)", secondary_y=True, gridcolor='lightgray')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # ========== TABLE: WHO IS MOVING (FIXED) ==========
            st.subheader(f"👥 Pergerakan Ultimate Holder - {target_stock}")
            
            # FIXED: Use date filter from sidebar for this table
            if len(sel_date) == 2:
                mask_date = (df_deep['Tanggal_Data'].dt.date >= sel_date[0]) & \
                           (df_deep['Tanggal_Data'].dt.date <= sel_date[1])
                df_deep_period = df_deep.loc[mask_date].copy()
            else:
                df_deep_period = df_deep.copy()
            
            # Calculate flow per ultimate holder
            flow_analysis = df_deep_period.groupby('REAL_OWNER').agg({
                'Net_Flow': 'sum',
                'Jumlah Saham (Curr)': 'last',
                'HOLDING_TYPE': 'first',
                'BANK_SOURCE': lambda x: ', '.join(set([b for b in x if b != '-']))[:30]
            }).reset_index()
            
            # Filter only active movers
            active_movers = flow_analysis[flow_analysis['Net_Flow'] != 0].sort_values('Net_Flow', ascending=False)
            
            if not active_movers.empty:
                # Top movers summary
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**🟢 Top Accumulator**")
                    top_buyers = active_movers[active_movers['Net_Flow'] > 0].head(3)
                    for _, row in top_buyers.iterrows():
                        st.write(f"- {row['REAL_OWNER'][:30]}: +{row['Net_Flow']:,.0f} saham")
                
                with col2:
                    st.markdown("**🔴 Top Distributor**")
                    top_sellers = active_movers[active_movers['Net_Flow'] < 0].head(3)
                    for _, row in top_sellers.iterrows():
                        st.write(f"- {row['REAL_OWNER'][:30]}: {row['Net_Flow']:,.0f} saham")
                
                st.divider()
                
                # Detailed table
                st.dataframe(
                    active_movers.style.format({
                        'Net_Flow': '{:+,.0f}',
                        'Jumlah Saham (Curr)': '{:,.0f}'
                    }).applymap(
                        lambda v: 'color: green' if v > 0 else 'color: red' if v < 0 else '',
                        subset=['Net_Flow']
                    ),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "REAL_OWNER": "Ultimate Holder",
                        "Net_Flow": "Net Flow (Periode)",
                        "Jumlah Saham (Curr)": "Posisi Akhir",
                        "HOLDING_TYPE": "Tipe",
                        "BANK_SOURCE": "Bank"
                    }
                )
            else:
                st.info("ℹ️ Tidak ada perubahan kepemilikan signifikan pada periode ini")
            
            # ========== PIE CHART: OWNERSHIP COMPOSITION ==========
            st.subheader("🥧 Komposisi Kepemilikan (Posisi Terakhir)")
            
            last_date_deep = df_deep['Tanggal_Data'].max()
            last_holdings = df_deep[df_deep['Tanggal_Data'] == last_date_deep]
            
            # Group by ultimate holder
            pie_data = last_holdings.groupby('REAL_OWNER')['Jumlah Saham (Curr)'].sum().reset_index()
            pie_data = pie_data.sort_values('Jumlah Saham (Curr)', ascending=False).head(8)
            
            fig_pie = px.pie(
                pie_data,
                values='Jumlah Saham (Curr)',
                names='REAL_OWNER',
                title=f"Komposisi Ultimate Holder - {target_stock} (per {last_date_deep.date()})",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)

# ==============================================================================
# TAB 3: ULTIMATE HOLDER - GLOBAL VIEW
# ==============================================================================
with tab3:
    st.header("👑 Ultimate Holder (Global View)")
    st.caption("Konsolidasi kepemilikan per ultimate holder (semua akun nominee digabung)")
    
    if not df_view.empty:
        last_date = df_view['Tanggal_Data'].max()
        df_last = df_view[df_view['Tanggal_Data'] == last_date]
        
        uh_group = df_last.groupby('REAL_OWNER').agg({
            'Jumlah Saham (Curr)': 'sum',
            'Kode Efek': 'nunique',
            'ACCOUNT_STATUS': lambda x: '⚠️ REPO' if any('PLEDGE' in str(s) for s in x) else 'CLEAN',
            'HOLDING_TYPE': lambda x: ' | '.join(set([t.split()[0] for t in x]))[:30]
        }).sort_values('Jumlah Saham (Curr)', ascending=False).head(50)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(
                uh_group.style.format({'Jumlah Saham (Curr)': '{:,.0f}'}),
                use_container_width=True,
                column_config={
                    "REAL_OWNER": "Ultimate Holder",
                    "Jumlah Saham (Curr)": "Total Holdings",
                    "Kode Efek": "Portfolio Size",
                    "ACCOUNT_STATUS": "Status",
                    "HOLDING_TYPE": "Tipe"
                }
            )
        
        with col2:
            # Top 5 pie chart
            top5 = uh_group.head(5).reset_index()
            fig = px.pie(
                top5,
                values='Jumlah Saham (Curr)',
                names='REAL_OWNER',
                title="Top 5 Ultimate Holder",
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Metrics
            total_value = df_last['Jumlah Saham (Curr)'].sum()
            st.metric("Total Market Value (>5%)", f"{total_value:,.0f}")
    else:
        st.warning("Tidak ada data dengan filter yang dipilih")

# ==============================================================================
# TAB 4: REPO MONITOR - PLEDGE DETECTION
# ==============================================================================
with tab4:
    st.header("⚠️ Repo & Pledge Monitor")
    st.caption("Mendeteksi akun dengan indikasi saham digadaikan (forced sell risk)")
    
    df_repo = df_view[df_view['ACCOUNT_STATUS'].str.contains("PLEDGE|REPO", na=False)]
    
    if not df_repo.empty:
        repo_last = df_repo[df_repo['Tanggal_Data'] == df_repo['Tanggal_Data'].max()]
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Saham Tergadai", f"{repo_last['Jumlah Saham (Curr)'].sum():,.0f}")
        with col2:
            st.metric("Ultimate Holders", f"{repo_last['REAL_OWNER'].nunique()}")
        with col3:
            pledge_ratio = repo_last['Jumlah Saham (Curr)'].sum() / df_view[df_view['Tanggal_Data'] == df_view['Tanggal_Data'].max()]['Jumlah Saham (Curr)'].sum() * 100
            st.metric("% of Holdings", f"{pledge_ratio:.1f}%")
        
        # Detail table
        st.subheader("📋 Daftar Akun Terindikasi Repo")
        st.dataframe(
            repo_last[['REAL_OWNER', 'Kode Efek', 'Nama Pemegang Saham', 'BANK_SOURCE', 'Jumlah Saham (Curr)']]
            .sort_values('Jumlah Saham (Curr)', ascending=False)
            .style.format({'Jumlah Saham (Curr)': '{:,.0f}'}),
            use_container_width=True,
            hide_index=True,
            column_config={
                "REAL_OWNER": "Ultimate Holder",
                "Kode Efek": "Saham",
                "Nama Pemegang Saham": "Kustodian",
                "BANK_SOURCE": "Bank",
                "Jumlah Saham (Curr)": "Jumlah"
            }
        )
        
        # Historical trend
        st.subheader("📈 Historical Repo Trend")
        repo_trend = df_repo.groupby('Tanggal_Data')['Jumlah Saham (Curr)'].sum().reset_index()
        fig = px.area(
            repo_trend,
            x='Tanggal_Data',
            y='Jumlah Saham (Curr)',
            title="Volume Saham Digadaikan Over Time",
            color_discrete_sequence=['#FF4B4B']
        )
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.success("✅ AMAN! Tidak ditemukan indikasi Repo/Gadai pada periode ini")

# ==============================================================================
# TAB 5: SMART MONEY - PORTFOLIO VALUATION (FIXED)
# ==============================================================================
with tab5:
    st.header("💰 Smart Money Leaderboard")
    st.caption("Estimasi nilai portofolio ultimate holder berdasarkan harga pasar terkini")
    
    if df_price.empty:
        st.warning("⚠️ Data harga tidak tersedia. Fitur ini membutuhkan file Kompilasi_Data_1Tahun.csv")
        
        if st.button("🔄 Coba Load Ulang Data Harga", type="primary"):
            st.cache_data.clear()
            st.rerun()
            
    else:
        if not df_view.empty:
            # FIXED: Use snapshot from last date - NO DOUBLE COUNTING!
            last_date = df_view['Tanggal_Data'].max()
            df_last = df_view[df_view['Tanggal_Data'] == last_date].copy()
            
            # Get latest price for each stock
            latest_prices = df_price.sort_values('Tanggal_Data').groupby('Kode Efek').last().reset_index()
            latest_prices = latest_prices[['Kode Efek', 'Harga_Close']]
            
            # Merge holdings with latest prices
            df_valuation = pd.merge(df_last, latest_prices, on='Kode Efek', how='left')
            
            # Calculate valuation
            df_valuation['Valuasi'] = df_valuation['Jumlah Saham (Curr)'] * df_valuation['Harga_Close']
            df_valuation = df_valuation.dropna(subset=['Valuasi'])
            
            if not df_valuation.empty:
                # Group by ultimate holder
                ranking = df_valuation.groupby('REAL_OWNER').agg({
                    'Valuasi': 'sum',
                    'Kode Efek': 'nunique',
                    'Jumlah Saham (Curr)': 'sum'
                }).sort_values('Valuasi', ascending=False).head(20)
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Portfolio Value", f"Rp {ranking['Valuasi'].sum():,.0f}")
                with col2:
                    st.metric("Top Holder", ranking.index[0][:20] if not ranking.empty else "-")
                with col3:
                    st.metric("Total Holdings", f"{ranking['Jumlah Saham (Curr)'].sum():,.0f}")
                
                # Leaderboard
                st.subheader("🏆 Top 20 Ultimate Holder by Portfolio Value")
                st.dataframe(
                    ranking.style.format({
                        'Valuasi': 'Rp {:,.0f}',
                        'Jumlah Saham (Curr)': '{:,.0f}'
                    }),
                    use_container_width=True,
                    column_config={
                        "REAL_OWNER": "Ultimate Holder",
                        "Valuasi": "💰 Estimasi Nilai Portfolio",
                        "Kode Efek": "📈 Jumlah Saham",
                        "Jumlah Saham (Curr)": "📦 Total Lembar"
                    }
                )
                
                # Bar chart
                fig = px.bar(
                    ranking.head(10).reset_index(),
                    x='REAL_OWNER',
                    y='Valuasi',
                    title="Top 10 Ultimate Holder by Portfolio Value",
                    color='Valuasi',
                    color_continuous_scale='Viridis',
                    text_auto='.2s'
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.warning("Tidak ada data valuasi yang bisa dihitung")
        else:
            st.warning("Tidak ada data dengan filter yang dipilih")

# ==============================================================================
# 8. FOOTER
# ==============================================================================
st.divider()
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.caption(f"""
    🦅 **Bandarmologi X-Ray Enterprise Edition v6.1**  
    Forensic Unmasking | Stock Screener | Deep Dive | Repo Hunter | Smart Money  
    Data Source: KSEI · Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
    *See through the nominees, track the smart money*
    """)
