"""
================================================================================
🦅 BANDARMOLOGI X-RAY - ENTERPRISE EDITION v6.2
================================================================================
✅ STOCK SCREENER: Deteksi saham akumulasi/distribusi otomatis
✅ DEEP DIVE: Analisa satu saham (Chart Harga vs Flow Bandar)
✅ UNMASKING ENGINE: Tetap aktif (Logic Prioritas Bank)
✅ P&L ANALYSIS: Unrealized Profit/Loss, Win Rate, Entry/Exit Price
✅ REPO MONITOR: Deteksi saham digadaikan
✅ ULTIMATE HOLDER: Konsolidasi kepemilikan
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
        if any(k in name for k in BandarXRay.NOMINEE_KEYWORDS):
            return False
        name_clean = name.replace('.', '').replace(',', '')
        return any(k in name_clean for k in BandarXRay.DIRECT_INDICATORS)

    @classmethod
    def classify_account(cls, row):
        """3-Layer Forensic Classification"""
        holder = str(row['Nama Pemegang Saham']).upper()
        account = str(row['Nama Rekening Efek']).upper() if pd.notna(row.get('Nama Rekening Efek')) else ""
        
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
        
        if is_pledge:
            holding_type += " [REPO]"
            
        return pd.Series([real_owner, holding_type, account_status, bank_source])

# ==============================================================================
# 3. PRICE ANALYZER ENGINE - P&L CALCULATION (RESTORED FROM v5.0!)
# ==============================================================================
class PriceAnalyzer:
    """Menganalisis performa trading Ultimate Holder dengan data harga"""
    
    @staticmethod
    def calculate_entry_exit_price(ksei_df, price_df):
        """Hitung harga beli/rata-rata untuk setiap transaksi"""
        merged = pd.merge(
            ksei_df,
            price_df[['Kode Efek', 'Tanggal_Data', 'Harga_Close']],
            on=['Kode Efek', 'Tanggal_Data'],
            how='left'
        )
        merged['Transaction_Value'] = merged['Net_Flow'] * merged['Harga_Close']
        return merged
    
    @staticmethod
    def calculate_performance_metrics(holder_df):
        """
        Hitung metrics performa per Ultimate Holder:
        - Average buy price
        - Average sell price
        - Unrealized P&L
        - Win rate
        - Return %
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
# 4. DATA LOADERS
# ==============================================================================
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

@st.cache_resource
def get_gdrive_service():
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
    
    unique_pairs = df[['Nama Pemegang Saham', 'Nama Rekening Efek']].drop_duplicates()
    
    forensic_results = unique_pairs.apply(
        BandarXRay.classify_account, 
        axis=1, 
        result_type='expand'
    )
    forensic_results.columns = ['REAL_OWNER', 'HOLDING_TYPE', 'ACCOUNT_STATUS', 'BANK_SOURCE']
    
    unique_pairs = unique_pairs.reset_index(drop=True)
    forensic_results = forensic_results.reset_index(drop=True)
    unique_pairs = pd.concat([unique_pairs, forensic_results], axis=1)
    
    df_result = pd.merge(
        df, 
        unique_pairs, 
        on=['Nama Pemegang Saham', 'Nama Rekening Efek'], 
        how='left'
    )
    
    return df_result

# ==============================================================================
# 5. MAIN APP - LOAD & PROCESS
# ==============================================================================
with st.spinner('🔄 Menghubungkan ke Google Drive & Memproses Data...'):
    df_ksei_raw, df_price = load_data_complete()
    
    if not df_ksei_raw.empty:
        df = process_forensics(df_ksei_raw)
    else:
        st.error("❌ GAGAL: Data KSEI tidak dapat dimuat.")
        st.stop()

# ==============================================================================
# 6. SIDEBAR - GLOBAL FILTERS
# ==============================================================================
with st.sidebar:
    st.title("🦅 X-RAY CONTROL")
    st.caption("Enterprise Edition v6.2")
    st.divider()
    
    all_stocks = sorted(df['Kode Efek'].unique())
    sel_stock = st.multiselect(
        "📈 Filter Saham (Global)", 
        all_stocks,
        help="Filter ini berlaku untuk Tab 3, 4, 5, 6"
    )
    
    min_date = df['Tanggal_Data'].min().date()
    max_date = df['Tanggal_Data'].max().date()
    sel_date = st.date_input(
        "📅 Periode Analisis",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    st.divider()
    
    min_flow_value = st.number_input(
        "💰 Minimal Net Flow (Rp)",
        min_value=0,
        value=1_000_000_000,
        step=1_000_000_000,
        format="%d",
        help="Tampilkan hanya saham dengan estimasi flow > nilai ini"
    )
    
    st.divider()
    st.subheader("📊 Data Summary")
    st.metric("Total Records", f"{len(df):,}")
    st.metric("Unique Stocks", f"{df['Kode Efek'].nunique()}")
    st.metric("Unique Holders", f"{df['REAL_OWNER'].nunique()}")
    
    if not df_price.empty:
        st.success(f"✅ Data Harga: {len(df_price):,} records")
    else:
        st.warning("⚠️ Data Harga Tidak Tersedia")

# ==============================================================================
# 7. APPLY GLOBAL FILTERS
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
# 8. DASHBOARD TABS - 6 FEATURES!
# ==============================================================================
st.title("🦅 Bandarmologi X-Ray: Enterprise Edition")
st.caption(f"Periode Global: {min_date} → {max_date} | Data: {len(df):,} records")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎯 STOCK SCREENER", 
    "🔬 DEEP DIVE", 
    "👑 ULTIMATE HOLDER",
    "⚠️ REPO MONITOR",
    "💰 PORTFOLIO VALUE",
    "📊 P&L ANALYSIS"  # NEW TAB - RESTORED!
])

# ==============================================================================
# TAB 1: STOCK SCREENER
# ==============================================================================
with tab1:
    st.header("🎯 Stock Screener: Radar Akumulasi")
    st.markdown("Mendeteksi saham yang sedang diakumulasi/distribusi oleh Pemegang >5%")
    
    df_screener = df.copy()
    if len(sel_date) == 2:
        df_screener = df_screener[
            (df_screener['Tanggal_Data'].dt.date >= sel_date[0]) & 
            (df_screener['Tanggal_Data'].dt.date <= sel_date[1])
        ]
    
    if not df_screener.empty:
        screener = df_screener.groupby('Kode Efek').agg({
            'Net_Flow': 'sum',
            'Jumlah Saham (Curr)': 'sum',
            'REAL_OWNER': 'nunique'
        }).reset_index()
        
        if not df_price.empty:
            latest_prices = df_price.sort_values('Tanggal_Data').groupby('Kode Efek').last().reset_index()
            latest_prices = latest_prices[['Kode Efek', 'Harga_Close']]
            screener = pd.merge(screener, latest_prices, on='Kode Efek', how='left')
        else:
            screener['Harga_Close'] = 0
        
        screener['Value_Flow'] = screener['Net_Flow'] * screener['Harga_Close']
        screener = screener[abs(screener['Value_Flow']) >= min_flow_value]
        screener['Status'] = screener['Net_Flow'].apply(
            lambda x: "AKUMULASI 🟢" if x > 0 else "DISTRIBUSI 🔴" if x < 0 else "NEUTRAL ⚪"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🟢 Top 10 Akumulasi")
            top_accum = screener[screener['Net_Flow'] > 0].sort_values('Value_Flow', ascending=False).head(10)
            if not top_accum.empty:
                st.dataframe(top_accum[['Kode Efek', 'Net_Flow', 'Value_Flow', 'Harga_Close', 'REAL_OWNER']]
                    .style.format({'Net_Flow': '{:,.0f}', 'Value_Flow': 'Rp {:,.0f}', 'Harga_Close': 'Rp {:,.0f}'}),
                    use_container_width=True, hide_index=True)
        
        with col2:
            st.subheader("🔴 Top 10 Distribusi")
            top_dist = screener[screener['Net_Flow'] < 0].sort_values('Value_Flow', ascending=True).head(10)
            if not top_dist.empty:
                st.dataframe(top_dist[['Kode Efek', 'Net_Flow', 'Value_Flow', 'Harga_Close', 'REAL_OWNER']]
                    .style.format({'Net_Flow': '{:,.0f}', 'Value_Flow': 'Rp {:,.0f}', 'Harga_Close': 'Rp {:,.0f}'}),
                    use_container_width=True, hide_index=True)

# ==============================================================================
# TAB 2: DEEP DIVE
# ==============================================================================
with tab2:
    st.header("🔬 Deep Dive Analysis")
    
    target_stock = st.selectbox("🔍 Pilih Saham untuk Deep Dive:", all_stocks)
    
    if target_stock:
        df_deep = df[df['Kode Efek'] == target_stock].copy().sort_values('Tanggal_Data')
        
        if not df_deep.empty:
            st.subheader("📈 Korelasi: Harga vs Kepemilikan Bandar")
            
            daily_holdings = df_deep.groupby('Tanggal_Data')['Jumlah Saham (Curr)'].sum().reset_index()
            price_deep = df_price[df_price['Kode Efek'] == target_stock].sort_values('Tanggal_Data') if not df_price.empty else pd.DataFrame()
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Scatter(x=daily_holdings['Tanggal_Data'], y=daily_holdings['Jumlah Saham (Curr)'], 
                          name="Kepemilikan Bandar", fill='tozeroy', line=dict(color='#00CC96')),
                secondary_y=False
            )
            
            if not price_deep.empty:
                min_d = daily_holdings['Tanggal_Data'].min()
                max_d = daily_holdings['Tanggal_Data'].max()
                price_filtered = price_deep[(price_deep['Tanggal_Data'] >= min_d) & (price_deep['Tanggal_Data'] <= max_d)]
                fig.add_trace(
                    go.Scatter(x=price_filtered['Tanggal_Data'], y=price_filtered['Harga_Close'],
                              name="Harga Saham", line=dict(color='#FFA500', width=2.5)),
                    secondary_y=True
                )
            
            fig.update_layout(height=500, hovermode="x unified")
            fig.update_yaxes(title_text="Jumlah Saham", secondary_y=False)
            fig.update_yaxes(title_text="Harga (Rp)", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)
            
            # Active movers table
            st.subheader(f"👥 Pergerakan Ultimate Holder")
            if len(sel_date) == 2:
                mask = (df_deep['Tanggal_Data'].dt.date >= sel_date[0]) & (df_deep['Tanggal_Data'].dt.date <= sel_date[1])
                df_deep_period = df_deep.loc[mask].copy()
            else:
                df_deep_period = df_deep.copy()
            
            flow_analysis = df_deep_period.groupby('REAL_OWNER').agg({
                'Net_Flow': 'sum', 'Jumlah Saham (Curr)': 'last'
            }).reset_index()
            active_movers = flow_analysis[flow_analysis['Net_Flow'] != 0].sort_values('Net_Flow', ascending=False)
            
            if not active_movers.empty:
                st.dataframe(active_movers.style.format({'Net_Flow': '{:+,.0f}', 'Jumlah Saham (Curr)': '{:,.0f}'})
                    .applymap(lambda v: 'color: green' if v > 0 else 'color: red' if v < 0 else '', subset=['Net_Flow']),
                    use_container_width=True, hide_index=True)

# ==============================================================================
# TAB 3: ULTIMATE HOLDER
# ==============================================================================
with tab3:
    st.header("👑 Ultimate Holder (Global View)")
    
    if not df_view.empty:
        last_date = df_view['Tanggal_Data'].max()
        df_last = df_view[df_view['Tanggal_Data'] == last_date]
        
        uh_group = df_last.groupby('REAL_OWNER').agg({
            'Jumlah Saham (Curr)': 'sum',
            'Kode Efek': 'nunique',
            'ACCOUNT_STATUS': lambda x: '⚠️ REPO' if any('PLEDGE' in str(s) for s in x) else 'CLEAN'
        }).sort_values('Jumlah Saham (Curr)', ascending=False).head(50)
        
        st.dataframe(uh_group.style.format({'Jumlah Saham (Curr)': '{:,.0f}'}), use_container_width=True)

# ==============================================================================
# TAB 4: REPO MONITOR
# ==============================================================================
with tab4:
    st.header("⚠️ Repo & Pledge Monitor")
    
    df_repo = df_view[df_view['ACCOUNT_STATUS'].str.contains("PLEDGE|REPO", na=False)]
    
    if not df_repo.empty:
        repo_last = df_repo[df_repo['Tanggal_Data'] == df_repo['Tanggal_Data'].max()]
        st.dataframe(
            repo_last[['REAL_OWNER', 'Kode Efek', 'Nama Pemegang Saham', 'BANK_SOURCE', 'Jumlah Saham (Curr)']]
            .sort_values('Jumlah Saham (Curr)', ascending=False)
            .style.format({'Jumlah Saham (Curr)': '{:,.0f}'}),
            use_container_width=True, hide_index=True
        )
    else:
        st.success("✅ Tidak ada data Repo pada filter ini")

# ==============================================================================
# TAB 5: PORTFOLIO VALUATION (FIXED - NO DOUBLE COUNT)
# ==============================================================================
with tab5:
    st.header("💰 Portfolio Valuation")
    st.caption("Estimasi nilai portofolio berdasarkan harga pasar terkini")
    
    if df_price.empty:
        st.warning("⚠️ Data harga tidak tersedia")
    else:
        if not df_view.empty:
            last_date = df_view['Tanggal_Data'].max()
            df_last = df_view[df_view['Tanggal_Data'] == last_date].copy()
            
            latest_prices = df_price.sort_values('Tanggal_Data').groupby('Kode Efek').last().reset_index()
            latest_prices = latest_prices[['Kode Efek', 'Harga_Close']]
            df_valuation = pd.merge(df_last, latest_prices, on='Kode Efek', how='left')
            df_valuation['Valuasi'] = df_valuation['Jumlah Saham (Curr)'] * df_valuation['Harga_Close']
            df_valuation = df_valuation.dropna(subset=['Valuasi'])
            
            if not df_valuation.empty:
                ranking = df_valuation.groupby('REAL_OWNER').agg({
                    'Valuasi': 'sum', 'Kode Efek': 'nunique'
                }).sort_values('Valuasi', ascending=False).head(20)
                
                st.dataframe(
                    ranking.style.format({'Valuasi': 'Rp {:,.0f}'}),
                    use_container_width=True,
                    column_config={"REAL_OWNER": "Ultimate Holder", "Valuasi": "Nilai Portfolio", "Kode Efek": "Jumlah Saham"}
                )

# ==============================================================================
# TAB 6: P&L ANALYSIS - RESTORED FROM v5.0!
# ==============================================================================
with tab6:
    st.header("📊 Profit & Loss Analysis")
    st.caption("Analisis performa trading Ultimate Holder: Entry Price, Unrealized P&L, Win Rate")
    
    if df_price.empty:
        st.warning("⚠️ Data harga tidak tersedia. Fitur ini membutuhkan file Kompilasi_Data_1Tahun.csv")
    else:
        if not df_view.empty:
            # Merge dengan data harga
            merged_df = pd.merge(
                df_view,
                df_price[['Kode Efek', 'Tanggal_Data', 'Harga_Close']],
                on=['Kode Efek', 'Tanggal_Data'],
                how='inner'
            )
            
            if not merged_df.empty:
                # Hitung performance metrics
                perf_df = PriceAnalyzer.calculate_performance_metrics(merged_df)
                
                if not perf_df.empty:
                    # Aggregate per ultimate holder
                    holder_perf = perf_df.groupby('REAL_OWNER').agg({
                        'Unrealized_PnL': 'sum',
                        'Unrealized_PnL_%': 'mean',
                        'Current_Value': 'sum',
                        'Win': 'sum',
                        'Has_Sell': 'sum',
                        'Kode Efek': 'nunique',
                        'Avg_Buy_Price': 'mean'
                    }).reset_index()
                    
                    holder_perf.columns = ['REAL_OWNER', 'Total_PnL', 'Avg_Return_%', 
                                          'Portfolio_Value', 'Win_Trades', 'Total_Sell_Trades',
                                          'Stocks_Held', 'Avg_Entry_Price']
                    
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
                        return f'color: {"green" if val > 0 else "red" if val < 0 else "white"}'
                    
                    st.dataframe(
                        holder_perf.head(20).style.format({
                            'Total_PnL': 'Rp {:,.0f}',
                            'Portfolio_Value': 'Rp {:,.0f}',
                            'Avg_Return_%': '{:.1f}%',
                            'Win_Rate': '{:.1f}%',
                            'Avg_Entry_Price': 'Rp {:,.0f}'
                        }).applymap(color_pnl, subset=['Total_PnL', 'Avg_Return_%']),
                        use_container_width=True,
                        column_config={
                            "REAL_OWNER": "👑 Ultimate Holder",
                            "Total_PnL": "💰 Unrealized P&L",
                            "Avg_Return_%": "📊 Avg Return",
                            "Portfolio_Value": "💵 Portfolio Value",
                            "Win_Rate": "🎯 Win Rate",
                            "Stocks_Held": "📈 Stocks",
                            "Avg_Entry_Price": "💰 Avg Entry Price"
                        }
                    )
                    
                    # CHARTS
                    col1, col2 = st.columns(2)
                    with col1:
                        fig_pnl = px.bar(
                            holder_perf.head(10),
                            x='REAL_OWNER', y='Total_PnL',
                            title="Top 10 Unrealized P&L",
                            color='Total_PnL', color_continuous_scale='RdYlGn',
                            text_auto='.2s'
                        )
                        fig_pnl.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig_pnl, use_container_width=True)
                    
                    with col2:
                        fig_win = px.scatter(
                            holder_perf.head(20),
                            x='Win_Rate', y='Total_PnL',
                            size='Portfolio_Value', color='Avg_Return_%',
                            hover_name='REAL_OWNER',
                            title="Win Rate vs Profitability",
                            labels={'Win_Rate': 'Win Rate (%)', 'Total_PnL': 'P&L (Rp)'},
                            color_continuous_scale='RdYlGn'
                        )
                        st.plotly_chart(fig_win, use_container_width=True)
                    
                    # DETAIL PER STOCK
                    with st.expander("📋 Detail Performa per Saham"):
                        stock_detail = perf_df.sort_values('Unrealized_PnL', ascending=False).head(50)
                        st.dataframe(
                            stock_detail[['REAL_OWNER', 'Kode Efek', 'Avg_Buy_Price', 
                                         'Current_Price', 'Current_Holding', 'Current_Value',
                                         'Unrealized_PnL', 'Unrealized_PnL_%']]
                            .style.format({
                                'Avg_Buy_Price': 'Rp {:,.0f}',
                                'Current_Price': 'Rp {:,.0f}',
                                'Current_Holding': '{:,.0f}',
                                'Current_Value': 'Rp {:,.0f}',
                                'Unrealized_PnL': 'Rp {:,.0f}',
                                'Unrealized_PnL_%': '{:.1f}%'
                            }).applymap(color_pnl, subset=['Unrealized_PnL', 'Unrealized_PnL_%']),
                            use_container_width=True
                        )
                else:
                    st.warning("Tidak cukup data untuk analisis performa")
            else:
                st.warning("Tidak ada data yang cocok antara KSEI dan data harga")
        else:
            st.warning("Tidak ada data dengan filter yang dipilih")

# ==============================================================================
# 9. FOOTER
# ==============================================================================
st.divider()
st.caption(f"""
🦅 **Bandarmologi X-Ray Enterprise Edition v6.2**  
Forensic Unmasking | Stock Screener | Deep Dive | Repo Hunter | Portfolio Value | P&L Analysis  
Data Source: KSEI · Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
*See through the nominees, track the smart money, analyze their performance*
""")
