"""
================================================================================
🦅 BANDARMOLOGI X-RAY - ENTERPRISE EDITION v7.0
================================================================================
Fitur Utama:
✅ Forensic Engine: Unmasking Nominee & Repo Detection (Logic Prioritas Benar)
✅ Price Integration: Menggunakan VWAP/Typical Price untuk Modal, Close untuk Valuasi
✅ Smart Money: Leaderboard PnL Estimasi
✅ Deep Dive: Chart Korelasi Bandar vs Harga
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
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. KONFIGURASI HALAMAN & TEMA
# ==============================================================================
st.set_page_config(
    layout="wide",
    page_title="Bandarmologi X-Ray",
    page_icon="🦅",
    initial_sidebar_state="expanded"
)

# Custom CSS (Fixed for Light Mode Visibility)
st.markdown("""
<style>
    /* Main container padding */
    .main > div { padding: 0rem 1rem; }
    
    /* Metric Value - HITAM BIAR JELAS */
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700;
        color: #000000 !important;
    }
    /* Metric Label - ABU TUA */
    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        font-weight: 500;
        color: #444444 !important;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #f0f2f6;
        padding: 0.5rem;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        padding: 0px 16px;
        border-radius: 8px;
        font-weight: 600;
        color: #31333F !important;
        background-color: #ffffff;
        border: 1px solid #ddd;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00CC96 !important;
        color: #ffffff !important;
        border-bottom: 2px solid #00FF00;
    }
    
    /* Headers - HITAM */
    h1, h2, h3 { color: #000000 !important; }
    
    /* Warning/Success Boxes */
    .stAlert { border-left: 5px solid #00CC96; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. BANDAR X-RAY ENGINE (FORENSIK)
# ==============================================================================
class BandarXRay:
    """Mesin Forensik: Membedah Nama Rekening Efek"""

    # Pola Regex untuk mendeteksi Bank Kustodian (Nominee)
    PATTERNS_NOMINEE = [
        (r'(?:HSBC|HPTS|HSTSRBACT|HSSTRBACT|HASDBACR|HDSIVBICSI|HINSVBECS).*?(?:PRIVATE|FUND|CLIENT|DIVISION).*?(?:PT\.?)?\s*(.+?)(?:\s*[-–—]\s*\d+[A-Z]*|\s*\([^)]+\)|$)', 'HSBC'),
        (r'(?:UBS\s+AG|USBTRS|U20B9S1|U20B2S3|UINBVSE|DINBVSE|UINBVS).*?(?:S/A|A/C|BRANCH|TR|SEPNOTRSE).*?(?:PT\.?)?\s*(.+?)(?:\s*[-–—]\s*\d+[A-Z]*|\s*\([^)]+\)|$)', 'UBS AG'),
        (r'(?:DB\s+AG|DEUTSCHE\s+BANK|D21B4|D20B4|D22B5S9).*?(?:A/C|S/A|CLT).*?(?:PT\.?)?\s*(.+?)(?:\s*[-–—]\s*\d+[A-Z]*|\s*\([^)]+\)|$)', 'Deutsche Bank'),
        (r'(?:CITIBANK|CITI).*?(?:S/A|CBHK|PBGSG).*?(?:PT\.?)?\s*(.+?)(?:\s*[-–—]\s*\d+[A-Z]*|\s*\([^)]+\)|$)', 'Citibank'),
        (r'(?:STANDARD\s+CHARTERED|SCB).*?(?:S/A|A/C|CUSTODY).*?(?:PT\.?)?\s*(.+?)(?:\s*[-–—]\s*\d+[A-Z]*|\s*\([^)]+\)|$)', 'Standard Chartered'),
        (r'(?:BOS\s+LTD|BANK\s+OF\s+SINGAPORE|BINOVSE).*?(?:S/A|A/C).*?(?:PT\.?)?\s*(.+?)(?:\s*[-–—]\s*\d+[A-Z]*|\s*\([^)]+\)|$)', 'Bank of Singapore'),
        (r'(?:JPMCB|JPMORGAN|JINPVMECSBT).*?(?:RE-|NA\s+RE-).*?(?:PT\.?)?\s*(.+?)(?:\s*[-–—]\s*\d+[A-Z]*|\s*\([^)]+\)|$)', 'JPMorgan'),
        (r'(?:BNYM|BNPP).*?RE\s+(.+?)(?:\s*[-–—]\s*\d+[A-Z]*|\s*\([^)]+\)|$)', 'BNY Mellon'),
        (r'(?:MAYBANK|M92A1Y7).*?S/A\s+(.+?)(?:\s*[-–—]\s*\d+[A-Z]*|\s*\([^)]+\)|$)', 'Maybank'),
        (r'.*?(?:S/A|QQ|OBO|BENEFICIARY)\s+(?:PT\.?)?\s*(.+?)(?:\s*[-–—]\s*\d+[A-Z]*|\s*\([^)]+\)|$)', 'Nominee General'),
    ]
    
    PLEDGE_KEYWORDS = ['PLEDGE', 'REPO', 'JAMINAN', 'AGUNAN', 'COLLATERAL', 'LOCKED', 'MARGIN', 'LENDING']
    
    DIRECT_INDICATORS = [
        ' PT', 'PT ', ' TBK', ' LTD', ' INC', ' CORP', ' COMPANY',
        'DRS.', 'DR.', 'IR.', 'H.', 'HJ.', 'YAYASAN', 'DANA PENSIUN', 'KOPERASI'
    ]

    @staticmethod
    def clean_name(text):
        """Membersihkan sampah referensi angka di belakang nama"""
        if pd.isna(text) or text == '-': return '-'
        text = str(text).strip()
        text = re.sub(r'\s*[-–—]\s*\d+.*$', '', text) # Hapus "- 20911..."
        text = re.sub(r'\s*\([A-Z0-9\s\-]+\)$', '', text) # Hapus "(ID123)"
        return text.strip().upper()

    @staticmethod
    def is_direct(name):
        name = str(name).upper()
        # Jika ada keyword bank, bukan direct
        if any(k in name for k in ['S/A', 'A/C', 'FOR', 'BRANCH', 'TRUST', 'CUSTODIAN']): return False
        # Jika ada indikator perusahaan/gelar, kemungkinan direct
        return any(k in name for k in BandarXRay.DIRECT_INDICATORS)

    @classmethod
    def classify_account(cls, row):
        holder = str(row['Nama Pemegang Saham']).upper()
        account = str(row['Nama Rekening Efek']).upper() if pd.notna(row.get('Nama Rekening Efek')) else ""
        
        # Default Values
        real_owner, holding_type, status, bank = cls.clean_name(holder), "DIRECT", "NORMAL", "-"
        
        # 1. CEK PLEDGE/REPO (Prioritas Utama)
        if any(k in account for k in cls.PLEDGE_KEYWORDS):
            status = "⚠️ PLEDGE/REPO"
        
        # 2. CEK NOMINEE/BANK (Unmasking)
        nominee_found = False
        if account and len(account) > 5:
            for pattern, source in cls.PATTERNS_NOMINEE:
                match = re.search(pattern, account, re.IGNORECASE)
                if match:
                    real_owner = cls.clean_name(match.group(1))
                    bank = source
                    holding_type = f"NOMINEE ({source})"
                    nominee_found = True
                    break
        
        # 3. CEK DIRECT
        if not nominee_found:
            if cls.is_direct(account):
                # Jika nama pemegang ada di dalam nama rekening (atau sebaliknya)
                if holder in account or account in holder:
                    holding_type = "DIRECT"
                    real_owner = cls.clean_name(holder)
                else:
                    holding_type = "DIRECT (VARIANT)"
                    real_owner = cls.clean_name(account)
            else:
                holding_type = "DIRECT (ASSUMED)"
        
        # Tambahkan label Repo ke tipe holding agar terlihat jelas
        if status != "NORMAL": 
            holding_type = f"{holding_type} [REPO]"
            
        return pd.Series([real_owner, holding_type, status, bank])

# ==============================================================================
# 3. PRICE ANALYZER (LOGIC PnL & VWAP)
# ==============================================================================
class PriceAnalyzer:
    """Menganalisis performa trading dengan Logic VWAP"""
    
    @staticmethod
    def calculate_entry_exit_price(ksei_df, price_df):
        # Merge KSEI dengan harga
        merged = pd.merge(
            ksei_df,
            price_df[['Kode Efek', 'Tanggal_Data', 'Harga_Close', 'Harga_Avg', 'Volume_Harian']],
            on=['Kode Efek', 'Tanggal_Data'],
            how='left'
        )
        
        # LOGIC MODAL: Net Flow (Lembar) * Harga Rata-Rata Harian (VWAP)
        merged['Transaction_Value'] = merged['Net_Flow'] * merged['Harga_Avg']
        
        return merged
    
    @staticmethod
    def calculate_performance_metrics(holder_df):
        results = []
        
        for (holder, stock), group in holder_df.groupby(['REAL_OWNER', 'Kode Efek']):
            group = group.sort_values('Tanggal_Data')
            
            # --- 1. MENGHITUNG MODAL BELI (AKUMULASI) ---
            # Kita pakai Harga_Avg (VWAP)
            buy_trades = group[group['Net_Flow'] > 0].copy()
            if not buy_trades.empty:
                # Total Rupiah yang dikeluarkan untuk beli
                total_money_in = (buy_trades['Net_Flow'] * buy_trades['Harga_Avg']).sum()
                total_shares_in = buy_trades['Net_Flow'].sum()
                
                # Harga Modal Rata-Rata (Average Buy Price)
                avg_buy_price = total_money_in / total_shares_in if total_shares_in > 0 else 0
            else:
                avg_buy_price = 0
                total_shares_in = 0
            
            # --- 2. MENGHITUNG PENJUALAN (DISTRIBUSI) ---
            sell_trades = group[group['Net_Flow'] < 0].copy()
            if not sell_trades.empty:
                # Total Rupiah yang didapat dari jual
                total_money_out = abs((sell_trades['Net_Flow'] * sell_trades['Harga_Avg']).sum())
                total_shares_out = abs(sell_trades['Net_Flow'].sum())
                
                avg_sell_price = total_money_out / total_shares_out if total_shares_out > 0 else 0
            else:
                avg_sell_price = 0
                total_shares_out = 0
            
            # --- 3. POSISI SEKARANG (VALUASI) ---
            # Sisa barang dihitung nilainya pakai HARGA CLOSE TERAKHIR (Mark-to-Market)
            # Karena data KSEI adalah posisi akhir hari, kita ambil data terakhir
            current_holding_shares = group.iloc[-1]['Jumlah Saham (Curr)'] # Ini satuan LEMBAR
            current_market_price = group.iloc[-1]['Harga_Close'] if 'Harga_Close' in group.columns else 0
            
            # Valuasi Portfolio = Lembar * Harga Close
            current_value_rp = current_holding_shares * current_market_price
            
            # --- 4. HITUNG PROFIT/LOSS (FLOATING) ---
            # PnL = (Harga Pasar Sekarang - Harga Modal Rata2) * Sisa Barang
            # Syarat: Harus punya barang dan punya harga modal
            if avg_buy_price > 0 and current_holding_shares > 0 and current_market_price > 0:
                unrealized_pnl_rp = (current_market_price - avg_buy_price) * current_holding_shares
                unrealized_pnl_pct = ((current_market_price - avg_buy_price) / avg_buy_price) * 100
            else:
                unrealized_pnl_rp = 0
                unrealized_pnl_pct = 0
            
            # Win/Loss Realized (Jual > Beli)
            wins = 0
            if avg_sell_price > 0 and avg_buy_price > 0:
                wins = 1 if avg_sell_price > avg_buy_price else 0
            
            results.append({
                'REAL_OWNER': holder,
                'Kode Efek': stock,
                'Avg_Buy_Price': avg_buy_price,   # Modal (VWAP)
                'Avg_Sell_Price': avg_sell_price, # Jualan (VWAP)
                'Current_Price': current_market_price, # Harga Pasar (Close)
                'Current_Holding_Shares': current_holding_shares, # Satuan Lembar
                'Current_Value_Rp': current_value_rp,
                'Unrealized_PnL_Rp': unrealized_pnl_rp,
                'Unrealized_PnL_%': unrealized_pnl_pct,
                'Total_Buy_Vol': total_shares_in,
                'Total_Sell_Vol': total_shares_out,
                'Win': wins,
                'Has_Sell': 1 if total_shares_out > 0 else 0
            })
        
        return pd.DataFrame(results)

# ==============================================================================
# 4. DATA LOADER
# ==============================================================================
@st.cache_resource
def get_gdrive_service():
    try:
        if "gdrive_creds" in st.secrets:
            creds = service_account.Credentials.from_service_account_info(
                st.secrets["gdrive_creds"], scopes=['https://www.googleapis.com/auth/drive.readonly'])
            return build('drive', 'v3', credentials=creds)
    except: pass
    return None

@st.cache_data(ttl=3600)
def load_data_complete():
    """Load KSEI and Price Data"""
    service = get_gdrive_service()
    if not service: return pd.DataFrame(), pd.DataFrame()
    
    # --- LOAD KSEI (5%) ---
    df_ksei = pd.DataFrame()
    try:
        q = f"name = 'MASTER_DATABASE_5persen.csv' and '{st.secrets['gdrive']['folder_id']}' in parents and trashed = false"
        res = service.files().list(q=q, fields="files(id)").execute()
        if res.get('files'):
            req = service.files().get_media(fileId=res['files'][0]['id'])
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, req)
            done = False
            while not done: _, done = downloader.next_chunk()
            fh.seek(0)
            
            df_ksei = pd.read_csv(fh, dtype={'Kode Efek': str})
            df_ksei['Tanggal_Data'] = pd.to_datetime(df_ksei['Tanggal_Data'])
            
            # Cleaning Angka & Hitung Flow
            for c in ['Jumlah Saham (Prev)', 'Jumlah Saham (Curr)']:
                df_ksei[c] = pd.to_numeric(df_ksei[c].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            df_ksei['Net_Flow'] = df_ksei['Jumlah Saham (Curr)'] - df_ksei['Jumlah Saham (Prev)']
            
            if 'Nama Rekening Efek' not in df_ksei.columns: df_ksei['Nama Rekening Efek'] = '-'
            df_ksei['Nama Rekening Efek'] = df_ksei['Nama Rekening Efek'].fillna('-')
    except Exception as e: st.error(f"KSEI Error: {e}")

    # --- LOAD PRICE (KOMPILASI) ---
    df_price = pd.DataFrame()
    try:
        q = f"name = 'Kompilasi_Data_1Tahun.csv' and '{st.secrets['gdrive']['folder_id']}' in parents and trashed = false"
        res = service.files().list(q=q, fields="files(id)").execute()
        if res.get('files'):
            req = service.files().get_media(fileId=res['files'][0]['id'])
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, req)
            done = False
            while not done: _, done = downloader.next_chunk()
            fh.seek(0)
            
            df_price = pd.read_csv(fh)
            
            # Mapping Kolom
            col_map = {
                'Stock Code': 'Kode Efek', 'Close': 'Harga_Close', 'Volume': 'Volume_Harian', 
                'Last Trading Date': 'Tanggal_Data', 'Date': 'Tanggal_Data'
            }
            for old, new in col_map.items():
                if old in df_price.columns: df_price[new] = df_price[old]
            
            # LOGIC HARGA TRANSAKSI (VWAP Logic)
            # Prioritas: VWAP > Typical Price > (High+Low+Close)/3 > Close
            if 'VWAP' in df_price.columns:
                df_price['Harga_Avg'] = df_price['VWAP']
            elif 'Typical Price' in df_price.columns:
                df_price['Harga_Avg'] = df_price['Typical Price']
            else:
                if all(c in df_price.columns for c in ['High', 'Low', 'Harga_Close']):
                    df_price['Harga_Avg'] = (df_price['High'] + df_price['Low'] + df_price['Harga_Close']) / 3
                else:
                    df_price['Harga_Avg'] = df_price['Harga_Close'] # Fallback
            
            df_price['Tanggal_Data'] = pd.to_datetime(df_price['Tanggal_Data'])
    except Exception as e: st.error(f"Price Error: {e}")

    return df_ksei, df_price

@st.cache_data(ttl=3600)
def process_forensics(df):
    """Menjalankan Klasifikasi Rekening"""
    if df.empty: return df
    
    unique = df[['Nama Pemegang Saham', 'Nama Rekening Efek']].drop_duplicates()
    
    res = unique.apply(BandarXRay.classify_account, axis=1, result_type='expand')
    res.columns = ['REAL_OWNER', 'HOLDING_TYPE', 'ACCOUNT_STATUS', 'BANK_SOURCE']
    
    unique = pd.concat([unique, res], axis=1)
    return pd.merge(df, unique, on=['Nama Pemegang Saham', 'Nama Rekening Efek'], how='left')

# ==============================================================================
# 5. DASHBOARD UTAMA
# ==============================================================================

# Load Data
with st.spinner('Menghubungkan Database & Analisa Forensik...'):
    df_ksei_raw, df_price = load_data_complete()
    if not df_ksei_raw.empty:
        df = process_forensics(df_ksei_raw)
    else:
        st.error("Gagal Load Data KSEI. Pastikan file ada di Google Drive.")
        st.stop()

# --- SIDEBAR CONTROL ---
with st.sidebar:
    st.title("🦅 X-RAY CONTROL")
    
    all_stocks = sorted(df['Kode Efek'].unique())
    sel_stock = st.multiselect("Filter Saham (Global)", all_stocks)
    
    min_d, max_d = df['Tanggal_Data'].min().date(), df['Tanggal_Data'].max().date()
    sel_date = st.date_input("Periode Analisa", [min_d, max_d])
    
    st.divider()
    st.info("Filter ini berlaku untuk Tab Ultimate Holder, Repo, dan Flow.")

# Filter Data Global
df_view = df.copy()
if sel_stock: df_view = df_view[df_view['Kode Efek'].isin(sel_stock)]
if len(sel_date) == 2:
    df_view = df_view[(df_view['Tanggal_Data'].dt.date >= sel_date[0]) & (df_view['Tanggal_Data'].dt.date <= sel_date[1])]

# --- TAB NAVIGASI ---
st.title("🦅 Bandarmologi X-Ray: Enterprise Edition")
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 STOCK SCREENER", "🔬 DEEP DIVE", "👑 ULTIMATE HOLDER", "⚠️ REPO MONITOR", "💰 SMART MONEY"
])

# ---------------------------------------------------------------------
# TAB 1: STOCK SCREENER
# ---------------------------------------------------------------------
with tab1:
    st.header("🎯 Stock Screener: Radar Akumulasi")
    st.markdown("Mendeteksi saham yang diakumulasi oleh 'Big Money' dalam periode terpilih.")
    
    if len(sel_date) == 2:
        # Group per saham
        screener = df_view.groupby('Kode Efek').agg({
            'Net_Flow': 'sum',
            'Jumlah Saham (Curr)': 'sum',
            'REAL_OWNER': 'nunique'
        }).reset_index()
        
        # Ambil harga terakhir untuk estimasi value
        if not df_price.empty:
            last_prices = df_price.sort_values('Tanggal_Data').groupby('Kode Efek')['Harga_Close'].last().reset_index()
            screener = pd.merge(screener, last_prices, on='Kode Efek', how='left')
        
        # Value Flow Estimate
        screener['Value Flow (Est)'] = screener['Net_Flow'] * screener['Harga_Close'] if 'Harga_Close' in screener.columns else 0
        
        # Sorting
        top_accum = screener.sort_values('Value Flow (Est)', ascending=False).head(10)
        top_dist = screener.sort_values('Value Flow (Est)', ascending=True).head(10)
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🟢 Top Akumulasi (Net Buy Terbesar)")
            st.dataframe(top_accum[['Kode Efek', 'Net_Flow', 'Value Flow (Est)', 'Harga_Close']].style.format({
                'Net_Flow': '{:,.0f}', 'Value Flow (Est)': 'Rp {:,.0f}', 'Harga_Close': '{:,.0f}'
            }), use_container_width=True)
            
        with c2:
            st.subheader("🔴 Top Distribusi (Net Sell Terbesar)")
            st.dataframe(top_dist[['Kode Efek', 'Net_Flow', 'Value Flow (Est)', 'Harga_Close']].style.format({
                'Net_Flow': '{:,.0f}', 'Value Flow (Est)': 'Rp {:,.0f}', 'Harga_Close': '{:,.0f}'
            }), use_container_width=True)

# ---------------------------------------------------------------------
# TAB 2: DEEP DIVE (SINGLE STOCK)
# ---------------------------------------------------------------------
with tab2:
    st.header("🔬 Deep Dive Analysis")
    target_stock = st.selectbox("🔍 Pilih Saham untuk Bedah Detail:", all_stocks)
    
    if target_stock:
        df_deep = df[df['Kode Efek'] == target_stock].sort_values('Tanggal_Data')
        
        # 1. CHART DUAL AXIS
        # Total Kepemilikan 5% (Garis Hijau) vs Harga Saham (Garis Oranye)
        daily_holdings = df_deep.groupby('Tanggal_Data')['Jumlah Saham (Curr)'].sum().reset_index()
        price_chart = df_price[df_price['Kode Efek'] == target_stock].sort_values('Tanggal_Data') if not df_price.empty else pd.DataFrame()
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Area Chart: Total Lembar di Tangan Bandar
        fig.add_trace(go.Scatter(
            x=daily_holdings['Tanggal_Data'], y=daily_holdings['Jumlah Saham (Curr)'], 
            name="Total Holdings >5%", fill='tozeroy', line=dict(color='#00CC96')
        ), secondary_y=False)
        
        # Line Chart: Harga Saham
        if not price_chart.empty:
            # Filter tanggal biar sama
            price_chart = price_chart[(price_chart['Tanggal_Data'] >= daily_holdings['Tanggal_Data'].min()) & 
                                      (price_chart['Tanggal_Data'] <= daily_holdings['Tanggal_Data'].max())]
            fig.add_trace(go.Scatter(
                x=price_chart['Tanggal_Data'], y=price_chart['Harga_Close'], 
                name="Harga Saham", line=dict(color='#FFA500', width=2)
            ), secondary_y=True)
            
        fig.update_layout(title=f"Korelasi Harga {target_stock} vs Akumulasi Bandar", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        
        # 2. PERUBAHAN PEMEGANG
        st.subheader("Perubahan Kepemilikan (Periode Ini)")
        if len(sel_date) == 2:
            df_period = df_deep[(df_deep['Tanggal_Data'].dt.date >= sel_date[0]) & (df_deep['Tanggal_Data'].dt.date <= sel_date[1])]
            movers = df_period.groupby('REAL_OWNER')['Net_Flow'].sum().reset_index().sort_values('Net_Flow', ascending=False)
            movers = movers[movers['Net_Flow'] != 0] # Hanya yg bergerak
            
            st.dataframe(movers.style.format({'Net_Flow': '{:+,.0f}'})
                         .applymap(lambda v: 'color: green' if v > 0 else 'color: red', subset=['Net_Flow']), 
                         use_container_width=True)

# ---------------------------------------------------------------------
# TAB 3: ULTIMATE HOLDER
# ---------------------------------------------------------------------
with tab3:
    st.header("👑 Ultimate Holder (Global View)")
    last_date = df_view['Tanggal_Data'].max()
    df_last = df_view[df_view['Tanggal_Data'] == last_date]
    
    uh_group = df_last.groupby('REAL_OWNER').agg({
        'Jumlah Saham (Curr)': 'sum',
        'Kode Efek': 'nunique',
        'ACCOUNT_STATUS': lambda x: '⚠️ REPO' if any('PLEDGE' in str(s) for s in x) else 'CLEAN'
    }).sort_values('Jumlah Saham (Curr)', ascending=False).head(50)
    
    st.dataframe(uh_group.style.format({'Jumlah Saham (Curr)': '{:,.0f}'}), use_container_width=True)

# ---------------------------------------------------------------------
# TAB 4: REPO MONITOR
# ---------------------------------------------------------------------
with tab4:
    st.header("⚠️ Repo & Pledge Monitor")
    df_repo = df_view[df_view['ACCOUNT_STATUS'].str.contains("PLEDGE|REPO")]
    
    if not df_repo.empty:
        repo_last = df_repo[df_repo['Tanggal_Data'] == df_repo['Tanggal_Data'].max()]
        st.dataframe(repo_last[['REAL_OWNER', 'Kode Efek', 'Nama Pemegang Saham', 'Jumlah Saham (Curr)']]
                     .sort_values('Jumlah Saham (Curr)', ascending=False)
                     .style.format({'Jumlah Saham (Curr)': '{:,.0f}'}), 
                     use_container_width=True)
    else:
        st.success("Tidak ada indikasi Repo pada filter saat ini.")

# ---------------------------------------------------------------------
# TAB 5: SMART MONEY (PROFITABILITY)
# ---------------------------------------------------------------------
with tab5:
    st.header("💰 Smart Money Leaderboard (PnL Estimasi)")
    st.caption("Menggunakan VWAP/Typical Price untuk Entry, dan Closing Price untuk Valuasi.")
    
    if df_price.empty:
        st.warning("Data harga belum tersedia. Silakan cek koneksi.")
    else:
        # Hitung PnL
        merged_df = PriceAnalyzer.calculate_entry_exit_price(df_view, df_price)
        perf_df = PriceAnalyzer.calculate_performance_metrics(merged_df)
        
        if not perf_df.empty:
            # Ranking by Unrealized PnL
            perf_agg = perf_df.groupby('REAL_OWNER').agg({
                'Unrealized_PnL_Rp': 'sum',
                'Unrealized_PnL_%': 'mean',
                'Current_Value_Rp': 'sum',
                'Total_Buy_Vol': 'sum'
            }).sort_values('Unrealized_PnL_Rp', ascending=False).head(20)
            
            def color_pnl(val):
                return 'color: green' if val > 0 else 'color: red' if val < 0 else ''
            
            st.dataframe(perf_agg.style.format({
                'Unrealized_PnL_Rp': 'Rp {:,.0f}',
                'Unrealized_PnL_%': '{:.1f}%',
                'Current_Value_Rp': 'Rp {:,.0f}',
                'Total_Buy_Vol': '{:,.0f}'
            }).applymap(color_pnl, subset=['Unrealized_PnL_Rp', 'Unrealized_PnL_%']), 
            use_container_width=True)
        else:
            st.info("Belum ada data transaksi yang cukup untuk menghitung PnL.")

# Footer
st.divider()
st.caption("Bandarmologi X-Ray v7.0 | Units: Lembar Saham | PnL Base: VWAP vs Close")
