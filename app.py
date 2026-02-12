"""
================================================================================
🦅 BANDARMOLOGI X-RAY - ENTERPRISE EDITION v6.0
================================================================================
New Features:
✅ STOCK SCREENER: Deteksi saham akumulasi/distribusi otomatis.
✅ DEEP DIVE: Analisa satu saham (Chart Harga vs Flow Bandar).
✅ UNMASKING ENGINE: Tetap aktif (Logic Prioritas Bank).
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

# Custom CSS
st.markdown("""
<style>
    .main > div { padding: 0rem 1rem; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem !important; color: #000000 !important; }
    div[data-testid="stMetricLabel"] { font-size: 0.9rem !important; color: #444444 !important; }
    .stTabs [data-baseweb="tab-list"] { background-color: #f0f2f6; padding: 0.5rem; border-radius: 10px; }
    .stTabs [data-baseweb="tab"] { background-color: #ffffff; color: #31333F !important; border: 1px solid #ddd; }
    .stTabs [aria-selected="true"] { background-color: #00CC96 !important; color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. BANDAR X-RAY ENGINE (FORENSIK)
# ==============================================================================
class BandarXRay:
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
    ]
    
    PLEDGE_KEYWORDS = ['PLEDGE', 'REPO', 'JAMINAN', 'AGUNAN', 'COLLATERAL', 'LOCKED', 'MARGIN']
    DIRECT_INDICATORS = [' PT', 'PT ', ' TBK', ' LTD', ' INC', ' CORP', 'DRS.', 'DR.', 'IR.', 'H.', 'YAYASAN', 'DANA PENSIUN']

    @staticmethod
    def clean_name(text):
        if pd.isna(text) or text == '-': return '-'
        text = str(text).strip()
        text = re.sub(r'\s*[-–—]\s*\d+.*$', '', text)
        text = re.sub(r'\s*\([A-Z0-9\s\-]+\)$', '', text)
        return text.strip().upper()

    @staticmethod
    def is_direct(name):
        name = str(name).upper()
        if any(k in name for k in ['S/A', 'A/C', 'FOR', 'BRANCH', 'TRUST', 'CUSTODIAN']): return False
        return any(k in name for k in BandarXRay.DIRECT_INDICATORS)

    @classmethod
    def classify_account(cls, row):
        holder = str(row['Nama Pemegang Saham']).upper()
        account = str(row['Nama Rekening Efek']).upper() if pd.notna(row.get('Nama Rekening Efek')) else ""
        
        real_owner, holding_type, status, bank = cls.clean_name(holder), "DIRECT", "NORMAL", "-"
        
        # 1. PLEDGE
        if any(k in account for k in cls.PLEDGE_KEYWORDS):
            status = "⚠️ PLEDGE/REPO"
        
        # 2. NOMINEE
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
        
        # 3. DIRECT
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
        
        if status != "NORMAL": holding_type += " [REPO]"
        return pd.Series([real_owner, holding_type, status, bank])

# ==============================================================================
# 3. DATA LOADERS
# ==============================================================================
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

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
    
    # Load KSEI
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
            for c in ['Jumlah Saham (Prev)', 'Jumlah Saham (Curr)']:
                df_ksei[c] = pd.to_numeric(df_ksei[c].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            df_ksei['Net_Flow'] = df_ksei['Jumlah Saham (Curr)'] - df_ksei['Jumlah Saham (Prev)']
            if 'Nama Rekening Efek' not in df_ksei.columns: df_ksei['Nama Rekening Efek'] = '-'
            df_ksei['Nama Rekening Efek'] = df_ksei['Nama Rekening Efek'].fillna('-')
    except Exception as e: st.error(f"KSEI Error: {e}")

    # Load Price
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
            # Map columns
            col_map = {'Stock Code': 'Kode Efek', 'Close': 'Harga_Close', 'Volume': 'Volume_Harian', 
                       'Last Trading Date': 'Tanggal_Data', 'Date': 'Tanggal_Data'}
            for old, new in col_map.items():
                if old in df_price.columns: df_price[new] = df_price[old]
            df_price['Tanggal_Data'] = pd.to_datetime(df_price['Tanggal_Data'])
    except Exception as e: st.error(f"Price Error: {e}")

    return df_ksei, df_price

@st.cache_data(ttl=3600)
def process_forensics(df):
    if df.empty: return df
    unique = df[['Nama Pemegang Saham', 'Nama Rekening Efek']].drop_duplicates()
    res = unique.apply(BandarXRay.classify_account, axis=1, result_type='expand')
    res.columns = ['REAL_OWNER', 'HOLDING_TYPE', 'ACCOUNT_STATUS', 'BANK_SOURCE']
    unique = pd.concat([unique, res], axis=1)
    return pd.merge(df, unique, on=['Nama Pemegang Saham', 'Nama Rekening Efek'], how='left')

# ==============================================================================
# 4. DASHBOARD UI
# ==============================================================================
with st.spinner('Menghubungkan Database & Analisa Forensik...'):
    df_ksei_raw, df_price = load_data_complete()
    if not df_ksei_raw.empty:
        df = process_forensics(df_ksei_raw)
    else:
        st.error("Gagal Load Data KSEI")
        st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.title("🦅 X-RAY CONTROL")
    all_stocks = sorted(df['Kode Efek'].unique())
    sel_stock = st.multiselect("Filter Saham (Global)", all_stocks)
    
    min_d, max_d = df['Tanggal_Data'].min().date(), df['Tanggal_Data'].max().date()
    sel_date = st.date_input("Periode", [min_d, max_d])
    
    st.divider()
    st.info("Filter di atas berlaku untuk Tab 3, 4, 5. Tab Screener & Deep Dive memiliki filter sendiri.")

# Apply Global Filter
df_view = df.copy()
if sel_stock: df_view = df_view[df_view['Kode Efek'].isin(sel_stock)]
if len(sel_date) == 2:
    df_view = df_view[(df_view['Tanggal_Data'].dt.date >= sel_date[0]) & (df_view['Tanggal_Data'].dt.date <= sel_date[1])]

# ==============================================================================
# 5. TABS
# ==============================================================================
st.title("🦅 Bandarmologi X-Ray: Enterprise Edition")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 STOCK SCREENER", 
    "🔬 DEEP DIVE", 
    "👑 ULTIMATE HOLDER",
    "⚠️ REPO MONITOR",
    "💰 SMART MONEY"
])

# --- TAB 1: STOCK SCREENER (POTENSI AKUMULASI) ---
with tab1:
    st.header("🎯 Stock Screener: Radar Akumulasi")
    st.markdown("Mendeteksi saham yang sedang diakumulasi/distribusi oleh Pemegang >5% dalam periode terpilih.")
    
    if len(sel_date) == 2:
        # Grouping by Stock
        screener = df_view.groupby('Kode Efek').agg({
            'Net_Flow': 'sum',
            'Jumlah Saham (Curr)': 'sum',
            'REAL_OWNER': 'nunique'
        }).reset_index()
        
        # Merge dengan Harga Terakhir (jika ada)
        if not df_price.empty:
            last_prices = df_price.sort_values('Tanggal_Data').groupby('Kode Efek')['Harga_Close'].last().reset_index()
            screener = pd.merge(screener, last_prices, on='Kode Efek', how='left')
        
        # Logic Klasifikasi Sederhana
        def classify_flow(val):
            if val > 0: return "AKUMULASI 🟢"
            elif val < 0: return "DISTRIBUSI 🔴"
            else: return "NEUTRAL ⚪"
            
        screener['Status'] = screener['Net_Flow'].apply(classify_flow)
        screener['Value Flow (Est)'] = screener['Net_Flow'] * screener['Harga_Close'] if 'Harga_Close' in screener.columns else 0
        
        # Sorting
        screener = screener.sort_values('Value Flow (Est)', ascending=False)
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🟢 Top Akumulasi (Big Money In)")
            top_accum = screener[screener['Net_Flow'] > 0].head(10)
            st.dataframe(top_accum[['Kode Efek', 'Status', 'Net_Flow', 'Value Flow (Est)', 'Harga_Close']].style.format({
                'Net_Flow': '{:,.0f}', 'Value Flow (Est)': 'Rp {:,.0f}', 'Harga_Close': '{:,.0f}'
            }), use_container_width=True, hide_index=True)
            
        with c2:
            st.subheader("🔴 Top Distribusi (Big Money Out)")
            top_dist = screener[screener['Net_Flow'] < 0].sort_values('Net_Flow', ascending=True).head(10)
            st.dataframe(top_dist[['Kode Efek', 'Status', 'Net_Flow', 'Value Flow (Est)', 'Harga_Close']].style.format({
                'Net_Flow': '{:,.0f}', 'Value Flow (Est)': 'Rp {:,.0f}', 'Harga_Close': '{:,.0f}'
            }), use_container_width=True, hide_index=True)

# --- TAB 2: DEEP DIVE (SINGLE STOCK ANALYSIS) ---
with tab2:
    st.header("🔬 Deep Dive Analysis")
    st.markdown("Bedah tuntas satu saham: Korelasi Harga vs Pergerakan Bandar.")
    
    # Pilih Saham (Override Global Filter)
    target_stock = st.selectbox("🔍 Pilih Saham untuk Deep Dive:", all_stocks)
    
    if target_stock:
        # Filter Data
        df_deep = df[df['Kode Efek'] == target_stock]
        df_deep = df_deep.sort_values('Tanggal_Data')
        
        if not df_deep.empty:
            # 1. CHART: HARGA VS BANDAR FLOW (Kumulatif)
            # Hitung total kepemilikan 5% per hari
            daily_holdings = df_deep.groupby('Tanggal_Data')['Jumlah Saham (Curr)'].sum().reset_index()
            
            # Ambil data harga
            price_deep = df_price[df_price['Kode Efek'] == target_stock].sort_values('Tanggal_Data') if not df_price.empty else pd.DataFrame()
            
            # Buat Chart Dual Axis
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Line 1: Kepemilikan Bandar (Area)
            fig.add_trace(
                go.Scatter(x=daily_holdings['Tanggal_Data'], y=daily_holdings['Jumlah Saham (Curr)'], 
                           name="Total Kepemilikan 5% (Bandar)", fill='tozeroy', line=dict(color='#00CC96')),
                secondary_y=False
            )
            
            # Line 2: Harga Saham (Candlestick/Line)
            if not price_deep.empty:
                # Filter tanggal harga sesuai data KSEI
                mask = (price_deep['Tanggal_Data'] >= daily_holdings['Tanggal_Data'].min()) & (price_deep['Tanggal_Data'] <= daily_holdings['Tanggal_Data'].max())
                price_chart = price_deep.loc[mask]
                
                fig.add_trace(
                    go.Scatter(x=price_chart['Tanggal_Data'], y=price_chart['Harga_Close'], 
                               name="Harga Saham", line=dict(color='#FFA500', width=2)),
                    secondary_y=True
                )
            
            fig.update_layout(title=f"Analisa Korelasi: Harga {target_stock} vs Akumulasi Bandar", hovermode="x unified")
            fig.update_yaxes(title_text="Lembar Saham Bandar", secondary_y=False)
            fig.update_yaxes(title_text="Harga Saham", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)
            
            # 2. WHO IS MOVING? (Perubahan Periode Ini)
            st.subheader(f"Perubahan Kepemilikan {target_stock} (Periode Terpilih)")
            
            # Filter periode global untuk tabel ini
            if len(sel_date) == 2:
                mask_date = (df_deep['Tanggal_Data'].dt.date >= sel_date[0]) & (df_deep['Tanggal_Data'].dt.date <= sel_date[1])
                df_deep_period = df_deep.loc[mask_date]
            else:
                df_deep_period = df_deep
                
            flow_analysis = df_deep_period.groupby('REAL_OWNER').agg({
                'Net_Flow': 'sum',
                'Jumlah Saham (Curr)': 'last', # Posisi akhir
                'HOLDING_TYPE': 'first'
            }).sort_values('Net_Flow', ascending=False)
            
            # Tampilkan yang ada pergerakan saja
            active_movers = flow_analysis[flow_analysis['Net_Flow'] != 0]
            
            if not active_movers.empty:
                st.dataframe(active_movers.style.format({'Net_Flow': '{:+,.0f}', 'Jumlah Saham (Curr)': '{:,.0f}'})
                             .applymap(lambda v: 'color: green' if v > 0 else 'color: red' if v < 0 else '', subset=['Net_Flow']),
                             use_container_width=True)
            else:
                st.info("Tidak ada perubahan kepemilikan signifikan pada periode ini.")
                
            # 3. KOMPOSISI PEMEGANG
            st.subheader("Peta Kekuatan (Komposisi)")
            # Ambil data hari terakhir
            last_day_data = df_deep[df_deep['Tanggal_Data'] == df_deep['Tanggal_Data'].max()]
            fig_pie = px.pie(last_day_data, values='Jumlah Saham (Curr)', names='REAL_OWNER', title="Komposisi Pemegang Saham", hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)

# --- TAB 3: ULTIMATE HOLDER (GLOBAL) ---
with tab3:
    st.header("👑 Ultimate Holder (Global View)")
    last_date = df_view['Tanggal_Data'].max()
    df_last = df_view[df_view['Tanggal_Data'] == last_date]
    
    uh_group = df_last.groupby('REAL_OWNER').agg({
        'Jumlah Saham (Curr)': 'sum',
        'Kode Efek': 'nunique',
        'ACCOUNT_STATUS': lambda x: '⚠️ REPO' if any('PLEDGE' in s for s in x) else 'CLEAN'
    }).sort_values('Jumlah Saham (Curr)', ascending=False).head(50)
    
    st.dataframe(uh_group.style.format({'Jumlah Saham (Curr)': '{:,.0f}'}), use_container_width=True)

# --- TAB 4: REPO MONITOR ---
with tab4:
    st.header("⚠️ Repo & Pledge Monitor")
    df_repo = df_view[df_view['ACCOUNT_STATUS'].str.contains("PLEDGE|REPO")]
    if not df_repo.empty:
        repo_last = df_repo[df_repo['Tanggal_Data'] == df_repo['Tanggal_Data'].max()]
        st.dataframe(repo_last[['REAL_OWNER', 'Kode Efek', 'Nama Pemegang Saham', 'Jumlah Saham (Curr)']]
                     .sort_values('Jumlah Saham (Curr)', ascending=False)
                     .style.format({'Jumlah Saham (Curr)': '{:,.0f}'}), use_container_width=True)
    else:
        st.success("Tidak ada data Repo pada filter ini.")

# --- TAB 5: SMART MONEY (PROFITABILITY) ---
with tab5:
    st.header("💰 Smart Money Leaderboard (Estimasi Profit)")
    if df_price.empty:
        st.warning("Data harga belum tersedia. Silakan cek koneksi Drive.")
    else:
        # Simple Profit Calculation Logic (Avg Price vs Current Price)
        # Merge View Data with Price
        merged = pd.merge(df_view, df_price[['Kode Efek', 'Tanggal_Data', 'Harga_Close']], on=['Kode Efek', 'Tanggal_Data'], how='left')
        merged['Valuation'] = merged['Jumlah Saham (Curr)'] * merged['Harga_Close']
        
        # Ranking by Portfolio Value
        rich_list = merged.groupby('REAL_OWNER')['Valuation'].sum().sort_values(ascending=False).head(20)
        st.subheader("Top 20 Portfolio Valuation (Estimasi)")
        st.dataframe(rich_list.to_frame().style.format("Rp {:,.0f}"), use_container_width=True)

# Footer
st.divider()
st.caption("Bandarmologi X-Ray v6.0 | Created for Professional Analysis")
