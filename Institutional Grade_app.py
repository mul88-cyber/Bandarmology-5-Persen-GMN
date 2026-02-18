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
# KONFIGURASI: LINK GOOGLE DRIVE
# =============================================================================
FILE_IDS = {
    'harian': '1t_wCljhepGBqZVrvleuZKldomQKop9DY',           # Kompilasi_Data_1Tahun.csv
    'ksei': '1eTUIC120SHTCzvBk77Q87w0X56F2HkWz',             # KSEI_Shareholder_Processed.csv
    'master_5_parquet': '1tb1umgJc1giaKYyMNuQWhH7R8cH75s2X', # Master 5% Parquet
    'master_5_light': '10CS5QJU5MHafIpanEH9XU6SpCEOVd-pb'    # Master 5% CSV Light
}

# =============================================================================
# FUNGSI LOAD DATA
# =============================================================================
def load_csv_from_gdrive(file_id, max_retries=3):
    """Load CSV dari Google Drive"""
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    for attempt in range(max_retries):
        try:
            session = requests.Session()
            response = session.get(url, stream=True, timeout=30)
            
            if 'Virus scan warning' in response.text or 'Quota exceeded' in response.text:
                import re
                match = re.search(r'confirm=([0-9A-Za-z]+)', response.text)
                if match:
                    confirm_token = match.group(1)
                    url = f"https://drive.google.com/uc?export=download&confirm={confirm_token}&id={file_id}"
                    response = session.get(url, stream=True, timeout=30)
            
            response.raise_for_status()
            content = response.content.decode('utf-8')
            df = pd.read_csv(StringIO(content))
            return df
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == 2:
                try:
                    url = f"https://drive.google.com/uc?id={file_id}"
                    df = pd.read_csv(url)
                    return df
                except:
                    pass
            time.sleep(2)
    
    raise Exception(f"Gagal load file ID {file_id}")

def load_parquet_from_gdrive(file_id):
    """Load Parquet dari Google Drive"""
    try:
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        buffer = BytesIO(response.content)
        df = pd.read_parquet(buffer)
        return df
    except Exception as e:
        st.warning(f"Gagal load Parquet: {e}. Mencoba format CSV...")
        return None

# =============================================================================
# CACHE DATA LOADING
# =============================================================================
@st.cache_data(ttl=3600, show_spinner="Loading data harian...")
def load_harian():
    """Load data harian"""
    try:
        df = load_csv_from_gdrive(FILE_IDS['harian'])
        df['Last Trading Date'] = pd.to_datetime(df['Last Trading Date'], errors='coerce')
        df = df.dropna(subset=['Last Trading Date'])
        return df
    except Exception as e:
        st.error(f"❌ Gagal load data harian: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=86400, show_spinner="Loading data KSEI...")
def load_ksei():
    """Load data KSEI"""
    try:
        df = load_csv_from_gdrive(FILE_IDS['ksei'])
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        return df
    except Exception as e:
        st.error(f"❌ Gagal load data KSEI: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=86400, show_spinner="Loading data kepemilikan 5%...")
def load_master_5():
    """Load data master 5%"""
    # Prioritaskan Parquet
    if 'master_5_parquet' in FILE_IDS:
        df = load_parquet_from_gdrive(FILE_IDS['master_5_parquet'])
        if df is not None:
            if 'Tanggal_Data' in df.columns:
                df['Tanggal_Data'] = pd.to_datetime(df['Tanggal_Data'], errors='coerce')
            st.success("✅ Load data 5% (Parquet)")
            return df
    
    # Fallback ke CSV
    try:
        df = load_csv_from_gdrive(FILE_IDS['master_5_light'])
        df['Tanggal_Data'] = pd.to_datetime(df['Tanggal_Data'], errors='coerce')
        st.success("✅ Load data 5% (CSV)")
        return df
    except:
        st.warning("⚠️ Data 5% tidak tersedia")
        return pd.DataFrame()

# =============================================================================
# FUNGSI KALKULASI ADVANCED METRICS
# =============================================================================

def calculate_float_pressure(df):
    """
    1️⃣ Float Pressure & Liquidity Stress Analysis
    """
    df = df.copy()
    
    # A. Float Turnover Ratio (Daily Volume / Tradeable Shares)
    df['Float_Turnover_Ratio'] = df['Volume'] / (df['Tradeble Shares'] + 1)
    
    # B. Float Market Cap Turnover (Value / Float Market Cap)
    df['Float_MCap'] = df['Close'] * df['Tradeble Shares']
    df['Float_MCap_Turnover'] = df['Value'] / (df['Float_MCap'] + 1)
    
    # C. Liquidity Score (Composite)
    # Komponen: 
    # 1. Volume vs MA20 (higher = better liquidity)
    # 2. Float Turnover (higher = better liquidity)
    # 3. Free Float % (higher = better liquidity)
    
    # Normalize each component to 0-100 scale
    vol_ratio = df['Volume'] / (df['MA20_vol'] + 1)
    vol_score = np.clip(vol_ratio * 20, 0, 100)  # Volume 5x MA20 = 100
    
    float_turnover_score = np.clip(df['Float_Turnover_Ratio'] * 10000, 0, 100)  # Turnover 1% = 100
    
    free_float_score = df['Free Float']  # Already in percentage
    
    # Composite Liquidity Score (0-100)
    df['Liquidity_Score'] = (
        vol_score * 0.4 +
        float_turnover_score * 0.3 +
        free_float_score * 0.3
    )
    
    # Classification
    conditions = [
        df['Liquidity_Score'] < 20,
        df['Liquidity_Score'] < 40,
        df['Liquidity_Score'] < 60,
        df['Liquidity_Score'] >= 60
    ]
    choices = ['💀 Sangat Kering (High Risk)', '⚠️ Kering', '🟡 Normal', '💧 Likuid']
    df['Liquidity_Status'] = np.select(conditions, choices, default='Unknown')
    
    # High Manipulation Risk Flag
    df['High_Manipulation_Risk'] = (
        (df['Free Float'] < 30) & 
        (df['Float_Turnover_Ratio'] < 0.005) & 
        (df['Liquidity_Score'] < 30)
    ).astype(int)
    
    return df

def calculate_smart_money_accumulation(df):
    """
    2️⃣ Smart Money Accumulation Detection
    """
    df = df.copy()
    
    # Foreign Net
    if 'Foreign Buy' in df.columns and 'Foreign Sell' in df.columns:
        df['Foreign_Net'] = df['Foreign Buy'] - df['Foreign Sell']
    else:
        df['Foreign_Net'] = 0
    
    # 20-day rolling sum of foreign net
    df['SMF_20D'] = df.groupby('Stock Code')['Foreign_Net'].transform(
        lambda x: x.rolling(20, min_periods=5).sum()
    )
    
    # Smart Money Accumulation Flags
    conditions = [
        # Strong Accumulation
        (df['Volume'] > df['MA20_vol'] * 2) & 
        (df['Change %'].abs() < 3) & 
        (df['Foreign_Net'] > 0) &
        (df['SMF_20D'] > 0),
        
        # Moderate Accumulation
        (df['Volume'] > df['MA20_vol'] * 1.5) & 
        (df['Change %'].abs() < 5) & 
        (df['Foreign_Net'] > 0),
        
        # Possible Accumulation
        (df['Volume'] > df['MA20_vol']) & 
        (df['Foreign_Net'] > 0)
    ]
    choices = ['💪 Strong Accumulation', '📈 Moderate Accumulation', '👀 Possible Accumulation']
    
    df['Smart_Money_Signal'] = np.select(conditions, choices, default='-')
    
    # Smart Money Score (0-100)
    score = 0
    if 'Volume' in df.columns and 'MA20_vol' in df.columns:
        vol_factor = np.clip((df['Volume'] / df['MA20_vol']) * 10, 0, 40)
        score += vol_factor
    
    if 'Foreign_Net' in df.columns:
        foreign_pos = (df['Foreign_Net'] > 0).astype(int) * 30
        score += foreign_pos
    
    if 'SMF_20D' in df.columns:
        smf_pos = (df['SMF_20D'] > 0).astype(int) * 30
        score += smf_pos
    
    df['Smart_Money_Score'] = np.clip(score, 0, 100)
    
    return df

def calculate_holder_movement(df_harian, df_master):
    """
    3️⃣ Big Holder Movement (Game Changer)
    """
    if df_master.empty:
        return df_harian
    
    df = df_harian.copy()
    
    # Aggregate holder changes per stock
    holder_agg = df_master.groupby('Kode Efek').agg({
        'Perubahan_Saham': 'sum',
        'Jumlah Saham (Curr)': 'last'
    }).reset_index()
    holder_agg.columns = ['Stock Code', 'Holder_Total_Change', 'Holder_Current_Total']
    
    df = df.merge(holder_agg, on='Stock Code', how='left')
    df['Holder_Total_Change'] = df['Holder_Total_Change'].fillna(0)
    df['Holder_Current_Total'] = df['Holder_Current_Total'].fillna(0)
    
    # Calculate % of listed shares
    df['Holder_Change_Pct_Listed'] = (df['Holder_Total_Change'] / (df['Listed Shares'] + 1)) * 100
    
    # Stealth Accumulation Detection
    conditions = [
        # Strong Stealth Accumulation
        (df['Holder_Total_Change'] > 0) &
        (df['Holder_Change_Pct_Listed'] > 0.5) &  # >0.5% of listed shares
        (df['Change %'].abs() < 3) &  # Price sideways
        (df['Volume'] > df['MA20_vol'] * 1.2),  # Volume increasing
        
        # Moderate Stealth Accumulation
        (df['Holder_Total_Change'] > 0) &
        (df['Holder_Change_Pct_Listed'] > 0.25) &
        (df['Change %'].abs() < 5) &
        (df['Volume'] > df['MA20_vol']),
        
        # Possible Accumulation
        (df['Holder_Total_Change'] > 0) &
        (df['Change %'].abs() < 7)
    ]
    choices = ['🎯 Strong Stealth Accumulation', '📊 Moderate Accumulation', '👀 Possible Accumulation']
    
    df['Holder_Movement_Signal'] = np.select(conditions, choices, default='-')
    
    # Holder Score
    holder_score = 0
    if 'Holder_Change_Pct_Listed' in df.columns:
        holder_score += np.clip(df['Holder_Change_Pct_Listed'] * 20, 0, 40)  # 2.5% change = 50
    
    if 'Volume' in df.columns and 'MA20_vol' in df.columns:
        vol_ratio = df['Volume'] / df['MA20_vol']
        holder_score += np.clip((vol_ratio - 1) * 20, 0, 30)  # Volume 2x = +20
    
    price_sideways = (df['Change %'].abs() < 3).astype(int) * 30
    holder_score += price_sideways
    
    df['Holder_Score'] = np.clip(holder_score, 0, 100)
    
    return df

def calculate_foreign_dominance(df):
    """
    4️⃣ Foreign Flow Dominance Model
    """
    df = df.copy()
    
    if 'Foreign Buy' in df.columns and 'Foreign Sell' in df.columns and 'Value' in df.columns:
        df['Foreign_Net'] = df['Foreign Buy'] - df['Foreign Sell']
        df['Foreign_Total'] = df['Foreign Buy'] + df['Foreign Sell']
        
        # Foreign Dominance (% of total value)
        df['Foreign_Dominance'] = (df['Foreign_Total'] / (df['Value'] + 1)) * 100
        df['Foreign_Net_Ratio'] = df['Foreign_Net'] / (df['Foreign_Total'] + 1)
        
        # Classification
        conditions = [
            df['Foreign_Dominance'] > 40,
            df['Foreign_Dominance'] > 25,
            df['Foreign_Dominance'] > 10,
            df['Foreign_Dominance'] <= 10
        ]
        choices = ['🔥 Sangat Dominan (>40%)', '💪 Dominan (25-40%)', '👀 Moderate (10-25%)', '💤 Rendah (<10%)']
        df['Foreign_Dominance_Level'] = np.select(conditions, choices, default='Unknown')
        
        # Foreign Sentiment (based on net)
        sentiment_conditions = [
            (df['Foreign_Net_Ratio'] > 0.3) & (df['Foreign_Net'] > 0),
            (df['Foreign_Net_Ratio'] > 0) & (df['Foreign_Net'] > 0),
            (df['Foreign_Net_Ratio'] < -0.3) & (df['Foreign_Net'] < 0),
            (df['Foreign_Net_Ratio'] < 0) & (df['Foreign_Net'] < 0)
        ]
        sentiment_choices = ['🚀 Very Strong Buy', '📈 Net Buy', '📉 Net Sell', '🔻 Very Strong Sell']
        df['Foreign_Sentiment'] = np.select(sentiment_conditions, sentiment_choices, default='⚖️ Neutral')
        
        # Foreign Dominance Score
        df['Foreign_Score'] = np.clip(df['Foreign_Dominance'] * 2, 0, 100)  # 50% dominance = 100
    else:
        df['Foreign_Dominance'] = 0
        df['Foreign_Sentiment'] = 'Data Tidak Tersedia'
        df['Foreign_Score'] = 0
    
    return df

def calculate_ubo_control_index(df_harian, df_master):
    """
    5️⃣ UBO Control Index (Advanced)
    """
    if df_master.empty:
        return df_harian
    
    df = df_harian.copy()
    
    # Get top 5 holders per stock from latest data
    latest_date = df_master['Tanggal_Data'].max()
    df_latest = df_master[df_master['Tanggal_Data'] == latest_date]
    
    # Calculate top 5 concentration
    top_holders = df_latest.groupby('Kode Efek').apply(
        lambda x: x.nlargest(5, 'Jumlah Saham (Curr)')['Jumlah Saham (Curr)'].sum()
    ).reset_index()
    top_holders.columns = ['Stock Code', 'Top5_Total_Shares']
    
    df = df.merge(top_holders, on='Stock Code', how='left')
    df['Top5_Total_Shares'] = df['Top5_Total_Shares'].fillna(0)
    
    # Concentration Ratio
    df['Concentration_Ratio'] = (df['Top5_Total_Shares'] / (df['Listed Shares'] + 1)) * 100
    
    # High Control Stock Flag
    df['High_Control_Stock'] = (
        (df['Concentration_Ratio'] > 60) &
        (df['Free Float'] < 35) &
        (df['Volume'] / (df['Tradeble Shares'] + 1) < 0.003)
    ).astype(int)
    
    # Control Score
    control_score = 0
    control_score += np.clip(df['Concentration_Ratio'] * 0.8, 0, 40)  # 50% = 40
    control_score += (100 - df['Free Float']) * 0.4  # Low float = high control
    control_score += np.clip((1 - df['Volume'] / (df['Tradeble Shares'] + 1) * 1000) * 20, 0, 20)
    
    df['Control_Score'] = np.clip(control_score, 0, 100)
    
    return df

def calculate_float_domination_score(df):
    """
    6️⃣ FLOAT DOMINATION SCORE (Composite Ultimate Indicator)
    """
    df = df.copy()
    
    # Ensure all component scores exist
    required_scores = ['Liquidity_Score', 'Smart_Money_Score', 'Holder_Score', 
                      'Foreign_Score', 'Control_Score']
    
    for score in required_scores:
        if score not in df.columns:
            df[score] = 0
    
    # Weight configuration
    weights = {
        'Liquidity_Score': 0.20,      # 20% - How tradable?
        'Smart_Money_Score': 0.25,     # 25% - Is smart money moving?
        'Holder_Score': 0.20,           # 20% - Are big holders accumulating?
        'Foreign_Score': 0.20,          # 20% - Foreign dominance?
        'Control_Score': 0.15           # 15% - Is stock easily controlled?
    }
    
    # Calculate weighted score
    df['Float_Domination_Score'] = 0
    for score, weight in weights.items():
        df['Float_Domination_Score'] += df[score] * weight
    
    # Classification
    conditions = [
        df['Float_Domination_Score'] >= 80,
        df['Float_Domination_Score'] >= 70,
        df['Float_Domination_Score'] >= 60,
        df['Float_Domination_Score'] >= 50,
        df['Float_Domination_Score'] < 50
    ]
    choices = [
        '🚀 PRIME - Perfect Setup',
        '💪 STRONG - Ready to Move',
        '📈 GOOD - Watch Closely',
        '👀 WATCH - Potential',
        '💤 SLEEP - No Signal'
    ]
    df['Domination_Level'] = np.select(conditions, choices, default='Unknown')
    
    # Detailed recommendation
    rec_conditions = [
        (df['Float_Domination_Score'] >= 75) & (df['Smart_Money_Score'] >= 70) & (df['Holder_Score'] >= 60),
        (df['Float_Domination_Score'] >= 65) & (df['Foreign_Score'] >= 50),
        (df['Float_Domination_Score'] >= 60) & (df['Control_Score'] >= 70),
        (df['Float_Domination_Score'] >= 50) & (df['Liquidity_Score'] < 30),
        df['Float_Domination_Score'] < 50
    ]
    rec_choices = [
        '🎯 STRONG BUY - Institutional Quality',
        '📈 BUY - Foreign Accumulation',
        '👀 SPEC BUY - High Control Stock',
        '⚠️ WATCH - Manipulation Risk',
        '⏸️ HOLD - Wait for Signal'
    ]
    df['Float_Domination_Recommendation'] = np.select(rec_conditions, rec_choices, default='🤔 Neutral')
    
    return df

# =============================================================================
# FORMATTER FUNCTIONS
# =============================================================================
def format_rupiah(angka):
    if pd.isna(angka) or angka == 0:
        return "Rp 0"
    return f"Rp {angka:,.0f}".replace(",", ".")

def format_lembar(angka):
    if pd.isna(angka) or angka == 0:
        return "0"
    return f"{angka:,.0f}".replace(",", ".")

def format_persen(angka):
    if pd.isna(angka):
        return "0.00%"
    return f"{angka:.2f}%"

def format_float(x):
    if pd.isna(x):
        return "0.00"
    return f"{x:.2f}"

# =============================================================================
# KONFIGURASI HALAMAN
# =============================================================================
st.set_page_config(
    page_title="Bandar Eye IDX - Institutional Grade",
    page_icon="🐋",
    layout="wide"
)

# =============================================================================
# LOAD ALL DATA
# =============================================================================
with st.spinner('Memuat data harga...'):
    df_harian = load_harian()
with st.spinner('Memuat data KSEI...'):
    df_ksei = load_ksei()
with st.spinner('Memuat data kepemilikan 5%...'):
    df_master = load_master_5()

if df_harian.empty:
    st.error("⚠️ Data harian tidak tersedia. Dashboard tidak dapat berjalan.")
    st.stop()

# =============================================================================
# APPLY ALL ADVANCED CALCULATIONS
# =============================================================================
with st.spinner('Menganalisis Float Pressure...'):
    df_harian = calculate_float_pressure(df_harian)

with st.spinner('Mendeteksi Smart Money...'):
    df_harian = calculate_smart_money_accumulation(df_harian)

with st.spinner('Menganalisis Pergerakan Holder...'):
    df_harian = calculate_holder_movement(df_harian, df_master)

with st.spinner('Menghitung Foreign Dominance...'):
    df_harian = calculate_foreign_dominance(df_harian)

with st.spinner('Menghitung UBO Control Index...'):
    df_harian = calculate_ubo_control_index(df_harian, df_master)

with st.spinner('Menghitung Float Domination Score...'):
    df_harian = calculate_float_domination_score(df_harian)

# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.image("https://img.icons8.com/fluency/96/whale.png", width=80)
st.sidebar.title("🐋 Bandar Eye")
st.sidebar.caption("v3.0 - Float Domination Edition")

# Date filters
min_date = df_harian['Last Trading Date'].min()
max_date = df_harian['Last Trading Date'].max()

st.sidebar.subheader("📅 Filter Tanggal")
start_date = st.sidebar.date_input("Dari", min_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("Sampai", max_date, min_value=min_date, max_value=max_date)

# Sector filter
if 'Sector' in df_harian.columns:
    sektor_list = sorted(df_harian['Sector'].dropna().unique())
    selected_sectors = st.sidebar.multiselect("🏭 Sektor", sektor_list, default=[])
else:
    selected_sectors = []

# Quick filters
st.sidebar.markdown("---")
st.sidebar.subheader("⚡ Quick Filter")

quick_filter = st.sidebar.selectbox(
    "Preset Filter",
    ["All Stocks", 
     "🚀 Prime Setup (Score >80)", 
     "💪 Strong Accumulation", 
     "🔥 High Manipulation Risk",
     "🎯 Stealth Accumulation",
     "🌍 Foreign Dominant"]
)

# Apply date filter
df_filtered = df_harian[
    (df_harian['Last Trading Date'] >= pd.to_datetime(start_date)) &
    (df_harian['Last Trading Date'] <= pd.to_datetime(end_date))
].copy()

if selected_sectors:
    df_filtered = df_filtered[df_filtered['Sector'].isin(selected_sectors)]

# Apply quick filter
if quick_filter == "🚀 Prime Setup (Score >80)":
    df_filtered = df_filtered[df_filtered['Float_Domination_Score'] >= 80]
elif quick_filter == "💪 Strong Accumulation":
    df_filtered = df_filtered[df_filtered['Smart_Money_Signal'] == '💪 Strong Accumulation']
elif quick_filter == "🔥 High Manipulation Risk":
    df_filtered = df_filtered[df_filtered['High_Manipulation_Risk'] == 1]
elif quick_filter == "🎯 Stealth Accumulation":
    df_filtered = df_filtered[df_filtered['Holder_Movement_Signal'] == '🎯 Strong Stealth Accumulation']
elif quick_filter == "🌍 Foreign Dominant":
    df_filtered = df_filtered[df_filtered['Foreign_Dominance_Level'] == '🔥 Sangat Dominan (>40%)']

# =============================================================================
# MAIN DASHBOARD
# =============================================================================
st.title("🐋 Bandar Eye IDX - Institutional Grade")
st.caption(f"Data terakhir: {max_date.strftime('%d %B %Y')} | Total Saham: {df_filtered['Stock Code'].nunique():,}")

# Top metrics
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("🚀 Prime Setup", f"{len(df_filtered[df_filtered['Float_Domination_Score'] >= 80]):,}")
with col2:
    st.metric("💪 Strong Accum", f"{len(df_filtered[df_filtered['Smart_Money_Signal'] == '💪 Strong Accumulation']):,}")
with col3:
    st.metric("🎯 Stealth Accum", f"{len(df_filtered[df_filtered['Holder_Movement_Signal'] == '🎯 Strong Stealth Accumulation']):,}")
with col4:
    st.metric("🔥 High Risk", f"{len(df_filtered[df_filtered['High_Manipulation_Risk'] == 1]):,}")
with col5:
    st.metric("🌍 Foreign Dominant", f"{len(df_filtered[df_filtered['Foreign_Dominance_Level'] == '🔥 Sangat Dominan (>40%)']):,}")

# =============================================================================
# TABS
# =============================================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔥 FLOAT DOMINATION SCORE",
    "📊 Market Microstructure",
    "🏦 Smart Money Flow",
    "👥 Holder Analysis",
    "🌍 Foreign Flow",
    "🎯 Scanner & Watchlist"
])

# =============================================================================
# TAB 1: FLOAT DOMINATION SCORE (MAIN INDICATOR)
# =============================================================================
with tab1:
    st.header("🔥 FLOAT DOMINATION SCORE - Ultimate Indicator")
    st.caption("Composite score dari 5 komponen: Liquidity, Smart Money, Holder Movement, Foreign Flow, dan Control Index")
    
    # Top stocks by Float Domination Score
    top_domination = df_filtered.nlargest(50, 'Float_Domination_Score')
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🏆 Top 50 Stocks by Float Domination Score")
        
        display_cols = ['Stock Code', 'Close', 'Change %', 'Float_Domination_Score', 
                       'Domination_Level', 'Float_Domination_Recommendation',
                       'Liquidity_Status', 'Smart_Money_Signal', 'Holder_Movement_Signal']
        
        # Filter to existing columns
        display_cols = [c for c in display_cols if c in top_domination.columns]
        
        st.dataframe(
            top_domination[display_cols].head(50),
            use_container_width=True,
            hide_index=True,
            column_config={
                'Close': st.column_config.NumberColumn(format="Rp %d"),
                'Change %': st.column_config.NumberColumn(format="%.2f%%"),
                'Float_Domination_Score': st.column_config.NumberColumn(format="%.1f")
            }
        )
    
    with col2:
        st.subheader("📊 Score Distribution")
        
        # Histogram
        fig = px.histogram(
            df_filtered,
            x='Float_Domination_Score',
            nbins=20,
            title="Distribusi Float Domination Score",
            labels={'Float_Domination_Score': 'Score'}
        )
        fig.add_vline(x=80, line_dash="dash", line_color="green", annotation_text="Prime")
        fig.add_vline(x=60, line_dash="dash", line_color="orange", annotation_text="Good")
        st.plotly_chart(fig, use_container_width=True)
        
        # Score breakdown for selected stock
        st.subheader("🔍 Score Breakdown")
        selected_stock = st.selectbox(
            "Pilih Saham untuk Breakdown",
            options=top_domination['Stock Code'].head(20).tolist()
        )
        
        if selected_stock:
            stock_data = df_filtered[df_filtered['Stock Code'] == selected_stock].iloc[0]
            
            # Radar chart for components
            components = pd.DataFrame({
                'Component': ['Liquidity', 'Smart Money', 'Holder', 'Foreign', 'Control'],
                'Score': [
                    stock_data['Liquidity_Score'],
                    stock_data['Smart_Money_Score'],
                    stock_data['Holder_Score'],
                    stock_data['Foreign_Score'],
                    stock_data['Control_Score']
                ]
            })
            
            fig = px.line_polar(
                components,
                r='Score',
                theta='Component',
                line_close=True,
                range_r=[0, 100],
                title=f"Score Components - {selected_stock}"
            )
            fig.update_traces(fill='toself')
            st.plotly_chart(fig, use_container_width=True)
            
            # Show all signals
            st.write(f"**Domination Level:** {stock_data['Domination_Level']}")
            st.write(f"**Recommendation:** {stock_data['Float_Domination_Recommendation']}")
            st.write(f"**Liquidity:** {stock_data['Liquidity_Status']}")
            st.write(f"**Smart Money:** {stock_data['Smart_Money_Signal']}")
            st.write(f"**Holder Movement:** {stock_data['Holder_Movement_Signal']}")
            st.write(f"**Foreign:** {stock_data['Foreign_Sentiment']} ({stock_data['Foreign_Dominance_Level']})")

# =============================================================================
# TAB 2: MARKET MICROSTRUCTURE (Float Pressure & Liquidity)
# =============================================================================
with tab2:
    st.header("📊 Market Microstructure Analysis")
    st.caption("Float Pressure, Liquidity Stress, dan Manipulation Risk")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💧 Liquidity Analysis")
        
        # Filter by liquidity status
        liq_status = st.multiselect(
            "Filter by Liquidity Status",
            ['💀 Sangat Kering (High Risk)', '⚠️ Kering', '🟡 Normal', '💧 Likuid'],
            default=['💀 Sangat Kering (High Risk)', '⚠️ Kering']
        )
        
        liq_df = df_filtered[df_filtered['Liquidity_Status'].isin(liq_status)]
        
        st.dataframe(
            liq_df[['Stock Code', 'Close', 'Volume', 'Free Float', 
                   'Float_Turnover_Ratio', 'Liquidity_Score', 'Liquidity_Status']].head(30),
            use_container_width=True,
            hide_index=True,
            column_config={
                'Close': st.column_config.NumberColumn(format="Rp %d"),
                'Volume': st.column_config.NumberColumn(format="%d"),
                'Float_Turnover_Ratio': st.column_config.NumberColumn(format="%.4f"),
                'Liquidity_Score': st.column_config.NumberColumn(format="%.1f")
            }
        )
    
    with col2:
        st.subheader("🔥 High Manipulation Risk Stocks")
        
        risk_df = df_filtered[df_filtered['High_Manipulation_Risk'] == 1]
        
        st.metric("Total High Risk Stocks", len(risk_df))
        
        if not risk_df.empty:
            st.dataframe(
                risk_df[['Stock Code', 'Close', 'Free Float', 'Float_Turnover_Ratio', 
                        'Liquidity_Score', 'Control_Score']].head(30),
                use_container_width=True,
                hide_index=True
            )
            
            # Scatter plot: Free Float vs Turnover
            fig = px.scatter(
                risk_df.head(100),
                x='Free Float',
                y='Float_Turnover_Ratio',
                color='Liquidity_Score',
                hover_name='Stock Code',
                title="High Risk Stocks: Free Float vs Turnover",
                labels={'Free Float': 'Free Float %', 'Float_Turnover_Ratio': 'Turnover Ratio'}
            )
            st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# TAB 3: SMART MONEY FLOW
# =============================================================================
with tab3:
    st.header("🏦 Smart Money Accumulation Detection")
    st.caption("Deteksi akumulasi smart money (foreign + volume spike + price stability)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💪 Strong Accumulation Candidates")
        
        strong_acc = df_filtered[df_filtered['Smart_Money_Signal'] == '💪 Strong Accumulation']
        
        st.metric("Total Strong Accumulation", len(strong_acc))
        
        if not strong_acc.empty:
            st.dataframe(
                strong_acc[['Stock Code', 'Close', 'Change %', 'Volume', 'MA20_vol',
                           'Foreign_Net', 'SMF_20D', 'Smart_Money_Score']].head(30),
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Close': st.column_config.NumberColumn(format="Rp %d"),
                    'Change %': st.column_config.NumberColumn(format="%.2f%%"),
                    'Foreign_Net': st.column_config.NumberColumn(format="Rp %d"),
                    'SMF_20D': st.column_config.NumberColumn(format="Rp %d")
                }
            )
    
    with col2:
        st.subheader("📈 Moderate Accumulation")
        
        mod_acc = df_filtered[df_filtered['Smart_Money_Signal'] == '📈 Moderate Accumulation']
        
        st.metric("Total Moderate Accumulation", len(mod_acc))
        
        if not mod_acc.empty:
            st.dataframe(
                mod_acc[['Stock Code', 'Close', 'Change %', 'Volume', 'Foreign_Net', 
                        'Smart_Money_Score']].head(30),
                use_container_width=True,
                hide_index=True
            )
    
    # Visualization
    st.subheader("📊 Smart Money Score Distribution")
    
    fig = px.scatter(
        df_filtered.nlargest(100, 'Smart_Money_Score'),
        x='Change %',
        y='Volume',
        size='Smart_Money_Score',
        color='Smart_Money_Signal',
        hover_name='Stock Code',
        title="Smart Money: Volume vs Price Change",
        labels={'Change %': 'Price Change %', 'Volume': 'Volume'}
    )
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# TAB 4: HOLDER ANALYSIS (UBO Movement)
# =============================================================================
with tab4:
    st.header("👥 Big Holder Movement Analysis")
    st.caption("Deteksi pergerakan pemegang saham 5% (Ultimate Beneficial Owner)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Strong Stealth Accumulation")
        
        stealth_df = df_filtered[df_filtered['Holder_Movement_Signal'] == '🎯 Strong Stealth Accumulation']
        
        st.metric("Total Stealth Accumulation", len(stealth_df))
        
        if not stealth_df.empty:
            st.dataframe(
                stealth_df[['Stock Code', 'Close', 'Change %', 'Volume', 
                           'Holder_Change_Pct_Listed', 'Holder_Score', 'Control_Score']].head(30),
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Holder_Change_Pct_Listed': st.column_config.NumberColumn(format="%.2f%%"),
                    'Holder_Score': st.column_config.NumberColumn(format="%.1f")
                }
            )
    
    with col2:
        st.subheader("📊 Moderate Accumulation")
        
        mod_holder = df_filtered[df_filtered['Holder_Movement_Signal'] == '📊 Moderate Accumulation']
        
        st.metric("Total Moderate Accumulation", len(mod_holder))
        
        if not mod_holder.empty:
            st.dataframe(
                mod_holder[['Stock Code', 'Close', 'Holder_Change_Pct_Listed', 
                           'Holder_Score']].head(30),
                use_container_width=True,
                hide_index=True
            )
    
    # UBO Control Index
    st.subheader("🎮 UBO Control Index (High Control Stocks)")
    
    high_control = df_filtered[df_filtered['High_Control_Stock'] == 1]
    
    st.metric("Total High Control Stocks", len(high_control))
    
    if not high_control.empty:
        st.dataframe(
            high_control[['Stock Code', 'Close', 'Free Float', 'Concentration_Ratio',
                         'Control_Score', 'Liquidity_Status']].head(30),
            use_container_width=True,
            hide_index=True,
            column_config={
                'Concentration_Ratio': st.column_config.NumberColumn(format="%.1f%%")
            }
        )

# =============================================================================
# TAB 5: FOREIGN FLOW
# =============================================================================
with tab5:
    st.header("🌍 Foreign Flow Dominance Model")
    st.caption("Analisis dominasi dan sentimen investor asing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔥 Sangat Dominan (>40%)")
        
        dom_df = df_filtered[df_filtered['Foreign_Dominance_Level'] == '🔥 Sangat Dominan (>40%)']
        
        st.metric("Total Sangat Dominan", len(dom_df))
        
        if not dom_df.empty:
            st.dataframe(
                dom_df[['Stock Code', 'Close', 'Change %', 'Foreign_Dominance',
                       'Foreign_Sentiment', 'Foreign_Score']].head(30),
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Foreign_Dominance': st.column_config.NumberColumn(format="%.1f%%")
                }
            )
    
    with col2:
        st.subheader("💪 Dominan (25-40%)")
        
        mod_dom = df_filtered[df_filtered['Foreign_Dominance_Level'] == '💪 Dominan (25-40%)']
        
        st.metric("Total Dominan", len(mod_dom))
        
        if not mod_dom.empty:
            st.dataframe(
                mod_dom[['Stock Code', 'Close', 'Foreign_Dominance', 
                        'Foreign_Sentiment']].head(30),
                use_container_width=True,
                hide_index=True
            )
    
    # Foreign Sentiment Distribution
    st.subheader("📊 Foreign Sentiment Distribution")
    
    sentiment_counts = df_filtered['Foreign_Sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    
    fig = px.pie(
        sentiment_counts,
        values='Count',
        names='Sentiment',
        title="Distribusi Sentimen Asing",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Foreign Net vs Price Change
    st.subheader("📈 Foreign Net vs Price Change")
    
    foreign_scatter = df_filtered.nlargest(200, 'Foreign_Score').copy()
    foreign_scatter['Foreign_Net_Display'] = foreign_scatter['Foreign_Net'] / 1e9  # Convert to billions
    
    fig = px.scatter(
        foreign_scatter,
        x='Foreign_Net_Display',
        y='Change %',
        color='Foreign_Sentiment',
        size='Foreign_Score',
        hover_name='Stock Code',
        title="Foreign Net (Rp Miliar) vs Price Change %",
        labels={'Foreign_Net_Display': 'Foreign Net (Rp Miliar)', 'Change %': 'Price Change %'}
    )
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# TAB 6: SCANNER & WATCHLIST
# =============================================================================
with tab6:
    st.header("🎯 Multi-Dimension Scanner")
    
    scan_mode = st.radio(
        "Pilih Mode Scanner:",
        ["🔍 Hunter: Divergensi (Harga Turun, Akumulasi Naik)",
         "📋 Custom Watchlist",
         "📊 Sector Rotation"],
        horizontal=True
    )
    
    if scan_mode == "🔍 Hunter: Divergensi (Harga Turun, Akumulasi Naik)":
        st.subheader("💎 Deteksi Saham 'Salah Harga'")
        
        lookback = st.slider("Periode Analisa (hari)", 10, 90, 30)
        
        # Get first and last date in period
        end_date_div = df_filtered['Last Trading Date'].max()
        start_date_div = end_date_div - timedelta(days=lookback)
        
        # Calculate price change
        df_period = df_filtered[
            (df_filtered['Last Trading Date'] >= start_date_div) &
            (df_filtered['Last Trading Date'] <= end_date_div)
        ]
        
        price_start = df_period.sort_values('Last Trading Date').groupby('Stock Code')['Close'].first()
        price_end = df_period.sort_values('Last Trading Date').groupby('Stock Code')['Close'].last()
        
        df_div = pd.DataFrame({'Price_Start': price_start, 'Price_End': price_end})
        df_div['Price_Chg_Pct'] = (df_div['Price_End'] - df_div['Price_Start']) / df_div['Price_Start'] * 100
        
        # Filter price down
        df_div = df_div[df_div['Price_Chg_Pct'] <= -5]
        
        # Add latest data
        latest = df_filtered.sort_values('Last Trading Date').groupby('Stock Code').last()
        df_div = df_div.merge(latest[['Smart_Money_Score', 'Holder_Score', 'Foreign_Score', 
                                      'Float_Domination_Score']], left_index=True, right_index=True, how='left')
        
        # Filter for accumulation scores
        df_div = df_div[
            (df_div['Smart_Money_Score'] >= 50) |
            (df_div['Holder_Score'] >= 50)
        ].sort_values('Float_Domination_Score', ascending=False)
        
        st.success(f"Ditemukan {len(df_div)} saham dengan potensi divergensi")
        
        if not df_div.empty:
            st.dataframe(
                df_div.reset_index()[['Stock Code', 'Price_Start', 'Price_End', 'Price_Chg_Pct',
                                     'Smart_Money_Score', 'Holder_Score', 'Float_Domination_Score']].head(50),
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Price_Start': st.column_config.NumberColumn(format="Rp %d"),
                    'Price_End': st.column_config.NumberColumn(format="Rp %d"),
                    'Price_Chg_Pct': st.column_config.NumberColumn(format="%.2f%%")
                }
            )
    
    elif scan_mode == "📋 Custom Watchlist":
        st.subheader("📋 Watchlist Personal")
        
        all_stocks = sorted(df_filtered['Stock Code'].unique())
        default_stocks = [s for s in ["BBCA", "BBRI", "ADRO", "TLKM"] if s in all_stocks]
        watchlist = st.multiselect("Pilih Saham:", all_stocks, default=default_stocks)
        
        if watchlist:
            df_watch = df_filtered[df_filtered['Stock Code'].isin(watchlist)].copy()
            
            # Get latest for each
            df_latest = df_watch.sort_values('Last Trading Date').groupby('Stock Code').last().reset_index()
            
            st.dataframe(
                df_latest[['Stock Code', 'Close', 'Change %', 'Float_Domination_Score',
                          'Domination_Level', 'Smart_Money_Signal', 'Holder_Movement_Signal',
                          'Foreign_Sentiment']],
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Close': st.column_config.NumberColumn(format="Rp %d"),
                    'Change %': st.column_config.NumberColumn(format="%.2f%%")
                }
            )
    
    else:  # Sector Rotation
        st.subheader("📊 Sector Rotation Analysis")
        
        if 'Sector' in df_filtered.columns:
            sector_agg = df_filtered.groupby('Sector').agg({
                'Float_Domination_Score': 'mean',
                'Smart_Money_Score': 'mean',
                'Holder_Score': 'mean',
                'Foreign_Score': 'mean',
                'Stock Code': 'count'
            }).round(2).reset_index()
            sector_agg.columns = ['Sector', 'Avg_Domination', 'Avg_SmartMoney', 
                                 'Avg_Holder', 'Avg_Foreign', 'Stock_Count']
            
            sector_agg = sector_agg.sort_values('Avg_Domination', ascending=False)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.dataframe(sector_agg, use_container_width=True, hide_index=True)
            
            with col2:
                fig = px.bar(
                    sector_agg.head(10),
                    x='Avg_Domination',
                    y='Sector',
                    orientation='h',
                    title="Top 10 Sectors by Domination Score",
                    color='Avg_Domination',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.caption("🐋 Bandar Eye IDX - Institutional Grade | Float Domination Edition v3.0")
st.caption(f"Last Updated: {datetime.now().strftime('%d-%m-%Y %H:%M')}")
