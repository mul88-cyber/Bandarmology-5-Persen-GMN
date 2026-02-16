import streamlit as st
import pandas as pd
import numpy as np
import gdown
import os

st.set_page_config(layout="wide")

# ==========================================================
# CONFIG
# ==========================================================

KOMPILASI_FILE_ID = "1t_wCljhepGBqZVrvleuZKldomQKop9DY"
MASTER_FILE_ID = "1mS7Xp_PMqFnLTikU7giDZ42mqcbsiYvx"

# ==========================================================
# LOAD FUNCTION
# ==========================================================

@st.cache_data(show_spinner=False)
def load_csv(file_id):
    filename = f"{file_id}.csv"
    if not os.path.exists(filename):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, filename, quiet=False)
    return pd.read_csv(filename)

with st.spinner("Loading databases..."):
    df = load_csv(KOMPILASI_FILE_ID)
    master = load_csv(MASTER_FILE_ID)

# ==========================================================
# PREPARE MARKET DATA
# ==========================================================

df['Tanggal'] = pd.to_datetime(df['Last Trading Date'], errors='coerce')
df = df.sort_values(['Stock Code', 'Tanggal'])

df['Foreign_Net'] = df['Foreign Buy'] - df['Foreign Sell']
df['Float_Turnover'] = df['Volume'] / (df['Tradeble Shares'] + 1)
df['Float_MarketCap'] = df['Close'] * df['Tradeble Shares']
df['Value_Turnover'] = df['Value'] / (df['Float_MarketCap'] + 1)
df['Vol_Spike'] = df['Volume'] / (df['MA20_vol'] + 1)

df['Value_MA20'] = df.groupby('Stock Code')['Value'].transform(lambda x: x.rolling(20).mean())
df['Value_Spike'] = df['Value'] / (df['Value_MA20'] + 1)

df['Freq_MA20'] = df.groupby('Stock Code')['Frequency'].transform(lambda x: x.rolling(20).mean())
df['Freq_Spike'] = df['Frequency'] / (df['Freq_MA20'] + 1)

df['Range_Pct'] = (df['High'] - df['Low']) / (df['Close'] + 1)
df['Bid_Offer_Ratio'] = df['Bid Volume'] / (df['Offer Volume'] + 1)
df['Est_MarketCap'] = df['Close'] * df['Listed Shares']

# ==========================================================
# SMART MONEY FLOW 20D
# ==========================================================

df['SMF_20D'] = df.groupby('Stock Code')['Foreign_Net'].transform(lambda x: x.rolling(20).sum())
df['SMF_Strength'] = df['SMF_20D'] / (df['Float_MarketCap'] + 1)

# ==========================================================
# HOLDER MOVEMENT DETECTOR
# ==========================================================

master['Delta_Shares'] = master['Jumlah Saham (Curr)'] - master['Jumlah Saham (Prev)']

holder_summary = master.groupby('Kode Efek')['Delta_Shares'].sum().reset_index()
holder_summary.columns = ['Stock Code', 'Holder_Delta']

df = df.merge(holder_summary, on='Stock Code', how='left')
df['Holder_Delta'] = df['Holder_Delta'].fillna(0)

df['Holder_Accum_Flag'] = (
    (df['Holder_Delta'] > 0) &
    (df['Change %'].abs() < 5)
).astype(int)

# ==========================================================
# EARLY WARNING SYSTEM FLAGS
# ==========================================================

df['Flag_Volume_Spike'] = np.select(
    [
        df['Vol_Spike'] > 3,
        df['Vol_Spike'] < 0.3
    ],
    [
        '🚨 Volume Spike Tinggi',
        '⚠️ Volume Sepi'
    ],
    default='Normal'
)

df['Flag_Price_Shock'] = np.select(
    [
        df['Change %'] > 7,
        df['Change %'] < -5
    ],
    [
        '🚀 Lonjakan Harga',
        '📉 Koreksi Dalam'
    ],
    default='Stabil'
)

df['Flag_Bid_Offer'] = np.select(
    [
        df['Bid_Offer_Ratio'] > 2,
        df['Bid_Offer_Ratio'] < 0.5
    ],
    [
        '✅ Tekanan Beli',
        '❌ Tekanan Jual'
    ],
    default='Seimbang'
)

df['Flag_SMF'] = np.select(
    [
        (df['SMF_20D'] > df.groupby('Stock Code')['SMF_20D'].transform('quantile', 0.9)) & (df['Change %'] > 2),
        (df['SMF_20D'] < df.groupby('Stock Code')['SMF_20D'].transform('quantile', 0.1)) & (df['Change %'] < -2)
    ],
    [
        '💰 Akumulasi Agresif',
        '💸 Distribusi Agresif'
    ],
    default='Netral'
)

# ==========================================================
# ANALISIS HOLDER LEBIH DALAM
# ==========================================================

df['Holder_Action_Type'] = 'Netral / Tidak Jelas'
accum_condition = (df['Holder_Delta'] > 0) & (df['Holder_Delta'].notna())
df.loc[accum_condition & (df['Change %'] > 2), 'Holder_Action_Type'] = '🏦 Akumulasi + Harga Naik (Kuat)'
df.loc[accum_condition & (df['Change %'].between(-2, 2)), 'Holder_Action_Type'] = '🏦 Akumulasi + Harga Stabil (Averaging)'
df.loc[accum_condition & (df['Change %'] < -2), 'Holder_Action_Type'] = '🏦 Akumulasi + Harga Turun (Value Trap?)'

dist_condition = (df['Holder_Delta'] < 0) & (df['Holder_Delta'].notna())
df.loc[dist_condition & (df['Change %'] > 2), 'Holder_Action_Type'] = '📉 Distribusi + Harga Naik (Distribusi)'
df.loc[dist_condition & (df['Change %'].between(-2, 2)), 'Holder_Action_Type'] = '📉 Distribusi + Harga Stabil (Peaking?)'
df.loc[dist_condition & (df['Change %'] < -2), 'Holder_Action_Type'] = '📉 Distribusi + Harga Turun (Sell-off)'

df['Konfirmasi_Akumulasi'] = ((df['SMF_20D'] > 0) & (df['Holder_Delta'] > 0)).astype(int)

# ==========================================================
# SCORING MODELS
# ==========================================================

# 1️⃣ Intraday Swing
df['Swing_Score'] = (
    (df['Vol_Spike'] > 2.5).astype(int) * 25 +
    (df['Range_Pct'] > 0.05).astype(int) * 20 +
    (df['Value_Spike'] > 2).astype(int) * 20 +
    (df['Bid_Offer_Ratio'] > 1.2).astype(int) * 15 +
    (df['Freq_Spike'] > 1.5).astype(int) * 10 +
    (df['SMF_20D'] > 0).astype(int) * 10
)

# 2️⃣ Mid-Term Accumulation
df['Accum_Score'] = (
    ((df['Vol_Spike'] > 1.3) & (df['Vol_Spike'] < 2)).astype(int) * 25 +
    (df['Change %'].abs() < 3).astype(int) * 20 +
    (df['SMF_20D'] > 0).astype(int) * 20 +
    (df['Float_Turnover'] < 0.005).astype(int) * 15 +
    (df['Holder_Accum_Flag'] == 1).astype(int) * 20
)

# 3️⃣ Goreng Risk
df['Goreng_Score'] = (
    (df['Free Float'] < 35).astype(int) * 30 +
    (df['Float_Turnover'] < 0.004).astype(int) * 30 +
    (df['Value_Turnover'] < 0.0015).astype(int) * 20 +
    (df['Est_MarketCap'] < 5_000_000_000_000).astype(int) * 10 +
    (df['Holder_Delta'] < 0).astype(int) * 10
)

# 4️⃣ Institutional Composite
df['Institutional_Score'] = (
    df['Swing_Score'] * 0.25 +
    df['Accum_Score'] * 0.35 +
    df['Goreng_Score'] * 0.25 +
    (df['SMF_Strength'] > 0).astype(int) * 15
)

# ==========================================================
# AI BREAKOUT PROBABILITY MODEL (RULE-BASED)
# ==========================================================

df['Breakout_Probability'] = (
    df['Swing_Score'] * 0.3 +
    df['Accum_Score'] * 0.3 +
    (df['SMF_20D'] > 0).astype(int) * 20 +
    (df['Holder_Accum_Flag'] == 1).astype(int) * 20
)

df['Breakout_Probability'] = np.clip(df['Breakout_Probability'], 0, 100)

# ==========================================================
# LATEST SNAPSHOT
# ==========================================================

latest_date = df['Tanggal'].max()
latest_df = df[df['Tanggal'] == latest_date]

# ==========================================================
# RINGKASAN SEKTORAL
# ==========================================================

sector_summary = latest_df.groupby('Sector').agg({
    'Breakout_Probability': lambda x: (x > 70).sum(),
    'Institutional_Score': 'mean',
    'SMF_20D': 'sum',
    'Stock Code': 'count',
    'Konfirmasi_Akumulasi': 'sum'
}).rename(columns={
    'Breakout_Probability': 'Jumlah_Breakout_Tinggi',
    'Stock Code': 'Total_Saham',
    'Institutional_Score': 'Rata_Institutional_Score',
    'SMF_20D': 'Total_SMF_20D',
    'Konfirmasi_Akumulasi': 'Total_Konfirmasi_Akumulasi'
}).round(2).reset_index()

sector_summary = sector_summary.sort_values('Jumlah_Breakout_Tinggi', ascending=False)

# ==========================================================
# UI
# ==========================================================

st.title("🏦 Institutional Intelligence Engine PRO+")
st.caption(f"Data terakhir: {latest_date.date()}")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Filter & Konfigurasi")
    
    # Filter Sektor
    sector_list = ["All"] + sorted(latest_df['Sector'].dropna().unique().tolist())
    selected_sector = st.selectbox("Pilih Sektor", sector_list)
    
    st.markdown("---")
    st.header("📊 Ringkasan Cepat")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Saham", f"{len(latest_df):,}")
    with col2:
        st.metric("Breakout >70", f"{len(latest_df[latest_df['Breakout_Probability'] > 70]):,}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Konfirmasi Akumulasi", f"{len(latest_df[latest_df['Konfirmasi_Akumulasi'] == 1]):,}")
    with col2:
        st.metric("Holder Accum", f"{len(latest_df[latest_df['Holder_Delta'] > 0]):,}")
    
    st.markdown("---")
    st.caption("🚀 PRO+ Version dengan Early Warning System")

# Filter berdasarkan sektor yang dipilih
filtered_df = latest_df.copy()
if selected_sector != "All":
    filtered_df = filtered_df[filtered_df['Sector'] == selected_sector]

# --- RINGKASAN SEKTORAL (EXPANDER) ---
with st.expander("📈 Analisis Sektoral (Top Sectors by Breakout Signals)", expanded=False):
    col1, col2 = st.columns([3, 2])
    with col1:
        st.dataframe(
            sector_summary.head(10),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Total_SMF_20D": st.column_config.NumberColumn(format="Rp %d")
            }
        )
    with col2:
        st.bar_chart(
            sector_summary.head(10).set_index('Sector')['Jumlah_Breakout_Tinggi'],
            use_container_width=True
        )

# --- TABS UTAMA ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "⚡ Intraday Swing",
    "🏦 Mid-Term Accumulation",
    "🔥 Goreng Risk",
    "🧠 Breakout Probability AI",
    "👥 Holder Movement",
    "🚨 Early Warning System"
])

# FUNGSI HELPERS UNTUK FORMAT DATA
def format_rupiah(df, columns):
    """Format kolom tertentu sebagai Rupiah di dataframe"""
    config = {}
    for col in columns:
        if col in df.columns:
            config[col] = st.column_config.NumberColumn(format="Rp %d")
    return config

# TAB 1: Intraday Swing
with tab1:
    st.subheader("⚡ Potensi Intraday Swing (Volume & Range Tinggi)")
    
    cols = ['Stock Code', 'Close', 'Change %', 'Vol_Spike', 'Range_Pct', 
            'Flag_Volume_Spike', 'Flag_Bid_Offer', 'Swing_Score']
    
    display_df = filtered_df.sort_values('Swing_Score', ascending=False).head(30)
    
    st.dataframe(
        display_df[cols],
        use_container_width=True,
        column_config=format_rupiah(display_df, ['Close'])
    )

# TAB 2: Mid-Term Accumulation
with tab2:
    st.subheader("🏦 Akumulasi Jangka Menengah (Institutional + Holder)")
    
    cols = ['Stock Code', 'Close', 'Change %', 'SMF_20D', 'Holder_Delta', 
            'Holder_Action_Type', 'Konfirmasi_Akumulasi', 'Accum_Score']
    
    display_df = filtered_df.sort_values('Accum_Score', ascending=False).head(30)
    
    st.dataframe(
        display_df[cols],
        use_container_width=True,
        column_config=format_rupiah(display_df, ['Close', 'SMF_20D'])
    )

# TAB 3: Goreng Risk
with tab3:
    st.subheader("🔥 Potensi Gorengan (Liquiditas Rendah, Holder Keluar)")
    
    cols = ['Stock Code', 'Close', 'Free Float', 'Float_Turnover', 
            'Value_Turnover', 'Holder_Delta', 'Goreng_Score']
    
    display_df = filtered_df.sort_values('Goreng_Score', ascending=False).head(30)
    
    st.dataframe(
        display_df[cols],
        use_container_width=True,
        column_config=format_rupiah(display_df, ['Close'])
    )
    
    st.caption("⚠️ Semakin tinggi skor, semakin berisiko sebagai saham gorengan")

# TAB 4: Breakout Probability
with tab4:
    st.subheader("🧠 Probabilitas Breakout (AI Model)")
    
    cols = ['Stock Code', 'Close', 'Breakout_Probability', 'Swing_Score', 
            'Accum_Score', 'SMF_20D', 'Holder_Delta', 'Konfirmasi_Akumulasi']
    
    display_df = filtered_df.sort_values('Breakout_Probability', ascending=False).head(30)
    
    # Color coding untuk probabilitas
    def color_prob(val):
        if val > 80:
            return 'background-color: #00ff0022'  # Hijau terang
        elif val > 60:
            return 'background-color: #ffff0022'  # Kuning
        elif val > 40:
            return 'background-color: #ffaa0022'  # Oranye
        else:
            return ''
    
    styled_df = display_df[cols].style.applymap(color_prob, subset=['Breakout_Probability'])
    
    st.dataframe(
        styled_df,
        use_container_width=True,
        column_config=format_rupiah(display_df, ['Close', 'SMF_20D'])
    )

# TAB 5: Holder Movement
with tab5:
    st.subheader("👥 Pergerakan Holder (5% Threshold)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**🏦 Top Accumulation**")
        acc_df = filtered_df.nlargest(15, 'Holder_Delta')
        st.dataframe(
            acc_df[['Stock Code', 'Holder_Delta', 'Change %', 'Holder_Action_Type']],
            use_container_width=True,
            hide_index=True,
            column_config={
                'Holder_Delta': st.column_config.NumberColumn(format="%d saham")
            }
        )
    
    with col2:
        st.write("**📉 Top Distribution**")
        dist_df = filtered_df.nsmallest(15, 'Holder_Delta')
        st.dataframe(
            dist_df[['Stock Code', 'Holder_Delta', 'Change %', 'Holder_Action_Type']],
            use_container_width=True,
            hide_index=True,
            column_config={
                'Holder_Delta': st.column_config.NumberColumn(format="%d saham")
            }
        )
    
    st.markdown("---")
    st.write("**📊 Statistik Holder**")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Saham dengan Accumulation", 
                 f"{len(filtered_df[filtered_df['Holder_Delta'] > 0])} saham")
    with col2:
        st.metric("Saham dengan Distribution", 
                 f"{len(filtered_df[filtered_df['Holder_Delta'] < 0])} saham")
    with col3:
        st.metric("Konfirmasi Asing + Holder", 
                 f"{filtered_df['Konfirmasi_Akumulasi'].sum()} saham")

# TAB 6: Early Warning System
with tab6:
    st.subheader("🚨 Early Warning System")
    st.caption("Deteksi dini anomali pasar dan pergerakan tidak wajar")
    
    # Filter multi-select untuk tipe peringatan
    warning_options = st.multiselect(
        "Pilih Tipe Peringatan",
        ['🚨 Volume Spike Tinggi', '⚠️ Volume Sepi', 
         '✅ Tekanan Beli', '❌ Tekanan Jual',
         '💰 Akumulasi Agresif', '💸 Distribusi Agresif',
         '📉 Koreksi Dalam', '🚀 Lonjakan Harga'],
        default=['🚨 Volume Spike Tinggi', '💰 Akumulasi Agresif', '✅ Tekanan Beli']
    )
    
    # Mapping flag ke kolom
    flag_columns = {
        '🚨 Volume Spike Tinggi': 'Flag_Volume_Spike',
        '⚠️ Volume Sepi': 'Flag_Volume_Spike',
        '✅ Tekanan Beli': 'Flag_Bid_Offer',
        '❌ Tekanan Jual': 'Flag_Bid_Offer',
        '💰 Akumulasi Agresif': 'Flag_SMF',
        '💸 Distribusi Agresif': 'Flag_SMF',
        '📉 Koreksi Dalam': 'Flag_Price_Shock',
        '🚀 Lonjakan Harga': 'Flag_Price_Shock'
    }
    
    # Buat kondisi filter
    if warning_options:
        condition = pd.Series([False] * len(filtered_df))
        for opt in warning_options:
            col = flag_columns[opt]
            condition = condition | (filtered_df[col] == opt)
        
        warning_df = filtered_df[condition].sort_values('Vol_Spike', ascending=False)
    else:
        warning_df = filtered_df
    
    # Tampilkan hasil
    if len(warning_df) > 0:
        st.success(f"Ditemukan {len(warning_df)} saham dengan peringatan")
        
        display_cols = ['Stock Code', 'Close', 'Change %', 'Volume', 
                       'Flag_Volume_Spike', 'Flag_Price_Shock', 
                       'Flag_Bid_Offer', 'Flag_SMF', 'Holder_Action_Type']
        
        st.dataframe(
            warning_df[display_cols].head(50),
            use_container_width=True,
            column_config={
                'Volume': st.column_config.NumberColumn(format="%d"),
                'Close': st.column_config.NumberColumn(format="Rp %d")
            }
        )
    else:
        st.info("Tidak ada saham dengan peringatan yang dipilih")

# --- FOOTER ---
st.markdown("---")
st.success("✅ System Active — Institutional Detection + AI Model + Early Warning System Running")

# Debug info di expander (opsional)
with st.expander("🔧 Debug Info", expanded=False):
    st.write(f"Total data: {len(df):,} baris")
    st.write(f"Unique stocks: {df['Stock Code'].nunique()}")
    st.write(f"Latest date: {latest_date}")
    st.write(f"Filtered stocks: {len(filtered_df)}")
    
    # Tampilkan kolom yang tersedia
    if st.checkbox("Tampilkan semua kolom"):
        st.write(list(df.columns))
