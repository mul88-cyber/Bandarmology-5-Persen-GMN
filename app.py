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
# UI
# ==========================================================

st.title("🏦 Institutional Intelligence Engine PRO+")
st.caption(f"Data terakhir: {latest_date.date()}")

sector_list = ["All"] + sorted(latest_df['Sector'].dropna().unique().tolist())
selected_sector = st.selectbox("Filter Sector", sector_list)

if selected_sector != "All":
    latest_df = latest_df[latest_df['Sector'] == selected_sector]

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "⚡ Intraday Swing",
    "🏦 Mid-Term Accumulation",
    "🔥 Goreng Risk",
    "🧠 Breakout Probability AI",
    "👥 Holder Movement"
])

# TAB 1
with tab1:
    swing = latest_df.sort_values('Swing_Score', ascending=False)
    st.dataframe(
        swing[['Stock Code','Close','Vol_Spike','SMF_20D',
               'Swing_Score']].head(30),
        use_container_width=True
    )

# TAB 2
with tab2:
    accum = latest_df.sort_values('Accum_Score', ascending=False)
    st.dataframe(
        accum[['Stock Code','Close','SMF_20D',
               'Holder_Delta','Accum_Score']].head(30),
        use_container_width=True
    )

# TAB 3
with tab3:
    goreng = latest_df.sort_values('Goreng_Score', ascending=False)
    st.dataframe(
        goreng[['Stock Code','Free Float','Float_Turnover',
               'Holder_Delta','Goreng_Score']].head(30),
        use_container_width=True
    )

# TAB 4
with tab4:
    breakout = latest_df.sort_values('Breakout_Probability', ascending=False)
    st.dataframe(
        breakout[['Stock Code','Breakout_Probability',
                  'Swing_Score','Accum_Score',
                  'SMF_20D','Holder_Delta']].head(30),
        use_container_width=True
    )

# TAB 5
with tab5:
    holder = latest_df.sort_values('Holder_Delta', ascending=False)
    st.dataframe(
        holder[['Stock Code','Holder_Delta',
                'Holder_Accum_Flag']].head(30),
        use_container_width=True
    )

st.success("System Active — Institutional Detection + AI Model Running")
