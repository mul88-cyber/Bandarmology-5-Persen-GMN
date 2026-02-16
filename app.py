import streamlit as st
import pandas as pd
import numpy as np
import gdown
from io import StringIO

st.set_page_config(layout="wide")

# ==========================================================
# CONFIG - GANTI DENGAN FILE ID ANDA
# ==========================================================

KOMPILASI_FILE_ID = "1t_wCljhepGBqZVrvleuZKldomQKop9DY"
MASTER_FILE_ID = "1mS7Xp_PMqFnLTikU7giDZ42mqcbsiYvx"

# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data
def load_csv_from_drive(file_id):
    url = f"https://drive.google.com/uc?id={file_id}"
    output = f"{file_id}.csv"
    gdown.download(url, output, quiet=False)
    df = pd.read_csv(output)
    return df

with st.spinner("Loading Data..."):
    df = load_csv_from_drive(KOMPILASI_FILE_ID)
    master = load_csv_from_drive(MASTER_FILE_ID)

# ==========================================================
# DATA PREP
# ==========================================================

df['Tanggal'] = pd.to_datetime(df['Last Trading Date'], errors='coerce')
df = df.sort_values(['Stock Code', 'Tanggal'])

df['Foreign_Net'] = df['Foreign Buy'] - df['Foreign Sell']
df['Float_Turnover'] = df['Volume'] / df['Tradeble Shares']
df['Float_MarketCap'] = df['Close'] * df['Tradeble Shares']
df['Value_Turnover'] = df['Value'] / df['Float_MarketCap']
df['Vol_Spike'] = df['Volume'] / df['MA20_vol']

# ==========================================================
# SCORING MODELS
# ==========================================================

# A) BREAKOUT SCORE
df['Breakout_Score'] = (
    (df['Vol_Spike'] > 2).astype(int) * 40 +
    (df['Change %'] > 3).astype(int) * 30 +
    (df['Foreign_Net'] > 0).astype(int) * 20 +
    (df['Bid Volume'] > df['Offer Volume']).astype(int) * 10
)

# B) STEALTH ACCUMULATION SCORE
df['Accum_Score'] = (
    (df['Vol_Spike'] > 1.5).astype(int) * 30 +
    (df['Change %'].abs() < 3).astype(int) * 30 +
    (df['Foreign_Net'] > 0).astype(int) * 20 +
    (df['Float_Turnover'] < 0.01).astype(int) * 20
)

# C) FLOAT CONTROL SCORE
df['Control_Score'] = (
    (df['Free Float'] < 35).astype(int) * 30 +
    (df['Float_Turnover'] < 0.005).astype(int) * 40 +
    (df['Value_Turnover'] < 0.002).astype(int) * 30
)

# D) INSTITUTIONAL SCORE (COMPOSITE)
df['Institutional_Score'] = (
    df['Breakout_Score'] * 0.3 +
    df['Accum_Score'] * 0.3 +
    df['Control_Score'] * 0.4
)

# ==========================================================
# FILTER LATEST DATE
# ==========================================================

latest_date = df['Tanggal'].max()
latest_df = df[df['Tanggal'] == latest_date]

# ==========================================================
# STREAMLIT UI
# ==========================================================

st.title("📊 Institutional Flow Intelligence Dashboard")

tab1, tab2, tab3, tab4 = st.tabs([
    "🚀 Breakout Radar",
    "🕵 Stealth Accumulation",
    "🎯 Easy To Move",
    "🏦 Institutional Composite"
])

# ----------------------------------------------------------
# TAB 1 - BREAKOUT
# ----------------------------------------------------------
with tab1:
    st.subheader("Top Breakout Candidates")
    breakout = latest_df.sort_values('Breakout_Score', ascending=False)
    st.dataframe(breakout[['Stock Code','Close','Volume','Vol_Spike','Foreign_Net','Breakout_Score']].head(30))

# ----------------------------------------------------------
# TAB 2 - ACCUMULATION
# ----------------------------------------------------------
with tab2:
    st.subheader("Stealth Accumulation Candidates")
    accum = latest_df.sort_values('Accum_Score', ascending=False)
    st.dataframe(accum[['Stock Code','Close','Volume','Change %','Foreign_Net','Accum_Score']].head(30))

# ----------------------------------------------------------
# TAB 3 - FLOAT CONTROL
# ----------------------------------------------------------
with tab3:
    st.subheader("High Float Control Risk")
    control = latest_df.sort_values('Control_Score', ascending=False)
    st.dataframe(control[['Stock Code','Free Float','Float_Turnover','Value_Turnover','Control_Score']].head(30))

# ----------------------------------------------------------
# TAB 4 - INSTITUTIONAL MODEL
# ----------------------------------------------------------
with tab4:
    st.subheader("Institutional Composite Ranking")
    inst = latest_df.sort_values('Institutional_Score', ascending=False)
    st.dataframe(inst[['Stock Code','Institutional_Score','Breakout_Score','Accum_Score','Control_Score']].head(30))

st.success(f"Data terakhir: {latest_date.date()}")
