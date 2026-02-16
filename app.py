import streamlit as st
import pandas as pd
import numpy as np
import gdown
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

st.set_page_config(layout="wide")

# ==========================================================
# CONFIG
# ==========================================================

KOMPILASI_FILE_ID = "1t_wCljhepGBqZVrvleuZKldomQKop9DY"
MASTER_FILE_ID = "1mS7Xp_PMqFnLTikU7giDZ42mqcbsiYvx"

# ==========================================================
# LOAD CSV
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
# PREP DATA
# ==========================================================

df['Tanggal'] = pd.to_datetime(df['Last Trading Date'], errors='coerce')
df = df.sort_values(['Stock Code','Tanggal'])

df['Foreign_Net'] = df['Foreign Buy'] - df['Foreign Sell']
df['Float_MarketCap'] = df['Close'] * df['Tradeble Shares']
df['Vol_Spike'] = df['Volume'] / (df['MA20_vol'] + 1)
df['Float_Turnover'] = df['Volume'] / (df['Tradeble Shares'] + 1)
df['Range_Pct'] = (df['High'] - df['Low']) / (df['Close'] + 1)
df['Est_MarketCap'] = df['Close'] * df['Listed Shares']

# Smart Money Flow 20D
df['SMF_20D'] = df.groupby('Stock Code')['Foreign_Net'].transform(lambda x: x.rolling(20).sum())

# Accumulation Condition (daily)
df['Daily_Accum'] = (
    (df['Vol_Spike'] > 1.3) &
    (df['Change %'].abs() < 3) &
    (df['Foreign_Net'] > 0)
).astype(int)

# Accumulation Persistence (rolling 10 hari)
df['Accum_5D'] = df.groupby('Stock Code')['Daily_Accum'].transform(lambda x: x.rolling(5).sum())
df['Accum_10D'] = df.groupby('Stock Code')['Daily_Accum'].transform(lambda x: x.rolling(10).sum())

# Holder Movement
master['Delta_Shares'] = master['Jumlah Saham (Curr)'] - master['Jumlah Saham (Prev)']
holder_sum = master.groupby('Kode Efek')['Delta_Shares'].sum().reset_index()
holder_sum.columns = ['Stock Code','Holder_Delta']

df = df.merge(holder_sum, on='Stock Code', how='left')
df['Holder_Delta'] = df['Holder_Delta'].fillna(0)

# ==========================================================
# MACHINE LEARNING BREAKOUT MODEL
# Target: Next Day Close > 3%
# ==========================================================

df['Future_Return'] = df.groupby('Stock Code')['Close'].shift(-1) / df['Close'] - 1
df['Target'] = (df['Future_Return'] > 0.03).astype(int)

feature_cols = [
    'Vol_Spike',
    'Float_Turnover',
    'Range_Pct',
    'SMF_20D',
    'Holder_Delta',
    'Accum_5D'
]

ml_df = df.dropna(subset=feature_cols + ['Target'])

X = ml_df[feature_cols]
y = ml_df['Target']

model = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(max_iter=1000))
])

if len(ml_df) > 1000:
    model.fit(X, y)
    df['Breakout_Prob_ML'] = model.predict_proba(df[feature_cols])[:,1] * 100
else:
    df['Breakout_Prob_ML'] = 0

# ==========================================================
# SECTOR MONEY ROTATION
# ==========================================================

sector_flow = df.groupby(['Tanggal','Sector'])['SMF_20D'].sum().reset_index()

latest_date = df['Tanggal'].max()
latest_df = df[df['Tanggal'] == latest_date]

sector_latest = sector_flow[sector_flow['Tanggal'] == latest_date]
sector_pivot = sector_latest.pivot_table(
    index='Sector',
    values='SMF_20D'
)

# ==========================================================
# UI
# ==========================================================

st.title("🏦 Smart Money Scanner SaaS Platform")
st.caption(f"Latest Data: {latest_date.date()}")

tab1, tab2, tab3, tab4 = st.tabs([
    "🧠 ML Breakout Probability",
    "📊 Sector Money Rotation",
    "🏦 Accumulation Persistence",
    "👥 Holder Movement Monitor"
])

# ==========================================================
# TAB 1 - ML
# ==========================================================

with tab1:
    st.subheader("Machine Learning Breakout Probability (>3% next day)")
    ml_rank = latest_df.sort_values('Breakout_Prob_ML', ascending=False)
    st.dataframe(
        ml_rank[['Stock Code','Breakout_Prob_ML',
                 'Vol_Spike','SMF_20D','Accum_5D','Holder_Delta']].head(30),
        use_container_width=True
    )

# ==========================================================
# TAB 2 - SECTOR ROTATION
# ==========================================================

with tab2:
    st.subheader("Sector Smart Money Rotation (20D Flow)")
    st.dataframe(sector_pivot.sort_values('SMF_20D', ascending=False),
                 use_container_width=True)

# ==========================================================
# TAB 3 - ACCUM PERSISTENCE
# ==========================================================

with tab3:
    st.subheader("5–10 Day Consistent Accumulation")
    persist = latest_df.sort_values('Accum_10D', ascending=False)
    st.dataframe(
        persist[['Stock Code','Accum_5D','Accum_10D',
                 'SMF_20D','Holder_Delta']].head(30),
        use_container_width=True
    )

# ==========================================================
# TAB 4 - HOLDER
# ==========================================================

with tab4:
    holder_rank = latest_df.sort_values('Holder_Delta', ascending=False)
    st.dataframe(
        holder_rank[['Stock Code','Holder_Delta',
                     'Breakout_Prob_ML']].head(30),
        use_container_width=True
    )

st.success("SaaS Smart Money Scanner Active — ML + Institutional Flow Running")
