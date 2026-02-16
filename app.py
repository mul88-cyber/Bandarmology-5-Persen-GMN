import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import plotly.express as px

st.set_page_config(layout="wide")

st.title("🚀 Smart Money Scanner Pro")
st.caption("Institutional Detection • Breakout Probability • Sector Rotation")

# =============================
# LOAD DATA
# =============================

@st.cache_data
def load_market_data():
    return pd.read_csv("Kompilasi_Data_1Tahun.csv")

@st.cache_data
def load_holder_data():
    return pd.read_csv("MASTER_DATABASE_5persen.csv")

df = load_market_data()
holder_df = load_holder_data()

# =============================
# BASIC CLEANING
# =============================

numeric_cols = [
    'Close', 'Volume', 'Value', 'Free Float',
    'Change %', 'Typical Price'
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.replace([np.inf, -np.inf], np.nan)
df.fillna(0, inplace=True)

# =============================
# FEATURE ENGINEERING
# =============================

# 1️⃣ Smart Money Flow (Value bias)
df['Smart_Money'] = df['Value'] * df['Change %']

df['Smart_Money_20D'] = (
    df.groupby('Stock Code')['Smart_Money']
    .rolling(20)
    .sum()
    .reset_index(level=0, drop=True)
)

# 2️⃣ Float Turnover
df['Free_Float_MarketCap'] = df['Close'] * df['Tradeble Shares']
df['Float_Turnover'] = df['Value'] / df['Free_Float_MarketCap']

# 3️⃣ Accumulation Persistence (5-day positive smart money streak)
df['Accum_Flag'] = np.where(df['Smart_Money'] > 0, 1, 0)

df['Accumulation_Persistence'] = (
    df.groupby('Stock Code')['Accum_Flag']
    .rolling(5)
    .sum()
    .reset_index(level=0, drop=True)
)

# 4️⃣ Holder Movement Detector (5% database)
holder_df['Net_Change'] = holder_df['Jumlah Saham (Curr)'] - holder_df['Jumlah Saham (Prev)']

holder_summary = (
    holder_df.groupby('Kode Efek')['Net_Change']
    .sum()
    .reset_index()
)

holder_summary.columns = ['Stock Code', 'Holder_Net_Change']

df = df.merge(holder_summary, on='Stock Code', how='left')
df['Holder_Net_Change'].fillna(0, inplace=True)

# =============================
# TARGET CREATION (Breakout label)
# =============================

df['Future_Return'] = (
    df.groupby('Stock Code')['Close']
    .shift(-5) / df['Close'] - 1
)

df['Breakout_Target'] = np.where(df['Future_Return'] > 0.05, 1, 0)

# =============================
# ML MODEL (Logistic Regression)
# =============================

feature_cols = [
    'Change %',
    'Volume',
    'Smart_Money_20D',
    'Float_Turnover',
    'Accumulation_Persistence',
    'Holder_Net_Change'
]

X = df[feature_cols].copy()
y = df['Breakout_Target']

X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

model = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000))
])

model.fit(X, y)

df['Breakout_Prob_ML'] = model.predict_proba(X)[:,1] * 100

# =============================
# DASHBOARD VIEW
# =============================

latest_date = df['Last Trading Date'].max()
latest_df = df[df['Last Trading Date'] == latest_date]

st.subheader("🔥 Top Breakout Probability")

top_breakout = latest_df.sort_values('Breakout_Prob_ML', ascending=False).head(20)

st.dataframe(
    top_breakout[[
        'Stock Code',
        'Close',
        'Breakout_Prob_ML',
        'Smart_Money_20D',
        'Accumulation_Persistence',
        'Float_Turnover',
        'Holder_Net_Change'
    ]]
)

# =============================
# SECTOR MONEY ROTATION
# =============================

st.subheader("📊 Sector Money Rotation Heatmap")

sector_flow = (
    latest_df.groupby('Sector')['Smart_Money_20D']
    .sum()
    .reset_index()
)

fig = px.treemap(
    sector_flow,
    path=['Sector'],
    values='Smart_Money_20D',
    color='Smart_Money_20D',
)

st.plotly_chart(fig, use_container_width=True)

# =============================
# RISK: SAHAM MUDAH DIGERAKKAN
# =============================

st.subheader("⚠ Saham Mudah Digerakkan (High Risk)")

goreng_risk = latest_df[
    (latest_df['Float_Turnover'] > 0.05) &
    (latest_df['Free Float'] < 40)
].sort_values('Float_Turnover', ascending=False)

st.dataframe(
    goreng_risk[[
        'Stock Code',
        'Close',
        'Float_Turnover',
        'Free Float',
        'Breakout_Prob_ML'
    ]].head(20)
)

st.success("System Stable • ML Active • Institutional Detection Running")
