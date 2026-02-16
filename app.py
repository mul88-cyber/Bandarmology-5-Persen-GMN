import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import BytesIO
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import plotly.express as px

st.set_page_config(layout="wide")
st.title("🚀 Smart Money Scanner Pro")

# =====================================================
# 🔐 CONFIG – GANTI DENGAN FILE ID ANDA
# =====================================================

MARKET_FILE_ID = "1t_wCljhepGBqZVrvleuZKldomQKop9DY"
HOLDER_FILE_ID = "1mS7Xp_PMqFnLTikU7giDZ42mqcbsiYvx"

# =====================================================
# 📥 LOAD FROM GOOGLE DRIVE
# =====================================================

@st.cache_data
def load_csv_from_drive(file_id):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url)
    return pd.read_csv(BytesIO(response.content), low_memory=False)

try:
    df = load_csv_from_drive(MARKET_FILE_ID)
    holder_df = load_csv_from_drive(HOLDER_FILE_ID)
except Exception as e:
    st.error("❌ Gagal load data dari Google Drive")
    st.stop()

# =====================================================
# 🧹 CLEANING
# =====================================================

numeric_cols = [
    'Close', 'Volume', 'Value', 'Free Float',
    'Change %', 'Typical Price',
    'Tradeble Shares'
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# =====================================================
# 📊 FEATURE ENGINEERING
# =====================================================

df['Smart_Money'] = df['Value'] * df['Change %']

df['Smart_Money_20D'] = (
    df.groupby('Stock Code')['Smart_Money']
    .rolling(20)
    .sum()
    .reset_index(level=0, drop=True)
)

df['Free_Float_MarketCap'] = df['Close'] * df['Tradeble Shares']
df['Float_Turnover'] = df['Value'] / df['Free_Float_MarketCap']

df['Accum_Flag'] = np.where(df['Smart_Money'] > 0, 1, 0)

df['Accumulation_Persistence'] = (
    df.groupby('Stock Code')['Accum_Flag']
    .rolling(5)
    .sum()
    .reset_index(level=0, drop=True)
)

# =====================================================
# 🏦 HOLDER MOVEMENT
# =====================================================

holder_df['Jumlah Saham (Prev)'] = pd.to_numeric(holder_df['Jumlah Saham (Prev)'], errors='coerce')
holder_df['Jumlah Saham (Curr)'] = pd.to_numeric(holder_df['Jumlah Saham (Curr)'], errors='coerce')

holder_df['Net_Change'] = holder_df['Jumlah Saham (Curr)'] - holder_df['Jumlah Saham (Prev)']

holder_summary = (
    holder_df.groupby('Kode Efek')['Net_Change']
    .sum()
    .reset_index()
)

holder_summary.columns = ['Stock Code', 'Holder_Net_Change']

df = df.merge(holder_summary, on='Stock Code', how='left')
df['Holder_Net_Change'].fillna(0, inplace=True)

# =====================================================
# 🎯 TARGET (5-Day Breakout >5%)
# =====================================================

df['Future_Return'] = (
    df.groupby('Stock Code')['Close']
    .shift(-5) / df['Close'] - 1
)

df['Breakout_Target'] = np.where(df['Future_Return'] > 0.05, 1, 0)

# =====================================================
# 🤖 MACHINE LEARNING MODEL
# =====================================================

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

X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(0, inplace=True)

model = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000))
])

model.fit(X, y)

df['Breakout_Prob_ML'] = model.predict_proba(X)[:, 1] * 100

# =====================================================
# 📊 DASHBOARD
# =====================================================

latest_date = df['Last Trading Date'].max()
latest_df = df[df['Last Trading Date'] == latest_date]

st.subheader("🔥 Top Breakout Probability")

top_breakout = latest_df.sort_values(
    'Breakout_Prob_ML', ascending=False
).head(20)

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

# =====================================================
# 📈 SECTOR ROTATION
# =====================================================

st.subheader("📊 Sector Money Rotation")

sector_flow = (
    latest_df.groupby('Sector')['Smart_Money_20D']
    .sum()
    .reset_index()
)

fig = px.treemap(
    sector_flow,
    path=['Sector'],
    values='Smart_Money_20D',
    color='Smart_Money_20D'
)

st.plotly_chart(fig, use_container_width=True)

# =====================================================
# ⚠ SAHAM MUDAH DIGERAKKAN
# =====================================================

st.subheader("⚠ Saham Mudah Digerakkan")

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

st.success("✅ System Stable • SaaS Mode Active")
