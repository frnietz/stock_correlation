import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import date, timedelta

st.set_page_config(page_title="Simple Stock Correlation", layout="wide")

st.title("📈 Simple Stock Correlation (Daily Close)")
st.caption("Enter tickers and a date range. We fetch **daily Close** prices, compute daily % returns, and display the correlation matrix.")

with st.sidebar:
    st.header("Inputs")
    tickers_raw = st.text_input("Tickers (comma-separated)", value="AAPL,MSFT,GOOGL,AMZN,NVDA")
    today = date.today()
    one_year_ago = today - timedelta(days=365)
    start_date = st.date_input("Start date", value=one_year_ago, max_value=today - timedelta(days=1))
    end_date = st.date_input("End date", value=today, min_value=start_date + timedelta(days=1))

# Parse tickers
tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]

@st.cache_data(show_spinner=True, ttl=1800)
def fetch_close_prices(tickers, start, end):
    if not tickers:
        return pd.DataFrame()
    df = yf.download(tickers, start=start, end=end + timedelta(days=1), progress=False, auto_adjust=False)
    # If multiple tickers, df has MultiIndex columns (field, ticker)
    if isinstance(df.columns, pd.MultiIndex):
        close = df.xs("Close", axis=1, level=0)
    else:
        # Single ticker: columns are fields
        if "Close" in df.columns:
            close = df[["Close"]].copy()
        else:
            # fallback if Close missing
            fallback = "Adj Close" if "Adj Close" in df.columns else df.columns[0]
            close = df[[fallback]].copy()
        close.columns = [tickers[0]]
    close.index = pd.to_datetime(close.index)
    return close.sort_index()

prices = fetch_close_prices(tickers, start_date, end_date)


