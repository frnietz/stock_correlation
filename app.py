import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import date, timedelta

st.set_page_config(page_title="Stock Correlation Matrix", layout="wide")

st.title("📈 Correlation Matrix of Selected Stocks")
st.caption("Pick tickers, date range, sampling interval, and (optionally) a rolling window. We'll compute returns and show a correlation heatmap.")

# -----------------------------
# Predefined baskets
# -----------------------------
PREDEFINED = {
    "None": [],
    "Tech Megacaps (US)": ["AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","ORCL","INTC","CSCO","ADBE"],
    "Airlines (US + Intl)": ["AAL","DAL","UAL","LUV","ALK","JBLU","RYAAY","BA","EADSY"],
    "US Money Center Banks": ["JPM","BAC","WFC","C","GS","MS","USB","PNC"],
    "Consumer Staples (US)": ["KO","PEP","PG","WMT","COST","MDLZ","KHC"],
    "Semiconductors": ["NVDA","AMD","AVGO","TSM","QCOM","INTC","MU","TXN"],
    "S&P Sectors (SPDR ETFs)": ["XLB","XLE","XLF","XLI","XLK","XLP","XLU","XLV","XLY","XLRE","XLC"],
    "Energy Majors": ["XOM","CVX","SHEL","BP","TTE"],
    "Media & Streaming": ["NFLX","DIS","PARA","WBD","ROKU","SPOT"],
}

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    default_universe = sorted(list(set(sum(PREDEFINED.values(), [])))) + [
        "MA","V","HD","NKE","CRM","PFE","BABA","T","BA"
    ]
    universe = st.text_area(
        "Universe (comma-separated tickers)",
        value=",".join(default_universe),
        help="Edit or paste your own list. Separate by commas."
    )
    all_tickers = sorted({t.strip().upper() for t in universe.split(",") if t.strip()})
    
    # Basket selection
    st.subheader("🎒 Predefined basket")
    basket_name = st.selectbox("Choose a basket", options=list(PREDEFINED.keys()), index=0)
    basket_ticks = PREDEFINED.get(basket_name, [])
    st.write("Basket tickers:", ", ".join(basket_ticks) if basket_ticks else "—")
    use_basket = st.checkbox("Use basket tickers", value=False, help="If checked, the basket tickers are used (you can also append them).")
    append_basket = st.checkbox("Append basket to manual selection", value=True, help="If unchecked while 'Use basket' is checked, only the basket tickers are used.")
    
    # Manual selection
    st.subheader("🧩 Manual selection")
    preselect = [t for t in ["AAPL","MSFT","GOOGL","AMZN","NVDA"] if t in all_tickers]
    manual = st.multiselect("Select tickers", options=all_tickers, default=preselect, help="Choose 2+ tickers")
    
    # Date range
    today = date.today()
    one_year_ago = today - timedelta(days=365)
    col_start, col_end = st.columns(2)
    with col_start:
        start_date = st.date_input("Start date", value=one_year_ago, max_value=today - timedelta(days=1))
    with col_end:
        end_date = st.date_input("End date", value=today, min_value=start_date + timedelta(days=1))
    
    # Interval / resampling
    interval_label = st.selectbox("Sampling interval", ["Daily","Weekly (Fri close)","Monthly (month-end)"], index=0)
    price_field = st.selectbox("Price field", ["Adj Close","Close","Open","High","Low"], index=0)
    returns_mode = st.radio("Returns mode", ["Daily % change","Daily log returns","Price levels (no returns)"], index=0,
                            help="Correlation on returns is usually more meaningful than on raw prices.")
    missing_handling = st.selectbox("Missing data", ["Drop rows with any NA","Forward-fill then drop remaining"], index=1)
    
    # Rolling window
    st.subheader("🧮 Rolling-window correlation")
    rolling_mode = st.checkbox("Enable rolling window", value=False, help="Compute correlation over a moving window of N periods.")
    window_size = st.slider("Window size (periods)", min_value=5, max_value=252, value=60, step=5, disabled=not rolling_mode,
                            help="Periods depend on sampling interval (e.g., daily ≈ trading days).")
    
# Resolve final ticker list
if use_basket and basket_ticks:
    if append_basket:
        tickers = sorted(set(manual).union(basket_ticks))
    else:
        tickers = list(dict.fromkeys(basket_ticks))  # preserve order & de-dupe
else:
    tickers = manual

# -----------------------------
# Data functions
# -----------------------------
@st.cache_data(show_spinner=True, ttl=3600)
def fetch_prices(tickers, start, end, field):
    if len(tickers) == 0:
        return pd.DataFrame()
    df = yf.download(tickers, start=start, end=end + timedelta(days=1), progress=False, auto_adjust=False)
    # Normalize to single-level columns of tickers
    if isinstance(df.columns, pd.MultiIndex):
        if field in df.columns.get_level_values(0):
            sub = df[field].copy()
        else:
            fallback = "Adj Close" if "Adj Close" in df.columns.get_level_values(0) else "Close"
            sub = df[fallback].copy()
        sub.columns = sub.columns.get_level_values(1)
    else:
        # Single ticker
        if field in df.columns:
            sub = df[[field]].copy()
        else:
            fallback = "Adj Close" if "Adj Close" in df.columns else "Close"
            sub = df[[fallback]].copy()
        sub.columns = tickers
    return sub.sort_index()

def resample_prices(prices_df, interval_label):
    if prices_df.empty:
        return prices_df
    if interval_label.startswith("Daily"):
        return prices_df
    elif interval_label.startswith("Weekly"):
        return prices_df.resample("W-FRI").last()
    elif interval_label.startswith("Monthly"):
        return prices_df.resample("M").last()
    return prices_df

def prepare_matrix_data(prices_df, mode, missing):
    if prices_df.empty or len(prices_df.columns) < 2:
        return None, None
    df = prices_df.ffill().dropna() if (missing == "Forward-fill then drop remaining") else prices_df.dropna()
    if df.empty: 
        return None, None
    if mode == "Daily % change":
        ret = df.pct_change().dropna(how="all")
    elif mode == "Daily log returns":
        ret = np.log(df).diff().dropna(how="all")
    else:
        ret = df
    ret = ret.dropna(axis=1, how="all")
    if ret.shape[1] < 2:
        return None, None
    return ret, ret.corr()

def rolling_corr_slice(ret, window, end_idx):
    # Return correlation matrix for a given window ending at end_idx (inclusive).
    if ret is None or ret.empty:
        return None
    if end_idx < 0 or end_idx >= len(ret):
        return None
    start_idx = max(0, end_idx - window + 1)
    window_df = ret.iloc[start_idx:end_idx+1]
    if window_df.shape[0] < 2:
        return None
    return window_df.corr(), window_df.index[0], window_df.index[-1]

# -----------------------------
# Fetch and compute
# -----------------------------
prices_raw = fetch_prices(tickers, start_date, end_date, price_field)
prices = resample_prices(prices_raw, interval_label)
returns_df, corr_df = prepare_matrix_data(prices, returns_mode, missing_handling)

# -----------------------------
# Top metrics
# -----------------------------
top = st.container()
with top:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Selected tickers", len(tickers))
    c2.metric("Observations", 0 if returns_df is None else returns_df.shape[0])
    c3.metric("Sampling", interval_label)
    c4.metric("Date range", f"{start_date} → {end_date}")

st.divider()

# -----------------------------
# Display
# -----------------------------
if returns_df is None:
    st.info("Select at least two tickers with available data in the chosen date range to compute a correlation matrix.")
else:
    if rolling_mode:
        st.subheader("Rolling-Window Correlation")
        n = returns_df.shape[0]
        if window_size >= n:
            st.warning(f"Window size ({window_size}) is too large for the available observations ({n}). Reduce the window or extend the date range.")
        else:
            # Slider over index positions to choose the window end
            end_pos = st.slider("Window end position", min_value=window_size-1, max_value=n-1, value=n-1, step=1)
            out = rolling_corr_slice(returns_df, window_size, end_pos)
            if not out:
                st.info("Not enough data within the selected window.")
            else:
                corr_w, win_start, win_end = out
                st.caption(f"Window: **{win_start.date()} → {win_end.date()}**  ·  Size: **{window_size} periods**")
                st.dataframe(corr_w.style.format("{:.2f}"), use_container_width=True, height=400)
                fig = go.Figure(data=go.Heatmap(
                    z=corr_w.values, x=corr_w.columns, y=corr_w.index, zmin=-1, zmax=1, colorbar=dict(title="ρ")
                ))
                fig.update_layout(height=700, margin=dict(l=50,r=50,b=50,t=40))
                st.plotly_chart(fig, use_container_width=True)
                corr_to_download = corr_w
    else:
        st.subheader("Correlation Matrix")
        st.dataframe(corr_df.style.format("{:.2f}"), use_container_width=True, height=400)
        fig = go.Figure(data=go.Heatmap(
            z=corr_df.values, x=corr_df.columns, y=corr_df.index, zmin=-1, zmax=1, colorbar=dict(title="ρ")
        ))
        fig.update_layout(height=700, margin=dict(l=50,r=50,b=50,t=40))
        st.plotly_chart(fig, use_container_width=True)
        corr_to_download = corr_df

    # Downloads
    st.download_button(
        "⬇️ Download correlation CSV",
        data=corr_to_download.to_csv(index=True).encode("utf-8"),
        file_name="correlation_matrix.csv",
        mime="text/csv"
    )
    st.download_button(
        "⬇️ Download returns CSV",
        data=returns_df.to_csv(index=True).encode("utf-8"),
        file_name="returns_data.csv",
        mime="text/csv"
    )

# Notes
with st.expander("Notes & Tips"):
    st.markdown('''
- **Sampling interval** controls resampling from daily data: weekly uses Friday close; monthly uses month-end close.
- Correlation of **returns** (percent or log) is generally more informative than correlation of price levels.
- The **rolling window** computes correlation on the last _N_ periods ending at a chosen index; move the slider to explore dynamics.
- You can also paste ETFs, crypto, or international tickers supported by Yahoo Finance (e.g., `RY.TO`, `^XAU`, `BTC-USD`).
''')
