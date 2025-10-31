import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import date, timedelta

st.set_page_config(page_title="Simple Stock Correlation", layout="wide")

st.title("📈 Simple Stock Correlation (Daily Close)")
st.caption("Enter tickers and a date range. We fetch **daily Close** prices, compute daily % returns, and display the correlation matrix.")

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


with st.sidebar:
    st.header("Inputs")
    tickers_raw = st.text_input("Tickers (comma-separated)", value="AAPL,MSFT,GOOGL,AMZN,NVDA")
    today = date.today()
    one_year_ago = today - timedelta(days=365)
    start_date = st.date_input("Start date", value=one_year_ago, max_value=today - timedelta(days=1))
    end_date = st.date_input("End date", value=today, min_value=start_date + timedelta(days=1))


# Basket selection
    st.subheader("🎒 Predefined basket")
    basket_name = st.selectbox("Choose a basket", options=list(PREDEFINED.keys()), index=0)
    basket_ticks = PREDEFINED.get(basket_name, [])
    st.write("Basket tickers:", ", ".join(basket_ticks) if basket_ticks else "—")
    use_basket = st.checkbox("Use basket tickers", value=False, help="If checked, the basket tickers are used (you can also append them).")
    append_basket = st.checkbox("Append basket to manual selection", value=True, help="If unchecked while 'Use basket' is checked, only the basket tickers are used.")

# Parse tickers
tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
# Resolve final ticker list
if use_basket and basket_ticks:
    if append_basket:
        tickers = sorted(set(manual).union(basket_ticks))
    else:
        tickers = list(dict.fromkeys(basket_ticks))  # preserve order & de-dupe
else:
    tickers = manual


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

if prices.empty or len(prices.columns) < 2:
    st.info("Enter at least two valid tickers with available data for the selected range.")
else:
    st.subheader("Daily Close Prices")
    st.dataframe(prices.tail(10), use_container_width=True)

    # Daily % returns and correlation
    returns = prices.pct_change().dropna(how="all")
    returns = returns.dropna(axis=1, how="all")
    if returns.shape[1] < 2:
        st.info("Not enough overlapping data to compute correlation. Try adjusting the date range or tickers.")
    else:
        corr = returns.corr()
        st.subheader("Correlation Matrix (Daily % Returns)")
        st.dataframe(corr.style.format("{:.2f}"), use_container_width=True, height=400)

        z = corr.values
        x = corr.columns
        y = corr.index
        
        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            zmin=-1,
            zmax=1,
            colorscale="rdylbu",          # vivid contrasting colors
            colorbar=dict(title="ρ"),
            hovertemplate='%{x} vs %{y}: <b>%{z:.2f}</b><extra></extra>',
        ))
        
        # Add text annotations (correlation values)
        for i in range(len(y)):
            for j in range(len(x)):
                fig.add_annotation(
                    x=x[j],
                    y=y[i],
                    text=f"{z[i][j]:.2f}",
                    showarrow=False,
                    font=dict(color="black", size=12, family="Arial") if abs(z[i][j]) < 0.7 else 
                         dict(color="white", size=12, family="Arial")  # high contrast
                )
        
        fig.update_layout(
            height=700,
            margin=dict(l=50,r=50,b=50,t=40),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        # Downloads
        st.download_button("⬇️ Download correlation CSV", data=corr.to_csv().encode("utf-8"),
                           file_name="correlation_matrix.csv", mime="text/csv")
        st.download_button("⬇️ Download prices CSV", data=prices.to_csv().encode("utf-8"),
                           file_name="daily_close_prices.csv", mime="text/csv")
