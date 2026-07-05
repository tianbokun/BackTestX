import streamlit as st
from data_fetcher import fetch_history

COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


@st.cache_data(ttl=300, show_spinner="正在获取数据...")
def cached_fetch(symbol, asset_type, start_str, end_str, adjust, _refresh_key=0):
    return fetch_history(
        asset_type=asset_type, symbol=symbol,
        start_date=start_str, end_date=end_str, adjust=adjust,
    )
