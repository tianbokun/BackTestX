"""
A股 定投回测系统
Streamlit Web UI
支持多频率对比 + 一次性投入对比 + 网格搜索
"""

import sys
from pathlib import Path
from datetime import date

import streamlit as st
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from data_fetcher import ASSET_TYPE_CONFIG, get_price_series
from ui._helpers import cached_fetch
from ui.dca_backtest import render_dca_backtest
from ui.grid_search import render_grid_search
from ui.rl_training import render_rl_training

st.set_page_config(
    page_title="A股定投回测系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "rl_agent" not in st.session_state:
    st.session_state.rl_agent = None
if "rl_model_info" not in st.session_state:
    st.session_state.rl_model_info = None
if "rl_model_just_saved" not in st.session_state:
    st.session_state.rl_model_just_saved = False
if "rl_hp_agent" not in st.session_state:
    st.session_state.rl_hp_agent = None
if "rl_hp_params" not in st.session_state:
    st.session_state.rl_hp_params = None
if "rl_trained_agent" not in st.session_state:
    st.session_state.rl_trained_agent = None
if "rl_dqn_result" not in st.session_state:
    st.session_state.rl_dqn_result = None
if "rl_bh_result" not in st.session_state:
    st.session_state.rl_bh_result = None
if "rl_train_meta" not in st.session_state:
    st.session_state.rl_train_meta = None

# ══════════════════════════════════════════
#  侧边栏导航 + 公共参数
# ══════════════════════════════════════════

mode = st.sidebar.radio("模式", ["📊 定投回测", "🎯 网格搜索", "🤖 强化学习"], index=0)

asset_type = st.sidebar.selectbox(
    "资产类型",
    options=list(ASSET_TYPE_CONFIG.keys()),
    format_func=lambda x: ASSET_TYPE_CONFIG[x]["label"],
)
asset_config = ASSET_TYPE_CONFIG[asset_type]
symbol = st.sidebar.text_input(
    "代码", placeholder=asset_config["search_hint"],
    help="输入资产代码, 如 000001, 510300, 110011",
).strip()

today = date.today()
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = col1.date_input(
        "开始日期", value=date(today.year - 5, 1, 1),
        min_value=date(1990, 1, 1), max_value=today,
    )
with col2:
    end_date = col2.date_input(
        "结束日期", value=today,
        min_value=date(1990, 1, 1), max_value=today,
    )

adjust = st.sidebar.selectbox(
    "复权方式",
    options={"qfq": "前复权", "hfq": "后复权", "": "不复权"},
    format_func=lambda x: {"qfq": "前复权", "hfq": "后复权", "": "不复权"}[x],
    index=0,
)


# ══════════════════════════════════════════
#  数据获取 (公共)
# ══════════════════════════════════════════

def _fetch_data():
    if not symbol:
        st.info(f"👈 请在侧边栏输入{asset_config['label']}代码后开始")
        st.stop()
    df = None
    try:
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        df = cached_fetch(symbol, asset_type, start_str, end_str, adjust)
    except Exception as e:
        st.error(f"数据获取失败: {e}")
        st.stop()
    if df.empty:
        st.warning(f"未获取到 {symbol} 的数据, 请检查代码是否正确")
        st.stop()
    price = get_price_series(df)
    if price is None or len(price) == 0:
        st.warning("无法从数据中提取价格序列")
        st.stop()
    return df, price


# ══════════════════════════════════════════
#  主路由
# ══════════════════════════════════════════

df, price_series = _fetch_data()

with st.expander("📋 原始数据预览", expanded=False):
    display_df = df.copy()
    if isinstance(display_df.index, pd.DatetimeIndex):
        display_df = display_df.reset_index()
    st.dataframe(display_df.tail(20), width='stretch')

if mode.startswith("📊"):
    render_dca_backtest(price_series, start_date, end_date)
elif mode.startswith("🤖"):
    render_rl_training(df, end_date, symbol, asset_type, adjust)
else:
    render_grid_search(price_series, start_date, end_date, symbol)

# ── 页脚 ──
st.markdown("---")
st.caption(
    "⚠️ 免责声明: 本工具仅供学习研究使用, 回测历史收益不代表未来表现, "
    "不构成任何投资建议。数据来源: AKShare (东方财富)"
)
