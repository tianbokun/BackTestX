"""
A股 定投回测系统
Streamlit Web UI
支持多频率对比 + 一次性投入对比 + 网格搜索
"""

import sys
import atexit
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
from ui.hierarchical_rl import render_hierarchical_rl
from ui.symbol_manager import render_symbol_manager
from ui.task_manager import render_task_manager
from ui.sentiment import render_sentiment_dashboard
from ui.intel_dca import render_intel_dca


# ══════════════════════════════════════════════════════════════
#  Graceful Shutdown Registration
# ══════════════════════════════════════════════════════════════

def _register_shutdown():
    """Register graceful shutdown handler."""
    atexit.register(lambda: TaskManager()._graceful_shutdown())


from backtest.rl.task_manager import TaskManager

_register_shutdown()


# ══════════════════════════════════════════════════════════════
#  Global CSS Injection
# ══════════════════════════════════════════════════════════════

def _inject_global_styles():
    st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }
    [data-testid^="stIcon"] {
        font-family: 'Material Symbols Rounded' !important;
    }

    /* ── Base ── */
    .stApp {
        background: linear-gradient(135deg, #f0f4ff 0%, #f8fafc 40%, #f0f4ff 100%);
    }
    .main > div {
        padding: 1rem 2rem 3rem;
        max-width: 1280px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }
    h1 { font-size: 1.75rem !important; color: #0f172a !important; }
    h2 { font-size: 1.35rem !important; color: #1e293b !important; }
    h3 { font-size: 1.1rem !important; color: #334155 !important; }
    p, .stMarkdown { color: #475569; line-height: 1.6; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: rgba(255,255,255,0.92) !important;
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(0,0,0,0.04);
    }
    section[data-testid="stSidebar"] > div {
        padding: 1.5rem 1rem;
    }
    section[data-testid="stSidebar"] .stMarkdown h3 {
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8 !important;
        margin-top: 1.5rem !important;
        padding-bottom: 0.25rem;
        border-bottom: 1px solid #e2e8f0;
    }
    section[data-testid="stSidebar"] .stSelectbox,
    section[data-testid="stSidebar"] .stTextInput,
    section[data-testid="stSidebar"] .stNumberInput,
    section[data-testid="stSidebar"] .stDateInput {
        margin-bottom: 0.5rem;
    }
    section[data-testid="stSidebar"] label {
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        color: #64748b !important;
    }
    section[data-testid="stSidebar"] hr {
        margin: 1.25rem 0;
        border-color: #e2e8f0;
    }

    /* ── Top Tab Navigation (Pill Tabs) ── */
    div[data-testid="stRadio"] {
        margin-bottom: 1.5rem;
    }
    div[data-testid="stRadio"] > div {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: wrap;
        gap: 4px;
        background: #eef2f6;
        border-radius: 14px;
        padding: 5px;
        width: fit-content;
        margin: 0 auto;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.04);
    }
    div[data-testid="stRadio"] label > div:first-child {
        display: none !important;
    }
    div[data-testid="stRadio"] label {
        padding: 10px 28px !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
        font-size: 15px !important;
        cursor: pointer !important;
        transition: all 0.25s ease !important;
        color: #64748b !important;
        background: transparent !important;
        border: none !important;
        margin: 0 !important;
        white-space: nowrap;
    }
    div[data-testid="stRadio"] label:hover {
        background: rgba(255,255,255,0.6) !important;
        color: #1e293b !important;
    }
    /* Active pill */
    div[data-testid="stRadio"] label:has(input:checked) {
        background: white !important;
        color: #2563eb !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
        font-weight: 600 !important;
    }
    div[data-testid="stRadio"] label > div:first-child ~ span {
        font-weight: 500 !important;
    }

    /* ── Buttons ── */
    div[data-testid="stButton"] button {
        border-radius: 10px !important;
        border: none !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 0.5rem 1.5rem !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
        transition: all 0.2s ease !important;
        letter-spacing: 0.01em;
    }
    div[data-testid="stButton"] button[kind="primary"] {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: white !important;
        box-shadow: 0 4px 14px rgba(37,99,235,0.3) !important;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(37,99,235,0.35) !important;
    }
    div[data-testid="stButton"] button[kind="secondary"] {
        background: white !important;
        color: #1e293b !important;
        border: 1px solid #e2e8f0 !important;
    }
    div[data-testid="stButton"] button[kind="secondary"]:hover {
        border-color: #2563eb !important;
        color: #2563eb !important;
    }

    /* ── Form Inputs ── */
    div[data-testid="stSelectbox"] > div,
    div[data-testid="stTextInput"] > div,
    div[data-testid="stNumberInput"] > div,
    div[data-testid="stDateInput"] > div {
        border-radius: 10px !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02) !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stSelectbox"] > div:focus-within,
    div[data-testid="stTextInput"] > div:focus-within,
    div[data-testid="stNumberInput"] > div:focus-within,
    div[data-testid="stDateInput"] > div:focus-within {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 3px rgba(37,99,235,0.12) !important;
    }

    /* ── Checkbox ── */
    div[data-testid="stCheckbox"] label {
        gap: 8px;
    }
    div[data-testid="stCheckbox"] label > div:first-child {
        border-radius: 6px !important;
        border: 1px solid #cbd5e1 !important;
        transition: all 0.15s ease;
    }

    /* ── Metrics ── */
    div[data-testid="stMetric"] {
        background: white;
        border-radius: 14px;
        padding: 1rem 1.25rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
        border: 1px solid rgba(255,255,255,0.6);
        transition: all 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        box-shadow: 0 4px 20px rgba(0,0,0,0.07);
        transform: translateY(-1px);
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        color: #0f172a !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        color: #94a3b8 !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 0.85rem !important;
    }

    /* ── Expander ── */
    div[data-testid="stExpander"] {
        border: none !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
        margin-bottom: 0.75rem;
        overflow: hidden;
        background: white;
    }
    div[data-testid="stExpander"] summary {
        padding: 0.75rem 1rem !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        color: #1e293b !important;
        background: white;
    }
    div[data-testid="stExpander"] summary:hover {
        background: #f8fafc !important;
    }
    div[data-testid="stExpander"] > div[data-testid="stExpanderContent"] {
        padding: 0 1rem 1rem !important;
        border-top: 1px solid #f1f5f9;
    }

    /* ── DataFrame ── */
    div[data-testid="stDataFrame"] {
        border-radius: 12px !important;
        overflow: hidden;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    }
    div[data-testid="stDataFrame"] table {
        font-size: 0.85rem !important;
    }
    div[data-testid="stDataFrame"] thead th {
        background: #f1f5f9 !important;
        color: #475569 !important;
        font-weight: 600 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        padding: 0.6rem 0.8rem !important;
    }
    div[data-testid="stDataFrame"] tbody tr:hover {
        background: #f8fafc !important;
    }

    /* ── Info / Success / Warning / Error boxes ── */
    .stAlert {
        border-radius: 12px !important;
        border: none !important;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
        padding: 0.75rem 1rem !important;
    }
    .stAlert > div:first-child {
        font-weight: 500;
    }

    /* ── Divider ── */
    hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent, #cbd5e1, transparent) !important;
        margin: 2rem 0 !important;
    }

    /* ── Plotly chart wrapper ── */
    .js-plotly-plot {
        border-radius: 12px;
        background: white;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
        padding: 0.5rem;
    }

    /* ── Footer ── */
    .footer-caption {
        text-align: center;
        font-size: 0.75rem;
        color: #94a3b8;
        padding: 1.5rem 0 0.5rem;
    }

    /* ── Hide Streamlit status widget (running-man + Stop) ── */
    [data-testid="stStatusWidget"] { display: none; }

    /* ── Spinner ── */
    .stSpinner {
        border-radius: 12px;
    }

    /* ── Progress bar ── */
    .stProgress > div {
        border-radius: 10px;
        overflow: hidden;
    }

    /* ── Responsive ── */
    @media (max-width: 768px) {
        .main > div { padding: 0.5rem 1rem 2rem; }
        div[data-testid="stRadio"] label { padding: 8px 16px !important; font-size: 13px !important; }
        div[data-testid="stMetric"] { padding: 0.75rem 1rem; }
        div[data-testid="stMetricValue"] { font-size: 1.35rem !important; }
        section[data-testid="stSidebar"] > div { padding: 1rem 0.75rem; }
    }
    @media (max-width: 480px) {
        div[data-testid="stRadio"] > div { justify-content: center; }
        div[data-testid="stRadio"] label { padding: 6px 12px !important; font-size: 12px !important; }
    }
    </style>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  Page Config + Styles
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="A股定投回测系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

_inject_global_styles()

# ── Session state init ──
for key in ("rl_agent", "rl_model_info", "rl_hp_agent", "rl_hp_params",
            "rl_trained_agent", "rl_dqn_result", "rl_bh_result", "rl_train_meta"):
    if key not in st.session_state:
        st.session_state[key] = None

for key in ("rl_model_just_saved",):
    if key not in st.session_state:
        st.session_state[key] = False


# ══════════════════════════════════════════════════════════════
#  顶部 Tab 导航 — Pill Tabs
# ══════════════════════════════════════════════════════════════

mode = st.radio(
    "_mode",
    ["📊 定投回测", "🎯 网格搜索", "🧮 智能定投", "🤖 强化学习", "🧠 分层RL", "📊 情绪数据", "📋 训练任务", "📋 代码管理"],
    horizontal=True,
    label_visibility="collapsed",
    key="mode_tab",
)


# ══════════════════════════════════════════════════════════════
#  侧边栏 — 公共参数 (按 Tab 条件显示)
# ══════════════════════════════════════════════════════════════

needs_sidebar_symbol = (mode.startswith("📊") and "情绪" not in mode) or mode.startswith("🎯") or mode.startswith("🧮")

if needs_sidebar_symbol:
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
            min_value=date(1970, 1, 1), max_value=today,
        )
    with col2:
        end_date = col2.date_input(
            "结束日期", value=today,
            min_value=date(1970, 1, 1), max_value=today,
        )
else:
    asset_type = "etf"
    symbol = ""
    today = date.today()
    start_date = date(today.year - 5, 1, 1)
    end_date = today

adjust = st.sidebar.selectbox(
    "复权方式",
    options={"qfq": "前复权", "hfq": "后复权", "": "不复权"},
    format_func=lambda x: {"qfq": "前复权", "hfq": "后复权", "": "不复权"}[x],
    index=0,
)


# ══════════════════════════════════════════════════════════════
#  数据获取 (仅 DCA / 网格搜索 需要)
# ══════════════════════════════════════════════════════════════

def _fetch_data():
    if not symbol:
        st.info(f"👈 请在侧边栏输入{ASSET_TYPE_CONFIG[asset_type]['label']}代码后开始")
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

if needs_sidebar_symbol:
    df, price_series = _fetch_data()

    with st.expander("📋 原始数据预览", expanded=False):
        display_df = df.copy()
        if isinstance(display_df.index, pd.DatetimeIndex):
            display_df = display_df.reset_index()
        st.dataframe(display_df.tail(20), width='stretch')


# ══════════════════════════════════════════════════════════════
#  主路由
# ══════════════════════════════════════════════════════════════

if mode == "📊 定投回测":
    render_dca_backtest(price_series, start_date, end_date)
elif mode.startswith("📊 情绪"):
    render_sentiment_dashboard()
elif mode.startswith("🎯"):
    render_grid_search(price_series, start_date, end_date, symbol)
elif mode.startswith("🧮"):
    render_intel_dca(price_series, start_date, end_date, symbol, asset_type)
elif mode.startswith("🤖"):
    render_rl_training(end_date, adjust)
elif mode.startswith("🧠"):
    render_hierarchical_rl(end_date, adjust)
elif mode == "📋 训练任务":
    render_task_manager()
elif mode.startswith("📋"):
    render_symbol_manager()


# ── 页脚 ──
st.markdown("---")
st.markdown(
    '<p class="footer-caption">'
    "⚠️ 本工具仅供学习研究使用，回测历史收益不代表未来表现，不构成任何投资建议。"
    "数据来源：AKShare / 东方财富（A股）</p>",
    unsafe_allow_html=True,
)
