"""
A股 定投回测系统
Streamlit Web UI
支持多频率对比 + 一次性投入对比
"""

import sys
from pathlib import Path
from datetime import datetime, date

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent))

from data_fetcher import ASSET_TYPE_CONFIG, fetch_history, get_price_series
from backtest.dca import run_dca_backtest, run_lump_sum_backtest, freq_map

# ── 页面配置 ──
st.set_page_config(
    page_title="A股定投回测系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 侧边栏: 参数输入 ──
st.sidebar.title("📊 定投回测")

asset_type = st.sidebar.selectbox(
    "资产类型",
    options=list(ASSET_TYPE_CONFIG.keys()),
    format_func=lambda x: ASSET_TYPE_CONFIG[x]["label"],
)

asset_config = ASSET_TYPE_CONFIG[asset_type]
symbol = st.sidebar.text_input(
    "代码",
    placeholder=asset_config["search_hint"],
    help="输入资产代码, 如 000001, 510300, 110011",
).strip()

st.sidebar.markdown("### 定投参数")
dca_amount = st.sidebar.number_input(
    "每期定投金额 (元)", min_value=100, value=1000, step=100
)

# ── 多频率选择 ──
all_freqs = ["daily", "weekly", "biweekly", "monthly", "quarterly", "yearly"]
default_freqs = ["weekly", "monthly", "quarterly"]
dca_freqs = st.sidebar.multiselect(
    "定投频率 (可多选对比)",
    options=all_freqs,
    default=default_freqs,
    format_func=lambda x: freq_map.get(x, x),
    help="选择多个频率可对比不同定投策略的效果",
)

dca_day = st.sidebar.number_input(
    "每月/季执行日", min_value=1, max_value=28, value=1,
    help="仅对 每月/每季度/每年 生效",
)

st.sidebar.markdown("### 回测区间")
today = date.today()

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = col1.date_input(
        "开始定投",
        value=date(today.year - 5, 1, 1),
        min_value=date(1990, 1, 1),
        max_value=today,
    )
with col2:
    end_date = col2.date_input(
        "结束持有",
        value=today,
        min_value=date(1990, 1, 1),
        max_value=today,
    )

adjust = st.sidebar.selectbox(
    "复权方式",
    options={"qfq": "前复权", "hfq": "后复权", "": "不复权"},
    format_func=lambda x: {"qfq": "前复权", "hfq": "后复权", "": "不复权"}[x],
    index=0,
)

include_lump_sum = st.sidebar.checkbox("对比: 一次性投入", value=True,
    help="在回测区间首日一次性投入同等金额, 与定投对比")

run_btn = st.sidebar.button("🚀 开始回测", type="primary", use_container_width=True)


# ══════════════════════════════════════════
#  主区域
# ══════════════════════════════════════════

st.title("📈 A股定投回测系统")
st.markdown(
    "基于真实历史数据, 回测定投策略的历史收益表现。"
    "数据来源: 东方财富 / AKShare"
)

if not symbol:
    st.info(f"👈 请在侧边栏输入{asset_config['label']}代码后开始回测")
    st.stop()

# ── 数据获取 ──
with st.spinner(f"正在获取 {symbol} 历史数据..."):
    try:
        df = fetch_history(
            asset_type=asset_type,
            symbol=symbol,
            start_date=start_date.strftime("%Y%m%d"),
            end_date=end_date.strftime("%Y%m%d"),
            adjust=adjust,
        )
    except Exception as e:
        st.error(f"数据获取失败: {e}")
        st.stop()

if df.empty:
    st.warning(f"未获取到 {symbol} 的数据, 请检查代码是否正确")
    st.stop()

price_series = get_price_series(df)
if price_series is None or len(price_series) == 0:
    st.warning("无法从数据中提取价格序列")
    st.stop()

# ── 原始数据预览 ──
with st.expander("📋 原始数据预览", expanded=False):
    display_df = df.copy()
    if isinstance(display_df.index, pd.DatetimeIndex):
        display_df = display_df.reset_index()
    st.dataframe(display_df.tail(20), use_container_width=True)

# ── 价格走势图 ──
st.subheader("📉 历史价格走势")
fig_price = go.Figure()
fig_price.add_trace(go.Scatter(
    x=price_series.index, y=price_series.values,
    mode="lines", name="价格",
    line=dict(color="#1f77b4", width=2),
))
fig_price.update_layout(
    xaxis_title="日期", yaxis_title="价格",
    hovermode="x unified", margin=dict(l=10, r=10, t=10, b=10), height=380,
)
st.plotly_chart(fig_price, use_container_width=True)


# ══════════════════════════════════════════
#  回测执行 (多策略)
# ══════════════════════════════════════════

def _run_all_backtests(price_series, start_date, end_date, amount, freqs, day):
    """运行所有选定策略, 返回 {strategy_name: result}"""
    results = {}
    for f in freqs:
        r = run_dca_backtest(
            price_series=price_series,
            start_date=start_date,
            end_date=end_date,
            frequency=f,
            amount=float(amount),
            day=day,
        )
        if not r["records"].empty:
            results[r["strategy"]] = r
    return results


if run_btn:
    if not dca_freqs:
        st.warning("请至少选择一个定投频率")
        st.stop()

    with st.spinner("执行回测对比..."):
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        results = _run_all_backtests(price_series, start_str, end_str,
                                     dca_amount, dca_freqs, dca_day)

        # 一次性投入
        if include_lump_sum:
            total_dca_amount = 0
            if results:
                sample = next(iter(results.values()))
                total_dca_amount = sample["total_invested"]
            lump = run_lump_sum_backtest(price_series, start_str, end_str,
                                         total_dca_amount)
            if lump["total_invested"] > 0:
                results["一次性投入"] = lump

    if not results:
        st.warning("在所选区间内没有找到符合条件的定投日期, 请调整参数")
        st.stop()

    # ── 策略对比表 ──
    st.subheader("🎯 策略对比")

    comp_data = []
    for name, r in results.items():
        comp_data.append({
            "策略": name,
            "总投入": r["total_invested"],
            "终值": r["final_value"],
            "总收益率%": r["total_return_pct"],
            "年化收益率%": r["annualized_return_pct"],
            "定投次数": r["num_investments"],
        })
    comp_df = pd.DataFrame(comp_data)

    # 高亮最佳收益率
    def _highlight_best(val, col):
        if col in ("总收益率%", "年化收益率%"):
            best = comp_df[col].max()
            return "background-color: #2ca02c33" if val == best else ""
        return ""

    st.dataframe(
        comp_df.style.apply(lambda row: [
            _highlight_best(row["总收益率%"], "总收益率%"),
            _highlight_best(row["年化收益率%"], "年化收益率%"),
            "", "", "", "",
        ], axis=1).format({
            "总投入": "{:,.2f}",
            "终值": "{:,.2f}",
            "总收益率%": "{:+.2f}%",
            "年化收益率%": "{:+.2f}%",
        }),
        use_container_width=True,
        hide_index=True,
        column_config={
            "总投入": st.column_config.NumberColumn(format="%.2f"),
            "终值": st.column_config.NumberColumn(format="%.2f"),
            "总收益率%": st.column_config.NumberColumn(format="+%.2f%%"),
            "年化收益率%": st.column_config.NumberColumn(format="+%.2f%%"),
        },
    )

    # ── 策略市值对比图 ──
    st.subheader("📊 持仓市值对比")

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
              "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    fig_compare = go.Figure()
    for i, (name, r) in enumerate(results.items()):
        color = colors[i % len(colors)]
        width = 4 if name == "一次性投入" else 2
        dash = "dash" if name == "一次性投入" else "solid"
        fig_compare.add_trace(go.Scatter(
            x=r["portfolio_series"].index,
            y=r["portfolio_series"].values,
            mode="lines",
            name=name,
            line=dict(color=color, width=width, dash=dash),
        ))

    fig_compare.update_layout(
        xaxis_title="日期", yaxis_title="持仓市值 (元)",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=10, b=10), height=450,
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig_compare, use_container_width=True)

    # ── 累计投入对比图 ──
    st.subheader("💰 累计投入对比")
    fig_inv = go.Figure()
    for i, (name, r) in enumerate(results.items()):
        color = colors[i % len(colors)]
        fig_inv.add_trace(go.Scatter(
            x=r["invested_series"].index,
            y=r["invested_series"].values,
            mode="lines", name=name,
            line=dict(color=color, width=2, dash="dot"),
        ))
    fig_inv.update_layout(
        xaxis_title="日期", yaxis_title="累计投入 (元)",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=10, b=10), height=350,
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig_inv, use_container_width=True)

    # ── 收益率对比图 ──
    st.subheader("📈 收益率走势对比")
    fig_ret = go.Figure()
    for i, (name, r) in enumerate(results.items()):
        ret_series = ((r["portfolio_series"] - r["invested_series"])
                      / r["invested_series"] * 100)
        ret_series = ret_series.replace([float("inf"), -float("inf")], 0)
        color = colors[i % len(colors)]
        fig_ret.add_trace(go.Scatter(
            x=ret_series.index, y=ret_series.values,
            mode="lines", name=name,
            line=dict(color=color, width=2),
        ))
    fig_ret.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="盈亏线")
    fig_ret.update_layout(
        xaxis_title="日期", yaxis_title="收益率 (%)",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=10, b=10), height=350,
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig_ret, use_container_width=True)

    # ── 各策略详情 (可折叠) ──
    st.subheader("📋 各策略定投明细")
    tabs = st.tabs(list(results.keys()))
    for ti, (name, r) in enumerate(results.items()):
        with tabs[ti]:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("总投入", f"{r['total_invested']:,.2f}")
            m2.metric("终值", f"{r['final_value']:,.2f}")
            m3.metric("总收益率", f"{r['total_return_pct']:+.2f}%")
            m4.metric("年化收益率", f"{r['annualized_return_pct']:+.2f}%")

            if not r["records"].empty:
                rec = r["records"].copy()
                rec["日期"] = rec["日期"].dt.strftime("%Y-%m-%d")
                st.dataframe(rec, use_container_width=True, hide_index=True)

else:
    st.info("👈 在侧边栏设置好参数后, 点击「开始回测」按钮")

# ── 页脚 ──
st.markdown("---")
st.caption(
    "⚠️ 免责声明: 本工具仅供学习研究使用, 回测历史收益不代表未来表现, "
    "不构成任何投资建议。数据来源: AKShare (东方财富)"
)
