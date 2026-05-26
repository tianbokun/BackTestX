"""
A股 定投回测系统
Streamlit Web UI
支持多频率对比 + 一次性投入对比 + 网格搜索
"""

import sys
from pathlib import Path
from datetime import datetime, date

import torch

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent))

from data_fetcher import ASSET_TYPE_CONFIG, fetch_history, get_price_series, add_premium_rate, fetch_etf_realtime_premium, ensure_ohlc
from backtest.dca import run_dca_backtest, run_lump_sum_backtest, freq_map
from backtest.strategies import (
    STRATEGY_CATALOG, run_dropbuy_backtest,
    run_ma_adjust_dca, run_cost_average_dca,
    run_value_averaging, run_trend_dca,
    run_alipay_smart_dca,
)
from backtest.rl.trainer import (
    train_dqn, evaluate, run_bh_baseline,
    predict_signal, compute_signal_history,
    hyperparam_search,
)
from backtest.rl.dqn_agent import DQNAgent
from backtest.grid_search import (
    run_grid_search, save_result, list_saved_results, load_result,
)

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

@st.cache_data(ttl=300, show_spinner="正在获取数据...")
def _cached_fetch(symbol, asset_type, start_str, end_str, adjust):
    return fetch_history(
        asset_type=asset_type, symbol=symbol,
        start_date=start_str, end_date=end_str, adjust=adjust,
    )


def _fetch_data():
    if not symbol:
        st.info(f"👈 请在侧边栏输入{asset_config['label']}代码后开始")
        st.stop()
    df = None
    try:
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        df = _cached_fetch(symbol, asset_type, start_str, end_str, adjust)
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
#  定投回测模式
# ══════════════════════════════════════════

def _render_dca_backtest(price_series):
    st.sidebar.markdown("### 定投参数")
    dca_amount = st.sidebar.number_input(
        "日均定投金额 (元)", min_value=10, value=100, step=10,
        help="实际每期投入 = 日均金额 × 对应频率的交易日乘数",
    )

    all_freqs = ["daily", "weekly", "biweekly", "monthly", "quarterly", "yearly"]
    default_freqs = ["weekly", "monthly", "quarterly"]
    dca_freqs = st.sidebar.multiselect(
        "定投频率 (可多选对比)", options=all_freqs,
        default=default_freqs,
        format_func=lambda x: freq_map.get(x, x),
    )

    dca_day = st.sidebar.number_input(
        "每月/季执行日", min_value=1, max_value=28, value=1,
    )

    dca_max_total = st.sidebar.number_input(
        "总投资上限 (元)", min_value=0, value=0, step=10000,
        help="0 表示不设上限",
    )

    with st.sidebar.expander("💰 费率设置", expanded=False):
        dca_commission = st.number_input("佣金费率", min_value=0.0, value=0.00025, step=0.00005, format="%.5f",
                                         help="默认万2.5")
        dca_min_commission = st.number_input("最低佣金(元)", min_value=0.0, value=5.0, step=1.0,
                                             help="每笔交易最低佣金, 默认5元")
        dca_stamp_duty = st.number_input("印花税率", min_value=0.0, value=0.001, step=0.0001, format="%.4f",
                                         help="仅卖出时收取, 默认千1")

    include_lump_sum = st.sidebar.checkbox("对比: 一次性投入", value=True)

    run_btn = st.sidebar.button("🚀 开始回测", type="primary", width='stretch')

    # 价格走势
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
    st.subheader("📉 历史价格走势")
    st.plotly_chart(fig_price, width='stretch')

    def _run_all(amount, freqs, day, max_total, commission, min_comm, stamp):
        results = {}
        for f in freqs:
            r = run_dca_backtest(
                price_series=price_series,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                frequency=f, amount=float(amount), day=day,
                max_total=float(max_total),
                commission_rate=commission, min_commission=min_comm, stamp_duty=stamp,
            )
            if not r["records"].empty:
                results[r["strategy"]] = r
        return results

    if not run_btn:
        st.info("👈 在侧边栏设置好参数后, 点击「开始回测」按钮")
        return

    if not dca_freqs:
        st.warning("请至少选择一个定投频率")
        return

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    results = {}
    try:
        with st.spinner("执行回测对比..."):
            results = _run_all(dca_amount, dca_freqs, dca_day, dca_max_total,
                               dca_commission, dca_min_commission, dca_stamp_duty)

            if include_lump_sum:
                if dca_max_total > 0:
                    lump_amount = float(dca_max_total)
                elif results:
                    lump_amount = next(iter(results.values()))["total_invested"]
                else:
                    lump_amount = 0
                if lump_amount > 0:
                    lump = run_lump_sum_backtest(price_series, start_str, end_str, lump_amount,
                                                 commission_rate=dca_commission,
                                                 min_commission=dca_min_commission,
                                                 stamp_duty=dca_stamp_duty)
                    if lump["total_invested"] > 0:
                        results["一次性投入"] = lump
    except Exception as e:
        st.error(f"回测执行失败: {e}")
        return

    if not results:
        st.warning("在所选区间内没有找到符合条件的定投日期, 请调整参数")
        return

    # 对比表
    st.subheader("🎯 策略对比")
    has_fees = "total_commissions" in next(iter(results.values()), {})
    comp_data = []
    for name, r in results.items():
        entry = {
            "策略": name, "总投入": r["total_invested"],
            "终值": r["final_value"], "总收益率%": r["total_return_pct"],
            "年化收益率%": r["annualized_return_pct"],
            "定投次数": r["num_investments"],
        }
        if has_fees:
            entry["交易费用"] = r.get("total_commissions", 0)
        comp_data.append(entry)
    comp_df = pd.DataFrame(comp_data)

    def _hlight(val, col):
        if col in ("总收益率%", "年化收益率%"):
            best = comp_df[col].max()
            return "background-color: #2ca02c33" if val == best else ""
        return ""

    cols_to_hlight = ["总收益率%", "年化收益率%"]
    blank_count = len(comp_df.columns) - len(cols_to_hlight)
    st.dataframe(
        comp_df.style.apply(lambda row: [
            _hlight(row["总收益率%"], "总收益率%"),
            _hlight(row["年化收益率%"], "年化收益率%"),
            *([""] * blank_count),
        ], axis=1).format({
            "总投入": "{:,.2f}", "终值": "{:,.2f}",
            "总收益率%": "{:+.2f}%", "年化收益率%": "{:+.2f}%",
            "交易费用": "{:,.2f}",
        }),
        width='stretch', hide_index=True,
        column_config={
            "总投入": st.column_config.NumberColumn(format="%.2f"),
            "终值": st.column_config.NumberColumn(format="%.2f"),
            "总收益率%": st.column_config.NumberColumn(format="+%.2f%%"),
            "年化收益率%": st.column_config.NumberColumn(format="+%.2f%%"),
            "交易费用": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
              "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    # 市值对比图
    st.subheader("📊 持仓市值对比")
    fig_c = go.Figure()
    for i, (name, r) in enumerate(results.items()):
        c = colors[i % len(colors)]
        fig_c.add_trace(go.Scatter(
            x=r["portfolio_series"].index, y=r["portfolio_series"].values,
            mode="lines", name=name,
            line=dict(color=c, width=4 if name == "一次性投入" else 2,
                      dash="dash" if name == "一次性投入" else "solid"),
        ))
    fig_c.update_layout(
        xaxis_title="日期", yaxis_title="持仓市值 (元)",
        hovermode="x unified", margin=dict(l=10, r=10, t=10, b=10), height=450,
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig_c, width='stretch')

    # 累计投入对比
    st.subheader("💰 累计投入对比")
    fig_inv = go.Figure()
    for i, (name, r) in enumerate(results.items()):
        c = colors[i % len(colors)]
        fig_inv.add_trace(go.Scatter(
            x=r["invested_series"].index, y=r["invested_series"].values,
            mode="lines", name=name, line=dict(color=c, width=2, dash="dot"),
        ))
    fig_inv.update_layout(
        xaxis_title="日期", yaxis_title="累计投入 (元)",
        hovermode="x unified", margin=dict(l=10, r=10, t=10, b=10), height=350,
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig_inv, width='stretch')

    # 收益率走势
    st.subheader("📈 收益率走势对比")
    fig_ret = go.Figure()
    for i, (name, r) in enumerate(results.items()):
        ret_s = ((r["portfolio_series"] - r["invested_series"]) / r["invested_series"] * 100)
        ret_s = ret_s.replace([float("inf"), -float("inf")], 0)
        c = colors[i % len(colors)]
        fig_ret.add_trace(go.Scatter(
            x=ret_s.index, y=ret_s.values, mode="lines", name=name, line=dict(color=c, width=2),
        ))
    fig_ret.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="盈亏线")
    fig_ret.update_layout(
        xaxis_title="日期", yaxis_title="收益率 (%)",
        hovermode="x unified", margin=dict(l=10, r=10, t=10, b=10), height=350,
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig_ret, width='stretch')

    # 明细
    st.subheader("📋 各策略定投明细")
    tabs = st.tabs(list(results.keys()))
    for ti, (name, r) in enumerate(results.items()):
        with tabs[ti]:
            has_fee_cols = "total_commissions" in r
            cols = st.columns(5 if has_fee_cols else 4)
            cols[0].metric("总投入", f"{r['total_invested']:,.2f}")
            cols[1].metric("终值", f"{r['final_value']:,.2f}")
            cols[2].metric("总收益率", f"{r['total_return_pct']:+.2f}%")
            cols[3].metric("年化收益率", f"{r['annualized_return_pct']:+.2f}%")
            if has_fee_cols:
                cols[4].metric("交易费用", f"{r.get('total_commissions', 0):,.2f}")
            if not r["records"].empty:
                rec = r["records"].copy()
                if "日期" in rec.columns and hasattr(rec["日期"], "dt"):
                    rec["日期"] = rec["日期"].dt.strftime("%Y-%m-%d")
                st.dataframe(rec, width='stretch', hide_index=True)


# ══════════════════════════════════════════
#  网格搜索模式
# ══════════════════════════════════════════

def _render_grid_search(price_series):
    st.title("🎯 网格搜索 — 下跌加仓策略")
    st.markdown(
        "对**下跌加仓策略**（前一日跌幅超过 X% 时买入 Y 元）进行超参数网格搜索，"
        "使用 Walk-Forward 交叉验证评估。"
    )

    # 策略清单 (可折叠)
    with st.expander("📚 常见智能定投策略一览"):
        for i, s in enumerate(STRATEGY_CATALOG, 1):
            st.markdown(f"**{i}. [{s['category']}] {s['name']}**")
            st.markdown(f"{s['description']}")
            st.markdown(f"*来源: {s['source']}*")
            c1, c2 = st.columns(2)
            c1.markdown(f"✅ 优点: {s['pros']}")
            c2.markdown(f"⚠️ 缺点: {s['cons']}")
            if i < len(STRATEGY_CATALOG):
                st.divider()

    st.markdown("---")

    # ═══════════════ 策略全能对比 ═══════════════

    st.subheader("🥊 策略全能对比")
    st.markdown(
        "一次性对比所有定投频率 + 下跌加仓策略 + 一次性投入的收益率表现。"
    )

    compare_max_total = st.number_input(
        "总投资上限 (元, 0=不限)", min_value=0, value=0, step=50000,
        key="compare_max_total",
        help="所有策略共享此总投资上限; 0 表示不设限",
    )
    compare_X = st.number_input("下跌加仓: 跌幅阈值 X (%)", min_value=0.1, value=1.0, step=0.1,
                                key="compare_x")
    compare_Y = st.number_input("下跌加仓: 每期买入 Y (元)", min_value=100, value=5000, step=100,
                                key="compare_y")
    compare_dca_amount = st.number_input(
        "日均定投金额 (元)", min_value=10, value=100, step=10,
        key="compare_dca_amount",
        help="实际每期投入 = 日均金额 × 对应频率的交易日乘数",
    )

    with st.expander("💰 费率设置", expanded=False):
        gs_commission = st.number_input("佣金费率", min_value=0.0, value=0.00025, step=0.00005, format="%.5f", key="gs_commission")
        gs_min_commission = st.number_input("最低佣金(元)", min_value=0.0, value=5.0, step=1.0, key="gs_min_comm")
        gs_stamp_duty = st.number_input("印花税率", min_value=0.0, value=0.001, step=0.0001, format="%.4f", key="gs_stamp")

    run_compare = st.button("🏁 运行全能对比", type="primary", width='stretch',
                            key="run_compare")

    if run_compare:
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        mt = float(compare_max_total)
        com, mcom, sd = float(gs_commission), float(gs_min_commission), float(gs_stamp_duty)
        results = {}

        with st.spinner("正在运行所有策略对比..."):
            dca_amt = float(compare_dca_amount)

            # 1. DCA 各频率
            all_freqs = ["daily", "weekly", "biweekly", "monthly", "quarterly", "yearly"]
            for f in all_freqs:
                r = run_dca_backtest(
                    price_series=price_series,
                    start_date=start_str, end_date=end_str,
                    frequency=f, amount=dca_amt, day=1,
                    max_total=mt,
                    commission_rate=com, min_commission=mcom, stamp_duty=sd,
                )
                if not r["records"].empty:
                    results[r["strategy"]] = r

            # 2. 智能策略 (通过 _build_smart_result 内部已支持佣金)
            sr = run_ma_adjust_dca(price_series, start_date=start_str, end_date=end_str,
                                   amount=dca_amt*22, ma_period=250, max_total=mt)
            if sr["num_investments"] > 0:
                results[sr["strategy"]] = sr

            sr = run_cost_average_dca(price_series, start_date=start_str, end_date=end_str,
                                      amount=dca_amt*22, max_total=mt)
            if sr["num_investments"] > 0:
                results[sr["strategy"]] = sr

            sr = run_value_averaging(price_series, start_date=start_str, end_date=end_str,
                                     amount=dca_amt*22, max_total=mt)
            if sr["num_investments"] > 0:
                results[sr["strategy"]] = sr

            sr = run_trend_dca(price_series, start_date=start_str, end_date=end_str,
                               amount=dca_amt*22, max_total=mt)
            if sr["num_investments"] > 0:
                results[sr["strategy"]] = sr

            sr = run_alipay_smart_dca(price_series, start_date=start_str, end_date=end_str,
                                      amount=dca_amt*22, max_total=mt)
            if sr["num_investments"] > 0:
                results[sr["strategy"]] = sr

            # 3. 下跌加仓
            db_res = run_dropbuy_backtest(
                price_series, X=compare_X, Y=compare_Y,
                start_date=start_str, end_date=end_str,
                max_total=mt,
                commission_rate=com, min_commission=mcom, stamp_duty=sd,
            )
            if db_res.num_investments > 0:
                label = f"下跌加仓 (X={compare_X:.1f}%, Y={compare_Y:.0f})"
                results[label] = {
                    "total_invested": db_res.total_invested,
                    "final_value": db_res.final_value,
                    "total_return_pct": db_res.total_return_pct,
                    "annualized_return_pct": db_res.annualized_return_pct,
                    "num_investments": db_res.num_investments,
                    "records": db_res.records,
                    "portfolio_series": db_res.portfolio_series,
                    "invested_series": db_res.invested_series,
                    "nav_series": db_res.nav_series,
                    "strategy": label,
                    "total_commissions": db_res.total_commissions,
                    "sell_commission": db_res.sell_commission,
                    "stamp_duty_paid": db_res.stamp_duty_paid,
                }

            # 4. 一次性投入
            lump_amount = mt if mt > 0 else (
                next(iter(results.values()))["total_invested"] if results else 0
            )
            if lump_amount > 0:
                lump = run_lump_sum_backtest(price_series, start_str, end_str, lump_amount,
                                             commission_rate=com, min_commission=mcom, stamp_duty=sd)
                if lump["total_invested"] > 0:
                    results["一次性投入"] = lump

        if not results:
            st.warning("所有策略均无有效交易, 请调整参数")
        else:
            # ── 对比表 ──
            st.subheader("🎯 收益率对比")
            has_gs_fees = "total_commissions" in next(iter(results.values()), {})
            comp_data = []
            for name, r in results.items():
                entry = {
                    "策略": name, "总投入": r["total_invested"],
                    "终值": r["final_value"], "总收益率%": r["total_return_pct"],
                    "年化收益率%": r["annualized_return_pct"],
                    "定投次数": r["num_investments"],
                }
                if has_gs_fees:
                    entry["交易费用"] = r.get("total_commissions", 0)
                comp_data.append(entry)
            comp_df = pd.DataFrame(comp_data)

            def _hlight(val, col):
                if col in ("总收益率%", "年化收益率%"):
                    best = comp_df[col].max()
                    return "background-color: #2ca02c33" if val == best else ""
                return ""

            gs_cols_to_hlight = ["总收益率%", "年化收益率%"]
            gs_blank = len(comp_df.columns) - len(gs_cols_to_hlight)
            st.dataframe(
                comp_df.style.apply(lambda row: [
                    _hlight(row["总收益率%"], "总收益率%"),
                    _hlight(row["年化收益率%"], "年化收益率%"),
                    *([""] * gs_blank),
                ], axis=1).format({
                    "总投入": "{:,.2f}", "终值": "{:,.2f}",
                    "总收益率%": "{:+.2f}%", "年化收益率%": "{:+.2f}%",
                    "交易费用": "{:,.2f}",
                }),
                width='stretch', hide_index=True,
            )

            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                      "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

            # ── 市值对比 ──
            st.subheader("📊 持仓市值对比")
            fig_c = go.Figure()
            for i, (name, r) in enumerate(results.items()):
                c = colors[i % len(colors)]
                fig_c.add_trace(go.Scatter(
                    x=r["portfolio_series"].index, y=r["portfolio_series"].values,
                    mode="lines", name=name,
                    line=dict(color=c, width=4 if name == "一次性投入" else 2,
                              dash="dash" if name == "一次性投入" else "solid"),
                ))
            fig_c.update_layout(
                xaxis_title="日期", yaxis_title="持仓市值 (元)",
                hovermode="x unified", margin=dict(l=10, r=10, t=10, b=10), height=450,
                legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
            )
            st.plotly_chart(fig_c, width='stretch')

            # ── 累计投入对比 ──
            st.subheader("💰 累计投入对比")
            fig_inv = go.Figure()
            for i, (name, r) in enumerate(results.items()):
                c = colors[i % len(colors)]
                fig_inv.add_trace(go.Scatter(
                    x=r["invested_series"].index, y=r["invested_series"].values,
                    mode="lines", name=name, line=dict(color=c, width=2, dash="dot"),
                ))
            fig_inv.update_layout(
                xaxis_title="日期", yaxis_title="累计投入 (元)",
                hovermode="x unified", margin=dict(l=10, r=10, t=10, b=10), height=350,
                legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
            )
            st.plotly_chart(fig_inv, width='stretch')

            # ── 收益率走势 ──
            st.subheader("📈 收益率走势对比")
            fig_ret = go.Figure()
            for i, (name, r) in enumerate(results.items()):
                ret_s = ((r["portfolio_series"] - r["invested_series"]) / r["invested_series"] * 100)
                ret_s = ret_s.replace([float("inf"), -float("inf")], 0)
                c = colors[i % len(colors)]
                fig_ret.add_trace(go.Scatter(
                    x=ret_s.index, y=ret_s.values, mode="lines", name=name,
                    line=dict(color=c, width=2),
                ))
            fig_ret.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="盈亏线")
            fig_ret.update_layout(
                xaxis_title="日期", yaxis_title="收益率 (%)",
                hovermode="x unified", margin=dict(l=10, r=10, t=10, b=10), height=350,
                legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
            )
            st.plotly_chart(fig_ret, width='stretch')

            # ── 明细 ──
            st.subheader("📋 各策略交易明细")
            tabs = st.tabs(list(results.keys()))
            for ti, (name, r) in enumerate(results.items()):
                with tabs[ti]:
                    has_gs_fee = "total_commissions" in r
                    gs_cols = st.columns(5 if has_gs_fee else 4)
                    gs_cols[0].metric("总投入", f"{r['total_invested']:,.2f}")
                    gs_cols[1].metric("终值", f"{r['final_value']:,.2f}")
                    gs_cols[2].metric("总收益率", f"{r['total_return_pct']:+.2f}%")
                    gs_cols[3].metric("年化收益率", f"{r['annualized_return_pct']:+.2f}%")
                    if has_gs_fee:
                        gs_cols[4].metric("交易费用", f"{r.get('total_commissions', 0):,.2f}")
                    if not r["records"].empty:
                        rec = r["records"].copy()
                        if "日期" in rec.columns and hasattr(rec["日期"], "dt"):
                            rec["日期"] = rec["日期"].dt.strftime("%Y-%m-%d")
                        st.dataframe(rec, width='stretch', hide_index=True)

        st.markdown("---")

    # 选中的策略: 下跌加仓
    st.subheader("📐 下跌加仓策略")
    st.markdown(
        "**逻辑**: 每个交易日检查, 若当日涨跌幅 < -X% (即跌幅超过 X%), "
        "则在当日以收盘价买入 Y 元。\n\n"
        "**关键参数**: X = 跌幅阈值 (%), Y = 每期买入金额 (元)"
    )

    # 快速预览
    with st.expander("⚡ 快速试运行 (当前参数)", expanded=False):
        preview_X = st.number_input("预览: 跌幅阈值 X (%)", min_value=0.1, value=1.0, step=0.1,
                                    key="preview_x")
        preview_Y = st.number_input("预览: 每期买入 Y (元)", min_value=100, value=5000, step=100,
                                    key="preview_y")
        if st.button("运行预览", key="preview_btn"):
            with st.spinner("回测中..."):
                res = run_dropbuy_backtest(
                    price_series, X=preview_X, Y=preview_Y,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                )
            if res.num_investments == 0:
                st.warning(f"所选参数在区间内无触发, 请降低 X 值")
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("触发次数", res.num_investments)
                c2.metric("总投入", f"{res.total_invested:,.2f}")
                c3.metric("终值", f"{res.final_value:,.2f}")
                c4.metric("年化收益", f"{res.annualized_return_pct:+.2f}%")

    st.markdown("---")

    # 网格搜索参数
    st.subheader("🔬 网格搜索设置")

    gcol1, gcol2 = st.columns(2)
    with gcol1:
        X_min = st.number_input("X 最小值 (%)", min_value=0.1, value=0.5, step=0.1)
        X_max = st.number_input("X 最大值 (%)", min_value=0.1, value=3.0, step=0.1)
        X_step = st.number_input("X 步长 (%)", min_value=0.1, value=0.5, step=0.1)
    with gcol2:
        Y_min = st.number_input("Y 最小值 (元)", min_value=100, value=2000, step=100)
        Y_max = st.number_input("Y 最大值 (元)", min_value=100, value=10000, step=100)
        Y_step = st.number_input("Y 步长 (元)", min_value=100, value=2000, step=100)

    n_folds = st.slider("Walk-Forward 折数", min_value=2, max_value=8, value=3,
                        help="将总区间切为 n 段, 每段依次作为验证集, 前段作为训练集")

    gs_max_total = st.number_input(
        "总投资上限 (元, 0=不限)",
        min_value=0, value=0, step=50000,
        help="该策略在整个回测区间内的累计投入上限 (训练+验证共享此额度); 0 表示不设限",
    )

    run_gs = st.button("🚀 启动网格搜索", type="primary", width='stretch')

    # ═══════════════════════ 执行搜索 ═══════════════════════

    if run_gs:
        X_vals = []
        x = X_min
        while x <= X_max + 1e-9:
            X_vals.append(round(x, 2))
            x += X_step

        Y_vals = []
        y = Y_min
        while y <= Y_max + 1e-9:
            Y_vals.append(round(y, 0))
            y += Y_step

        if len(X_vals) * len(Y_vals) > 1000:
            st.error(f"组合过多 ({len(X_vals)}×{len(Y_vals)}={len(X_vals)*len(Y_vals)}), "
                     f"超过上限 1000, 请增大步长或缩小范围")
            st.stop()
        elif len(X_vals) * len(Y_vals) > 300:
            st.warning(f"组合数较大 ({len(X_vals)}×{len(Y_vals)}={len(X_vals)*len(Y_vals)}), "
                       f"预计耗时可能较长")

        total_runs = len(X_vals) * len(Y_vals) * n_folds
        st.info(f"共有 {len(X_vals)}×{len(Y_vals)} = {len(X_vals)*len(Y_vals)} 种参数组合, "
                f"共 {total_runs} 次回测, 预计 30~120 秒")

        status_text = st.empty()
        status_text.info("⏳ 正在运行网格搜索...")

        try:
            gs_result = run_grid_search(
                price_series, symbol=symbol,
                total_start=start_date.strftime("%Y-%m-%d"),
                total_end=end_date.strftime("%Y-%m-%d"),
                X_range=X_vals, Y_range=Y_vals,
                n_folds=n_folds,
                max_total=gs_max_total,
            )
            status_text.success("✅ 搜索完成!")
        except Exception as e:
            st.error(f"网格搜索失败: {e}")
            st.stop()

        # 保存结果
        saved_path = save_result(gs_result)

        # 📊 结果显示
        st.markdown("---")
        st.subheader("📊 搜索结果")

        # 全局最优
        st.success(
            f"**全局最优参数**: X = **{gs_result.best_params_overall['X']}%**, "
            f"Y = **{gs_result.best_params_overall['Y']:.0f} 元**, "
            f"各折验证集平均年化 = **{gs_result.avg_val_return:.2f}%**"
        )

        # 每折详情
        st.markdown("#### 每折最优参数")
        fold_rows = []
        for fold in gs_result.folds:
            dur = f"{fold.train_start[:7]} ~ {fold.train_end[:7]}" if fold.train_end > fold.train_start else "—"
            fold_rows.append({
                "折": fold.fold_id + 1,
                "训练集": dur,
                "验证集": f"{fold.val_start[:7]} ~ {fold.val_end[:7]}",
                "最优 X": fold.best_params.get("X", "—"),
                "最优 Y": f"{fold.best_params.get('Y', 0):.0f}",
                "验证年化": f"{fold.best_val_return:.2f}%",
            })
        st.dataframe(pd.DataFrame(fold_rows), hide_index=True, width='stretch')

        # 热力图: 各参数组合的平均验证年化
        st.markdown("#### 参数热力图 (各组合平均验证年化)")
        pivot = gs_result.trials_df.groupby(["X", "Y"])["val_annualized"].mean().reset_index()
        pivot_tbl = pivot.pivot_table(
            index="X", columns="Y", values="val_annualized", aggfunc="mean",
        )
        pivot_tbl = pivot_tbl.round(2)
        fig_heat = go.Figure(data=go.Heatmap(
            z=pivot_tbl.values,
            x=[f"{c:.0f}" for c in pivot_tbl.columns],
            y=[f"{i}%" for i in pivot_tbl.index],
            colorscale="RdYlGn",
            text=pivot_tbl.values,
            texttemplate="%{text:.1f}",
            hovertemplate="X=%{y}, Y=%{x}<br>年化=%{z:.2f}%<extra></extra>",
        ))
        fig_heat.update_layout(
            xaxis_title="Y (每期买入金额)",
            yaxis_title="X (跌幅阈值 %)",
            height=400,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_heat, width='stretch')

        # 全面结果表
        st.markdown("#### 全部网格搜索结果")
        display_cols = {
            "fold": "折", "X": "X(%)", "Y": "Y(元)",
            "train_num_trades": "训练触发次数", "train_annualized": "训练年化(%)",
            "val_num_trades": "验证触发次数", "val_annualized": "验证年化(%)",
        }
        display_df = gs_result.trials_df[list(display_cols.keys())].rename(columns=display_cols)
        st.dataframe(display_df, hide_index=True, width='stretch')

        st.caption(f"结果已保存至: `{saved_path}`")

    # ═══════════════════════ 历史结果 ═══════════════════════

    st.markdown("---")
    st.subheader("📁 历史搜索结果")

    saved_list = list_saved_results(symbol=symbol if symbol else None)
    if not saved_list:
        st.info("暂无历史搜索记录")
    else:
        names = []
        for item in saved_list:
            ts = item.get("timestamp", "?")
            bp = item.get("best_params_overall", {})
            av = item.get("avg_val_return", 0)
            names.append(f"{ts}  |  X={bp.get('X','?')}  Y={bp.get('Y','?')}  |  {av}%")
        sel_name = st.selectbox("选择历史记录查看", names, key="gs_history")
        if sel_name and st.button("加载选中记录"):
            idx = names.index(sel_name)
            item = saved_list[idx]
            result = load_result(item["dir"])
            st.json({
                "symbol": result.symbol,
                "区间": f"{result.total_start} ~ {result.total_end}",
                "折数": result.n_folds,
                "最优参数": result.best_params_overall,
                "平均验证年化": result.avg_val_return,
            })
            if not result.trials_df.empty:
                st.dataframe(result.trials_df, width='stretch')


# ══════════════════════════════════════════
#  强化学习模式
# ══════════════════════════════════════════

def _render_rl_training(df_full):
    st.title("🤖 DQN 强化学习训练系统")

    # 归一化列名
    rename_map = {"开盘": "开盘价", "收盘": "收盘价", "最高": "最高价", "最低": "最低价"}
    df = df_full.copy()
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # 场外基金/净值型资产只有收盘价, 用同一价格填充 OHLC 四列
    df = ensure_ohlc(df)

    # 对 ETF/LOF 添加溢价率列
    df = add_premium_rate(df, symbol, asset_type)
    has_premium = "溢价率" in df.columns

    # 侧边栏参数
    st.sidebar.markdown("### 🤖 强化学习参数")

    train_start = st.sidebar.date_input("训练集开始", value=df.index[0].date() if len(df) > 0 else date.today())
    train_end = st.sidebar.date_input("训练集结束", value=df.index[len(df)//3].date() if len(df) > 2 else date.today())
    val_start = st.sidebar.date_input("验证集开始", value=train_end)
    val_end = st.sidebar.date_input("验证集结束",
                                     value=df.index[2*len(df)//3].date() if len(df) > 2 else date.today())
    test_start = st.sidebar.date_input("测试集开始", value=val_end)
    test_end = st.sidebar.date_input("测试集结束", value=df.index[-1].date())

    system_version = st.sidebar.selectbox(
        "系统版本",
        options=["basic", "1.0", "2.0"],
        format_func=lambda x: {
            "basic": "基础版 (仅价格)",
            "1.0": "系统 1.0 (+技术指标)",
            "2.0": "系统 2.0 (+SVM+XGBoost)",
        }[x],
        index=1,
        help="basic=仅过去30日收盘价, 1.0=加入技术指标, 2.0=加入SVM/XGBoost涨跌信号",
    )

    with st.sidebar.expander("💰 费率设置", expanded=False):
        rl_commission = st.number_input("佣金费率", min_value=0.0, value=0.00025, step=0.00005, format="%.5f",
                                        key="rl_commission")
        rl_min_commission = st.number_input("最低佣金(元)", min_value=0.0, value=5.0, step=1.0,
                                            key="rl_min_comm")
        rl_stamp_duty = st.number_input("印花税率", min_value=0.0, value=0.001, step=0.0001, format="%.4f",
                                        key="rl_stamp")

    with st.sidebar.expander("⚙️ DQN 超参数", expanded=False):
        n_episodes = st.number_input("训练轮数", min_value=10, value=64, step=10)
        batch_size = st.number_input("Batch 大小", min_value=32, value=200, step=32)
        lr = st.text_input("学习率", value="1e-5")
        gamma = st.text_input("折扣因子 γ", value="0.98")
        hidden = st.number_input("隐藏层维度", min_value=32, value=128, step=32)
        epsilon_start = st.text_input("ε 初始值", value="0.9")
        epsilon_end = st.text_input("ε 终值", value="0.01")
        epsilon_decay = st.number_input("ε 衰减步数", min_value=100, value=500, step=100)
        target_update = st.number_input("目标网络更新间隔", min_value=10, value=50, step=10)
        buffer_capacity = st.number_input("经验回放容量", min_value=1000, value=10000, step=1000)

    search_btn = st.sidebar.button("🔍 超参搜索", width='stretch')
    run_btn = st.sidebar.button("🚀 开始训练", type="primary", width='stretch')

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📂 已保存模型")
    model_dir = Path("saved_models/rl")
    model_files = sorted(model_dir.glob("*.pt"), reverse=True) if model_dir.exists() else []
    if model_files:
        names = [m.stem for m in model_files]
        selected_name = st.sidebar.selectbox("选择模型", names, key="rl_model_selector")
        col_s1, col_s2 = st.sidebar.columns(2)
        if col_s1.button("📥 加载", width='stretch', key="rl_load_btn"):
            selected_path = str(model_dir / f"{selected_name}.pt")
            loaded = DQNAgent.load(selected_path)
            st.session_state.rl_agent = loaded
            meta = torch.load(selected_path, map_location="cpu", weights_only=False).get("metadata", {})
            st.session_state.rl_model_info = {"path": selected_path, "name": selected_name, **meta}
            st.rerun()
        if col_s2.button("🗑 删除", width='stretch', key="rl_del_btn"):
            (model_dir / f"{selected_name}.pt").unlink()
            if st.session_state.rl_model_info and st.session_state.rl_model_info.get("name") == selected_name:
                st.session_state.rl_agent = None
                st.session_state.rl_model_info = None
            st.rerun()
    elif st.session_state.rl_model_just_saved:
        st.sidebar.success("✅ 模型已保存！刷新页面后显示在列表中")
    else:
        st.sidebar.caption("暂无已保存的模型")

    # ── 划分数据集 ──
    df_train = df[(df.index >= pd.Timestamp(train_start)) & (df.index <= pd.Timestamp(train_end))].copy()
    df_val = df[(df.index >= pd.Timestamp(val_start)) & (df.index <= pd.Timestamp(val_end))].copy()
    df_test = df[(df.index >= pd.Timestamp(test_start)) & (df.index <= pd.Timestamp(test_end))].copy()

    if len(df_train) < 50:
        st.error(f"训练数据不足 ({len(df_train)} 行)，请扩大训练集")
        st.stop()
    if len(df_test) < 20:
        st.error(f"测试数据不足 ({len(df_test)} 行)，请扩大测试集")
        st.stop()

    fee_params = dict(commission_rate=float(rl_commission),
                      min_commission=float(rl_min_commission),
                      stamp_duty=float(rl_stamp_duty))

    # ── 超参搜索 ──
    if search_btn:
        if len(df_val) < 20:
            st.error(f"验证集数据不足 ({len(df_val)} 行)，至少需要 20 行")
            st.stop()
        df_hp = pd.concat([df_train, df_val]).sort_index()
        total_days = len(df_hp)
        st.info(f"超参搜索窗口: {df_hp.index[0].date()} ~ {df_hp.index[-1].date()} ({total_days} 行)")

        hp_header = st.empty()
        hp_header.info(f"⏳ 超参搜索: 324 组合 × 3 折 = 972 次训练, 请耐心等待...")

        progress_bar = st.progress(0)
        fold_bar = st.progress(0)
        detail_box = st.empty()
        log_box = st.empty()

        from collections import deque
        import time
        _hp_start = time.time()
        _hp_recent = deque(maxlen=5)

        def _hp_fold_callback(ci, total, fi, nf, params, fold_sharpe):
            elapsed = time.time() - _hp_start
            total_folds = total * nf
            folds_done = ci * nf + fi + 1
            pct = folds_done / total_folds * 100

            progress_bar.progress(folds_done / total_folds)
            fold_bar.progress((fi + 1) / nf)

            avg_time = elapsed / folds_done
            remaining = avg_time * (total_folds - folds_done)
            remaining_str = f"{remaining/60:.0f} 分" if remaining < 3600 else f"{remaining/3600:.1f} 时"

            p_str = ", ".join(f"{k}={v}" for k, v in sorted(params.items()))
            sign = "+" if fold_sharpe >= 0 else ""
            detail_box.code(
                f"组合 {ci+1}/{total} | 折 {fi+1}/{nf}\n"
                f"参数: {p_str}\n"
                f"本折夏普: {sign}{fold_sharpe:.4f}\n"
                f"已用: {elapsed/60:.1f} 分 | 预计剩余: ~{remaining_str}\n"
                f"总体: {folds_done}/{total_folds} 训练 ({pct:.2f}%)"
            )

        def _hp_combo_callback(ci, total, best_params, best_score):
            nf = 3
            folds_done = (ci + 1) * nf
            total_folds = total * nf
            pct = folds_done / total_folds * 100

            fold_bar.progress(0)
            elapsed = time.time() - _hp_start

            hp_header.info(
                f"⏳ 搜索中: {folds_done}/{total_folds} ({pct:.2f}%) | "
                f"已用 {elapsed/60:.1f} 分 | "
                f"当前最优: 夏普={best_score:.4f}"
            )

            if best_params and best_score > -999:
                _hp_recent.append((ci, best_score))

            # 最近 5 个 combo 的得分日志
            log_lines = ["最近 5 个最优得分 (更新时):"]
            for idx, sc in _hp_recent:
                log_lines.append(f"  #{idx+1:>3d}  夏普={sc:+.4f}")
            log_lines.append(f"  🏆 当前最优: 夏普={best_score:.4f}")
            log_box.code("\n".join(log_lines))

        try:
            hp_result = hyperparam_search(
                df_hp, system_version=system_version,
                progress_callback=None,
                combo_callback=_hp_combo_callback,
                fold_callback=_hp_fold_callback,
                **fee_params,
            )
        except Exception as e:
            st.error(f"超参搜索失败: {e}")
            st.stop()

        progress_bar.empty()
        fold_bar.empty()
        detail_box.empty()
        log_box.empty()
        hp_header.empty()

        bp = hp_result.get("best_params")
        bs = hp_result.get("best_score", -999)

        if bp is None:
            st.error(hp_result.get("error", "搜索失败"))
            st.stop()

        elapsed_total = hp_result.get("elapsed_sec", 0)
        st.success(f"✅ 搜索完成! 耗时 {elapsed_total/60:.1f} 分 | "
                   f"最优: lr={bp['lr']}, gamma={bp['gamma']}, "
                   f"hidden={bp['hidden']}, n_episodes={bp['n_episodes']}, "
                   f"epsilon_decay={bp['epsilon_decay']}  |  验证夏普={bs:.4f}")

        # 用最优参数在训练集上训练, 在验证集上评估
        st.markdown("### 📊 验证集回测结果 (最优参数)")
        with st.spinner("正在训练/回测..."):
            best_agent, _ = train_dqn(
                df_train, system_version=system_version,
                n_episodes=bp["n_episodes"], lr=bp["lr"], gamma=bp["gamma"],
                hidden=bp["hidden"], epsilon_decay=bp["epsilon_decay"],
                progress_callback=None, **fee_params,
            )
            val_result = evaluate(best_agent, df_val, system_version=system_version, **fee_params)
            bh_val = run_bh_baseline(df_val)

        comp_val = pd.DataFrame([
            {"策略": "DQN", "最终金额": val_result["final_value"],
             "收益率%": val_result["total_return_pct"], "夏普比率": val_result["sharpe_ratio"],
             "最大回撤%": val_result["max_drawdown_pct"], "交易次数": val_result["num_trades"]},
            {"策略": "买入持有(BH)", "最终金额": bh_val["final_value"],
             "收益率%": bh_val["total_return_pct"], "夏普比率": bh_val["sharpe_ratio"],
             "最大回撤%": bh_val["max_drawdown_pct"], "交易次数": "-"},
        ])
        st.dataframe(comp_val, width='stretch', hide_index=True)

        if not val_result["trades"].empty:
            with st.expander("📝 交易记录"):
                td = val_result["trades"].copy()
                td["日期"] = td["日期"].dt.strftime("%Y-%m-%d")
                st.dataframe(td, width='stretch', hide_index=True)

        st.session_state.rl_hp_agent = best_agent
        st.session_state.rl_hp_params = bp
        st.session_state.rl_hp_score = bs

    # ── 训练 ──
    if not run_btn and not search_btn:
        st.info("👈 在侧边栏设置好参数后，点击「开始训练」或「超参搜索」")
        _render_rl_signal(df)
        st.stop()

    if run_btn:
        st.markdown("---")
        st.subheader("📊 训练与测试")

        # 训练进度
        progress_bar = st.progress(0)
        loss_chart = st.empty()
        loss_data = []

        def _progress(ep, total, loss):
            progress_bar.progress((ep + 1) / total)
            if loss > 0:
                loss_data.append((ep, loss))
            if len(loss_data) > 1:
                import plotly.graph_objects as go
                ldf = pd.DataFrame(loss_data, columns=["ep", "loss"])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ldf["ep"], y=ldf["loss"], mode="lines", name="Loss"))
                fig.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10),
                                  xaxis_title="Step", yaxis_title="Loss")
                loss_chart.plotly_chart(fig, width='stretch')

        try:
            params = {
                "n_episodes": int(n_episodes),
                "batch_size": int(batch_size),
                "lr": float(lr),
                "gamma": float(gamma),
                "hidden": int(hidden),
                "epsilon_start": float(epsilon_start),
                "epsilon_end": float(epsilon_end),
                "epsilon_decay": int(epsilon_decay),
                "target_update": int(target_update),
                "buffer_capacity": int(buffer_capacity),
            }
        except ValueError:
            st.error("超参数格式错误，请检查数字格式")
            st.stop()

        with st.spinner("正在训练 DQN 智能体..."):
            agent, _ = train_dqn(
                df_train, system_version=system_version,
                progress_callback=_progress,
                **params, **fee_params,
            )

        st.success("✅ 训练完成！")

        # 测试集评估
        st.markdown("---")
        st.subheader("📊 测试集回测结果")

        with st.spinner("正在回测..."):
            result_dqn = evaluate(agent, df_test, system_version=system_version, **fee_params)
            result_bh = run_bh_baseline(df_test)

        # 指标对比表
        st.markdown("#### 📋 策略指标对比")
        comp = pd.DataFrame([
            {"策略": "DQN", "最终金额": result_dqn["final_value"],
             "收益率%": result_dqn["total_return_pct"],
             "夏普比率": result_dqn["sharpe_ratio"],
             "最大回撤%": result_dqn["max_drawdown_pct"],
             "交易次数": result_dqn["num_trades"]},
            {"策略": "买入持有(BH)", "最终金额": result_bh["final_value"],
             "收益率%": result_bh["total_return_pct"],
             "夏普比率": result_bh["sharpe_ratio"],
             "最大回撤%": result_bh["max_drawdown_pct"],
             "交易次数": "-"},
        ])
        st.dataframe(comp, width='stretch', hide_index=True)

        # 累计利润图
        st.markdown("#### 📈 累计利润对比")
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_test.index, y=result_dqn["equity_curve"],
            mode="lines", name=f"DQN ({system_version})",
            line=dict(color="#1f77b4", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=df_test.index, y=result_bh["equity_curve"],
            mode="lines", name="买入持有(BH)",
            line=dict(color="#ff7f0e", width=2, dash="dash"),
        ))
        fig.add_hline(y=1.0, line_dash="dot", line_color="gray", annotation_text="初始本金")
        fig.update_layout(
            xaxis_title="日期", yaxis_title="账户总值 (初始=1.0)",
            hovermode="x unified", height=400,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig, width='stretch')

        # 交易记录
        if not result_dqn["trades"].empty:
            st.markdown("#### 📝 交易记录")
            trades_df = result_dqn["trades"].copy()
            trades_df["日期"] = trades_df["日期"].dt.strftime("%Y-%m-%d")
            st.dataframe(trades_df, width='stretch', hide_index=True)

        # 保存模型
        save_col1, save_col2 = st.columns([1, 5])
        with save_col1:
            if st.button("💾 保存模型", type="primary"):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = Path(f"saved_models/rl/{symbol}_{system_version}_{ts}.pt")
                save_path.parent.mkdir(parents=True, exist_ok=True)
                agent.save(str(save_path), {
                    "symbol": symbol,
                    "system_version": system_version,
                    "train_start": str(train_start),
                    "train_end": str(train_end),
                    "test_return": result_dqn["total_return_pct"],
                    "sharpe": result_dqn["sharpe_ratio"],
                })
                st.session_state.rl_agent = agent
                meta = {"name": save_path.stem, "path": str(save_path),
                        "symbol": symbol, "system_version": system_version}
                st.session_state.rl_model_info = meta
                st.session_state.rl_model_just_saved = True
                st.success(f"模型已保存: `{save_path.name}`")
        with save_col2:
            st.caption("保存当前训练的 DQN 模型到磁盘，之后可在侧边栏加载使用")

        # 实时信号面板（有加载模型时显示）
        _render_rl_signal(df)

# ══════════════════════════════════════════
#  实时信号面板
# ══════════════════════════════════════════

def _render_rl_signal(df_full):
    if st.session_state.rl_agent is None:
        st.info("💡 训练完成后保存模型，或在侧边栏「已保存模型」中加载一个已有模型，即可查看实时信号")
        return

    st.markdown("---")
    st.subheader("📡 实时交易信号")

    info = st.session_state.rl_model_info or {}
    meta_cols = st.columns(4)
    with meta_cols[0]:
        st.metric("加载模型", info.get("name", "未知")[:30])
    with meta_cols[1]:
        st.metric("资产代码", info.get("symbol", symbol))
    with meta_cols[2]:
        ver = info.get("system_version", "1.0")
        st.metric("系统版本", {"basic": "基础版", "1.0": "系统1.0", "2.0": "系统2.0"}.get(ver, ver))
    with meta_cols[3]:
        st.metric("最新日期", str(df_full.index[-1].date()) if len(df_full) > 0 else "-")

    rename_map = {"开盘": "开盘价", "收盘": "收盘价", "最高": "最高价", "最低": "最低价"}
    df = df_full.copy()
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
    df = ensure_ohlc(df)

    agent = st.session_state.rl_agent
    ver = info.get("system_version", "1.0")

    with st.spinner("正在计算信号..."):
        sig = predict_signal(agent, df, system_version=ver)

    action_map = {-1: ("🔴 卖出", "#ef4444"), 0: ("⚪ 持有", "#6b7280"), 1: ("🟢 买入", "#22c55e")}
    label, color = action_map.get(sig, ("❓ 未知", "#888888"))
    st.markdown(
        f"<div style='text-align:center; padding:24px; background:{color}22; "
        f"border-radius:12px; border:2px solid {color}'>"
        f"<span style='font-size:48px; font-weight:bold; color:{color}'>{label}</span>"
        f"<br><span style='font-size:16px; color:{color}99'>"
        f"基于 {df.index[-1].date()} 日数据</span></div>",
        unsafe_allow_html=True,
    )

    # 显示当前溢价率 (仅 ETF/LOF)
    if "溢价率" in df.columns:
        latest_premium = float(df["溢价率"].iloc[-1])
        premium_color = "#22c55e" if abs(latest_premium) < 1 else "#ef4444"
        st.markdown(
            f"<div style='text-align:center; padding:12px; margin:8px 0; "
            f"border-radius:8px; border:1px solid {premium_color}'>"
            f"<span style='font-size:14px; color:#888'>当前溢价率</span><br>"
            f"<span style='font-size:32px; font-weight:bold; color:{premium_color}'>"
            f"{latest_premium:+.2f}%</span></div>",
            unsafe_allow_html=True,
        )
        # 实时折价率 (仅 ETF)
        if asset_type == "etf":
            rt_premium = fetch_etf_realtime_premium(symbol)
            if rt_premium != 0.0:
                st.caption(f"实时折价率: {rt_premium:+.2f}% (来自东方财富)")

    with st.expander("📈 近期信号历史", expanded=False):
        with st.spinner("正在回放信号..."):
            signals = compute_signal_history(agent, df, system_version=ver)
        sig_df = pd.DataFrame({
            "日期": df.index[-60:],
            "信号": [action_map.get(s, ("?", "#888"))[0] for s in signals[-60:]],
        })
        st.dataframe(sig_df, width='stretch', hide_index=True)

    with st.expander("📋 最新行情", expanded=False):
        tail = df[["开盘价", "收盘价", "最高价", "最低价"]].tail(10)
        st.dataframe(tail, width='stretch')

    if st.button("🔄 刷新信号", width='stretch'):
        st.rerun()


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
    _render_dca_backtest(price_series)
elif mode.startswith("🤖"):
    _render_rl_training(df)
else:
    _render_grid_search(price_series)

# ── 页脚 ──
st.markdown("---")
st.caption(
    "⚠️ 免责声明: 本工具仅供学习研究使用, 回测历史收益不代表未来表现, "
    "不构成任何投资建议。数据来源: AKShare (东方财富)"
)
