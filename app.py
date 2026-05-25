"""
A股 定投回测系统
Streamlit Web UI
支持多频率对比 + 一次性投入对比 + 网格搜索
"""

import sys
from pathlib import Path
from datetime import datetime, date

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent))

from data_fetcher import ASSET_TYPE_CONFIG, fetch_history, get_price_series
from backtest.dca import run_dca_backtest, run_lump_sum_backtest, freq_map
from backtest.strategies import (
    STRATEGY_CATALOG, run_dropbuy_backtest,
    run_ma_adjust_dca, run_cost_average_dca,
    run_value_averaging, run_trend_dca,
    run_alipay_smart_dca,
)
from backtest.grid_search import (
    run_grid_search, save_result, list_saved_results, load_result,
)

st.set_page_config(
    page_title="A股定投回测系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════
#  侧边栏导航 + 公共参数
# ══════════════════════════════════════════

mode = st.sidebar.radio("模式", ["📊 定投回测", "🎯 网格搜索"], index=0)

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
        with st.spinner(f"正在获取 {symbol} 历史数据..."):
            df = fetch_history(
                asset_type=asset_type, symbol=symbol,
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

    include_lump_sum = st.sidebar.checkbox("对比: 一次性投入", value=True)

    run_btn = st.sidebar.button("🚀 开始回测", type="primary", use_container_width=True)

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
    st.plotly_chart(fig_price, use_container_width=True)

    def _run_all(amount, freqs, day, max_total):
        results = {}
        for f in freqs:
            r = run_dca_backtest(
                price_series=price_series,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                frequency=f, amount=float(amount), day=day,
                max_total=float(max_total),
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
            results = _run_all(dca_amount, dca_freqs, dca_day, dca_max_total)

            if include_lump_sum:
                if dca_max_total > 0:
                    lump_amount = float(dca_max_total)
                elif results:
                    lump_amount = next(iter(results.values()))["total_invested"]
                else:
                    lump_amount = 0
                if lump_amount > 0:
                    lump = run_lump_sum_backtest(price_series, start_str, end_str, lump_amount)
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
    comp_data = []
    for name, r in results.items():
        comp_data.append({
            "策略": name, "总投入": r["total_invested"],
            "终值": r["final_value"], "总收益率%": r["total_return_pct"],
            "年化收益率%": r["annualized_return_pct"],
            "定投次数": r["num_investments"],
        })
    comp_df = pd.DataFrame(comp_data)

    def _hlight(val, col):
        if col in ("总收益率%", "年化收益率%"):
            best = comp_df[col].max()
            return "background-color: #2ca02c33" if val == best else ""
        return ""

    st.dataframe(
        comp_df.style.apply(lambda row: [
            _hlight(row["总收益率%"], "总收益率%"),
            _hlight(row["年化收益率%"], "年化收益率%"),
            "", "", "", "",
        ], axis=1).format({
            "总投入": "{:,.2f}", "终值": "{:,.2f}",
            "总收益率%": "{:+.2f}%", "年化收益率%": "{:+.2f}%",
        }),
        use_container_width=True, hide_index=True,
        column_config={
            "总投入": st.column_config.NumberColumn(format="%.2f"),
            "终值": st.column_config.NumberColumn(format="%.2f"),
            "总收益率%": st.column_config.NumberColumn(format="+%.2f%%"),
            "年化收益率%": st.column_config.NumberColumn(format="+%.2f%%"),
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
    st.plotly_chart(fig_c, use_container_width=True)

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
    st.plotly_chart(fig_inv, use_container_width=True)

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
    st.plotly_chart(fig_ret, use_container_width=True)

    # 明细
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

    run_compare = st.button("🏁 运行全能对比", type="primary", use_container_width=True,
                            key="run_compare")

    if run_compare:
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        mt = float(compare_max_total)
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
                )
                if not r["records"].empty:
                    results[r["strategy"]] = r

            # 2. 智能策略
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
                }

            # 4. 一次性投入
            lump_amount = mt if mt > 0 else (
                next(iter(results.values()))["total_invested"] if results else 0
            )
            if lump_amount > 0:
                lump = run_lump_sum_backtest(price_series, start_str, end_str, lump_amount)
                if lump["total_invested"] > 0:
                    results["一次性投入"] = lump

        if not results:
            st.warning("所有策略均无有效交易, 请调整参数")
        else:
            # ── 对比表 ──
            st.subheader("🎯 收益率对比")
            comp_data = []
            for name, r in results.items():
                comp_data.append({
                    "策略": name, "总投入": r["total_invested"],
                    "终值": r["final_value"], "总收益率%": r["total_return_pct"],
                    "年化收益率%": r["annualized_return_pct"],
                    "定投次数": r["num_investments"],
                })
            comp_df = pd.DataFrame(comp_data)

            def _hlight(val, col):
                if col in ("总收益率%", "年化收益率%"):
                    best = comp_df[col].max()
                    return "background-color: #2ca02c33" if val == best else ""
                return ""

            st.dataframe(
                comp_df.style.apply(lambda row: [
                    _hlight(row["总收益率%"], "总收益率%"),
                    _hlight(row["年化收益率%"], "年化收益率%"),
                    "", "", "", "",
                ], axis=1).format({
                    "总投入": "{:,.2f}", "终值": "{:,.2f}",
                    "总收益率%": "{:+.2f}%", "年化收益率%": "{:+.2f}%",
                }),
                use_container_width=True, hide_index=True,
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
            st.plotly_chart(fig_c, use_container_width=True)

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
            st.plotly_chart(fig_inv, use_container_width=True)

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
            st.plotly_chart(fig_ret, use_container_width=True)

            # ── 明细 ──
            st.subheader("📋 各策略交易明细")
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
                        if "日期" in rec.columns and hasattr(rec["日期"], "dt"):
                            rec["日期"] = rec["日期"].dt.strftime("%Y-%m-%d")
                        st.dataframe(rec, use_container_width=True, hide_index=True)

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

    run_gs = st.button("🚀 启动网格搜索", type="primary", use_container_width=True)

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
        st.dataframe(pd.DataFrame(fold_rows), hide_index=True, use_container_width=True)

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
        st.plotly_chart(fig_heat, use_container_width=True)

        # 全面结果表
        st.markdown("#### 全部网格搜索结果")
        display_cols = {
            "fold": "折", "X": "X(%)", "Y": "Y(元)",
            "train_num_trades": "训练触发次数", "train_annualized": "训练年化(%)",
            "val_num_trades": "验证触发次数", "val_annualized": "验证年化(%)",
        }
        display_df = gs_result.trials_df[list(display_cols.keys())].rename(columns=display_cols)
        st.dataframe(display_df, hide_index=True, use_container_width=True)

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
                st.dataframe(result.trials_df, use_container_width=True)


# ══════════════════════════════════════════
#  主路由
# ══════════════════════════════════════════

df, price_series = _fetch_data()

with st.expander("📋 原始数据预览", expanded=False):
    display_df = df.copy()
    if isinstance(display_df.index, pd.DatetimeIndex):
        display_df = display_df.reset_index()
    st.dataframe(display_df.tail(20), use_container_width=True)

if mode.startswith("📊"):
    _render_dca_backtest(price_series)
else:
    _render_grid_search(price_series)

# ── 页脚 ──
st.markdown("---")
st.caption(
    "⚠️ 免责声明: 本工具仅供学习研究使用, 回测历史收益不代表未来表现, "
    "不构成任何投资建议。数据来源: AKShare (东方财富)"
)
