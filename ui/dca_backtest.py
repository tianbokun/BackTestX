import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from backtest.dca import run_dca_backtest, run_lump_sum_backtest, freq_map
from ui._helpers import COLORS


def render_dca_backtest(price_series, start_date, end_date):
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

    # 市值对比图
    st.subheader("📊 持仓市值对比")
    fig_c = go.Figure()
    for i, (name, r) in enumerate(results.items()):
        c = COLORS[i % len(COLORS)]
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
        c = COLORS[i % len(COLORS)]
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
        c = COLORS[i % len(COLORS)]
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
