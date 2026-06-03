import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils.i18n import t
from backtest.dca import run_dca_backtest, run_lump_sum_backtest
from ui._helpers import COLORS


def render_dca_backtest(price_series, start_date, end_date):
    st.sidebar.markdown(t("dca.sidebar.header"))
    dca_amount = st.sidebar.number_input(
        t("dca.param.daily_amount"), min_value=10, value=100, step=10,
        help=t("dca.param.daily_amount.help"),
    )

    all_freqs = ["daily", "weekly", "biweekly", "monthly", "quarterly", "yearly"]
    default_freqs = ["weekly", "monthly", "quarterly"]
    dca_freqs = st.sidebar.multiselect(
        t("dca.param.frequency"), options=all_freqs,
        default=default_freqs,
        format_func=lambda x: t("freq." + x),
    )

    dca_day = st.sidebar.number_input(
        t("dca.param.exec_day"), min_value=1, max_value=28, value=1,
    )

    dca_max_total = st.sidebar.number_input(
        t("dca.param.max_invest"), min_value=0, value=0, step=10000,
        help=t("dca.param.max_invest.help"),
    )

    with st.sidebar.expander(t("dca.fee.header"), expanded=False):
        dca_commission = st.number_input(t("dca.fee.commission"), min_value=0.0, value=0.00025, step=0.00005, format="%.5f",
                                         help=t("dca.fee.commission.help"))
        dca_min_commission = st.number_input(t("dca.fee.min_commission"), min_value=0.0, value=5.0, step=1.0,
                                             help=t("dca.fee.min_commission.help"))
        dca_stamp_duty = st.number_input(t("dca.fee.stamp"), min_value=0.0, value=0.001, step=0.0001, format="%.4f",
                                         help=t("dca.fee.stamp.help"))

    include_lump_sum = st.sidebar.checkbox(t("dca.compare.lump_sum"), value=True)

    run_btn = st.sidebar.button(t("dca.btn.run"), type="primary", width='stretch')

    # 价格走势
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=price_series.index, y=price_series.values,
        mode="lines", name=t("dca.trace.price"),
        line=dict(color="#1f77b4", width=2),
    ))
    fig_price.update_layout(
        xaxis_title=t("dca.axis.date"), yaxis_title=t("dca.axis.price"),
        hovermode="x unified", margin=dict(l=10, r=10, t=10, b=10), height=380,
    )
    st.subheader(t("dca.title.price"))
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
        st.info(t("dca.info.waiting"))
        return

    if not dca_freqs:
        st.warning(t("dca.warning.no_freq"))
        return

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    results = {}
    try:
        with st.spinner(t("dca.spinner.running")):
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
        st.error(t("dca.error.run_failed", error=e))
        return

    if not results:
        st.warning(t("dca.warning.no_dates"))
        return

    # 对比表
    st.subheader(t("dca.title.compare"))
    has_fees = "total_commissions" in next(iter(results.values()), {})
    _entry_strat = t("dca.col.strategy")
    _entry_inv = t("dca.col.total_invest")
    _entry_fv = t("dca.col.final_value")
    _entry_tr = t("dca.col.total_return")
    _entry_ar = t("dca.col.annual_return")
    _entry_ic = t("dca.col.invest_count")
    _entry_tf = t("dca.col.trade_fee")
    comp_data = []
    for name, r in results.items():
        entry = {
            _entry_strat: name, _entry_inv: r["total_invested"],
            _entry_fv: r["final_value"], _entry_tr: r["total_return_pct"],
            _entry_ar: r["annualized_return_pct"],
            _entry_ic: r["num_investments"],
        }
        if has_fees:
            entry[_entry_tf] = r.get("total_commissions", 0)
        comp_data.append(entry)
    comp_df = pd.DataFrame(comp_data)

    def _hlight(val, col):
        if col in (_entry_tr, _entry_ar):
            best = comp_df[col].max()
            return "background-color: #2ca02c33" if val == best else ""
        return ""

    cols_to_hlight = [_entry_tr, _entry_ar]
    blank_count = len(comp_df.columns) - len(cols_to_hlight)
    st.dataframe(
        comp_df.style.apply(lambda row: [
            _hlight(row[_entry_tr], _entry_tr),
            _hlight(row[_entry_ar], _entry_ar),
            *([""] * blank_count),
        ], axis=1).format({
            _entry_inv: "{:,.2f}", _entry_fv: "{:,.2f}",
            _entry_tr: "{:+.2f}%", _entry_ar: "{:+.2f}%",
            _entry_tf: "{:,.2f}",
        }),
        width='stretch', hide_index=True,
        column_config={
            _entry_inv: st.column_config.NumberColumn(format="%.2f"),
            _entry_fv: st.column_config.NumberColumn(format="%.2f"),
            _entry_tr: st.column_config.NumberColumn(format="+%.2f%%"),
            _entry_ar: st.column_config.NumberColumn(format="+%.2f%%"),
            _entry_tf: st.column_config.NumberColumn(format="%.2f"),
        },
    )

    # 市值对比图
    st.subheader(t("dca.title.position"))
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
        xaxis_title=t("dca.axis.date"), yaxis_title=t("dca.axis.position"),
        hovermode="x unified", margin=dict(l=10, r=10, t=10, b=10), height=450,
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig_c, width='stretch')

    # 累计投入对比
    st.subheader(t("dca.title.invest"))
    fig_inv = go.Figure()
    for i, (name, r) in enumerate(results.items()):
        c = COLORS[i % len(COLORS)]
        fig_inv.add_trace(go.Scatter(
            x=r["invested_series"].index, y=r["invested_series"].values,
            mode="lines", name=name, line=dict(color=c, width=2, dash="dot"),
        ))
    fig_inv.update_layout(
        xaxis_title=t("dca.axis.date"), yaxis_title=t("dca.axis.invest"),
        hovermode="x unified", margin=dict(l=10, r=10, t=10, b=10), height=350,
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig_inv, width='stretch')

    # 收益率走势
    st.subheader(t("dca.title.return"))
    fig_ret = go.Figure()
    for i, (name, r) in enumerate(results.items()):
        ret_s = ((r["portfolio_series"] - r["invested_series"]) / r["invested_series"] * 100)
        ret_s = ret_s.replace([float("inf"), -float("inf")], 0)
        c = COLORS[i % len(COLORS)]
        fig_ret.add_trace(go.Scatter(
            x=ret_s.index, y=ret_s.values, mode="lines", name=name, line=dict(color=c, width=2),
        ))
    fig_ret.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text=t("dca.chart.zero_line"))
    fig_ret.update_layout(
        xaxis_title=t("dca.axis.date"), yaxis_title=t("dca.axis.return"),
        hovermode="x unified", margin=dict(l=10, r=10, t=10, b=10), height=350,
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig_ret, width='stretch')

    # 明细
    st.subheader(t("dca.title.detail"))
    tabs = st.tabs(list(results.keys()))
    for ti, (name, r) in enumerate(results.items()):
        with tabs[ti]:
            has_fee_cols = "total_commissions" in r
            cols = st.columns(5 if has_fee_cols else 4)
            cols[0].metric(t("dca.metric.total_invest"), f"{r['total_invested']:,.2f}")
            cols[1].metric(t("dca.metric.final_value"), f"{r['final_value']:,.2f}")
            cols[2].metric(t("dca.metric.total_return"), f"{r['total_return_pct']:+.2f}%")
            cols[3].metric(t("dca.metric.annual_return"), f"{r['annualized_return_pct']:+.2f}%")
            if has_fee_cols:
                cols[4].metric(t("dca.metric.trade_fee"), f"{r.get('total_commissions', 0):,.2f}")
            if not r["records"].empty:
                rec = r["records"].copy()
                _entry_date = t("dca.axis.date")
                if _entry_date in rec.columns and hasattr(rec[_entry_date], "dt"):
                    rec[_entry_date] = rec[_entry_date].dt.strftime("%Y-%m-%d")
                st.dataframe(rec, width='stretch', hide_index=True)
