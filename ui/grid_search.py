import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from backtest.dca import run_dca_backtest, run_lump_sum_backtest
from backtest.strategies import (
    STRATEGY_CATALOG, run_dropbuy_backtest,
    run_ma_adjust_dca, run_cost_average_dca,
    run_value_averaging, run_trend_dca,
    run_alipay_smart_dca,
)
from backtest.grid_search import (
    run_grid_search, save_result, list_saved_results, load_result,
)
from ui._helpers import COLORS
from utils.i18n import t, tt


def render_grid_search(price_series, start_date, end_date, symbol):
    st.title(t("grid.title"))
    st.markdown(t("grid.desc"))

    # 策略清单 (可折叠)
    with st.expander(t("grid.strategy_catalog")):
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

    st.subheader(t("grid.compare_all"))
    st.markdown(t("grid.compare_all.desc"))

    compare_max_total = st.number_input(
        t("grid.param.max_invest"), min_value=0, value=0, step=50000,
        key="compare_max_total",
        help=t("grid.param.max_invest.help"),
    )
    compare_X = st.number_input(t("grid.param.drop_threshold"), min_value=0.1, value=1.0, step=0.1,
                                key="compare_x")
    compare_Y = st.number_input(t("grid.param.drop_amount"), min_value=100, value=5000, step=100,
                                key="compare_y")
    compare_dca_amount = st.number_input(
        t("grid.param.daily_amount"), min_value=10, value=100, step=10,
        key="compare_dca_amount",
        help=t("grid.param.daily_amount.help"),
    )

    with st.expander(t("grid.fee.header"), expanded=False):
        gs_commission = st.number_input(t("dca.fee.commission"), min_value=0.0, value=0.00025, step=0.00005, format="%.5f", key="gs_commission")
        gs_min_commission = st.number_input(t("dca.fee.min_commission"), min_value=0.0, value=5.0, step=1.0, key="gs_min_comm")
        gs_stamp_duty = st.number_input(t("dca.fee.stamp"), min_value=0.0, value=0.001, step=0.0001, format="%.4f", key="gs_stamp")

    run_compare = st.button(t("grid.btn.run_all"), type="primary", width='stretch',
                            key="run_compare")

    if run_compare:
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        mt = float(compare_max_total)
        com, mcom, sd = float(gs_commission), float(gs_min_commission), float(gs_stamp_duty)
        results = {}

        with st.spinner(t("grid.spinner.compare")):
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
            st.warning(t("grid.warning.no_trades"))
        else:
            # ── 对比表 ──
            st.subheader(t("grid.title.return_compare"))
            has_gs_fees = "total_commissions" in next(iter(results.values()), {})

            col_strategy = t("dca.col.strategy")
            col_total_invest = t("dca.col.total_invest")
            col_final_value = t("dca.col.final_value")
            col_total_return = t("dca.col.total_return")
            col_annual_return = t("dca.col.annual_return")
            col_invest_count = t("dca.col.invest_count")
            col_trade_fee = t("dca.col.trade_fee")

            comp_data = []
            for name, r in results.items():
                entry = {
                    col_strategy: name, col_total_invest: r["total_invested"],
                    col_final_value: r["final_value"], col_total_return: r["total_return_pct"],
                    col_annual_return: r["annualized_return_pct"],
                    col_invest_count: r["num_investments"],
                }
                if has_gs_fees:
                    entry[col_trade_fee] = r.get("total_commissions", 0)
                comp_data.append(entry)
            comp_df = pd.DataFrame(comp_data)

            def _hlight(val, col):
                if col in (col_total_return, col_annual_return):
                    best = comp_df[col].max()
                    return "background-color: #2ca02c33" if val == best else ""
                return ""

            gs_cols_to_hlight = [col_total_return, col_annual_return]
            gs_blank = len(comp_df.columns) - len(gs_cols_to_hlight)
            st.dataframe(
                comp_df.style.apply(lambda row: [
                    _hlight(row[col_total_return], col_total_return),
                    _hlight(row[col_annual_return], col_annual_return),
                    *([""] * gs_blank),
                ], axis=1).format({
                    col_total_invest: "{:,.2f}", col_final_value: "{:,.2f}",
                    col_total_return: "{:+.2f}%", col_annual_return: "{:+.2f}%",
                    col_trade_fee: "{:,.2f}",
                }),
                width='stretch', hide_index=True,
            )

            # ── 市值对比 ──
            st.subheader(t("grid.title.position_compare"))
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

            # ── 累计投入对比 ──
            st.subheader(t("grid.title.invest_compare"))
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

            # ── 收益率走势 ──
            st.subheader(t("grid.title.return_trend"))
            fig_ret = go.Figure()
            for i, (name, r) in enumerate(results.items()):
                ret_s = ((r["portfolio_series"] - r["invested_series"]) / r["invested_series"] * 100)
                ret_s = ret_s.replace([float("inf"), -float("inf")], 0)
                c = COLORS[i % len(COLORS)]
                fig_ret.add_trace(go.Scatter(
                    x=ret_s.index, y=ret_s.values, mode="lines", name=name,
                    line=dict(color=c, width=2),
                ))
            fig_ret.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text=t("dca.chart.zero_line"))
            fig_ret.update_layout(
                xaxis_title=t("dca.axis.date"), yaxis_title=t("dca.axis.return"),
                hovermode="x unified", margin=dict(l=10, r=10, t=10, b=10), height=350,
                legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
            )
            st.plotly_chart(fig_ret, width='stretch')

            # ── 明细 ──
            st.subheader(t("grid.title.detail"))
            tabs = st.tabs(list(results.keys()))
            for ti, (name, r) in enumerate(results.items()):
                with tabs[ti]:
                    has_gs_fee = "total_commissions" in r
                    gs_cols = st.columns(5 if has_gs_fee else 4)
                    gs_cols[0].metric(t("dca.metric.total_invest"), f"{r['total_invested']:,.2f}")
                    gs_cols[1].metric(t("dca.metric.final_value"), f"{r['final_value']:,.2f}")
                    gs_cols[2].metric(t("dca.metric.total_return"), f"{r['total_return_pct']:+.2f}%")
                    gs_cols[3].metric(t("dca.metric.annual_return"), f"{r['annualized_return_pct']:+.2f}%")
                    if has_gs_fee:
                        gs_cols[4].metric(t("dca.metric.trade_fee"), f"{r.get('total_commissions', 0):,.2f}")
                    if not r["records"].empty:
                        rec = r["records"].copy()
                        if "日期" in rec.columns and hasattr(rec["日期"], "dt"):
                            rec["日期"] = rec["日期"].dt.strftime("%Y-%m-%d")
                        st.dataframe(rec, width='stretch', hide_index=True)

        st.markdown("---")

    # 选中的策略: 下跌加仓
    st.subheader(t("grid.title.drop_strategy"))
    st.markdown(t("grid.drop_strategy.desc"))

    # 快速预览
    with st.expander(t("grid.quick_run"), expanded=False):
        preview_X = st.number_input(t("grid.param.preview_x"), min_value=0.1, value=1.0, step=0.1,
                                    key="preview_x")
        preview_Y = st.number_input(t("grid.param.preview_y"), min_value=100, value=5000, step=100,
                                    key="preview_y")
        if st.button(t("grid.btn.preview"), key="preview_btn"):
            with st.spinner(t("grid.spinner.backtest")):
                res = run_dropbuy_backtest(
                    price_series, X=preview_X, Y=preview_Y,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                )
            if res.num_investments == 0:
                st.warning(t("grid.warning.no_trigger"))
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric(t("grid.metric.trigger_count"), res.num_investments)
                c2.metric(t("grid.metric.total_invest"), f"{res.total_invested:,.2f}")
                c3.metric(t("grid.metric.final_value"), f"{res.final_value:,.2f}")
                c4.metric(t("grid.metric.annual_return"), f"{res.annualized_return_pct:+.2f}%")

    st.markdown("---")

    # 网格搜索参数
    st.subheader(t("grid.search.title"))

    gcol1, gcol2 = st.columns(2)
    with gcol1:
        X_min = st.number_input(t("grid.param.x_min"), min_value=0.1, value=0.5, step=0.1)
        X_max = st.number_input(t("grid.param.x_max"), min_value=0.1, value=3.0, step=0.1)
        X_step = st.number_input(t("grid.param.x_step"), min_value=0.1, value=0.5, step=0.1)
    with gcol2:
        Y_min = st.number_input(t("grid.param.y_min"), min_value=100, value=2000, step=100)
        Y_max = st.number_input(t("grid.param.y_max"), min_value=100, value=10000, step=100)
        Y_step = st.number_input(t("grid.param.y_step"), min_value=100, value=2000, step=100)

    n_folds = st.slider(t("grid.param.wf_folds"), min_value=2, max_value=8, value=3,
                        help=t("grid.param.wf_folds.help"))

    gs_max_total = st.number_input(
        t("grid.param.max_invest_wf"),
        min_value=0, value=0, step=50000,
        help=t("grid.param.max_invest_wf.help"),
    )

    run_gs = st.button(t("grid.btn.start_search"), type="primary", width='stretch')

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

        n = len(X_vals)
        m = len(Y_vals)
        product = n * m

        if product > 1000:
            st.error(t("grid.error.too_many", n=n, m=m, product=product))
            st.stop()
        elif product > 300:
            st.warning(t("grid.warning.large_search", n=n, m=m, product=product))

        total_runs = product * n_folds
        st.info(t("grid.info.search_size", n=n, m=m, product=product, total=total_runs))

        status_text = st.empty()
        status_text.info(t("grid.info.searching"))

        try:
            gs_result = run_grid_search(
                price_series, symbol=symbol,
                total_start=start_date.strftime("%Y-%m-%d"),
                total_end=end_date.strftime("%Y-%m-%d"),
                X_range=X_vals, Y_range=Y_vals,
                n_folds=n_folds,
                max_total=gs_max_total,
            )
            status_text.success(t("grid.success.done"))
        except Exception as e:
            st.error(t("grid.error.search_failed", error=e))
            st.stop()

        # 保存结果
        saved_path = save_result(gs_result)

        # 📊 结果显示
        st.markdown("---")
        st.subheader(t("grid.result.title"))

        # 全局最优
        st.success(t("grid.result.best",
                     x=gs_result.best_params_overall['X'],
                     y=gs_result.best_params_overall['Y'],
                     r=gs_result.avg_val_return))

        # 每折详情
        st.markdown(t("grid.result.best_per_fold"))
        col_fold = t("grid.col.fold")
        col_train_set = t("grid.col.train_set")
        col_val_set = t("grid.col.val_set")
        col_best_x = t("grid.col.best_x")
        col_best_y = t("grid.col.best_y")
        col_val_return = t("grid.col.val_return")
        fold_rows = []
        for fold in gs_result.folds:
            dur = f"{fold.train_start[:7]} ~ {fold.train_end[:7]}" if fold.train_end > fold.train_start else "—"
            fold_rows.append({
                col_fold: fold.fold_id + 1,
                col_train_set: dur,
                col_val_set: f"{fold.val_start[:7]} ~ {fold.val_end[:7]}",
                col_best_x: fold.best_params.get("X", "—"),
                col_best_y: f"{fold.best_params.get('Y', 0):.0f}",
                col_val_return: f"{fold.best_val_return:.2f}%",
            })
        st.dataframe(pd.DataFrame(fold_rows), hide_index=True, width='stretch')

        # 热力图
        st.markdown(t("grid.result.heatmap"))
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
            xaxis_title=t("grid.heatmap.x_title"),
            yaxis_title=t("grid.heatmap.y_title"),
            height=400,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_heat, width='stretch')

        # 全面结果表
        st.markdown(t("grid.result.all"))
        display_cols = {
            "fold": col_fold,
            "X": t("grid.col.x"),
            "Y": t("grid.col.y"),
            "train_num_trades": t("grid.col.train_triggers"),
            "train_annualized": t("grid.col.train_return"),
            "val_num_trades": t("grid.col.val_triggers"),
            "val_annualized": t("grid.col.val_return_pct"),
        }
        display_df = gs_result.trials_df[list(display_cols.keys())].rename(columns=display_cols)
        st.dataframe(display_df, hide_index=True, width='stretch')

        st.caption(t("grid.result.saved", path=saved_path))

    # ═══════════════════════ 历史结果 ═══════════════════════

    st.markdown("---")
    st.subheader(t("grid.history.title"))

    saved_list = list_saved_results(symbol=symbol if symbol else None)
    if not saved_list:
        st.info(t("grid.history.empty"))
    else:
        names = []
        for item in saved_list:
            ts = item.get("timestamp", "?")
            bp = item.get("best_params_overall", {})
            av = item.get("avg_val_return", 0)
            names.append(f"{ts}  |  X={bp.get('X','?')}  Y={bp.get('Y','?')}  |  {av}%")
        sel_name = st.selectbox(t("grid.history.select"), names, key="gs_history")
        if sel_name and st.button(t("grid.history.load")):
            idx = names.index(sel_name)
            item = saved_list[idx]
            result = load_result(item["dir"])
            st.json({
                "symbol": result.symbol,
                tt("区间", "Range"): f"{result.total_start} ~ {result.total_end}",
                tt("折数", "Folds"): result.n_folds,
                tt("最优参数", "Best Params"): result.best_params_overall,
                tt("平均验证年化", "Avg Val Return"): result.avg_val_return,
            })
            if not result.trials_df.empty:
                st.dataframe(result.trials_df, width='stretch')
