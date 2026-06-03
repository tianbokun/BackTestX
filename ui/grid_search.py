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


def render_grid_search(price_series, start_date, end_date, symbol):
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

            # ── 市值对比 ──
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

            # ── 累计投入对比 ──
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

            # ── 收益率走势 ──
            st.subheader("📈 收益率走势对比")
            fig_ret = go.Figure()
            for i, (name, r) in enumerate(results.items()):
                ret_s = ((r["portfolio_series"] - r["invested_series"]) / r["invested_series"] * 100)
                ret_s = ret_s.replace([float("inf"), -float("inf")], 0)
                c = COLORS[i % len(COLORS)]
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

        # 热力图
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
