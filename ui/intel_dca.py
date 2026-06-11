import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date

from backtest.intel_dca import (
    calc_all_strategies, backtest_ma_deviation,
    simulate_portfolios, calc_cagr, _safe_series,
)
from data.symbol_registry import SymbolRegistry
from ui._helpers import cached_fetch
from data_fetcher import get_price_series


def _signal_emoji(signal: str) -> str:
    emoji_map = {
        "买入": "🟢", "加仓": "🟢", "补仓": "🟢", "首次定投": "🆕",
        "正常": "🔵", "多头持有": "🔵", "金叉加仓": "🟢",
        "减少": "🟡", "减仓": "🟡", "空头减仓": "🟡", "死叉减仓": "🔴",
        "暂停": "⏸️", "未触发": "⚪", "观望": "⚪",
        "卖出": "🔴", "减仓/卖出": "🔴",
        "不支持": "❌", "数据不足": "⚠️", "参数错误": "⚠️",
        "触发买入": "🔴",
    }
    return emoji_map.get(signal, "⚪")


def _signal_color(signal: str) -> str:
    if signal in ("买入", "加仓", "补仓", "首次定投", "金叉加仓", "触发买入"):
        return "rgba(34,197,94,0.15)"
    if signal in ("卖出", "减仓/卖出", "死叉减仓"):
        return "rgba(239,68,68,0.15)"
    if signal in ("减少", "减仓", "空头减仓"):
        return "rgba(234,179,8,0.15)"
    if signal in ("不支持", "参数错误", "数据不足"):
        return "rgba(239,68,68,0.08)"
    return "transparent"


def render_intel_dca():
    st.subheader("🧮 智能定投实时测算")

    # ── 从代码库选择标的 ──
    symbols = SymbolRegistry.list()
    if not symbols:
        st.warning("⚠️ 代码库为空, 请先在「📋 代码管理」页面添加资产")
        return

    options = {s["symbol"]: s for s in symbols}
    symbol_labels = {
        s["symbol"]: f"{s['symbol']}  {s['name']}  ({s['asset_type']})"
        for s in symbols
    }
    selected = st.selectbox(
        "选择投资标的",
        options=list(options.keys()),
        format_func=lambda x: symbol_labels.get(x, x),
        key="intel_symbol",
    )
    entry = options[selected]

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        adjust = st.selectbox(
            "复权方式",
            options={"qfq": "前复权", "hfq": "后复权", "": "不复权"},
            format_func=lambda x: {"qfq": "前复权", "hfq": "后复权", "": "不复权"}[x],
            index=0, key="intel_adjust",
        )
    today = date.today()
    with col2:
        start_date = col2.date_input(
            "开始日期", value=date(today.year - 5, 1, 1),
            min_value=date(1970, 1, 1), max_value=today,
            key="intel_start",
        )
    with col3:
        end_date = col3.date_input(
            "结束日期", value=today,
            min_value=date(1970, 1, 1), max_value=today,
            key="intel_end",
        )

    # ── 获取数据 ──
    with st.spinner("正在获取数据..."):
        try:
            start_str = start_date.strftime("%Y%m%d")
            end_str = end_date.strftime("%Y%m%d")
            df = cached_fetch(selected, entry["asset_type"], start_str, end_str, adjust)
        except Exception as e:
            st.error(f"数据获取失败: {e}")
            return

    if df is None or df.empty:
        st.warning(f"未获取到 {selected} 的数据, 请检查代码或日期范围")
        return

    price_series = get_price_series(df)
    if price_series is None or len(price_series) < 5:
        st.warning("价格数据不足, 请选择有效的股票/ETF代码")
        return

    symbol = selected
    asset_type = entry["asset_type"]
    current_price = float(price_series.iloc[-1])
    prev_close = float(price_series.iloc[-2]) if len(price_series) >= 2 else current_price
    daily_change = (current_price - prev_close) / prev_close * 100

    # ── 当前行情 ──
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("最新价", f"{current_price:.4f}", f"{daily_change:+.2f}%")
    if len(price_series) >= 250:
        ma250 = price_series.rolling(250).mean().iloc[-1]
        col2.metric("MA250", f"{ma250:.4f}", f"{(current_price-ma250)/ma250*100:+.2f}%")
    else:
        col2.metric("MA250", "N/A")
    ytd = price_series.index[-1] - pd.Timedelta(days=365)
    ytd_data = price_series[price_series.index >= ytd]
    if len(ytd_data) > 0:
        ytd_change = (current_price - ytd_data.iloc[0]) / ytd_data.iloc[0] * 100
        col3.metric("近一年涨跌", f"{ytd_change:+.2f}%")
    else:
        col3.metric("近一年涨跌", "N/A")
    col4.metric("数据区间", f"{price_series.index[0].strftime('%Y%m%d')} ~ {price_series.index[-1].strftime('%Y%m%d')}")

    # ── 定投参数 ──
    st.markdown("---")
    st.markdown("### 定投参数设置")
    st.caption("基准金额自动取上下限中点, 即价格等于均线时的正常投额")
    col1, col2 = st.columns(2)
    with col1:
        min_amount = st.number_input("最低金额 (元)", min_value=0, value=500, step=100, key="intel_min2")
    with col2:
        max_amount = st.number_input("最高金额 (元)", min_value=100, value=3000, step=100, key="intel_max2")
    base_amount = (min_amount + max_amount) / 2

    avg_cost = st.number_input(
        "当前持仓平均成本 (可选, 留空则自动模拟)", min_value=0.0, value=0.0,
        step=0.01, format="%.4f", key="intel_cost2",
        help="成本定投法需要此参数; 留空则模拟近24期定投的加权平均成本",
    )
    existing_shares = st.number_input(
        "当前持有份额 (可选, 价值平均法需要)", min_value=0.0, value=0.0,
        step=100.0, key="intel_shares2",
    )

    # ── 策略专属参数 ──
    with st.expander("🔧 均线偏离法 参数", expanded=True):
        col1, col2 = st.columns(2)
        ma_period = col1.selectbox("均线周期", options=[20, 60, 120, 250, 500], index=3, key="intel_ma2")
        ma_adjust = col2.slider("调整斜率", min_value=0.5, max_value=5.0, value=2.0, step=0.5, key="intel_ma_adj2",
                                help="越大 → 偏离均线时调整幅度越激进")

    with st.expander("🔧 估值定投法 参数"):
        col1, col2, col3 = st.columns(3)
        low_pct = col1.number_input("低估阈值 (%)", min_value=10, max_value=50, value=30, step=5, key="intel_low2")
        high_pct = col2.number_input("高估阈值 (%)", min_value=50, max_value=90, value=70, step=5, key="intel_high2")
        col3.markdown("适用: 指数/ETF")
        if asset_type not in ("index", "etf", "lof"):
            st.info("💡 当前资产不是指数/ETF, 估值策略可能不适用")

    with st.expander("🔧 成本定投法 参数"):
        col1, col2 = st.columns(2)
        cost_min_rate = col1.slider("最低比例", min_value=0.1, max_value=1.0, value=0.5, step=0.1, key="intel_cmin2",
                                    help="低于成本时最多可减少到基准金额的此比例")
        cost_max_rate = col2.slider("最高比例", min_value=1.0, max_value=5.0, value=2.0, step=0.5, key="intel_cmax2",
                                    help="高于成本时最多可增加到基准金额的此比例")

    with st.expander("🔧 价值平均法 参数"):
        col1, col2 = st.columns(2)
        target_inc = col1.number_input("每期目标增值 (元)", min_value=100, value=1000, step=100, key="intel_target2")
        periods = col2.number_input("已执行期数", min_value=0, value=1, step=1, key="intel_periods2",
                                    help="已执行了多少期定投")

    with st.expander("🔧 下跌加仓法 参数"):
        col1, col2, col3 = st.columns(3)
        drop_th = col1.slider("跌幅阈值 (%)", min_value=1.0, max_value=15.0, value=3.0, step=0.5, key="intel_dropth2")
        drop_base = col2.number_input("基准买入金额 (元)", min_value=100, value=1000, step=100, key="intel_dropbase2")
        cooldown = col3.number_input("冷静期 (天)", min_value=0, max_value=10, value=1, step=1, key="intel_cooldown2")

    with st.expander("🔧 网格交易法 参数"):
        col1, col2, col3 = st.columns(3)
        default_lower = round(current_price * 0.8, 4)
        default_upper = round(current_price * 1.2, 4)
        grid_lower = col1.number_input("价格下限", min_value=0.01, value=default_lower, step=0.01, format="%.4f",
                                       key="intel_glow2")
        grid_upper = col2.number_input("价格上限", max_value=999999.0, value=default_upper, step=0.01, format="%.4f",
                                       key="intel_ghigh2")
        grid_count = col3.number_input("网格层数", min_value=2, max_value=50, value=10, step=1, key="intel_gcnt2")
        col1_2, col2_2, _ = st.columns(3)
        amount_per_grid = col1_2.number_input("每格金额 (元)", min_value=100, value=500, step=100, key="intel_ga2")

    with st.expander("🔧 趋势定投法 参数"):
        col1, col2 = st.columns(2)
        short_p = col1.selectbox("短均线周期", options=[5, 10, 20, 30, 60], index=2, key="intel_short2")
        long_p = col2.selectbox("长均线周期", options=[60, 120, 250, 500], index=1, key="intel_long2")

    # ── 计算按钮 ──
    st.markdown("---")
    calc_btn = st.button("💡 计算当前建议", type="primary", use_container_width=True)

    if calc_btn:
        if max_amount <= min_amount:
            st.warning("最高金额应大于基准金额")
        else:
            call_avg_cost = avg_cost if avg_cost > 0 else None

            params = {
                "ma_period": ma_period,
                "ma_adjustment": ma_adjust,
                "low_percentile": low_pct,
                "high_percentile": high_pct,
                "cost_min_rate": cost_min_rate,
                "cost_max_rate": cost_max_rate,
                "target_increment": target_inc,
                "drop_threshold": drop_th,
                "drop_buy_base": drop_base,
                "cooldown_days": cooldown,
                "grid_lower": grid_lower,
                "grid_upper": grid_upper,
                "grid_count": int(grid_count),
                "amount_per_grid": amount_per_grid,
                "short_period": short_p,
                "long_period": long_p,
            }

            results = calc_all_strategies(
                price_series=price_series,
                base_amount=float(base_amount),
                min_amount=float(min_amount),
                max_amount=float(max_amount),
                params=params,
                symbol=symbol,
                asset_type=asset_type,
                avg_cost=call_avg_cost,
                existing_shares=float(existing_shares),
                periods_elapsed=int(periods),
            )

            if not results:
                st.error("计算失败, 请检查数据")
            else:
                st.markdown("### 📊 各策略计算结果对比")
                rows = []
                for key, r in results.items():
                    rows.append({
                        "策略": r.name,
                        "信号": f"{_signal_emoji(r.signal)} {r.signal}",
                        "建议金额": f"{r.amount:.0f} 元" if r.amount > 0 else "—",
                        "占基准": f"{r.amount_pct:.0f}%" if r.amount > 0 else "—",
                        "说明": r.explanation,
                    })
                if rows:
                    display_df = pd.DataFrame(rows)
                    st.dataframe(display_df, use_container_width=True, hide_index=True)

                st.markdown("### 💰 策略排行 (按建议金额)")
                ranked = [(k, r) for k, r in results.items() if r.amount > 0]
                ranked.sort(key=lambda x: x[1].amount, reverse=True)
                if ranked:
                    cols = st.columns(len(ranked))
                    for i, (col, (k, r)) in enumerate(zip(cols, ranked)):
                        with col:
                            st.metric(
                                label=f"{_signal_emoji(r.signal)} {r.name}",
                                value=f"{r.amount:.0f}元",
                                delta=f"{r.amount_pct:.0f}%",
                            )

                st.markdown("### 📋 各策略详细计算过程")
                for order_name in [
                    "ma_deviation", "valuation", "cost_average", "value_averaging",
                    "drop_trigger", "grid_trading", "trend_following",
                ]:
                    r = results.get(order_name)
                    if r is None:
                        continue
                    bg = _signal_color(r.signal)
                    with st.container():
                        st.markdown(f"""
                        <div style="background:{bg}; border-radius:12px; padding:0.75rem 1rem; margin-bottom:0.5rem; border:1px solid #e2e8f0;">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <div><strong>{r.name}</strong>
                                    <span style="margin-left:8px; font-size:0.9rem;">{_signal_emoji(r.signal)} {r.signal}</span>
                                </div>
                                <div style="font-size:1.2rem; font-weight:700;">
                                    {f"{r.amount:.0f} 元" if r.amount > 0 else "—"}
                                    <span style="font-size:0.8rem; color:#64748b; margin-left:4px;">
                                        {f"({r.amount_pct:.0f}%)" if r.amount > 0 else ""}
                                    </span>
                                </div>
                            </div>
                            <div style="margin-top:4px; color:#475569; font-size:0.85rem;">
                                {r.explanation}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        with st.expander("查看详细计算过程", expanded=False):
                            if r.key_metrics:
                                st.markdown("**关键指标**")
                                for mk, mv in r.key_metrics.items():
                                    st.markdown(f"- {mk}: **{mv}**")
                            if r.detail:
                                st.markdown("**计算过程**")
                                st.code(r.detail)
    else:
        st.info("👆 设置好参数后点击「计算当前建议」查看各策略结果")

    # ── 价格走势 + MA 参考图 ──
    st.markdown("### 📈 价格走势及均线参考")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=price_series.index, y=price_series.values,
        mode="lines", name="收盘价",
        line=dict(color="#1f77b4", width=2),
    ))
    if len(price_series) >= ma_period:
        ma_vals = price_series.rolling(ma_period).mean()
        fig.add_trace(go.Scatter(
            x=price_series.index, y=ma_vals.values,
            mode="lines", name=f"MA{ma_period}",
            line=dict(color="#ff7f0e", width=1.5, dash="dash"),
        ))
    if len(price_series) >= short_p:
        short_ma = price_series.rolling(short_p).mean()
        fig.add_trace(go.Scatter(
            x=price_series.index, y=short_ma.values,
            mode="lines", name=f"MA{short_p}",
            line=dict(color="#2ca02c", width=1.5),
        ))
    if len(price_series) >= long_p:
        long_ma = price_series.rolling(long_p).mean()
        fig.add_trace(go.Scatter(
            x=price_series.index, y=long_ma.values,
            mode="lines", name=f"MA{long_p}",
            line=dict(color="#d62728", width=1.5),
        ))
    fig.update_layout(
        hovermode="x unified", height=400,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", y=1.02),
        xaxis_title="日期", yaxis_title="价格",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ════════════════════════════════════════════════
    #  🕰️ 历史回测验证 — 均线偏离法
    # ════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 🕰️ 历史回测验证 — 均线偏离法")

    col_a, col_b = st.columns([1, 2])
    total_budget = col_a.number_input(
        "总投入本金上限 (元)", min_value=0, value=0, step=10000,
        help="0 表示不设上限, 各策略实际总投入作为对比基准",
    )
    bt_expander = st.expander("回测设置", expanded=True)
    with bt_expander:
        col1, col2 = st.columns(2)
        weekly_full = price_series.resample("W").last().dropna()
        max_bt_weeks = max(0, len(weekly_full) - max(ma_period // 5, 1))
        bt_weeks = col1.slider(
            "回测周数", min_value=12, max_value=max_bt_weeks or 12,
            value=min(52, max_bt_weeks or 52), step=4, key="bt_weeks",
        )
        col2.markdown("**策略**: 均线偏离法")
        run_bt = st.button("📊 运行回测", type="primary", use_container_width=True)

    if run_bt:
        with st.spinner(f"正在回测过去 {bt_weeks} 周..."):
            bt_df = backtest_ma_deviation(
                price_series,
                base_daily=float(base_amount),
                min_daily=float(min_amount),
                max_daily=float(max_amount),
                ma_period=ma_period,
                adjustment_factor=ma_adjust,
                lookback_weeks=bt_weeks,
                trade_days_per_week=5,
                total_budget=float(total_budget),
            )

        if bt_df.empty:
            st.warning("回测数据不足, 请增大回测周数或检查数据范围")
        else:
            # ── 统计卡片 ──
            buy_count = int((bt_df["signal"] == "买入").sum())
            pause_count = int((bt_df["signal"] == "暂停").sum())
            total_periods = len(bt_df)
            avg_amount = bt_df["amount"].mean()
            above_baseline = (bt_df["amount"] > bt_df["baseline"]).sum()
            below_baseline = (bt_df["amount"] < bt_df["baseline"]).sum()

            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("回测期数", f"{total_periods}")
            c2.metric("平均建议(周)", f"{avg_amount:.0f}")
            c3.metric("买入次数", f"{buy_count}")
            c4.metric("暂停次数", f"{pause_count}")
            c5.metric("超基准", f"{above_baseline}")
            c6.metric("低于基准", f"{below_baseline}")

            # ── 回测图表 ──
            fig_bt = go.Figure()
            color_map = {"买入": "#22c55e", "暂停": "#ef4444", "数据不足": "#94a3b8"}
            signals = bt_df["signal"].unique()

            fig_bt.add_trace(go.Scatter(
                x=bt_df["date"], y=bt_df["amount"],
                mode="lines+markers", name="均线法建议金额",
                line=dict(color="#3b82f6", width=2),
                marker=dict(
                    size=8, color=bt_df["signal"].map(color_map),
                    symbol="circle",
                ),
                text=[
                    f"信号: {s}<br>金额: {a:.0f}<br>MA偏离: {d:+.2f}%<br>占基准: {p:.0f}%"
                    for s, a, d, p in zip(
                        bt_df["signal"], bt_df["amount"],
                        bt_df["deviation_pct"], bt_df["amount_pct"],
                    )
                ],
                hoverinfo="text+x+y",
            ))
            fig_bt.add_hline(
                y=float(base_amount) * 5,
                line_dash="dash", line_color="#94a3b8",
                annotation_text=f"固定周投 ({float(base_amount)*5:.0f})",
                annotation_position="bottom right",
            )
            fig_bt.update_layout(
                title="逐周建议金额 vs 固定周投",
                xaxis_title="日期", yaxis_title="建议金额 (元/周)",
                hovermode="x unified", height=420,
                margin=dict(l=10, r=10, t=30, b=10),
                legend=dict(orientation="h", y=1.02),
            )
            st.plotly_chart(fig_bt, use_container_width=True)

            # ── 偏离度辅助图 ──
            fig_dev = go.Figure()
            fig_dev.add_trace(go.Scatter(
                x=bt_df["date"], y=bt_df["deviation_pct"],
                mode="lines", name="偏离度(%)",
                line=dict(color="#f59e0b", width=1.5),
                fill="tozeroy", fillcolor="rgba(245,158,11,0.12)",
            ))
            fig_dev.add_hline(y=0, line_dash="dot", line_color="#94a3b8")
            fig_dev.update_layout(
                title="均线偏离度变化",
                xaxis_title="日期", yaxis_title="偏离度 (%)",
                height=260, margin=dict(l=10, r=10, t=20, b=10),
                hovermode="x unified",
            )
            st.plotly_chart(fig_dev, use_container_width=True)

            # ── 双轴对比图: 建议金额 vs 净值 ──
            fig_dual = go.Figure()

            fig_dual.add_trace(go.Scatter(
                x=bt_df["date"], y=bt_df["amount"],
                mode="lines+markers", name="建议金额",
                line=dict(color="#3b82f6", width=2),
                marker=dict(size=6, color=bt_df["signal"].map(color_map), symbol="circle"),
                text=[
                    f"信号: {s}<br>金额: {a:.0f}<br>MA偏离: {d:+.2f}%"
                    for s, a, d in zip(bt_df["signal"], bt_df["amount"], bt_df["deviation_pct"])
                ],
                hoverinfo="text+x+y",
            ))
            fig_dual.add_trace(go.Scatter(
                x=bt_df["date"], y=bt_df["price"],
                mode="lines", name="净值",
                line=dict(color="#10b981", width=2),
                yaxis="y2",
            ))
            fig_dual.add_trace(go.Scatter(
                x=bt_df["date"], y=bt_df["ma"],
                mode="lines", name=f"MA{ma_period}",
                line=dict(color="#f59e0b", width=1.5, dash="dash"),
                yaxis="y2",
            ))
            fig_dual.update_layout(
                title="建议金额 vs 净值走势",
                xaxis=dict(title="日期"),
                yaxis=dict(title="建议金额 (元/周)", side="left"),
                yaxis2=dict(title="净值", side="right", overlaying="y"),
                hovermode="x unified", height=420,
                margin=dict(l=10, r=10, t=30, b=10),
                legend=dict(orientation="h", y=1.02),
            )
            st.plotly_chart(fig_dual, use_container_width=True)

            # ── 收益对比: 智能 vs 固定 vs 梭哈 ──
            port_df = simulate_portfolios(bt_df, max_daily=float(max_amount), total_budget=float(total_budget))
            if not port_df.empty:
                st.markdown("### 📊 收益对比 (总投入本金对齐)")

                last = port_df.iloc[-1]
                days = (port_df["date"].iloc[-1] - port_df["date"].iloc[0]).total_seconds() / 86400

                def _fmt(v): return f"{v:.2f}%"
                def _cagr(v, inv): return calc_cagr(v, inv, days)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("智能定投",
                          _fmt(last["smart_return"]),
                          f"年化 {_cagr(last['smart_value'], last['smart_invested'])*100:.2f}%")
                c2.metric("固定定投",
                          _fmt(last["fixed_return"]),
                          f"年化 {_cagr(last['fixed_value'], last['fixed_invested'])*100:.2f}%")
                c3.metric("一次性梭哈",
                          _fmt(last["lump_return"]),
                          f"年化 {_cagr(last['lump_value'], last['lump_invested'])*100:.2f}%")
                c4.metric("最大值定投",
                          _fmt(last["max_return"]),
                          f"年化 {_cagr(last['max_value'], last['max_invested'])*100:.2f}%")

                fig_ret = go.Figure()

                fig_ret.add_trace(go.Scatter(
                    x=port_df["date"], y=port_df["smart_return"],
                    mode="lines", name="智能定投",
                    line=dict(color="#3b82f6", width=2.5),
                    fill="tozeroy", fillcolor="rgba(59,130,246,0.08)",
                    text=[f"收益率: {r:.2f}%<br>投入: {v:.0f}" for r, v in zip(port_df["smart_return"], port_df["smart_invested"])],
                    hoverinfo="text+x",
                ))
                fig_ret.add_trace(go.Scatter(
                    x=port_df["date"], y=port_df["fixed_return"],
                    mode="lines", name="固定定投",
                    line=dict(color="#10b981", width=2),
                    text=[f"收益率: {r:.2f}%<br>投入: {v:.0f}" for r, v in zip(port_df["fixed_return"], port_df["fixed_invested"])],
                    hoverinfo="text+x",
                ))
                fig_ret.add_trace(go.Scatter(
                    x=port_df["date"], y=port_df["lump_return"],
                    mode="lines", name="一次性梭哈",
                    line=dict(color="#f59e0b", width=2, dash="dash"),
                    text=[f"收益率: {r:.2f}%<br>投入: {v:.0f}" for r, v in zip(port_df["lump_return"], port_df["lump_invested"])],
                    hoverinfo="text+x",
                ))
                fig_ret.add_trace(go.Scatter(
                    x=port_df["date"], y=port_df["max_return"],
                    mode="lines", name="最大值定投",
                    line=dict(color="#8b5cf6", width=2, dash="dash"),
                    text=[f"收益率: {r:.2f}%<br>投入: {v:.0f}" for r, v in zip(port_df["max_return"], port_df["max_invested"])],
                    hoverinfo="text+x",
                ))
                fig_ret.add_hline(y=0, line_dash="dot", line_color="#94a3b8")
                fig_ret.update_layout(
                    title="累计收益率对比",
                    xaxis_title="日期", yaxis_title="累计收益率 (%)",
                    hovermode="x unified", height=400,
                    margin=dict(l=10, r=10, t=30, b=10),
                    legend=dict(orientation="h", y=1.02),
                )
                st.plotly_chart(fig_ret, use_container_width=True)

            # ── 明细表 ──
            with st.expander("查看逐周明细数据"):
                display_bt = bt_df[["date", "price", "deviation_pct", "amount", "baseline", "signal"]].copy()
                display_bt["date"] = display_bt["date"].dt.strftime("%Y-%m-%d")
                display_bt.columns = ["日期", "价格", "偏离度(%)", "建议金额", "固定周投", "信号"]
                display_bt["偏离度(%)"] = display_bt["偏离度(%)"].map("{:+.2f}".format)
                display_bt["建议金额"] = display_bt["建议金额"].map("{:.0f}".format)
                display_bt["固定周投"] = display_bt["固定周投"].map("{:.0f}".format)
                st.dataframe(display_bt, use_container_width=True, hide_index=True)
