import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backtest.intel_dca import calc_all_strategies, _safe_series
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


def render_intel_dca(price_series, start_date, end_date, symbol, asset_type):
    st.subheader("🧮 智能定投实时测算")

    if not _safe_series(price_series, 5):
        st.warning("价格数据不足, 请选择有效的股票/ETF代码")
        return

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
    col1, col2, col3 = st.columns(3)
    with col1:
        base_amount = st.number_input("基准定投金额 (元)", min_value=100, value=1000, step=100, key="intel_base")
    with col2:
        min_amount = st.number_input("最低金额 (元)", min_value=0, value=500, step=100, key="intel_min")
    with col3:
        max_amount = st.number_input("最高金额 (元)", min_value=base_amount, value=3000, step=100, key="intel_max")

    avg_cost = st.number_input(
        "当前持仓平均成本 (可选, 留空则自动模拟)", min_value=0.0, value=0.0,
        step=0.01, format="%.4f", key="intel_cost",
        help="成本定投法需要此参数; 留空则模拟近24期定投的加权平均成本",
    )
    existing_shares = st.number_input(
        "当前持有份额 (可选, 价值平均法需要)", min_value=0.0, value=0.0,
        step=100.0, key="intel_shares",
    )

    # ── 策略专属参数 ──
    with st.expander("🔧 均线偏离法 参数", expanded=True):
        col1, col2 = st.columns(2)
        ma_period = col1.selectbox("均线周期", options=[20, 60, 120, 250, 500], index=3, key="intel_ma")
        ma_adjust = col2.slider("调整斜率", min_value=0.5, max_value=5.0, value=2.0, step=0.5, key="intel_ma_adj",
                                help="越大 → 偏离均线时调整幅度越激进")

    with st.expander("🔧 估值定投法 参数"):
        col1, col2, col3 = st.columns(3)
        low_pct = col1.number_input("低估阈值 (%)", min_value=10, max_value=50, value=30, step=5, key="intel_low")
        high_pct = col2.number_input("高估阈值 (%)", min_value=50, max_value=90, value=70, step=5, key="intel_high")
        col3.markdown("适用: 指数/ETF")
        if asset_type not in ("index", "etf", "lof"):
            st.info("💡 当前资产不是指数/ETF, 估值策略可能不适用")

    with st.expander("🔧 成本定投法 参数"):
        col1, col2 = st.columns(2)
        cost_min_rate = col1.slider("最低比例", min_value=0.1, max_value=1.0, value=0.5, step=0.1, key="intel_cmin",
                                    help="低于成本时最多可减少到基准金额的此比例")
        cost_max_rate = col2.slider("最高比例", min_value=1.0, max_value=5.0, value=2.0, step=0.5, key="intel_cmax",
                                    help="高于成本时最多可增加到基准金额的此比例")

    with st.expander("🔧 价值平均法 参数"):
        col1, col2 = st.columns(2)
        target_inc = col1.number_input("每期目标增值 (元)", min_value=100, value=1000, step=100, key="intel_target")
        periods = col2.number_input("已执行期数", min_value=0, value=1, step=1, key="intel_periods",
                                    help="已执行了多少期定投")

    with st.expander("🔧 下跌加仓法 参数"):
        col1, col2, col3 = st.columns(3)
        drop_th = col1.slider("跌幅阈值 (%)", min_value=1.0, max_value=15.0, value=3.0, step=0.5, key="intel_dropth")
        drop_base = col2.number_input("基准买入金额 (元)", min_value=100, value=1000, step=100, key="intel_dropbase")
        cooldown = col3.number_input("冷静期 (天)", min_value=0, max_value=10, value=1, step=1, key="intel_cooldown")

    with st.expander("🔧 网格交易法 参数"):
        col1, col2, col3 = st.columns(3)
        default_lower = round(current_price * 0.8, 4)
        default_upper = round(current_price * 1.2, 4)
        grid_lower = col1.number_input("价格下限", min_value=0.01, value=default_lower, step=0.01, format="%.4f",
                                       key="intel_glow")
        grid_upper = col2.number_input("价格上限", max_value=999999.0, value=default_upper, step=0.01, format="%.4f",
                                       key="intel_ghigh")
        grid_count = col3.number_input("网格层数", min_value=2, max_value=50, value=10, step=1, key="intel_gcnt")

    with st.expander("🔧 趋势定投法 参数"):
        col1, col2 = st.columns(2)
        short_p = col1.selectbox("短均线周期", options=[5, 10, 20, 30, 60], index=2, key="intel_short")
        long_p = col2.selectbox("长均线周期", options=[60, 120, 250, 500], index=1, key="intel_long")

    # ── 计算按钮 ──
    st.markdown("---")
    calc_btn = st.button("💡 计算当前建议", type="primary", use_container_width=True)

    if not calc_btn:
        st.info("👆 设置好参数后点击「计算当前建议」查看各策略结果")
        return

    if max_amount <= base_amount:
        st.warning("最高金额应大于基准金额")
        return

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
        "amount_per_grid": drop_base,
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
        return

    # ── 结果表格 ──
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

    # ── 总览: 建议金额排行 ──
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

    # ── 各策略详情 (展开) ──
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

            with st.expander(f"查看详细计算过程", expanded=False):
                if r.key_metrics:
                    st.markdown("**关键指标**")
                    for mk, mv in r.key_metrics.items():
                        st.markdown(f"- {mk}: **{mv}**")
                if r.detail:
                    st.markdown("**计算过程**")
                    st.code(r.detail)

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
