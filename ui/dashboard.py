import streamlit as st
from data.pe_fetcher import get_latest_pe, get_pe_percentile, manual_set_pe


def _format_pct(pct):
    if pct is None:
        return "N/A"
    return f"{pct:.1f}%"


def _pe_color(pe_value):
    if pe_value is None:
        return "off"
    if pe_value < 20:
        return "normal"
    if pe_value < 30:
        return "moderate"
    return "high"


def render_home_dashboard():
    st.subheader("🏠 首页")

    # ── 刷新 / 手动输入 ──
    cols = st.columns([1, 1, 3])
    refresh = cols[0].button("🔄 刷新", type="secondary")
    manual_mode = cols[1].button("✏️ 手动输入", type="secondary")

    pe_value = get_latest_pe(force_refresh=refresh)

    if manual_mode:
        manual_val = st.number_input(
            "输入当前 纳指100 PE(TTM)",
            min_value=0.0, max_value=200.0, value=pe_value or 30.0,
            step=0.1, format="%.1f",
            key="manual_pe_input",
        )
        if st.button("✅ 确认", type="primary"):
            manual_set_pe(manual_val)
            pe_value = manual_val
            st.rerun()

    percentile = get_pe_percentile(pe_value, force_refresh=refresh)

    # ── 指标卡片 ──
    col1, col2, col3 = st.columns(3)

    with col1:
        if pe_value is not None:
            st.metric(
                "纳指100 PE(TTM)",
                f"{pe_value:.2f}",
                delta=None,
            )
        else:
            st.metric("纳指100 PE(TTM)", "暂无数据")

    with col2:
        if percentile and percentile["pct"] is not None:
            st.metric(
                "历史百分位",
                f"{percentile['pct']:.1f}%",
                delta=None,
                help=f"基于最近 {percentile['count']} 个数据点, "
                     f"范围 {percentile['min']:.1f} ~ {percentile['max']:.1f}",
            )
        else:
            st.metric("历史百分位", "暂无数据")

    with col3:
        if percentile and percentile["count"] > 0:
            st.metric(
                "数据样本量",
                f"{percentile['count']}",
                delta=None,
                help=f"最小 {percentile['min']:.1f} / 最大 {percentile['max']:.1f}",
            )
        else:
            st.metric("数据样本量", "0")

    # ── 数据源说明 ──
    with st.expander("ℹ️ 数据说明", expanded=False):
        st.markdown("""
        - **PE(TTM)**: 滚动市盈率 (Trailing Twelve Months), 即当前总市值 / 过去12个月净利润
        - **目标指数**: 纳斯达克100 (NASDAQ-100, ^NDX)
        - **数据来源**: 通过 QQQ 持仓估值获取 (yfinance)
        - **百分位**: 当前 PE 在历史数据中的排序位置, 值越低代表越低估
        - **注意**: 若自动获取失败, 可手动输入 PE 值; 输入值仅缓存到本地, 无历史序列则不显示百分位
        """)

    # ── 简单的颜色提示 ──
    color = _pe_color(pe_value)
    if color == "high":
        st.info("📈 当前 PE 偏高 (＞30), 估值处于历史较高水平")
    elif color == "moderate":
        st.info("📊 当前 PE 适中 (20~30)")
    elif color == "normal" and pe_value is not None:
        st.success("📉 当前 PE 偏低 (＜20), 估值处于历史较低水平")
