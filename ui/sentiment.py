"""情绪数据看板 — 查看原始股吧帖子 + 新闻 + 情感指标聚合图表."""

from datetime import date, datetime

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data.sentiment.guba import GubaSource
from data.sentiment.news import NewsSource
from data.symbol_registry import SymbolRegistry

_SOURCES = {
    "guba": GubaSource(),
    "news": NewsSource(),
}


def _color_score(s: float) -> str:
    if s > 0.1:
        return "🟢"
    if s < -0.1:
        return "🔴"
    return "⚪"


def _render_metrics(daily: pd.DataFrame):
    if daily.empty:
        return
    latest = daily.iloc[-1]
    cols = st.columns(4)
    cols[0].metric("情感得分", f"{latest['sentiment_score']:.3f}",
                   delta=f"{latest['sentiment_score'] - daily.iloc[-2]['sentiment_score']:.3f}" if len(daily) > 1 else None)
    cols[1].metric("日均帖量", f"{latest['post_volume']:.0f}",
                   delta=f"{latest['post_volume'] - daily.iloc[-2]['post_volume']:.0f}" if len(daily) > 1 else None)
    cols[2].metric("多空比", f"{latest['bull_bear_ratio']:.2f}")
    cols[3].metric("分歧度", f"{latest['disagreement']:.3f}")


def _render_charts(daily: pd.DataFrame):
    if daily.empty or len(daily) < 2:
        st.info("数据不足，无法绘图")
        return

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("情感得分", "每日帖量", "多空比", "分歧度"),
    )

    fig.add_trace(
        go.Scatter(x=daily.index, y=daily["sentiment_score"],
                   mode="lines+markers", name="sentiment_score",
                   line=dict(color="#2563eb", width=2),
                   marker=dict(size=4, color=daily["sentiment_score"],
                               cmax=1, cmin=-1, colorscale="RdBu")),
        row=1, col=1,
    )
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=1)

    fig.add_trace(
        go.Bar(x=daily.index, y=daily["post_volume"],
               name="post_volume", marker_color="#3b82f6"),
        row=2, col=1,
    )

    fig.add_trace(
        go.Scatter(x=daily.index, y=daily["bull_bear_ratio"],
                   mode="lines+markers", name="bull_bear_ratio",
                   line=dict(color="#10b981", width=2),
                   marker=dict(size=4)),
        row=3, col=1,
    )
    fig.add_hline(y=1, line_dash="dot", line_color="gray", row=3, col=1)

    fig.add_trace(
        go.Scatter(x=daily.index, y=daily["disagreement"],
                   mode="lines+markers", name="disagreement",
                   line=dict(color="#f59e0b", width=2),
                   marker=dict(size=4),
                   fill="tozeroy", fillcolor="rgba(245,158,11,0.15)"),
        row=4, col=1,
    )

    fig.update_layout(
        height=600, hovermode="x unified",
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=False,
    )
    fig.update_yaxes(title_text="得分 (-1~1)", row=1, col=1)
    fig.update_yaxes(title_text="帖数", row=2, col=1)
    fig.update_yaxes(title_text="比值", row=3, col=1)
    fig.update_yaxes(title_text="0~1", row=4, col=1)

    st.plotly_chart(fig, width='stretch')


def _render_raw_posts(df: pd.DataFrame, source_name: str):
    if df.empty:
        st.caption("暂无原始数据")
        return

    display = df[["date", "score", "sentiment", "text", "read_count", "comment_count"]].copy()
    display["score"] = display["score"].round(3)
    display["date"] = pd.to_datetime(display["date"]).dt.strftime("%m-%d %H:%M")
    display["情感"] = display["score"].apply(_color_score)
    display["标题"] = display["text"].str[:80]
    display["阅读"] = display["read_count"].astype(int)
    display["评论"] = display["comment_count"].astype(int)

    st.dataframe(
        display[["date", "情感", "score", "标题", "阅读", "评论"]],
        column_config={
            "date": "时间",
            "情感": st.column_config.TextColumn("", width="small"),
            "score": "得分",
            "标题": st.column_config.TextColumn("标题", width="large"),
            "阅读": "阅读",
            "评论": "评论",
        },
        width='stretch', hide_index=True,
    )

    csv = display.to_csv(index=False)
    st.download_button(
        f"📥 下载 {source_name} 原始数据 CSV",
        data=csv,
        file_name=f"{source_name}_{date.today()}.csv",
        mime="text/csv",
    )


def render_sentiment_dashboard():
    st.title("📊 情绪数据看板")
    st.caption("查看东方财富股吧讨论和财经新闻的原始数据与情感分析结果")

    all_symbols = SymbolRegistry.list()
    if not all_symbols:
        st.error("尚未添加任何代码。请先在「📋 代码管理」中添加。")
        return

    type_filter = st.sidebar.selectbox(
        "资产类型",
        ["全部"] + sorted(set(s["asset_type"] for s in all_symbols)),
        key="sent_type",
    )
    filtered = all_symbols if type_filter == "全部" else [s for s in all_symbols if s["asset_type"] == type_filter]
    symbol_options = {f"{s['symbol']} - {s['name']}": s["symbol"] for s in filtered}
    selected_label = st.sidebar.selectbox("选择代码", list(symbol_options.keys()), key="sent_symbol")
    symbol = symbol_options[selected_label]

    source_name = st.sidebar.radio("数据来源", ["股吧", "新闻"], horizontal=True, key="sent_source")
    src_key = "guba" if source_name == "股吧" else "news"

    use_llm = st.sidebar.checkbox("使用 LLM 分析 (需 API key)", value=False, key="sent_llm")

    llm_client = None
    if use_llm:
        api_key = st.sidebar.text_input("DeepSeek API Key", type="password", key="sent_api_key")
        if api_key:
            from data.sentiment.deepseek_client import DeepSeekClient
            llm_client = DeepSeekClient(api_key)

    if st.sidebar.button("🔍 拉取数据", type="primary", width='stretch'):
        with st.spinner(f"正在获取 {symbol} {source_name} 数据..."):
            source = _SOURCES[src_key]
            daily = source.fetch(
                symbol,
                start_date="20200101",
                end_date=datetime.now().strftime("%Y%m%d"),
                use_llm=use_llm,
                llm_client=llm_client,
            )
            raw = source.fetch_raw_posts(symbol) if hasattr(source, "fetch_raw_posts") else pd.DataFrame()

        st.session_state.sent_daily = daily
        st.session_state.sent_raw = raw
        st.session_state.sent_fetched_symbol = symbol
        st.session_state.sent_fetched_source = source_name
        st.rerun()

    daily = st.session_state.get("sent_daily")
    raw = st.session_state.get("sent_raw")

    if daily is None:
        st.info("👈 在侧边栏选择代码后点击「拉取数据」")
        return

    if st.session_state.get("sent_fetched_symbol") != symbol:
        st.info("代码已切换，请重新拉取数据")
        return

    # ── 概览指标 ──
    st.subheader("📈 聚合概览")
    _render_metrics(pd.DataFrame(daily))

    # ── 趋势图表 ──
    st.subheader("📉 情感趋势")
    _render_charts(pd.DataFrame(daily))

    # ── 原始数据 ──
    st.subheader(f"📋 原始{source_name}帖子 (最近 {len(raw)} 条)")
    _render_raw_posts(pd.DataFrame(raw), source_name)
