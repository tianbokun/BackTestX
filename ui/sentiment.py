"""情绪数据看板 — 个股/板块双模式."""

from datetime import date, datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data.sentiment.guba import GubaSource
from data.sentiment.news import NewsSource
from data.sentiment.sector_sentiment import compute_sector_sentiment
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


# ── 板块模式 ──────────────────────────────────────────────

def _format_inflow(v: float) -> str:
    if pd.isna(v) or v == 0:
        return "—"
    if abs(v) >= 1e8:
        return f"{v / 1e8:.2f}亿"
    if abs(v) >= 1e4:
        return f"{v / 1e4:.2f}万"
    return f"{v:.0f}"


def _sentiment_bar_color(score: float) -> str:
    if score >= 0.3:
        return "#22c55e"
    if score >= 0.05:
        return "#86efac"
    if score <= -0.3:
        return "#ef4444"
    if score <= -0.05:
        return "#fca5a5"
    return "#d1d5db"


def _render_sector_dashboard():
    st.subheader("🏭 板块情绪排行")
    st.caption("综合多个数据源计算出每个板块的情绪得分")

    with st.sidebar:
        board_type = st.radio("板块类型", ["概念板块", "行业板块", "全部"],
                              horizontal=True, key="sector_board_type")
        type_map = {"概念板块": "concept", "行业板块": "industry", "全部": "all"}
        top_n = st.selectbox("显示数量", [20, 30, 50, 100, "全部"], index=1, key="sector_top_n")

    if st.sidebar.button("🔍 刷新板块数据", type="primary", use_container_width=True,
                         key="sector_fetch_btn"):
        st.session_state.sector_error = None
        with st.spinner("正在获取板块数据..."):
            try:
                result = compute_sector_sentiment(board_type=type_map[board_type])
                if result.empty:
                    st.session_state.sector_error = "API 返回空数据，东方财富接口可能暂时不可用"
                st.session_state.sector_result = result
            except Exception as e:
                st.session_state.sector_result = None
                st.session_state.sector_error = f"连接失败：{type(e).__name__}"
        st.rerun()

    result = st.session_state.get("sector_result")
    error = st.session_state.get("sector_error")

    if error:
        st.error(f"⚠️ {error}")
        st.info("👈 点击侧边栏「刷新板块数据」重试（东方财富接口有反爬限制，两次操作之间建议间隔 5 秒以上）")
        if st.button("🔄 重试", use_container_width=True):
            del st.session_state.sector_error
            st.rerun()
        return

    if result is None or result.empty:
        st.info("👈 点击侧边栏「刷新板块数据」")
        return

    # ── 概览指标 ──
    top = result.head(5)
    cols = st.columns(4)
    cols[0].metric("总板块数", f"{len(result)}")
    cols[1].metric("最正面", top.iloc[0]["board_name"],
                   delta=f"得分 {top.iloc[0]['sentiment_score']:.3f}")
    cols[2].metric("最负面", result.iloc[-1]["board_name"],
                   delta=f"得分 {result.iloc[-1]['sentiment_score']:.3f}",
                   delta_color="inverse")
    pos_count = (result["sentiment_score"] > 0).sum()
    cols[3].metric("正面/负面", f"{pos_count}/{len(result) - pos_count}")

    # ── 分布直方图 ──
    st.subheader("📊 情绪得分分布")
    bins = min(30, max(10, len(result) // 5))
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=result["sentiment_score"],
        nbinsx=bins,
        marker_color="#3b82f6",
        hovertemplate="得分区间: %{x:.2f}<br>数量: %{y}<extra></extra>",
    ))
    fig_hist.add_vline(x=0, line_dash="dot", line_color="gray")
    fig_hist.update_layout(
        height=250, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="情绪得分", yaxis_title="板块数量",
        hovermode="x",
    )
    st.plotly_chart(fig_hist, width='stretch')

    # ── 排行表格 ──
    st.subheader("🏆 板块情绪排行")

    show_all = top_n == "全部"
    display = result.copy()
    if not show_all:
        display = display.head(top_n)

    display["情绪"] = display["sentiment_score"].apply(
        lambda s: f"{'🟢' if s > 0.05 else '🔴' if s < -0.05 else '⚪'} {s:.3f}"
    )
    display["涨跌幅"] = display["change_pct"].apply(lambda v: f"{v:.2f}%")
    display["主力资金"] = display["main_force_net_inflow"].apply(_format_inflow)
    display["多空比"] = display["adv_dec_ratio"].apply(
        lambda v: f"{v:.2f}" if pd.notna(v) and v != 0 else "—"
    )
    display["宽度"] = display["breadth_ratio"].apply(lambda v: f"{v:+.3f}")
    display["异动"] = display["anomaly_count"].apply(lambda v: f"{int(v)}次" if pd.notna(v) else "—")
    display["板块类型"] = display["board_type"].map({"concept": "概念", "industry": "行业"}).fillna("")

    table_cols = ["排名", "板块名称", "板块类型", "情绪", "涨跌幅", "主力资金",
                   "多空比", "宽度", "异动"]
    table_df = display[[c for c in table_cols if c in display.columns or c == "排名"]]

    st.dataframe(
        table_df,
        column_config={
            "排名": st.column_config.NumberColumn("排名", width="small"),
            "板块名称": st.column_config.TextColumn("板块名称", width="medium"),
            "板块类型": st.column_config.TextColumn("类型", width="small"),
            "情绪": st.column_config.TextColumn("情绪得分", width="small"),
            "涨跌幅": st.column_config.TextColumn("涨跌幅", width="small"),
            "主力资金": st.column_config.TextColumn("主力资金", width="small"),
            "多空比": st.column_config.TextColumn("多空比", width="small"),
            "宽度": st.column_config.TextColumn("宽度指标", width="small"),
            "异动": st.column_config.TextColumn("异动次数", width="small"),
        },
        width='stretch', hide_index=True,
    )

    # ── 情绪 vs 涨跌幅散点图 ──
    st.subheader("📈 情绪 vs 涨跌幅")
    scatter = result.copy()
    scatter["color"] = scatter["sentiment_score"].apply(_sentiment_bar_color)
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=scatter["sentiment_score"],
        y=scatter["change_pct"],
        mode="markers+text",
        text=scatter["board_name"],
        textposition="top center",
        marker=dict(
            size=8,
            color=scatter["sentiment_score"],
            cmax=1, cmin=-1,
            colorscale="RdBu",
            line=dict(width=0.5, color="gray"),
        ),
        hovertemplate="<b>%{text}</b><br>情绪: %{x:.3f}<br>涨跌幅: %{y:.2f}%<extra></extra>",
    ))
    fig_scatter.add_hline(y=0, line_dash="dot", line_color="gray")
    fig_scatter.add_vline(x=0, line_dash="dot", line_color="gray")
    fig_scatter.update_layout(
        height=400, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="情绪得分", yaxis_title="涨跌幅 (%)",
        hovermode="closest",
    )
    st.plotly_chart(fig_scatter, width='stretch')

    # ── 下载 ──
    csv = result.to_csv(index=False)
    st.download_button(
        "📥 下载板块情绪数据 CSV",
        data=csv,
        file_name=f"sector_sentiment_{date.today()}.csv",
        mime="text/csv",
    )


# ── 个股模式 ──────────────────────────────────────────────

def _render_stock_dashboard():
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

    if st.sidebar.button("🔍 拉取数据", type="primary", use_container_width=True):
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

    st.subheader("📈 聚合概览")
    _render_metrics(pd.DataFrame(daily))

    st.subheader("📉 情感趋势")
    _render_charts(pd.DataFrame(daily))

    st.subheader(f"📋 原始{source_name}帖子 (最近 {len(raw)} 条)")
    _render_raw_posts(pd.DataFrame(raw), source_name)


# ── 入口 ──────────────────────────────────────────────────

def render_sentiment_dashboard():
    st.title("📊 情绪数据看板")
    st.caption("个股帖子情感分析 / 板块综合情绪排行")

    mode = st.radio(
        "模式", ["个股分析", "板块排行"],
        horizontal=True, label_visibility="collapsed",
        key="sent_mode",
    )

    if mode == "板块排行":
        _render_sector_dashboard()
    else:
        _render_stock_dashboard()
