"""情绪数据看板 — 个股/板块双模式."""

from datetime import date, datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data.sentiment.guba import GubaSource
from data.sentiment.news import NewsSource
from data.sentiment.sector_sentiment import (
    compute_sector_sentiment,
    fetch_board_constituents,
)
from data.sentiment import history as hist
from data.symbol_registry import SymbolRegistry

_SOURCES = {
    "guba": GubaSource(),
    "news": NewsSource(),
}

# ── 常量 ──
_HOT_CACHE_TTL = timedelta(minutes=5)


def _fmt_error(e: Exception) -> str:
    """格式化异常为可读的错误详情 (含简短 traceback)."""
    import traceback
    tb_lines = traceback.format_exception(type(e), e, e.__traceback__, limit=5)
    # 取关键帧: 文件路径:行号 → 函数名
    frames = []
    for line in tb_lines:
        if line.startswith("  File "):
            frames.append(line.strip())
    parts = [f"{type(e).__name__}"]
    if str(e):
        parts.append(str(e))
    if frames:
        parts.append("├─ 调用栈:")
        for f in frames[-3:]:
            parts.append(f"   {f}")
    cause = getattr(e, "__cause__", None) or getattr(e, "__context__", None)
    if cause:
        parts.append(f"├─ 原因: {type(cause).__name__}")
        if str(cause):
            parts.append(f"   └─ {cause}")
    return "\n".join(parts)


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


# ── 板块热帖函数 ──────────────────────────────────────────

def _render_sector_hot_posts(board_code: str, board_name: str):
    """在 expander 内渲染板块成分股的热门帖子."""
    if not board_code:
        st.caption("无板块代码，无法获取成分股")
        return

    cache_key = f"posts_{board_code}"
    now = datetime.now()
    cached = st.session_state.get(f"_hot_posts_{board_code}")
    if cached is not None:
        data, ts = cached
        if now - ts < _HOT_CACHE_TTL:
            _display_hot_posts(data, board_name)
            return

    with st.spinner(f"获取 {board_name} 成分股..."):
        constituents = fetch_board_constituents(board_code, top_n=5)

    if constituents.empty:
        st.caption("无法获取成分股数据")
        return

    results = []
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _fetch_stock_posts(row):
        sym = row["symbol"]
        try:
            src = _SOURCES["guba"]
            posts = src.fetch_raw_posts(sym)
            if posts is not None and not posts.empty:
                posts = posts.copy()
                posts["stock_symbol"] = sym
                posts["stock_name"] = row.get("name", sym)
                posts["stock_change"] = row.get("change_pct", None)
                return posts
        except Exception:
            pass
        return None

    progress_text = st.empty()
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(_fetch_stock_posts, row): i
                   for i, (_, row) in enumerate(constituents.iterrows())}
        for f in as_completed(futures):
            idx = futures[f]
            sym = constituents.iloc[idx]["symbol"]
            progress_text.caption(f"已获取 {sym} 的帖子...")
            res = f.result()
            if res is not None:
                results.append(res)

    if not results:
        st.caption("暂无帖子数据")
        return

    try:
        combined = pd.concat(results, ignore_index=True)
    except Exception:
        st.caption("整合帖子数据时出错，部分数据可能不完整")
        return
    st.session_state[f"_hot_posts_{board_code}"] = (combined, now)
    _display_hot_posts(combined, board_name)


def _display_hot_posts(combined: pd.DataFrame, board_name: str):
    """展示板块热帖 (按股票分组)."""
    st.caption(f"共 {len(combined)} 条帖子")
    for stock_symbol, group in combined.groupby("stock_symbol"):
        name = group.iloc[0].get("stock_name", stock_symbol)
        change = group.iloc[0].get("stock_change", None)
        label = f"{name} ({stock_symbol})"
        if pd.notna(change):
            label += f"  {change:+.2f}%"
        with st.expander(label):
            _render_raw_posts(group.reset_index(drop=True), "股吧")


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


# ── 导出图片 (matplotlib) ────────────────────────────────

_CJK_FONT = None  # lazy cache


def _cjk_font_name() -> str | None:
    """查找 or 下载 CJK 字体, 返回 matplotlib 字体族名称."""
    import os
    global _CJK_FONT
    if _CJK_FONT is not None:
        return _CJK_FONT if _CJK_FONT else None

    from matplotlib.font_manager import FontProperties
    import matplotlib.font_manager as fm

    # 已知可用的 CJK 字体名称 (不含中文字符, 无法通过字符范围检测)
    _CJK_NAMES = {"WenQuanYi Micro Hei", "WenQuanYi Zen Hei", "Noto Sans SC",
                   "Noto Sans CJK SC", "Source Han Sans SC", "Droid Sans Fallback"}

    def _search() -> str | None:
        for f in fm.fontManager.ttflist:
            if f.name in _CJK_NAMES:
                return f.name
        # fallback: 文件名含 wqy / noto / cjk 的
        for f in fm.fontManager.ttflist:
            lower = f.fname.lower()
            if any(kw in lower for kw in ("wqy", "wenquan", "noto", "cjk", "han", "chinese")):
                return f.name
        return None

    name = _search()
    if name:
        _CJK_FONT = name
        return name

    cache_dir = os.path.join(os.path.dirname(__file__), "..", ".cache", "fonts")
    os.makedirs(cache_dir, exist_ok=True)
    font_path = os.path.join(cache_dir, "wqy-microhei.ttc")

    if not os.path.exists(font_path):
        try:
            import urllib.request, tarfile
            deb_path = font_path + ".deb"
            url = "https://mirrors.tuna.tsinghua.edu.cn/debian/pool/main/f/fonts-wqy-microhei/fonts-wqy-microhei_0.2.0-beta-3.1_all.deb"
            urllib.request.urlretrieve(url, deb_path)
            with open(deb_path, "rb") as f:
                data = f.read()
            idx = data.find(b"data.tar")
            if idx >= 0:
                start = data.find(b"\x60\x0a", idx)
                if start >= 0:
                    start += 2
                    next_entry = data.find(b"\x60\x0a", data.find(b"data.tar", start) + 8)
                    if next_entry >= 0:
                        with tarfile.open(fileobj=__import__("io").BytesIO(data[start:next_entry])) as tar:
                            for m in tar:
                                if m.name.endswith(".ttc"):
                                    f2 = tar.extractfile(m)
                                    if f2:
                                        with open(font_path, "wb") as out:
                                            out.write(f2.read())
                                    break
            os.remove(deb_path)
        except Exception:
            pass

    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        name = _search()
        if not name:
            # addfont 未加入 ttflist, 直接从文件读名字
            try:
                name = FontProperties(fname=font_path).get_name()
            except Exception:
                pass
        _CJK_FONT = name or ""
        return _CJK_FONT if _CJK_FONT else None

    _CJK_FONT = ""
    return None


def _build_export_image(result: pd.DataFrame) -> bytes | None:
    """用 matplotlib 构建排版优秀的导出图并返回 PNG bytes."""
    import io, os
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.table as tbl
        from matplotlib.font_manager import FontProperties
    except ImportError:
        return None

    # 设置 CJK 字体
    cjk = _cjk_font_name()
    if cjk:
        plt.rcParams["font.sans-serif"] = [cjk, "DejaVu Sans"]
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["axes.unicode_minus"] = False

    total = len(result)
    pos_count = int((result["sentiment_score"] > 0).sum())
    neg_count = total - pos_count
    top_name = result.iloc[0]["board_name"]
    top_score = result.iloc[0]["sentiment_score"]
    bot_name = result.iloc[-1]["board_name"]
    bot_score = result.iloc[-1]["sentiment_score"]

    fig = plt.figure(figsize=(14, 24), facecolor="white")
    gs = fig.add_gridspec(5, 1, height_ratios=[0.07, 0.25, 0.16, 0.42, 0.10], hspace=0.65)

    # ── 标题行 ──
    ax_title = fig.add_subplot(gs[0, 0])
    ax_title.axis("off")
    ax_title.text(
        0.5, 0.3,
        f"板块情绪分析报告  —  {date.today()}",
        ha="center", va="center", fontsize=18, fontweight="bold", color="#1f2937",
        transform=ax_title.transAxes,
    )
    ax_title.text(
        0.5, -0.2,
        f"总数 {total}   ·   正面 {pos_count}   ·   负面 {neg_count}   ·   "
        f"最正面 {top_name} ({top_score:.3f})   ·   最负面 {bot_name} ({bot_score:.3f})",
        ha="center", va="center", fontsize=9, color="#6b7280",
        transform=ax_title.transAxes,
    )

    # ── 散点图 ──
    ax_sc = fig.add_subplot(gs[1, 0])
    sc = result.copy()
    sc["score_abs"] = sc["sentiment_score"].abs()
    n_ex = min(40, max(20, total // 3))
    ex_idx = sc["score_abs"].sort_values(ascending=False).index[:n_ex]
    ex = sc.loc[ex_idx]

    ax_sc.scatter(sc["sentiment_score"], sc["change_pct"],
                  s=8, c="lightgray", alpha=0.3, edgecolors="none", zorder=1)
    pos = ex[ex["sentiment_score"] > 0]
    neg = ex[ex["sentiment_score"] <= 0]
    if not pos.empty:
        ax_sc.scatter(pos["sentiment_score"], pos["change_pct"],
                      s=40, c="#22c55e", edgecolors="white", linewidth=0.5, zorder=2, label="正面")
    if not neg.empty:
        ax_sc.scatter(neg["sentiment_score"], neg["change_pct"],
                      s=40, c="#ef4444", edgecolors="white", linewidth=0.5, zorder=2, label="负面")
    # 标注情绪得分最高/最低各 5 个 — 自动防重叠
    ann_idx = pd.concat([sc.nlargest(5, "sentiment_score"),
                          sc.nsmallest(5, "sentiment_score")]).index
    ann_rows = sc.loc[ann_idx].sort_values("change_pct")
    yvals = ann_rows["change_pct"].values
    offsets = [(0, 8)] * len(yvals)
    for i in range(len(yvals) - 1):
        if abs(yvals[i] - yvals[i + 1]) < 1.5:
            offsets[i] = (0, -14)
            offsets[i + 1] = (0, 8)
    for idx, (_, r) in enumerate(ann_rows.iterrows()):
        ax_sc.annotate(
            r["board_name"],
            (r["sentiment_score"], r["change_pct"]),
            xytext=offsets[idx], textcoords="offset points",
            ha="center", fontsize=7, color="#1f2937",
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.5),
        )
    ax_sc.axhline(y=0, linestyle="--", color="gray", linewidth=0.8)
    ax_sc.axvline(x=0, linestyle="--", color="gray", linewidth=0.8)
    ax_sc.set_xlabel("情绪得分", fontsize=10)
    ax_sc.set_ylabel("涨跌幅 (%)", fontsize=10)
    ax_sc.set_title("情绪 vs 涨跌幅", fontsize=12, fontweight="bold", pad=8)
    ax_sc.legend(fontsize=8, loc="best")
    ax_sc.spines["top"].set_visible(False)
    ax_sc.spines["right"].set_visible(False)

    # ── 直方图 ──
    ax_hist = fig.add_subplot(gs[2, 0])
    bins = min(30, max(10, total // 5))
    ax_hist.hist(result["sentiment_score"], bins=bins, color="#3b82f6", edgecolor="white", linewidth=0.5)
    ax_hist.axvline(x=0, linestyle="--", color="gray", linewidth=0.8)
    ax_hist.set_xlabel("情绪得分", fontsize=10)
    ax_hist.set_ylabel("板块数量", fontsize=10)
    ax_hist.set_title("情绪得分分布", fontsize=12, fontweight="bold", pad=8)
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)

    # ── 排行表 ──
    ax_tbl = fig.add_subplot(gs[3, 0])
    ax_tbl.axis("off")
    ax_tbl.set_title("板块排行 Top / Bottom 20", fontsize=12, fontweight="bold", pad=10)

    # ── 免责声明 ──
    ax_footer = fig.add_subplot(gs[4, 0])
    ax_footer.axis("off")
    ax_footer.text(
        0.02, 0.85,
        "数据来源: 东方财富 (EastMoney) API via AKShare",
        ha="left", va="top", fontsize=7.5, color="#9ca3af",
        transform=ax_footer.transAxes,
    )
    ax_footer.text(
        0.02, 0.55,
        "算法说明: 5 因子加权情绪得分 — 宽度 (30%) · 主力资金 (25%) · 异动 (20%) · 涨跌幅 (15%) · 热度 (10%)",
        ha="left", va="top", fontsize=7.5, color="#9ca3af",
        transform=ax_footer.transAxes,
    )
    ax_footer.text(
        0.02, 0.20,
        "免责声明: 本报告仅供参考学习，不构成任何投资建议。股市有风险，投资需谨慎。",
        ha="left", va="top", fontsize=7.5, color="#9ca3af",
        transform=ax_footer.transAxes,
    )

    n = min(20, total)
    top = result.head(n)[["rank", "board_name", "sentiment_score", "change_pct"]].copy()
    bot = result.tail(n)[["rank", "board_name", "sentiment_score", "change_pct"]].copy()
    data_rows = []
    for _, r in top.iterrows():
        data_rows.append([int(r["rank"]), r["board_name"], f"{r['sentiment_score']:.4f}", f"{r['change_pct']:.2f}%"])
    data_rows.append(["…", "", "", ""])
    for _, r in bot.iterrows():
        data_rows.append([int(r["rank"]), r["board_name"], f"{r['sentiment_score']:.4f}", f"{r['change_pct']:.2f}%"])

    col_labels = ["排名", "板块名称", "情绪得分", "涨跌幅(%)"]
    cell_text = [[str(r[ci]) for ci in range(4)] for r in data_rows]
    row_colors = (["#f0fdf4"] * n + ["#ffffff"] + ["#fef2f2"] * n)

    table = ax_tbl.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        colWidths=[0.08, 0.35, 0.15, 0.15],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1f2937")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor(row_colors[row - 1] if row - 1 < len(row_colors) else "#ffffff")
        cell.set_edgecolor("#e5e7eb")
        cell.set_height(0.024)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


# ── 共用散点图 ───────────────────────────────────────────

def _render_sector_scatter(result: pd.DataFrame):
    """情绪 vs 涨跌幅散点图 (聚焦极端值)."""
    st.subheader("📈 情绪 vs 涨跌幅")
    scatter = result.copy()
    scatter["score_abs"] = scatter["sentiment_score"].abs()

    total = len(scatter)
    n_extreme = min(40, max(20, total // 3))
    n_mid = min(60, total - n_extreme) if total > n_extreme + 30 else 0

    sorted_idx = scatter["score_abs"].sort_values(ascending=False).index
    extreme_idx = sorted_idx[:n_extreme]

    if n_mid > 0:
        mid_idx = sorted_idx[n_extreme:]
        sampled = scatter.loc[mid_idx].sample(n=n_mid, random_state=42)
        plot_idx = extreme_idx.union(sampled.index)
        plot_data = scatter.loc[plot_idx].copy()
        plot_data["is_extreme"] = plot_data.index.isin(extreme_idx)
    else:
        plot_data = scatter.loc[extreme_idx].copy()
        plot_data["is_extreme"] = True

    fig_scatter = go.Figure()
    mid_data = plot_data[~plot_data["is_extreme"]]
    if not mid_data.empty:
        fig_scatter.add_trace(go.Scattergl(
            x=mid_data["sentiment_score"],
            y=mid_data["change_pct"],
            mode="markers",
            marker=dict(size=5, color="rgba(150,150,150,0.35)", line=dict(width=0)),
            showlegend=False, hoverinfo="skip",
        ))
    pos = plot_data[(plot_data["is_extreme"]) & (plot_data["sentiment_score"] > 0)]
    neg = plot_data[(plot_data["is_extreme"]) & (plot_data["sentiment_score"] <= 0)]
    for subset, color, label in [(pos, "#22c55e", "正面"), (neg, "#ef4444", "负面")]:
        if not subset.empty:
            fig_scatter.add_trace(go.Scattergl(
                x=subset["sentiment_score"],
                y=subset["change_pct"],
                mode="markers",
                marker=dict(size=10, color=color, line=dict(width=1, color="white")),
                name=label,
                text=subset["board_name"],
                hovertemplate="<b>%{text}</b><br>情绪: %{x:.3f}<br>涨跌幅: %{y:.2f}%<extra></extra>",
            ))
    # 标注情绪得分最高/最低各 5 个 — 自动防重叠
    ann_pos = scatter.nlargest(5, "sentiment_score")
    ann_neg = scatter.nsmallest(5, "sentiment_score")
    ann_data = pd.concat([ann_pos, ann_neg]).sort_values("change_pct")
    yv = ann_data["change_pct"].values
    positions = ["top center"] * len(yv)
    for i in range(len(yv) - 1):
        if abs(yv[i] - yv[i + 1]) < 1.5:
            positions[i] = "bottom center"
            positions[i + 1] = "top center"
    fig_scatter.add_trace(go.Scattergl(
        x=ann_data["sentiment_score"],
        y=ann_data["change_pct"],
        mode="markers+text",
        marker=dict(size=12, color="rgba(0,0,0,0)", line=dict(width=0)),
        text=ann_data["board_name"],
        textposition=positions,
        textfont=dict(size=10, color="#1f2937"),
        showlegend=False,
        hoverinfo="skip",
    ))
    fig_scatter.add_hline(y=0, line_dash="dot", line_color="gray")
    fig_scatter.add_vline(x=0, line_dash="dot", line_color="gray")
    fig_scatter.update_layout(
        height=400, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="情绪得分", yaxis_title="涨跌幅 (%)",
        hovermode="closest",
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig_scatter, width='stretch')


# ── 下载 (CSV + 导出图片) ──────────────────────────────

def _render_sector_downloads(result: pd.DataFrame):
    """CSV + 一键导出 PNG 按钮."""
    csv = result.to_csv(index=False)
    dc1, dc2 = st.columns(2)
    with dc1:
        st.download_button(
            "📥 下载板块情绪数据 CSV",
            data=csv,
            file_name=f"sector_sentiment_{date.today()}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dc2:
        _render_export_button(result)


def _render_export_button(result: pd.DataFrame):
    """渲染一键导出按钮."""
    import io
    buf = _build_export_image(result)
    if buf is None:
        st.caption("💡 安装 matplotlib 后可导出图片")
        return

    st.download_button(
        "📸 一键导出图片 (PNG)",
        data=buf,
        file_name=f"sector_sentiment_{date.today()}.png",
        mime="image/png",
        use_container_width=True,
    )
    with st.popover("👁️ 预览"):
        st.image(buf, use_container_width=True)


def _render_sector_history():
    """展示历史板块快照."""
    st.subheader("📅 历史板块快照")
    dates = hist.list_sector_snapshot_dates()
    if not dates:
        st.info("尚无历史板块快照，先刷新一次数据")
        return
    selected = st.selectbox("选择日期", dates, key="sector_hist_date")
    df = hist.load_sector_history(selected)
    if df.empty:
        st.warning("该日期无数据")
        return

    st.caption(f"快照时间: {selected}  |  共 {len(df)} 个板块")
    top = df.head(5)
    cols = st.columns(4)
    cols[0].metric("总板块数", f"{len(df)}")
    cols[1].metric("最正面", top.iloc[0]["board_name"],
                   delta=f"得分 {top.iloc[0]['sentiment_score']:.3f}")
    cols[2].metric("最负面", df.iloc[-1]["board_name"],
                   delta=f"得分 {df.iloc[-1]['sentiment_score']:.3f}",
                   delta_color="inverse")
    pos_count = (df["sentiment_score"] > 0).sum()
    cols[3].metric("正面/负面", f"{pos_count}/{len(df) - pos_count}")

    display = df.head(100).copy()
    display["情绪"] = display["sentiment_score"].apply(
        lambda s: f"{'🟢' if s > 0.05 else '🔴' if s < -0.05 else '⚪'} {s:.3f}"
    )
    display["涨跌幅"] = display["change_pct"].apply(lambda v: f"{v:.2f}%")
    display["主力资金"] = display["main_force_net_inflow"].apply(_format_inflow)
    display["排名"] = display["rank"]
    st.dataframe(
        display[["排名", "board_name", "情绪", "涨跌幅", "主力资金"]],
        column_config={
            "rank": "排名",
            "board_name": "板块名称",
            "情绪": "情绪得分",
            "涨跌幅": "涨跌幅",
            "主力资金": "主力资金",
        },
        width='stretch', hide_index=True,
    )

    _render_sector_scatter(df)
    _render_sector_downloads(df)


def _render_sector_dashboard():
    st.subheader("🏭 板块情绪排行")
    st.caption("综合多个数据源计算出每个板块的情绪得分")

    with st.sidebar:
        board_type = st.radio("板块类型", ["概念板块", "行业板块", "全部"],
                              horizontal=True, key="sector_board_type")
        type_map = {"概念板块": "concept", "行业板块": "industry", "全部": "all"}
        top_n = st.selectbox("显示数量", [20, 30, 50, 100, "全部"], index=1, key="sector_top_n")

    col1, col2 = st.sidebar.columns(2)
    if col1.button("🔍 刷新", type="primary", use_container_width=True,
                   key="sector_fetch_btn"):
        st.session_state.sector_error = None
        with st.spinner("正在获取板块数据..."):
            try:
                result = compute_sector_sentiment(board_type=type_map[board_type])
                if result.empty:
                    st.session_state.sector_error = "API 返回空数据，东方财富接口可能暂时不可用"
                else:
                    hist.append_sector_snapshot(result)
                st.session_state.sector_result = result
            except Exception as e:
                st.session_state.sector_result = None
                st.session_state.sector_error = _fmt_error(e)
        st.rerun()

    history_mode = col2.checkbox("历史快照", key="sector_hist_mode")
    if history_mode:
        _render_sector_history()
        return

    result = st.session_state.get("sector_result")
    error = st.session_state.get("sector_error")

    if error:
        st.error(f"⚠️ 数据获取失败")
        st.code(error, language="")
        st.info("👈 点击侧边栏「刷新」重试（东方财富接口有反爬限制，两次操作之间建议间隔 5 秒以上）")
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

    # ── 排行 + 每行 expander ──
    st.subheader("🏆 板块情绪排行")

    show_all = top_n == "全部"
    display = result.copy()
    if not show_all:
        display = display.head(top_n)

    for _, row in display.iterrows():
        cols = st.columns([1, 3, 1.5, 1.5, 1.5, 1.5, 1.5, 1])
        score = row["sentiment_score"]
        emoji = "🟢" if score > 0.05 else "🔴" if score < -0.05 else "⚪"
        cols[0].markdown(f"**{row['rank']:.0f}**")
        cols[1].markdown(f"**{row['board_name']}**")
        cols[2].markdown(f"{emoji} {score:.3f}")
        cols[3].markdown(f"{row['change_pct']:.2f}%")
        cols[4].markdown(_format_inflow(row.get("main_force_net_inflow", 0)))
        adv = row.get("advance", 0)
        dec = row.get("decline", 0)
        cols[5].markdown(f"{adv}/{dec}" if pd.notna(adv) and pd.notna(dec) else "—")
        cols[6].markdown(f"{int(row.get('anomaly_count', 0))}次" if pd.notna(row.get('anomaly_count')) else "—")

        board_code = row.get("board_code", None)
        with cols[7]:
            if board_code:
                expander_key = f"exp_{row['rank']:.0f}_{board_code}"
                with st.expander("📄", expanded=False):
                    _render_sector_hot_posts(board_code, row["board_name"])
            else:
                st.write("")

        st.divider()

    _render_sector_scatter(result)
    _render_sector_downloads(result)


# ── 个股模式 ──────────────────────────────────────────────

def _render_stock_history(symbol: str, src_key: str, source_name: str):
    """展示个股历史情绪数据."""
    st.subheader(f"📜 {symbol} 历史情绪")
    c1, c2 = st.columns(2)
    default_start = (date.today() - timedelta(days=90)).isoformat()
    default_end = date.today().isoformat()
    start = c1.date_input("开始日期", value=pd.to_datetime(default_start), key="sent_hist_start")
    end = c2.date_input("结束日期", value=pd.to_datetime(default_end), key="sent_hist_end")

    if c1.button("📊 查看历史", use_container_width=True, key="sent_hist_btn"):
        with st.spinner("加载历史数据..."):
            daily = hist.load_stock_daily(
                symbol, src_key,
                start=start.isoformat(), end=end.isoformat(),
            )

        if daily.empty:
            st.info("该时间范围内无历史数据")
            return

        st.subheader("📈 历史聚合概览")
        _render_metrics(daily)
        _render_charts(daily)

        raw = hist.load_stock_raw(
            symbol, src_key,
            start=start.isoformat(), end=end.isoformat(),
        )
        if not raw.empty:
            st.subheader(f"📋 历史原始帖子 ({len(raw)} 条)")
            _render_raw_posts(raw, source_name)

        st.caption("💾 数据来自本地持久化存储，无需重新拉取")


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

    view_mode = st.sidebar.radio(
        "视图", ["最新数据", "历史数据"],
        horizontal=True, key="sent_view_mode",
    )

    if view_mode == "历史数据":
        _render_stock_history(symbol, src_key, source_name)
        return

    use_llm = st.sidebar.checkbox("使用 LLM 分析 (需 API key)", value=False, key="sent_llm")

    llm_client = None
    if use_llm:
        api_key = st.sidebar.text_input("DeepSeek API Key", type="password", key="sent_api_key")
        if api_key:
            from data.sentiment.deepseek_client import DeepSeekClient
            llm_client = DeepSeekClient(api_key)

    if st.sidebar.button("🔍 拉取数据", type="primary", use_container_width=True):
        st.session_state.sent_error = None
        with st.spinner(f"正在获取 {symbol} {source_name} 数据..."):
            try:
                source = _SOURCES[src_key]
                daily = source.fetch(
                    symbol,
                    start_date="20200101",
                    end_date=datetime.now().strftime("%Y%m%d"),
                    use_llm=use_llm,
                    llm_client=llm_client,
                )
                raw = source.fetch_raw_posts(symbol) if hasattr(source, "fetch_raw_posts") else pd.DataFrame()

                hist.append_stock_daily(symbol, src_key, daily)
                if not raw.empty:
                    hist.append_stock_raw(symbol, src_key, raw)

                st.session_state.sent_daily = daily
                st.session_state.sent_raw = raw
                st.session_state.sent_fetched_symbol = symbol
                st.session_state.sent_fetched_source = source_name
            except Exception as e:
                st.session_state.sent_error = _fmt_error(e)
        st.rerun()

    sent_error = st.session_state.get("sent_error")
    if sent_error:
        st.error("⚠️ 数据获取失败")
        st.code(sent_error, language="")
        st.info("👈 点击侧边栏「拉取数据」重试（东方财富接口有反爬限制，两项操作之间建议间隔 5 秒以上）")
        if st.button("🔄 重试", use_container_width=True, key="sent_retry_btn"):
            del st.session_state.sent_error
            st.rerun()
        return

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
