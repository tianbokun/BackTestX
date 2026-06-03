"""情绪数据看板 — 个股/板块双模式."""

from datetime import date, datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.i18n import t

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
    cols[0].metric(t("sentiment.metric.sentiment"), f"{latest['sentiment_score']:.3f}",
                   delta=f"{latest['sentiment_score'] - daily.iloc[-2]['sentiment_score']:.3f}" if len(daily) > 1 else None)
    cols[1].metric(t("sentiment.metric.daily_posts"), f"{latest['post_volume']:.0f}",
                   delta=f"{latest['post_volume'] - daily.iloc[-2]['post_volume']:.0f}" if len(daily) > 1 else None)
    cols[2].metric(t("sentiment.metric.long_short"), f"{latest['bull_bear_ratio']:.2f}")
    cols[3].metric(t("sentiment.metric.divergence"), f"{latest['disagreement']:.3f}")


def _render_charts(daily: pd.DataFrame):
    if daily.empty or len(daily) < 2:
        st.info(t("sentiment.chart.no_data"))
        return

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(t("sentiment.chart.sentiment"), t("sentiment.chart.posts"), t("sentiment.chart.ls_ratio"), t("sentiment.chart.divergence")),
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
    fig.update_yaxes(title_text=t("sentiment.chart.axis.score"), row=1, col=1)
    fig.update_yaxes(title_text=t("sentiment.chart.axis.count"), row=2, col=1)
    fig.update_yaxes(title_text=t("sentiment.chart.axis.ratio"), row=3, col=1)
    fig.update_yaxes(title_text=t("sentiment.chart.axis.divergence"), row=4, col=1)

    st.plotly_chart(fig, width='stretch')


def _render_raw_posts(df: pd.DataFrame, source_name: str):
    if df.empty:
        st.caption(t("sentiment.raw_posts.empty"))
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
            "date": t("sentiment.col.time"),
            "情感": st.column_config.TextColumn("", width="small"),
            "score": t("sentiment.col.score"),
            "标题": st.column_config.TextColumn(t("sentiment.col.title"), width="large"),
            "阅读": t("sentiment.col.reads"),
            "评论": t("sentiment.col.comments"),
        },
        width='stretch', hide_index=True,
    )

    csv = display.to_csv(index=False)
    st.download_button(
        t("sentiment.raw_posts.download", source=source_name),
        data=csv,
        file_name=f"{source_name}_{date.today()}.csv",
        mime="text/csv",
    )


# ── 板块热帖函数 ──────────────────────────────────────────

def _render_sector_hot_posts(board_code: str, board_name: str):
    """在 expander 内渲染板块成分股的热门帖子."""
    if not board_code:
        st.caption(t("sentiment.hot_posts.no_code"))
        return

    cache_key = f"posts_{board_code}"
    now = datetime.now()
    cached = st.session_state.get(f"_hot_posts_{board_code}")
    if cached is not None:
        data, ts = cached
        if now - ts < _HOT_CACHE_TTL:
            _display_hot_posts(data, board_name)
            return

    with st.spinner(t("sentiment.hot_posts.fetching", sym=board_name)):
        constituents = fetch_board_constituents(board_code, top_n=5)

    if constituents.empty:
        st.caption(t("sentiment.hot_posts.no_constituents"))
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
            progress_text.caption(t("sentiment.hot_posts.fetching", sym=sym))
            res = f.result()
            if res is not None:
                results.append(res)

    if not results:
        st.caption(t("sentiment.hot_posts.empty"))
        return

    try:
        combined = pd.concat(results, ignore_index=True)
    except Exception:
        st.caption(t("sentiment.hot_posts.error"))
        return
    st.session_state[f"_hot_posts_{board_code}"] = (combined, now)
    _display_hot_posts(combined, board_name)


def _display_hot_posts(combined: pd.DataFrame, board_name: str):
    """展示板块热帖 (按股票分组)."""
    st.caption(t("sentiment.hot_posts.count", n=len(combined)))
    for stock_symbol, group in combined.groupby("stock_symbol"):
        name = group.iloc[0].get("stock_name", stock_symbol)
        change = group.iloc[0].get("stock_change", None)
        label = f"{name} ({stock_symbol})"
        if pd.notna(change):
            label += f"  {change:+.2f}%"
        with st.expander(label):
            _render_raw_posts(group.reset_index(drop=True), t("sentiment.stock.source.guba"))


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


# ── Export image helpers ──────────────────────────────────

def _export_prepare_cjk():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.font_manager import FontProperties
    except ImportError:
        return False
    cjk = _cjk_font_name()
    if cjk:
        plt.rcParams["font.sans-serif"] = [cjk, "DejaVu Sans"]
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["axes.unicode_minus"] = False
    return True


def _export_title_ax(ax, result, cfg):
    c, fs = cfg["colors"], cfg["font_sizes"]
    total = len(result)
    pos_count = int((result["sentiment_score"] > 0).sum())
    neg_count = total - pos_count
    top_name = result.iloc[0]["board_name"]
    top_score = result.iloc[0]["sentiment_score"]
    bot_name = result.iloc[-1]["board_name"]
    bot_score = result.iloc[-1]["sentiment_score"]

    ax.axis("off")
    ax.text(0.5, 0.3, f"板块情绪分析报告  —  {date.today()}",
            ha="center", va="center", fontsize=fs["title"], fontweight="bold", color=c["title"],
            transform=ax.transAxes)

    if cfg.get("show_kpi_cards", False):
        kpi_y = -0.1
        ax.text(0.12, kpi_y, f"+{pos_count}", ha="center", va="center",
                fontsize=fs["subtitle"], fontweight="bold", color=c["bg"],
                bbox=dict(boxstyle="round,pad=0.3", facecolor=c["pos"], edgecolor="none"),
                transform=ax.transAxes)
        ax.text(0.28, kpi_y, t("sentiment.sector.label.positive"), ha="left", va="center",
                fontsize=fs["subtitle"] - 1, color=c["subtitle"], transform=ax.transAxes)
        ax.text(0.47, kpi_y, f"–{neg_count}", ha="center", va="center",
                fontsize=fs["subtitle"], fontweight="bold", color=c["bg"],
                bbox=dict(boxstyle="round,pad=0.3", facecolor=c["neg"], edgecolor="none"),
                transform=ax.transAxes)
        ax.text(0.63, kpi_y, t("sentiment.sector.label.negative"), ha="left", va="center",
                fontsize=fs["subtitle"] - 1, color=c["subtitle"], transform=ax.transAxes)
        ax.text(0.85, kpi_y, f"最高  {top_name}  ({top_score:.3f})", ha="center", va="center",
                fontsize=fs["subtitle"], color=c["subtitle"], transform=ax.transAxes)
    else:
        ax.text(0.5, -0.2,
                f"{t('sentiment.metric.total')} {total}   ·   {t('sentiment.sector.label.positive')} {pos_count}   ·   {t('sentiment.sector.label.negative')} {neg_count}   ·   "
                f"{t('sentiment.metric.most_positive')} {top_name} ({top_score:.3f})   ·   {t('sentiment.metric.most_negative')} {bot_name} ({bot_score:.3f})",
                ha="center", va="center", fontsize=fs["subtitle"], color=c["subtitle"],
                transform=ax.transAxes)

    if cfg.get("decorations") == "double_line":
        ax.axhline(y=-0.4, xmin=0.05, xmax=0.95, color=c["grid"], linewidth=0.5)
        ax.axhline(y=-0.42, xmin=0.05, xmax=0.95, color=c["grid"], linewidth=0.5)
    elif cfg.get("decorations") == "header_bar":
        ax.axhline(y=-0.35, xmin=0, xmax=1, color=c["table_header_bg"], linewidth=3)


def _export_scatter_ax(ax, result, cfg):
    c, fs = cfg["colors"], cfg["font_sizes"]
    sc = result.copy()
    sc["score_abs"] = sc["sentiment_score"].abs()
    n_ex = min(40, max(20, len(result) // 3))
    ex_idx = sc["score_abs"].sort_values(ascending=False).index[:n_ex]
    ex = sc.loc[ex_idx]

    ax.scatter(sc["sentiment_score"], sc["change_pct"],
               s=8, c=c["grid"], alpha=0.3, edgecolors="none", zorder=1)
    pos = ex[ex["sentiment_score"] > 0]
    neg = ex[ex["sentiment_score"] <= 0]
    if not pos.empty:
        ax.scatter(pos["sentiment_score"], pos["change_pct"],
                   s=40, c=c["pos"], edgecolors="white", linewidth=0.5, zorder=2, label=t("sentiment.sector.label.positive"))
    if not neg.empty:
        ax.scatter(neg["sentiment_score"], neg["change_pct"],
                   s=40, c=c["neg"], edgecolors="white", linewidth=0.5, zorder=2, label=t("sentiment.sector.label.negative"))

    if cfg.get("show_annotations", True):
        ann_n = cfg.get("annotations_n", 5)
        ann_idx = pd.concat([sc.nlargest(ann_n, "sentiment_score"),
                              sc.nsmallest(ann_n, "sentiment_score")]).index
        ann_rows = sc.loc[ann_idx].sort_values("change_pct")
        yvals = ann_rows["change_pct"].values
        offsets = [(0, 8)] * len(yvals)
        for i in range(len(yvals) - 1):
            if abs(yvals[i] - yvals[i + 1]) < 1.5:
                offsets[i] = (0, -14)
                offsets[i + 1] = (0, 8)
        for idx, (_, r) in enumerate(ann_rows.iterrows()):
            ax.annotate(r["board_name"],
                        (r["sentiment_score"], r["change_pct"]),
                        xytext=offsets[idx], textcoords="offset points",
                        ha="center", fontsize=fs["annotation"], color=c["annotation"],
                        arrowprops=dict(arrowstyle="->", color=c["grid"], lw=0.5))

    ax.axhline(y=0, linestyle="--", color=c["grid"], linewidth=0.8)
    ax.axvline(x=0, linestyle="--", color=c["grid"], linewidth=0.8)
    ax.set_xlabel(t("sentiment.sector.axis.sentiment"), fontsize=fs["axis"])
    ax.set_ylabel(t("sentiment.sector.axis.change"), fontsize=fs["axis"])
    ax.set_title(t("sentiment.sector.scatter"), fontsize=fs["axis"] + 2, fontweight="bold", pad=8)
    ax.legend(fontsize=fs["axis"] - 2, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor(c["bg"])


def _export_histogram_ax(ax, result, cfg):
    c, fs = cfg["colors"], cfg["font_sizes"]
    total = len(result)
    bins = min(30, max(10, total // 5))
    ax.hist(result["sentiment_score"], bins=bins, color=c["hist"], edgecolor=c["bg"], linewidth=0.5)
    ax.axvline(x=0, linestyle="--", color=c["grid"], linewidth=0.8)
    ax.set_xlabel(t("sentiment.sector.axis.sentiment"), fontsize=fs["axis"])
    ax.set_ylabel(t("sentiment.chart.axis.count"), fontsize=fs["axis"])
    ax.set_title(t("sentiment.sector.distribution"), fontsize=fs["axis"] + 2, fontweight="bold", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor(c["bg"])


def _export_table_ax(ax, result, cfg):
    c, fs = cfg["colors"], cfg["font_sizes"]
    total = len(result)
    n = min(cfg["table_n"], total)
    ax.axis("off")
    ax.set_title(f"板块排行 Top / Bottom {n}", fontsize=fs["axis"] + 2, fontweight="bold", pad=10)

    top = result.head(n)[["rank", "board_name", "sentiment_score", "change_pct"]].copy()
    bot = result.tail(n)[["rank", "board_name", "sentiment_score", "change_pct"]].copy()
    data_rows = []
    for _, r in top.iterrows():
        data_rows.append([int(r["rank"]), r["board_name"], f"{r['sentiment_score']:.4f}", f"{r['change_pct']:.2f}%"])
    data_rows.append(["…", "", "", ""])
    for _, r in bot.iterrows():
        data_rows.append([int(r["rank"]), r["board_name"], f"{r['sentiment_score']:.4f}", f"{r['change_pct']:.2f}%"])

    col_labels = [t("sentiment.col.rank"), t("sentiment.col.board_name"), t("sentiment.sector.axis.sentiment"), t("sentiment.sector.axis.change")]
    cell_text = [[str(r[ci]) for ci in range(4)] for r in data_rows]
    row_colors = ([c["table_pos_bg"]] * n + [c["bg"]] + [c["table_neg_bg"]] * n)

    table = ax.table(
        cellText=cell_text, colLabels=col_labels,
        cellLoc="center", loc="center",
        colWidths=[0.08, 0.35, 0.15, 0.15],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fs["table"])
    total_rows = n * 2 + 2
    row_height = 1.0 / total_rows
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor(c["table_header_bg"])
            cell.set_text_props(color=c["table_header_text"], fontweight="bold")
        else:
            idx = row - 1
            cell.set_facecolor(row_colors[idx] if idx < len(row_colors) else c["bg"])
        cell.set_edgecolor(c["table_border"])
        cell.set_height(row_height * 0.95)


def _export_footer_ax(ax, cfg):
    c, fs = cfg["colors"], cfg["font_sizes"]
    ax.axis("off")
    for y, t in [
        (0.85, "数据来源: 东方财富 (EastMoney) API via AKShare"),
        (0.55, "算法说明: 5 因子加权情绪得分 — 宽度 (30%) · 主力资金 (25%) · 异动 (20%) · 涨跌幅 (15%) · 热度 (10%)"),
        (0.20, "免责声明: 本报告仅供参考学习，不构成任何投资建议。股市有风险，投资需谨慎。"),
    ]:
        ax.text(0.02, y, t, ha="left", va="top", fontsize=fs["footer"], color=c["footer"],
                transform=ax.transAxes)


# ── Layout variants ─────────────────────────────────────

def _render_column(result, cfg):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=cfg["figsize"], facecolor=cfg["colors"]["bg"])
    has_hist = cfg.get("show_histogram", True)
    if has_hist:
        nrows, ratios = 5, cfg["height_ratios"] if len(cfg["height_ratios"]) >= 5 else [0.07, 0.25, 0.16, 0.42, 0.10]
    else:
        nrows, ratios = 4, [0.08, 0.30, 0.48, 0.14]
    gs = fig.add_gridspec(nrows, 1, height_ratios=ratios, hspace=cfg.get("hspace", 0.65))
    _export_title_ax(fig.add_subplot(gs[0, 0]), result, cfg)
    _export_scatter_ax(fig.add_subplot(gs[1, 0]), result, cfg)
    if has_hist:
        _export_histogram_ax(fig.add_subplot(gs[2, 0]), result, cfg)
        _export_table_ax(fig.add_subplot(gs[3, 0]), result, cfg)
        _export_footer_ax(fig.add_subplot(gs[4, 0]), cfg)
    else:
        _export_table_ax(fig.add_subplot(gs[2, 0]), result, cfg)
        _export_footer_ax(fig.add_subplot(gs[3, 0]), cfg)
    return fig


def _render_column_compact(result, cfg):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=cfg["figsize"], facecolor=cfg["colors"]["bg"])
    gs = fig.add_gridspec(4, 1, height_ratios=cfg["height_ratios"], hspace=cfg.get("hspace", 0.55))
    _export_title_ax(fig.add_subplot(gs[0, 0]), result, cfg)
    _export_scatter_ax(fig.add_subplot(gs[1, 0]), result, cfg)
    _export_table_ax(fig.add_subplot(gs[2, 0]), result, cfg)
    _export_footer_ax(fig.add_subplot(gs[3, 0]), cfg)
    return fig


def _render_grid(result, cfg):
    import matplotlib.pyplot as plt
    c, fs = cfg["colors"], cfg["font_sizes"]
    fig = plt.figure(figsize=cfg["figsize"], facecolor=c["bg"])
    gs = fig.add_gridspec(4, 2, height_ratios=cfg["height_ratios"], hspace=cfg.get("hspace", 0.50),
                           width_ratios=[0.5, 0.5])
    _export_title_ax(fig.add_subplot(gs[0, :]), result, cfg)
    _export_scatter_ax(fig.add_subplot(gs[1, 0]), result, cfg)
    _export_histogram_ax(fig.add_subplot(gs[1, 1]), result, cfg)
    _export_table_ax(fig.add_subplot(gs[2, 1]), result, cfg)
    ax_kpi = fig.add_subplot(gs[2, 0])
    ax_kpi.axis("off")
    total = len(result)
    pos_count = int((result["sentiment_score"] > 0).sum())
    neg_count = total - pos_count
    mean_score = result["sentiment_score"].mean()
    top_name = result.iloc[0]["board_name"]
    bot_name = result.iloc[-1]["board_name"]
    for i, (label, value, color) in enumerate([
        (t("sentiment.metric.total"), str(total), c["title"]),
        ("平均情绪", f"{mean_score:.4f}", c["pos"] if mean_score > 0 else c["neg"]),
        (t("sentiment.metric.ratio"), f"{pos_count}/{neg_count}", c["subtitle"]),
        (t("sentiment.metric.most_positive"), top_name, c["pos"]),
        (t("sentiment.metric.most_negative"), bot_name, c["neg"]),
    ]):
        y_pos = 0.85 - i * 0.17
        ax_kpi.text(0.1, y_pos, label, fontsize=fs["axis"] - 1, color=c["subtitle"],
                     transform=ax_kpi.transAxes, va="center")
        ax_kpi.text(0.55, y_pos, value, fontsize=fs["axis"] + 2, color=color,
                     fontweight="bold", transform=ax_kpi.transAxes, va="center")
    _export_footer_ax(fig.add_subplot(gs[3, :]), cfg)
    return fig


# ── Main entry ──────────────────────────────────────────

def _build_export_image(
    result: pd.DataFrame,
    template: str = "minimal",
    preview: bool = False,
    **overrides,
) -> bytes | None:
    from data.sentiment.templates import resolve_template

    if not _export_prepare_cjk():
        return None
    import io
    import matplotlib.pyplot as plt

    cfg = resolve_template(template, overrides, preview=preview)
    layout = cfg.get("layout", "column")

    if layout == "column_compact":
        fig = _render_column_compact(result, cfg)
    elif layout == "grid":
        fig = _render_grid(result, cfg)
    else:
        fig = _render_column(result, cfg)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=cfg["_dpi"], bbox_inches="tight", facecolor=cfg["colors"]["bg"])
    plt.close(fig)
    return buf.getvalue()


# ── 共用散点图 ───────────────────────────────────────────

def _render_sector_scatter(result: pd.DataFrame):
    """情绪 vs 涨跌幅散点图 (聚焦极端值)."""
    st.subheader(t("sentiment.sector.scatter"))
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
    for subset, color, label in [(pos, "#22c55e", t("sentiment.sector.label.positive")), (neg, "#ef4444", t("sentiment.sector.label.negative"))]:
        if not subset.empty:
            fig_scatter.add_trace(go.Scattergl(
                x=subset["sentiment_score"],
                y=subset["change_pct"],
                mode="markers",
                marker=dict(size=10, color=color, line=dict(width=1, color="white")),
                name=label,
                text=subset["board_name"],
                hovertemplate=f"<b>%{{text}}</b><br>{t('sentiment.sector.axis.sentiment')}: %{{x:.3f}}<br>{t('sentiment.sector.axis.change')}: %{{y:.2f}}%<extra></extra>",
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
        xaxis_title=t("sentiment.sector.axis.sentiment"), yaxis_title=t("sentiment.sector.axis.change"),
        hovermode="closest",
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig_scatter, width='stretch')


# ── 下载 (CSV + 导出图片) ──────────────────────────────

def _render_sector_downloads(result: pd.DataFrame):
    """CSV + 导出图片编辑器."""
    csv = result.to_csv(index=False)
    st.download_button(
        t("sentiment.export.btn_csv"),
        data=csv,
        file_name=f"sector_sentiment_{date.today()}.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.divider()
    _render_export_editor(result)


def _render_export_editor(result: pd.DataFrame):
    """导出图片编辑器: 模板选择 + 自定义 + 实时预览 + 下载."""
    from data.sentiment.templates import template_default_overrides

    if "export_template" not in st.session_state:
        st.session_state.export_template = "minimal"
    if "export_overrides" not in st.session_state:
        st.session_state.export_overrides = {}
    if "export_preview" not in st.session_state:
        st.session_state.export_preview = None
    if "export_full" not in st.session_state:
        st.session_state.export_full = None

    st.markdown(t("sentiment.export.title"))

    TEMPLATE_KEYS = ["minimal", "dark", "warm", "xiaohongshu", "dashboard"]
    default_idx = TEMPLATE_KEYS.index(st.session_state.export_template) if st.session_state.export_template in TEMPLATE_KEYS else 0
    template_name = st.selectbox(
        t("sentiment.export.template"),
        options=TEMPLATE_KEYS,
        format_func=lambda x: t("template." + x),
        index=default_idx,
        key="export_template_sel",
    )

    with st.expander(t("sentiment.export.customize"), expanded=False):
        defaults = template_default_overrides(template_name)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(t("sentiment.export.color_section"))
            bg = st.color_picker(t("sentiment.export.bg"), value=defaults.get("bg", "#ffffff"))
            title_c = st.color_picker(t("sentiment.export.title_color"), value=defaults.get("title", "#1f2937"))
            pos_c = st.color_picker(t("sentiment.export.pos_color"), value=defaults.get("pos", "#22c55e"))
            neg_c = st.color_picker(t("sentiment.export.neg_color"), value=defaults.get("neg", "#ef4444"))
            hist_c = st.color_picker(t("sentiment.export.hist_color"), value=defaults.get("hist", "#3b82f6"))
            tbl_header = st.color_picker(t("sentiment.export.table_header"), value=defaults.get("table_header_bg", "#1f2937"))
        with c2:
            st.markdown(t("sentiment.export.layout_section"))
            title_size = st.slider(t("sentiment.export.title_size"), 12, 28, value=defaults.get("title_size", 18), key="es_title_sz")
            table_size = st.slider(t("sentiment.export.table_size"), 5, 12, value=defaults.get("table", 9), key="es_tbl_sz")
            ann_size = st.slider(t("sentiment.export.ann_size"), 5, 12, value=defaults.get("annotation", 7), key="es_ann_sz")
            table_n = st.selectbox(t("sentiment.export.table_n"), [5, 10, 20],
                                   index=[5, 10, 20].index(
                                       defaults.get("table_n", 20) if defaults.get("table_n") in [5, 10, 20] else 2),
                                   key="es_table_n")
            show_hist = st.checkbox(t("sentiment.export.show_hist"), value=defaults.get("show_histogram", True), key="es_hist")
            show_kpi = st.checkbox(t("sentiment.export.show_kpi"), value=defaults.get("show_kpi_cards", False), key="es_kpi")

        st.caption(t("sentiment.export.hint"))
        col_a, col_b = st.columns([1, 3])
        with col_a:
            if st.button(t("sentiment.export.apply"), type="primary", use_container_width=True, key="es_apply"):
                overrides = {
                    "bg": bg, "title": title_c, "pos": pos_c, "neg": neg_c,
                    "hist": hist_c, "table_header_bg": tbl_header,
                    "title_size": title_size, "table": table_size, "annotation": ann_size,
                    "table_n": int(table_n),
                    "show_histogram": show_hist,
                    "show_kpi_cards": show_kpi,
                }
                st.session_state.export_template = template_name
                st.session_state.export_overrides = overrides
                st.session_state.export_preview = _build_export_image(
                    result, template=template_name, preview=True, **overrides)
                st.session_state.export_full = _build_export_image(
                    result, template=template_name, preview=False, **overrides)
                st.rerun()

    changed = st.session_state.export_template != template_name
    if st.session_state.export_preview is not None and not changed:
        st.image(st.session_state.export_preview, use_container_width=True)
    else:
        with st.spinner(t("sentiment.export.generating")):
            preview = _build_export_image(result, template=template_name, preview=True)
            full = _build_export_image(result, template=template_name, preview=False)
            st.session_state.export_template = template_name
            st.session_state.export_preview = preview
            st.session_state.export_full = full
        st.image(preview, use_container_width=True)

    if st.session_state.export_full is not None:
        st.download_button(
            t("sentiment.export.btn_png"),
            data=st.session_state.export_full,
            file_name=f"sector_sentiment_{date.today()}.png",
            mime="image/png",
            use_container_width=True,
        )


def _render_sector_history():
    """展示历史板块快照."""
    st.subheader(t("sentiment.sector.history.title"))
    dates = hist.list_sector_snapshot_dates()
    if not dates:
        st.info(t("sentiment.sector.history.empty"))
        return
    selected = st.selectbox(t("sentiment.sector.history.select"), dates, key="sector_hist_date")
    df = hist.load_sector_history(selected)
    if df.empty:
        st.warning(t("sentiment.sector.history.no_data"))
        return

    st.caption(t("sentiment.sector.history.caption", date=selected, n=len(df)))
    top = df.head(5)
    cols = st.columns(4)
    cols[0].metric(t("sentiment.metric.total"), f"{len(df)}")
    cols[1].metric(t("sentiment.metric.most_positive"), top.iloc[0]["board_name"],
                   delta=f"{t('sentiment.col.score')} {top.iloc[0]['sentiment_score']:.3f}")
    cols[2].metric(t("sentiment.metric.most_negative"), df.iloc[-1]["board_name"],
                   delta=f"{t('sentiment.col.score')} {df.iloc[-1]['sentiment_score']:.3f}",
                   delta_color="inverse")
    pos_count = (df["sentiment_score"] > 0).sum()
    cols[3].metric(t("sentiment.metric.ratio"), f"{pos_count}/{len(df) - pos_count}")

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
            "rank": t("sentiment.col.rank"),
            "board_name": t("sentiment.col.board_name"),
            "情绪": t("sentiment.col.sentiment"),
            "涨跌幅": t("sentiment.col.change"),
            "主力资金": t("sentiment.col.main_force"),
        },
        width='stretch', hide_index=True,
    )

    _render_sector_scatter(df)
    _render_sector_downloads(df)


def _render_sector_dashboard():
    st.subheader(t("sentiment.sector.title"))
    st.caption(t("sentiment.sector.desc"))

    with st.sidebar:
        BOARD_TYPE_KEYS = ["concept", "industry", "all"]
        board_type = st.radio(t("sentiment.sector.board_type"), BOARD_TYPE_KEYS,
                              format_func=lambda x: t("sentiment.sector.type." + x),
                              horizontal=True, key="sector_board_type")
        type_map = {"concept": "concept", "industry": "industry", "all": "all"}
        top_n_options = [20, 30, 50, 100, "all"]
        top_n = st.selectbox(t("sentiment.sector.show_count"), top_n_options,
                             format_func=lambda x: t("sentiment.sector.type.all") if x == "all" else str(x),
                             index=1, key="sector_top_n")

    col1, col2 = st.sidebar.columns(2)
    if col1.button(t("sentiment.sector.btn_refresh"), type="primary", use_container_width=True,
                   key="sector_fetch_btn"):
        st.session_state.sector_error = None
        with st.spinner("正在获取板块数据..."):
            try:
                result = compute_sector_sentiment(board_type=type_map[board_type])
                if result.empty:
                    st.session_state.sector_error = _fmt_error(Exception("API 返回空数据，东方财富接口可能暂时不可用"))
                else:
                    hist.append_sector_snapshot(result)
                st.session_state.sector_result = result
            except Exception as e:
                st.session_state.sector_result = None
                st.session_state.sector_error = _fmt_error(e)
        st.rerun()

    history_mode = col2.checkbox(t("sentiment.sector.history_mode"), key="sector_hist_mode")
    if history_mode:
        _render_sector_history()
        return

    result = st.session_state.get("sector_result")
    error = st.session_state.get("sector_error")

    if error:
        st.error(f"{t('sentiment.sector.error.fetch_failed')}")
        st.code(error, language="")
        st.info(t("sentiment.sector.info.refresh"))
        if st.button(t("sentiment.sector.btn_retry"), use_container_width=True):
            del st.session_state.sector_error
            st.rerun()
        return

    if result is None or result.empty:
        st.info(t("sentiment.sector.info.no_data"))
        return

    # ── 概览指标 ──
    top = result.head(5)
    cols = st.columns(4)
    cols[0].metric(t("sentiment.metric.total"), f"{len(result)}")
    cols[1].metric(t("sentiment.metric.most_positive"), top.iloc[0]["board_name"],
                   delta=f"{t('sentiment.col.score')} {top.iloc[0]['sentiment_score']:.3f}")
    cols[2].metric(t("sentiment.metric.most_negative"), result.iloc[-1]["board_name"],
                   delta=f"{t('sentiment.col.score')} {result.iloc[-1]['sentiment_score']:.3f}",
                   delta_color="inverse")
    pos_count = (result["sentiment_score"] > 0).sum()
    cols[3].metric(t("sentiment.metric.ratio"), f"{pos_count}/{len(result) - pos_count}")

    # ── 分布直方图 ──
    st.subheader(t("sentiment.sector.distribution"))
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
        xaxis_title=t("sentiment.sector.axis.sentiment"), yaxis_title=t("sentiment.sector.axis.change"),
        hovermode="x",
    )
    st.plotly_chart(fig_hist, width='stretch')

    # ── 排行 + 每行 expander ──
    st.subheader(t("sentiment.sector.ranking"))

    show_all = top_n == "all"
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
    st.subheader(t("sentiment.stock.history.title", symbol=symbol))
    c1, c2 = st.columns(2)
    default_start = (date.today() - timedelta(days=90)).isoformat()
    default_end = date.today().isoformat()
    start = c1.date_input(t("sentiment.stock.history.start"), value=pd.to_datetime(default_start), key="sent_hist_start")
    end = c2.date_input(t("sentiment.stock.history.end"), value=pd.to_datetime(default_end), key="sent_hist_end")

    if c1.button(t("sentiment.stock.history.btn"), use_container_width=True, key="sent_hist_btn"):
        with st.spinner(t("sentiment.stock.history.loading")):
            daily = hist.load_stock_daily(
                symbol, src_key,
                start=start.isoformat(), end=end.isoformat(),
            )

        if daily.empty:
            st.info(t("sentiment.stock.history.empty"))
            return

        st.subheader(t("sentiment.stock.history.overview"))
        _render_metrics(daily)
        _render_charts(daily)

        raw = hist.load_stock_raw(
            symbol, src_key,
            start=start.isoformat(), end=end.isoformat(),
        )
        if not raw.empty:
            st.subheader(t("sentiment.stock.history.posts", n=len(raw)))
            _render_raw_posts(raw, source_name)

        st.caption(t("sentiment.stock.history.local"))


def _render_stock_dashboard():
    all_symbols = SymbolRegistry.list()
    if not all_symbols:
        st.error(t("sentiment.stock.error.no_symbols"))
        return

    type_filter = st.sidebar.selectbox(
        t("sentiment.sector.board_type"),
        [t("status.all")] + sorted(set(s["asset_type"] for s in all_symbols)),
        key="sent_type",
    )
    all_text = t("status.all")
    filtered = all_symbols if type_filter == all_text else [s for s in all_symbols if s["asset_type"] == type_filter]
    symbol_options = {f"{s['symbol']} - {s['name']}": s["symbol"] for s in filtered}
    selected_label = st.sidebar.selectbox(t("rl.sidebar.symbol_select"), list(symbol_options.keys()), key="sent_symbol")
    symbol = symbol_options[selected_label]

    source_name = st.sidebar.radio(t("sentiment.stock.data_source"),
                                   [t("sentiment.stock.source.guba"), t("sentiment.stock.source.news")],
                                   horizontal=True, key="sent_source")
    guba_text = t("sentiment.stock.source.guba")
    src_key = "guba" if source_name == guba_text else "news"

    view_mode = st.sidebar.radio(
        t("sentiment.stock.view"),
        [t("sentiment.stock.view.latest"), t("sentiment.stock.view.history")],
        horizontal=True, key="sent_view_mode",
    )

    if view_mode == t("sentiment.stock.view.history"):
        _render_stock_history(symbol, src_key, source_name)
        return

    use_llm = st.sidebar.checkbox(t("sentiment.stock.use_llm"), value=False, key="sent_llm")

    llm_client = None
    if use_llm:
        api_key = st.sidebar.text_input(t("sentiment.stock.deepseek_key"), type="password", key="sent_api_key")
        if api_key:
            from data.sentiment.deepseek_client import DeepSeekClient
            llm_client = DeepSeekClient(api_key)

    if st.sidebar.button(t("sentiment.stock.btn_fetch"), type="primary", use_container_width=True):
        st.session_state.sent_error = None
        with st.spinner(t("sentiment.hot_posts.fetching", sym=f"{symbol} {source_name}")):
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
        st.error(t("sentiment.stock.error.fetch_failed"))
        st.code(sent_error, language="")
        st.info(t("sentiment.stock.info.retry"))
        if st.button(t("sentiment.stock.btn_retry"), use_container_width=True, key="sent_retry_btn"):
            del st.session_state.sent_error
            st.rerun()
        return

    daily = st.session_state.get("sent_daily")
    raw = st.session_state.get("sent_raw")

    if daily is None:
        st.info(t("sentiment.stock.info.waiting"))
        return

    if st.session_state.get("sent_fetched_symbol") != symbol:
        st.info(t("sentiment.stock.info.switched"))
        return

    st.subheader(t("sentiment.stock.overview"))
    _render_metrics(pd.DataFrame(daily))

    st.subheader(t("sentiment.stock.trend"))
    _render_charts(pd.DataFrame(daily))

    st.subheader(t("sentiment.stock.raw_posts", source=source_name, n=len(raw)))
    _render_raw_posts(pd.DataFrame(raw), source_name)


# ── 入口 ──────────────────────────────────────────────────

def render_sentiment_dashboard():
    st.title(t("sentiment.title"))
    st.caption(t("sentiment.caption"))

    mode = st.radio(
        "模式", [t("sentiment.mode.stock"), t("sentiment.mode.sector")],
        horizontal=True, label_visibility="collapsed",
        key="sent_mode",
    )

    if mode == t("sentiment.mode.sector"):
        _render_sector_dashboard()
    else:
        _render_stock_dashboard()
