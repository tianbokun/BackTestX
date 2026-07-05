import io
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from core.holdings_store import save_record, load_record, load_meta, list_records

INDEX_KEYWORDS = [
    "指数", "ETF联接", "ETF发起式联接",
    "纳指", "纳斯达克", "标普",
    "中证", "沪深300", "日经225", "红利低波",
    "中证A500",
]

REBALANCE_GROUPS = ["指数基金", "主动基", "黄金+现金"]

REBALANCE_COLORS = {
    "指数基金": "#3b82f6",
    "主动基": "#8b5cf6",
    "黄金+现金": "#f59e0b",
}

MERGED_CATEGORY_MAP = {
    "活期现金": "黄金+现金",
    "黄金ETF": "黄金+现金",
    "指数基金": "指数基金",
    "主动基": "主动基",
}


def classify_fund(name: str) -> str:
    if "货币" in name:
        return "活期现金"
    if "黄金" in name:
        return "黄金ETF"
    for kw in INDEX_KEYWORDS:
        if kw in name:
            return "指数基金"
    return "主动基"


def parse_holdings(file_bytes: bytes) -> pd.DataFrame | None:
    try:
        df = pd.read_excel(io.BytesIO(file_bytes), sheet_name="持有信息", header=None, skiprows=5)
    except Exception as e:
        st.error(f"无法解析 Excel 文件: {e}")
        return None

    if df.shape[1] < 13:
        st.error("Excel 列数不足，请确认文件为基金E账户导出的标准格式")
        return None

    cols = df.iloc[:, [1, 2, 12]].copy()
    cols.columns = ["基金代码", "基金名称", "资产情况"]
    cols = cols.dropna(subset=["基金代码", "基金名称", "资产情况"])
    cols["基金代码"] = cols["基金代码"].astype(int).astype(str)
    cols["资产情况"] = pd.to_numeric(cols["资产情况"], errors="coerce")

    if cols["资产情况"].isna().all():
        st.error("资产情况列无有效数值，请确认列索引是否正确")
        return None

    cols["分类"] = cols["基金名称"].apply(classify_fund)
    return cols


def _render_analysis(df: pd.DataFrame):
    """展示分析结果（饼图 + metric + 明细表），按三大类显示。"""
    total_all = df["资产情况"].sum()
    df["再平衡分类"] = df["分类"].map(MERGED_CATEGORY_MAP)

    grp = df.groupby("再平衡分类", sort=False).agg(
        总市值=("资产情况", "sum"),
        数量=("基金代码", "nunique"),
    )
    grp["占比"] = grp["总市值"] / total_all
    grp = grp.reindex([c for c in REBALANCE_GROUPS if c in grp.index])

    fig = px.pie(
        grp, names=grp.index, values="总市值",
        color=grp.index, color_discrete_map=REBALANCE_COLORS,
        hole=0.45,
    )
    fig.update_traces(
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>市值: ¥%{value:,.2f}<br>占比: %{percent}",
    )
    fig.update_layout(height=350, margin=dict(t=0, b=0, l=0, r=0), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    cols = st.columns(3)
    for i, cat in enumerate(REBALANCE_GROUPS):
        if cat in grp.index:
            row = grp.loc[cat]
            with cols[i]:
                st.metric(label=cat, value=f"¥{row['总市值']:,.0f}",
                          delta=f"{row['占比']:.1%} · {int(row['数量'])}只")
        else:
            with cols[i]:
                st.metric(label=cat, value="¥0", delta="0.0% · 0只")

    st.dataframe(
        df.assign(资产情况=df["资产情况"].apply(lambda x: f"¥{x:,.2f}")),
        column_config={
            "基金代码": "代码",
            "基金名称": "名称",
            "资产情况": "资产(元)",
            "分类": "类型",
            "再平衡分类": "再平衡大类",
        },
        use_container_width=True,
        hide_index=True,
    )


def _render_rebalance(df: pd.DataFrame, total_asset: float):
    """再平衡计算器。"""
    st.subheader("⚖️ 再平衡计算")

    df["再平衡分类"] = df["分类"].map(MERGED_CATEGORY_MAP)
    current = df.groupby("再平衡分类")["资产情况"].sum()

    for c in REBALANCE_GROUPS:
        if c not in current.index:
            current[c] = 0.0

    col1, col2 = st.columns([1, 1])
    with col1:
        idx_pct = st.slider("指数基金 %", 0, 100, value=60, key="reb_idx",
                            help="目标占比")
        act_pct = st.slider("主动基 %", 0, 100 - idx_pct, value=30, key="reb_act",
                            help="目标占比")
        cash_pct = 100 - idx_pct - act_pct
        st.metric("黄金+现金 %", f"{cash_pct}%",
                  help=f"自动计算：100% - {idx_pct}% - {act_pct}% = {cash_pct}%")

    with col2:
        targets = {
            "指数基金": idx_pct / 100,
            "主动基": act_pct / 100,
            "黄金+现金": cash_pct / 100,
        }
        rows = []
        for cat in REBALANCE_GROUPS:
            cur = current.get(cat, 0.0)
            tgt_val = total_asset * targets[cat]
            diff = tgt_val - cur
            rows.append({
                "大类": cat,
                "当前市值": cur,
                "目标市值": tgt_val,
                "差额": diff,
                "操作": "买入" if diff > 0 else ("卖出" if diff < 0 else "—"),
            })
        tbl = pd.DataFrame(rows)
        tbl["当前市值"] = tbl["当前市值"].apply(lambda x: f"¥{x:,.0f}")
        tbl["目标市值"] = tbl["目标市值"].apply(lambda x: f"¥{x:,.0f}")
        tbl["差额"] = tbl["差额"].apply(lambda x: f"¥{x:+,.0f}")
        st.dataframe(tbl, use_container_width=True, hide_index=True)

    fig = go.Figure()
    cats = REBALANCE_GROUPS
    cur_vals = [current.get(c, 0.0) for c in cats]
    tgt_vals = [total_asset * targets[c] for c in cats]
    fig.add_trace(go.Bar(name="当前", x=cats, y=cur_vals,
                         marker_color=[REBALANCE_COLORS[c] for c in cats]))
    fig.add_trace(go.Bar(name="目标", x=cats, y=tgt_vals,
                         marker_color=[REBALANCE_COLORS[c] for c in cats],
                         opacity=0.4))
    fig.update_layout(
        barmode="group", height=300,
        margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(orientation="h", y=1.1),
        yaxis_title="市值 (元)",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_holdings_analyzer():
    st.subheader("📂 持仓类型分析")

    uploaded = st.file_uploader(
        "上传基金E账户导出的 Excel 文件",
        type=["xlsx"],
        help="请从基金E账户App导出「投资者公募基金持有信息」Excel 文件后上传",
    )

    history = list_records()
    history_options = []
    if history:
        history_options = [
            f"{h['timestamp']} — {h['filename']} (¥{h['total_asset']:,.0f})"
            for h in history
        ]

    selected_history = None
    if history_options:
        selected_label = st.selectbox(
            "📋 历史记录", ["(当前上传)"] + history_options,
            key="holdings_history_selector",
        )
        if selected_label and selected_label != "(当前上传)":
            selected_history = selected_label.split(" — ")[0]

    current_df = None
    current_filename = ""

    if selected_history:
        meta = load_meta(selected_history)
        df = load_record(selected_history)
        if df is not None:
            current_df = df
            current_filename = meta["filename"] if meta else selected_history

    if uploaded:
        file_sig = (uploaded.name, uploaded.size)
        if st.session_state.get("holdings_last_saved") != file_sig:
            with st.spinner("正在分析持仓..."):
                raw = uploaded.read()
                df = parse_holdings(raw)
            if df is not None and not df.empty:
                save_record(df, uploaded.name)
                st.session_state.holdings_last_saved = file_sig
                st.success(f"已保存分析记录: {uploaded.name}")
        else:
            df = parse_holdings(uploaded.getvalue())
        if df is not None and not df.empty:
            current_df = df
            current_filename = uploaded.name

    if current_df is None:
        st.info("⬆️ 请上传文件开始分析")
        return

    if current_filename:
        st.caption(f"当前: {current_filename}")

    _render_analysis(current_df)

    total = current_df["资产情况"].sum()
    st.divider()
    _render_rebalance(current_df, total)
