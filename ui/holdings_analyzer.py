import io
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

INDEX_KEYWORDS = [
    "指数", "ETF联接", "ETF发起式联接",
    "纳指", "纳斯达克", "标普",
    "中证", "沪深300", "日经225", "红利低波",
    "中证A500",
]


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


def render_holdings_analyzer():
    st.subheader("📂 持仓类型分析")

    uploaded = st.file_uploader(
        "上传基金E账户导出的 Excel 文件",
        type=["xlsx"],
        help="请从基金E账户App导出「投资者公募基金持有信息」Excel 文件后上传",
    )

    if not uploaded:
        st.info("⬆️ 请上传文件开始分析")
        example_path = Path("/home/tianbo/home/各类文件/投资/参考文献/基金E账户App投资者公募基金持有信息-2026-06-08.xlsx")
        if example_path.exists():
            with open(example_path, "rb") as f:
                example_bytes = f.read()
            if st.button("📎 使用示例文件"):
                uploaded = type("_FakeUpload", (), {
                    "name": example_path.name,
                    "read": lambda self=None: example_bytes,
                })()
                st.rerun()

    if not uploaded:
        return

    with st.spinner("正在分析持仓..."):
        raw = uploaded.read() if hasattr(uploaded, "read") else uploaded
        df = parse_holdings(raw)

    if df is None or df.empty:
        return

    total_all = df["资产情况"].sum()
    grp = df.groupby("分类", sort=False).agg(
        总市值=("资产情况", "sum"),
        数量=("基金代码", "nunique"),
    )
    grp["占比"] = grp["总市值"] / total_all

    category_order = ["活期现金", "黄金ETF", "指数基金", "主动基"]
    grp = grp.reindex([c for c in category_order if c in grp.index])

    color_map = {
        "活期现金": "#10b981",
        "黄金ETF": "#f59e0b",
        "指数基金": "#3b82f6",
        "主动基": "#8b5cf6",
    }

    fig = px.pie(
        grp,
        names=grp.index,
        values="总市值",
        color=grp.index,
        color_discrete_map=color_map,
        hole=0.45,
    )
    fig.update_traces(
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>市值: ¥%{value:,.2f}<br>占比: %{percent}",
    )
    fig.update_layout(
        height=400,
        margin=dict(t=0, b=0, l=0, r=0),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    cols = st.columns(4)
    for i, (cat, row) in enumerate(grp.iterrows()):
        with cols[i]:
            st.metric(
                label=f"{cat}",
                value=f"¥{row['总市值']:,.0f}",
                delta=f"{row['占比']:.1%} · {int(row['数量'])}只",
            )

    st.dataframe(
        df.assign(资产情况=df["资产情况"].apply(lambda x: f"¥{x:,.2f}")),
        column_config={
            "基金代码": "代码",
            "基金名称": "名称",
            "资产情况": "资产(元)",
            "分类": "类型",
        },
        use_container_width=True,
        hide_index=True,
    )
