from datetime import date

import streamlit as st
import pandas as pd

from data.symbol_registry import SymbolRegistry, migrate


def _run_migration():
    added = migrate()
    if added:
        st.toast(f"已导入 {added} 个默认 ETF 到代码注册表")


def render_symbol_manager():
    _run_migration()

    st.title("📋 代码管理")
    st.markdown("""
    <div style="background:#f0f4ff;border-radius:10px;padding:1rem 1.25rem;margin-bottom:1.5rem;font-size:0.9rem;color:#1e293b">
    统一管理所有交易标的代码。添加后的代码可在强化学习中以勾选方式选择使用。<br>
    数据优先使用本地缓存 (<code>cache/</code>)，仅当日期范围超出缓存覆盖时才重新下载。
    </div>
    """, unsafe_allow_html=True)

    all_tags = set()
    for s in SymbolRegistry.list():
        for t in s.get("tags", []):
            all_tags.add(t)
    all_tags = sorted(all_tags)

    col_filter, col_add = st.columns([2, 1])
    with col_filter:
        filter_type = st.selectbox("资产类型过滤", ["全部", "stock", "etf", "lof", "open_fund", "index"])
        filter_tag = st.selectbox("标签过滤", ["全部"] + all_tags)

    with col_add:
        st.markdown("")
        st.markdown("")
        add_btn = st.button("➕ 添加代码", type="primary", use_container_width=True)

    symbols = SymbolRegistry.list()
    if filter_type != "全部":
        symbols = [s for s in symbols if s["asset_type"] == filter_type]
    if filter_tag != "全部":
        symbols = [s for s in symbols if filter_tag in s.get("tags", [])]

    st.markdown("#### 已注册代码")
    cols_header = st.columns([0.5, 2.5, 1, 1.2, 1.2, 1.8, 0.8])
    cols_header[0].checkbox("选", key="sel_all", label_visibility="collapsed")
    cols_header[1].markdown("**代码 / 名称**")
    cols_header[2].markdown("**类型**")
    cols_header[3].markdown("**开始日期**")
    cols_header[4].markdown("**数据状态**")
    cols_header[5].markdown("**标签**")
    cols_header[6].markdown("**操作**")

    selected = []
    for s in symbols:
        cache_info = SymbolRegistry.get_cache_info(s["symbol"])
        with st.container(border=True):
            cols = st.columns([0.5, 2.5, 1, 1.2, 1.2, 1.8, 0.8])
            checked = cols[0].checkbox("", key=f"sel_{s['symbol']}", label_visibility="collapsed")
            if checked:
                selected.append(s["symbol"])
            cols[1].markdown(f"**{s['symbol']}**<br><small>{s['name']}</small>", unsafe_allow_html=True)
            cols[2].markdown(f"`{s['asset_type']}`")
            cols[3].text(s.get("start_date", "-"))
            status = cache_info.get("status", "-")
            cols[4].caption(status)
            if cache_info["status"] == "cached":
                cols[4].text(f"{cache_info['end']}")
            tags_str = ", ".join(s.get("tags", [])) or "-"
            cols[5].markdown(f"<small>{tags_str}</small>", unsafe_allow_html=True)

            with cols[6]:
                with st.popover("⚙", use_container_width=True):
                    with st.expander("✏️ 编辑", expanded=True):
                        new_name = st.text_input("名称", value=s["name"], key=f"ed_name_{s['symbol']}")
                        new_notes = st.text_area("备注", value=s.get("notes", ""), key=f"ed_notes_{s['symbol']}")
                        new_tags_str = st.text_input(
                            "标签 (逗号分隔)",
                            value=", ".join(s.get("tags", [])),
                            key=f"ed_tags_{s['symbol']}",
                        )
                        if st.button("保存修改", key=f"ed_save_{s['symbol']}"):
                            new_tags = [t.strip() for t in new_tags_str.split(",") if t.strip()]
                            SymbolRegistry.update(
                                s["symbol"],
                                name=new_name,
                                notes=new_notes,
                                tags=new_tags,
                            )
                            st.success("已更新")
                            st.rerun()

                    if st.button("🗑 删除", type="secondary", key=f"del_{s['symbol']}"):
                        SymbolRegistry.remove(s["symbol"])
                        st.success(f"已删除 {s['symbol']}")
                        st.rerun()

                    st.markdown("---")
                    st.caption("同步数据")
                    if st.button("🔄 立即同步", key=f"sync_{s['symbol']}"):
                        with st.spinner(f"正在获取 {s['symbol']} 数据..."):
                            df = SymbolRegistry.fetch_data(s["symbol"])
                        if df is not None and not df.empty:
                            st.success(f"同步完成: {len(df)} 行, "
                                       f"{str(df.index[0])[:10]} ~ {str(df.index[-1])[:10]}")
                            st.rerun()
                        else:
                            st.error("同步失败")

    if selected:
        n = len(selected)
        st.markdown("---")
        bar = st.columns([2, 1, 5])
        bar[0].markdown(f"已选 {n} 个代码")
        with bar[1]:
            with st.popover("🗑 批量删除", use_container_width=True):
                st.warning(f"确定要删除以下 {n} 个代码吗？此操作不可撤销。")
                for sym in selected:
                    entry = SymbolRegistry.get(sym)
                    label = entry["name"] if entry else sym
                    st.markdown(f"- **{sym}** {label}")
                if st.button("确认删除", type="primary", key="batch_del_confirm"):
                    for sym in selected:
                        SymbolRegistry.remove(sym)
                    st.success(f"已删除 {n} 个代码")
                    st.rerun()

    if add_btn:
        st.session_state.show_add_form = True

    if st.session_state.get("show_add_form", False):
        with st.container(border=True):
            st.markdown("#### ➕ 添加新代码")
            add_col1, add_col2 = st.columns(2)
            with add_col1:
                new_symbol = st.text_input("代码", key="add_symbol")
                new_name = st.text_input("名称", key="add_name")
                new_asset_type = st.selectbox(
                    "资产类型",
                    ["etf", "stock", "lof", "open_fund", "index"],
                    key="add_type",
                )
            with add_col2:
                today_default = date.today()
                new_start = st.date_input("开始日期", value=date(today_default.year - 5, 1, 1),
                                          key="add_start")
                new_tags_str = st.text_input("标签 (逗号分隔)", key="add_tags",
                                             help="如: 宽基ETF, 科技, 新能源")
                new_notes = st.text_area("备注", key="add_notes", height=68)

            col_confirm, col_cancel = st.columns([1, 5])
            with col_confirm:
                if st.button("确认添加", type="primary", key="add_confirm"):
                    if not new_symbol.strip():
                        st.error("请输入代码")
                    else:
                        tags = [t.strip() for t in new_tags_str.split(",") if t.strip()]
                        ok = SymbolRegistry.add(
                            symbol=new_symbol.strip(),
                            name=new_name.strip() or new_symbol.strip(),
                            asset_type=new_asset_type,
                            start_date=new_start.strftime("%Y%m%d"),
                            tags=tags,
                            notes=new_notes.strip(),
                        )
                        if ok:
                            st.success(f"已添加 {new_symbol}")
                            st.session_state.show_add_form = False
                            st.rerun()
                        else:
                            st.error(f"代码 {new_symbol} 已存在")
            with col_cancel:
                if st.button("取消", key="add_cancel"):
                    st.session_state.show_add_form = False
                    st.rerun()

    mirror = SymbolRegistry.list()
    type_counts = {}
    for s in mirror:
        t = s["asset_type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    dist_str = ", ".join(f"{t}:{c}" for t, c in sorted(type_counts.items()))
    st.markdown("---")
    st.caption(f"注册表共 {len(mirror)} 个代码 | 类型分布: {dist_str}")
