import streamlit as st
import pandas as pd

from data.symbol_registry import SymbolRegistry
from data.asset_config import ASSET_TYPE_CONFIG
from utils.i18n import t, tt


def render_symbol_manager():
    st.title(t("symbol.title"))
    st.markdown(t("symbol.desc"), unsafe_allow_html=True)

    all_tags = set()
    for s in SymbolRegistry.list():
        for tag in s.get("tags", []):
            all_tags.add(tag)
    all_tags = sorted(all_tags)

    def _asset_label(x):
        if x == "__all":
            return t("status.all")
        cfg = ASSET_TYPE_CONFIG[x]
        if st.session_state.get("_lang", "zh") == "en":
            return cfg.get("label_en", cfg["label"])
        return cfg["label"]

    FILTER_TYPES = ["__all", "stock", "etf", "lof", "open_fund", "index", "us"]
    filter_type = st.selectbox(t("symbol.filter_type"), FILTER_TYPES, format_func=_asset_label)
    filter_tag = st.selectbox(
        t("symbol.filter_tag"),
        ["__all"] + all_tags,
        format_func=lambda x: t("status.all") if x == "__all" else x,
    )

    st.markdown(t("symbol.add_header"))
    with st.container(border=True):
        ai1, ai2, ai3, ai4 = st.columns([1.5, 1, 2, 1])
        with ai1:
            new_symbol = st.text_input(t("sidebar.symbol_code"), placeholder=t("symbol.code_placeholder"), key="add_symbol")
        with ai2:
            new_asset_type = st.selectbox(t("symbol.add_type"), ["etf", "stock", "lof", "open_fund", "index", "us"], key="add_type")
        with ai3:
            new_tags_str = st.text_input(t("symbol.add_tags"), placeholder=t("symbol.tags_placeholder"), key="add_tags")
        with ai4:
            st.markdown("")
            st.markdown("")
            if st.button(t("symbol.add_btn"), type="primary", use_container_width=True, key="add_confirm"):
                if not new_symbol.strip():
                    st.error(t("symbol.error.no_code"))
                else:
                    tags = [tag.strip() for tag in new_tags_str.split(",") if tag.strip()]
                    with st.spinner(t("symbol.spinner.fetching", symbol=new_symbol.strip())):
                        try:
                            meta = SymbolRegistry.autofetch_meta(new_symbol.strip(), new_asset_type)
                        except ValueError as e:
                            st.error(str(e))
                            st.stop()
                    ok = SymbolRegistry.add(
                        symbol=new_symbol.strip(),
                        name=meta["name"],
                        asset_type=new_asset_type,
                        start_date=meta["start_date"],
                        tags=tags,
                    )
                    if ok:
                        st.success(t("symbol.success.added", sym=new_symbol, name=meta["name"], date=meta["start_date"]))
                        st.rerun()
                    else:
                        st.error(t("symbol.error.exists", sym=new_symbol))

    symbols = SymbolRegistry.list()
    if filter_type != "__all":
        symbols = [s for s in symbols if s["asset_type"] == filter_type]
    if filter_tag != "__all":
        symbols = [s for s in symbols if filter_tag in s.get("tags", [])]

    st.markdown(t("symbol.list_header"))
    cols_header = st.columns([0.5, 2.5, 1, 1.2, 1.2, 1.8, 0.8])
    cols_header[0].checkbox(t("symbol.col.select"), key="sel_all", label_visibility="collapsed")
    cols_header[1].markdown(t("symbol.col.code_name"))
    cols_header[2].markdown(t("symbol.col.type"))
    cols_header[3].markdown(t("symbol.col.start"))
    cols_header[4].markdown(t("symbol.col.status"))
    cols_header[5].markdown(t("symbol.col.tags"))
    cols_header[6].markdown(t("symbol.col.actions"))

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
                    with st.expander(t("symbol.edit"), expanded=True):
                        new_name = st.text_input(t("symbol.edit.name"), value=s["name"], key=f"ed_name_{s['symbol']}")
                        new_notes = st.text_area(t("symbol.edit.notes"), value=s.get("notes", ""), key=f"ed_notes_{s['symbol']}")
                        new_tags_str = st.text_input(
                            t("symbol.edit.tags"),
                            value=", ".join(s.get("tags", [])),
                            key=f"ed_tags_{s['symbol']}",
                        )
                        if st.button(t("symbol.edit.save"), key=f"ed_save_{s['symbol']}"):
                            new_tags = [tag.strip() for tag in new_tags_str.split(",") if tag.strip()]
                            SymbolRegistry.update(
                                s["symbol"],
                                name=new_name,
                                notes=new_notes,
                                tags=new_tags,
                            )
                            st.success(t("symbol.success.updated"))
                            st.rerun()

                    if st.button(t("symbol.edit.delete"), type="secondary", key=f"del_{s['symbol']}"):
                        SymbolRegistry.remove(s["symbol"])
                        st.success(t("symbol.success.deleted", sym=s["symbol"]))
                        st.rerun()

                    st.markdown("---")
                    st.caption(t("symbol.edit.sync"))
                    if st.button(t("symbol.edit.sync_btn"), key=f"sync_{s['symbol']}"):
                        with st.spinner(t("symbol.spinner.fetching", symbol=s["symbol"])):
                            df = SymbolRegistry.fetch_data(s["symbol"])
                        if df is not None and not df.empty:
                            st.success(t("symbol.sync.complete", n=len(df),
                                         start=str(df.index[0])[:10], end=str(df.index[-1])[:10]))
                            st.rerun()
                        else:
                            st.error(t("symbol.error.sync_failed"))

    if selected:
        n = len(selected)
        st.markdown("---")
        bar = st.columns([2, 1, 5])
        bar[0].markdown(t("symbol.selected_count", n=n))
        with bar[1]:
            with st.popover(tt("🗑 批量删除", "🗑 Batch Delete"), use_container_width=True):
                st.warning(t("symbol.batch_delete.warning", n=n))
                for sym in selected:
                    entry = SymbolRegistry.get(sym)
                    label = entry["name"] if entry else sym
                    st.markdown(f"- **{sym}** {label}")
                if st.button(t("symbol.batch_delete.confirm"), type="primary", key="batch_del_confirm"):
                    for sym in selected:
                        SymbolRegistry.remove(sym)
                    st.success(t("symbol.batch_delete.success", n=n))
                    st.rerun()

    mirror = SymbolRegistry.list()
    type_counts = {}
    for s in mirror:
        at = s["asset_type"]
        type_counts[at] = type_counts.get(at, 0) + 1
    dist_str = ", ".join(f"{at}:{c}" for at, c in sorted(type_counts.items()))
    st.markdown("---")
    st.caption(t("symbol.footer", n=len(mirror), dist=dist_str))
