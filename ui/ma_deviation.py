import json
import time
from datetime import date, timedelta
from pathlib import Path

import streamlit as st
import pandas as pd

from data.fetcher import fetch_history, get_price_series
from data.symbol_registry import SymbolRegistry

SETTINGS_FILE = Path(".cache/ma_deviation_settings.json")
ASSET_TYPES = ["stock", "etf", "index", "us", "lof", "open_fund"]
DEFAULT_SETTINGS = {"symbols": [], "ma_period": 250, "adjustment_factor": 2.0}


def _settings_path() -> Path:
    p = SETTINGS_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_settings() -> dict:
    p = _settings_path()
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return dict(DEFAULT_SETTINGS)


def save_settings(settings: dict) -> None:
    _settings_path().write_text(
        json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _lookup_name(code: str) -> str:
    entry = SymbolRegistry.get(code)
    if entry:
        return entry.get("name", code)
    return code


def _calc_dca_amount(deviation_pct: float, adjustment_factor: float) -> int:
    base = 5000
    ratio = 1 - (deviation_pct / 100) * adjustment_factor
    return max(0, min(10000, round(base * ratio)))


def _gauge_html(deviation_pct: float) -> str:
    pos = max(0, min(100, (deviation_pct + 50) / 100 * 100))
    marker_style = (
        f"position:absolute;left:{pos}%;top:-5px;"
        "transform:translateX(-50%);"
        "width:14px;height:30px;background:#1e293b;border-radius:4px;"
        "box-shadow:0 1px 4px rgba(0,0,0,0.3);z-index:2;"
    )
    return f"""
    <div style="margin:4px 0">
      <div style="position:relative;height:20px;background:linear-gradient(to right,#10b981,#f59e0b,#ef4444);border-radius:10px;width:100%">
        <div style="{marker_style}"></div>
      </div>
      <div style="display:flex;justify-content:space-between;font-size:11px;color:#94a3b8;padding:0 2px">
        <span>-50%</span><span>0%</span><span>+50%</span>
      </div>
    </div>"""


def _remove_sym(code: str) -> None:
    s = load_settings()
    s["symbols"] = [sym for sym in s["symbols"] if sym["code"] != code]
    save_settings(s)


def _fetch_dates() -> tuple[str, str]:
    end = date.today().strftime("%Y%m%d")
    start = (date.today() - timedelta(days=5 * 365)).strftime("%Y%m%d")
    return start, end


@st.cache_data(ttl=300, show_spinner=False)
def _cached_fetch(
    symbol: str, asset_type: str, start_date: str, end_date: str, _refresh_key: float = 0
) -> pd.DataFrame:
    return fetch_history(
        asset_type=asset_type,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        adjust="qfq",
    )


def _resolve_symbol(
    code: str, start_date: str, end_date: str, refresh_key: float = 0
):
    settings = load_settings()
    sym_info = next((s for s in settings["symbols"] if s["code"] == code), None)
    known_type = sym_info.get("type") if sym_info else None

    if known_type:
        try:
            df = _cached_fetch(code, known_type, start_date, end_date, _refresh_key=refresh_key)
            if df is not None and not df.empty:
                return df, known_type
        except Exception:
            pass

    for asset_type in ASSET_TYPES:
        if asset_type == known_type:
            continue
        try:
            df = _cached_fetch(code, asset_type, start_date, end_date, _refresh_key=refresh_key)
            if df is not None and not df.empty:
                if sym_info:
                    sym_info["type"] = asset_type
                else:
                    settings["symbols"].append({"code": code, "type": asset_type})
                save_settings(settings)
                return df, asset_type
        except Exception:
            continue

    return None, None


def calc_deviation(price_series: pd.Series, ma_period: int):
    current_price = float(price_series.iloc[-1])
    ma = price_series.rolling(ma_period).mean().iloc[-1]
    if pd.isna(ma):
        ma = price_series.mean()
    deviation_pct = (current_price - ma) / ma * 100
    return current_price, float(ma), deviation_pct


def render_ma_deviation_section() -> None:
    settings = load_settings()
    start_date, end_date = _fetch_dates()

    st.subheader("📈 均线偏离监控")

    refresh_col, _ = st.columns([1, 8])
    refresh = refresh_col.button("🔄 刷新", type="secondary", key="ma_dev_refresh")

    if "ma_dev_rk" not in st.session_state:
        st.session_state.ma_dev_rk = 0.0
    if refresh:
        st.session_state.ma_dev_rk = time.time()
    refresh_key = st.session_state.ma_dev_rk

    with st.expander("⚙️ 设置", expanded=False):
        registry_syms = SymbolRegistry.list()
        if registry_syms:
            reg_options = {f"{s['symbol']} - {s['name']}": s['symbol'] for s in registry_syms}
            selected_label = st.selectbox(
                "从代码管理导入", ["(选择添加...)"] + list(reg_options.keys()),
                key="ma_dev_reg_select",
            )
            if st.button("✅ 确认添加", key="ma_dev_confirm_add"):
                if selected_label and selected_label != "(选择添加...)" and selected_label in reg_options:
                    code = reg_options[selected_label]
                    if code not in [s["code"] for s in settings["symbols"]]:
                        settings["symbols"].append({"code": code})
                        save_settings(settings)
                    st.rerun()

        syms = settings.get("symbols", [])
        if syms:
            st.markdown("**已添加:**")
            for i, sym in enumerate(syms):
                if i % 4 == 0:
                    row_cols = st.columns(4)
                with row_cols[i % 4]:
                    st.button(
                        f"✕ {sym['code']}", key=f"ma_dev_rm_{sym['code']}",
                        on_click=_remove_sym, args=(sym["code"],),
                    )

        settings["ma_period"] = st.slider(
            "均线周期", 20, 500, value=settings["ma_period"], key="ma_dev_period",
        )
        settings["adjustment_factor"] = st.slider(
            "斜率系数", 0.5, 5.0, value=settings["adjustment_factor"],
            step=0.1, key="ma_dev_adj",
        )
        save_settings(settings)

    if not settings["symbols"]:
        st.info("⬆️ 请点击 ⚙️ 设置 添加要监控的代码")
        return

    if refresh:
        pbar = st.progress(0)

    results = []
    for i, sym in enumerate(settings["symbols"]):
        code = sym["code"]
        if refresh:
            pbar.progress(i / len(settings["symbols"]), text=f"正在获取 {code}...")

        try:
            df, at = _resolve_symbol(code, start_date, end_date, refresh_key=refresh_key)
            if df is None or df.empty:
                results.append({"code": code, "error": "无法获取数据"})
                continue

            ps = get_price_series(df)
            if ps is None or len(ps) == 0:
                results.append({"code": code, "error": "无价格数据"})
                continue

            price, ma, dev = calc_deviation(ps, settings["ma_period"])
            results.append({
                "code": code, "price": price, "ma": ma, "deviation": dev,
            })
        except Exception as e:
            results.append({"code": code, "error": str(e)})

    if refresh:
        pbar.progress(1.0, text="刷新完成")

    if not results:
        st.warning("所有代码均无法获取数据，请检查网络或代码是否正确")
        return

    for r in results:
        if "error" in r:
            st.warning(f"{r['code']}: {r['error']}")
            continue

        name = _lookup_name(r["code"])
        dev = r["deviation"]
        dca = _calc_dca_amount(dev, settings["adjustment_factor"])

        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(
                f"**{r['code']}** {name}"
                f" &nbsp;&nbsp; 偏离: <span style='color:{'#ef4444' if dev>0 else '#10b981'};font-weight:600'>{dev:+.2f}%</span>",
                unsafe_allow_html=True,
            )
            st.markdown(_gauge_html(dev), unsafe_allow_html=True)
            st.caption(f"现价 {r['price']:.2f} / MA{settings['ma_period']} {r['ma']:.2f}")
        with col2:
            st.metric(
                "推荐定投",
                f"¥{dca:,}",
                delta=f"¥{dca - 5000:+,}" if dca != 5000 else None,
            )
        st.divider()
