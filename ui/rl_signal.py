import streamlit as st
import pandas as pd

from utils.i18n import t
from data_fetcher import ensure_ohlc, fetch_etf_realtime_premium
from backtest.rl.trainer import predict_signal, compute_signal_history


def render_rl_signal(df_full, symbol, asset_type):
    if st.session_state.rl_agent is None:
        st.info(t("signal.hint"))
        return

    st.markdown("---")
    st.subheader(t("signal.title"))

    info = st.session_state.rl_model_info or {}
    meta_cols = st.columns(4)
    with meta_cols[0]:
        st.metric(t("signal.metric.model"), info.get("name", t("signal.action.unknown"))[:30])
    with meta_cols[1]:
        st.metric(t("signal.metric.symbol"), info.get("symbol", symbol))
    with meta_cols[2]:
        ver = info.get("system_version", "1.0")
        version_labels = {"basic": t("rl.sidebar.system.basic"), "1.0": t("rl.sidebar.system.v1"), "2.0": t("rl.sidebar.system.v2")}
        st.metric(t("signal.metric.version"), version_labels.get(ver, ver))
    with meta_cols[3]:
        st.metric(t("signal.metric.latest_date"), str(df_full.index[-1])[:10] if len(df_full) > 0 else "-")

    rename_map = {"开盘": "开盘价", "收盘": "收盘价", "最高": "最高价", "最低": "最低价"}
    df = df_full.copy()
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
    df = ensure_ohlc(df)

    agent = st.session_state.rl_agent
    ver = info.get("system_version", "1.0")
    fgs = info.get("feature_groups")

    with st.spinner("正在计算信号..."):
        sig = predict_signal(agent, df, system_version=ver, feature_groups=fgs)

    action_map = {-1: (t("signal.action.sell"), "#ef4444"), 0: (t("signal.action.hold"), "#6b7280"), 1: (t("signal.action.buy"), "#22c55e"), 2: (t("signal.action.sell"), "#ef4444")}
    label, color = action_map.get(sig, (t("signal.action.unknown"), "#888888"))
    st.markdown(
        f"<div style='text-align:center; padding:24px; background:{color}22; "
        f"border-radius:12px; border:2px solid {color}'>"
        f"<span style='font-size:48px; font-weight:bold; color:{color}'>{label}</span>"
        f"<br><span style='font-size:16px; color:{color}99'>"
        f"{t('signal.data_based_on', date=df.index[-1].strftime('%Y-%m-%d'))}</span></div>",
        unsafe_allow_html=True,
    )

    # 显示当前溢价率 (仅 ETF/LOF)
    if "溢价率" in df.columns:
        latest_premium = float(df["溢价率"].iloc[-1])
        premium_color = "#22c55e" if abs(latest_premium) < 1 else "#ef4444"
        st.markdown(
            f"<div style='text-align:center; padding:12px; margin:8px 0; "
            f"border-radius:8px; border:1px solid {premium_color}'>"
            f"<span style='font-size:14px; color:#888'>{t('signal.premium.label')}</span><br>"
            f"<span style='font-size:32px; font-weight:bold; color:{premium_color}'>"
            f"{latest_premium:+.2f}%</span></div>",
            unsafe_allow_html=True,
        )
        # 实时折价率 (仅 ETF)
        if asset_type == "etf":
            rt_premium = fetch_etf_realtime_premium(symbol)
            if rt_premium != 0.0:
                st.caption(t("signal.premium.discount", rate=rt_premium))

    with st.expander(t("signal.history"), expanded=False):
        with st.spinner("正在回放信号..."):
            signals = compute_signal_history(agent, df, system_version=ver, feature_groups=fgs)
        sig_df = pd.DataFrame({
            "日期": df.index[-60:],
            "信号": [action_map.get(s, ("?", "#888"))[0] for s in signals[-60:]],
        })
        st.dataframe(sig_df, width='stretch', hide_index=True)

    with st.expander(t("signal.latest_quotes"), expanded=False):
        tail = df[["开盘价", "收盘价", "最高价", "最低价"]].tail(10)
        st.dataframe(tail, width='stretch')

    if st.button(t("signal.refresh"), width='stretch'):
        st.rerun()
