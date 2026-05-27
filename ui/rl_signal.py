import streamlit as st
import pandas as pd

from data_fetcher import ensure_ohlc, fetch_etf_realtime_premium
from backtest.rl.trainer import predict_signal, compute_signal_history


def render_rl_signal(df_full, symbol, asset_type):
    if st.session_state.rl_agent is None:
        st.info("💡 训练完成后保存模型，或在侧边栏「已保存模型」中加载一个已有模型，即可查看实时信号")
        return

    st.markdown("---")
    st.subheader("📡 实时交易信号")

    info = st.session_state.rl_model_info or {}
    meta_cols = st.columns(4)
    with meta_cols[0]:
        st.metric("加载模型", info.get("name", "未知")[:30])
    with meta_cols[1]:
        st.metric("资产代码", info.get("symbol", symbol))
    with meta_cols[2]:
        ver = info.get("system_version", "1.0")
        st.metric("系统版本", {"basic": "基础版", "1.0": "系统1.0", "2.0": "系统2.0"}.get(ver, ver))
    with meta_cols[3]:
        st.metric("最新日期", str(df_full.index[-1])[:10] if len(df_full) > 0 else "-")

    rename_map = {"开盘": "开盘价", "收盘": "收盘价", "最高": "最高价", "最低": "最低价"}
    df = df_full.copy()
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
    df = ensure_ohlc(df)

    agent = st.session_state.rl_agent
    ver = info.get("system_version", "1.0")
    fgs = info.get("feature_groups")

    with st.spinner("正在计算信号..."):
        sig = predict_signal(agent, df, system_version=ver, feature_groups=fgs)

    action_map = {-1: ("🔴 卖出", "#ef4444"), 0: ("⚪ 持有", "#6b7280"), 1: ("🟢 买入", "#22c55e"), 2: ("🔴 卖出", "#ef4444")}
    label, color = action_map.get(sig, ("❓ 未知", "#888888"))
    st.markdown(
        f"<div style='text-align:center; padding:24px; background:{color}22; "
        f"border-radius:12px; border:2px solid {color}'>"
        f"<span style='font-size:48px; font-weight:bold; color:{color}'>{label}</span>"
        f"<br><span style='font-size:16px; color:{color}99'>"
        f"基于 {str(df.index[-1])[:10]} 日数据</span></div>",
        unsafe_allow_html=True,
    )

    # 显示当前溢价率 (仅 ETF/LOF)
    if "溢价率" in df.columns:
        latest_premium = float(df["溢价率"].iloc[-1])
        premium_color = "#22c55e" if abs(latest_premium) < 1 else "#ef4444"
        st.markdown(
            f"<div style='text-align:center; padding:12px; margin:8px 0; "
            f"border-radius:8px; border:1px solid {premium_color}'>"
            f"<span style='font-size:14px; color:#888'>当前溢价率</span><br>"
            f"<span style='font-size:32px; font-weight:bold; color:{premium_color}'>"
            f"{latest_premium:+.2f}%</span></div>",
            unsafe_allow_html=True,
        )
        # 实时折价率 (仅 ETF)
        if asset_type == "etf":
            rt_premium = fetch_etf_realtime_premium(symbol)
            if rt_premium != 0.0:
                st.caption(f"实时折价率: {rt_premium:+.2f}% (来自东方财富)")

    with st.expander("📈 近期信号历史", expanded=False):
        with st.spinner("正在回放信号..."):
            signals = compute_signal_history(agent, df, system_version=ver, feature_groups=fgs)
        sig_df = pd.DataFrame({
            "日期": df.index[-60:],
            "信号": [action_map.get(s, ("?", "#888"))[0] for s in signals[-60:]],
        })
        st.dataframe(sig_df, width='stretch', hide_index=True)

    with st.expander("📋 最新行情", expanded=False):
        tail = df[["开盘价", "收盘价", "最高价", "最低价"]].tail(10)
        st.dataframe(tail, width='stretch')

    if st.button("🔄 刷新信号", width='stretch'):
        st.rerun()
