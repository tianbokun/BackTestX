from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import torch

from backtest.rl.task_manager import TaskManager, TaskStatus
from backtest.rl.dqn_agent import DQNAgent


STATUS_EMOJI = {
    TaskStatus.PENDING.value: "⏳",
    TaskStatus.RUNNING.value: "🔄",
    TaskStatus.COMPLETED.value: "✅",
    TaskStatus.FAILED.value: "❌",
    TaskStatus.CANCELLED.value: "🚫",
    TaskStatus.EARLY_STOPPED.value: "⏹️",
}


def _ensure_trades(trades):
    if isinstance(trades, pd.DataFrame):
        return trades
    if isinstance(trades, list):
        return pd.DataFrame(trades) if trades else pd.DataFrame()
    return pd.DataFrame()


def _ensure_dates(dates):
    if isinstance(dates, pd.DatetimeIndex):
        return dates
    if isinstance(dates, list):
        return pd.DatetimeIndex([pd.Timestamp(d) for d in dates])
    return dates


def _render_rl_save(result: dict, task: dict = None):
    auto = (task or {}).get("auto_saved_models", {})
    if auto:
        st.subheader("💾 保存模型")
        for label, path in auto.items():
            p = Path(path)
            st.code(f"{p.name}  ({p.parent})", language="")
            st.caption(f"✅ 已自动保存为 {label}")
        st.caption("可在左侧「已保存模型」中加载使用")
        return

    agent = result.get("agent")
    if agent is None:
        st.info("💡 模型对象仅在训练会话中可用（页面刷新后丢失），请在此时保存")
        return
    meta = result.get("meta", {})
    agent_best = result.get("agent_best")
    dqn = result["result_dqn"]
    dqn_best = result.get("result_dqn_best")
    sym = meta.get("symbol", "unknown")
    sv = meta.get("system_version", "1.0")
    _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _default = f"{sym}_{sv}_{_ts}"
    st.subheader("💾 保存模型")
    _name = st.text_input("模型名称", value=_default, key="task_rl_save_name")
    choice = "最终权重"
    if agent_best is not None:
        choice = st.radio("保存哪个模型?", ["最终权重", "最佳episode权重"],
                          index=0, horizontal=True, key="task_rl_save_choice")
    if st.button("💾 保存模型", type="primary", key="task_rl_save_btn"):
        save_agent = agent_best if (choice == "最佳episode权重" and agent_best is not None) else agent
        save_result = dqn_best if (choice == "最佳episode权重" and dqn_best is not None) else dqn
        p = Path(f"saved_models/rl/{_name}.pt")
        p.parent.mkdir(parents=True, exist_ok=True)
        save_agent.save(str(p), {
            "symbol": sym, "system_version": sv,
            "feature_groups": meta.get("feature_groups", []),
            "train_start": meta.get("train_start", ""),
            "train_end": meta.get("train_end", ""),
            "test_return": save_result["total_return_pct"],
            "sharpe": save_result["sharpe_ratio"],
        })
        st.session_state.rl_agent = save_agent
        st.session_state.rl_model_info = {
            "name": p.stem, "path": str(p),
            "symbol": sym, "system_version": sv,
            "feature_groups": meta.get("feature_groups", []),
        }
        st.success(f"✅ 模型已保存至 `{p}`")
        st.rerun()


def _render_hrl_save(result: dict, task: dict = None):
    auto = (task or {}).get("auto_saved_models", {})
    if auto:
        st.subheader("💾 保存模型")
        for label, path in auto.items():
            p = Path(path)
            st.code(f"{p.name}  ({p.parent})", language="")
            st.caption(f"✅ 已自动保存为 {label}")
        st.caption("可在左侧「已保存模型」中加载使用")
        return

    trainer = result.get("agent")
    if trainer is None:
        st.info("💡 模型对象仅在训练会话中可用（页面刷新后丢失），请在此时保存")
        return
    syms = result.get("selected_symbols", [])
    label = "_".join(syms[:3]) if syms else "hrl"
    _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _default = f"{label}_{_ts}"
    st.subheader("💾 保存模型")
    _name = st.text_input("模型名称", value=_default, key="task_hrl_save_name")
    if st.button("💾 保存模型", type="primary", key="task_hrl_save_btn"):
        p = Path(f"saved_models/rl/{_name}.pt")
        p.parent.mkdir(parents=True, exist_ok=True)
        trainer.save(str(p))
        st.success(f"✅ HRL 模型已保存至 `{p}` (含 PPO + DQN 权重)")
        st.rerun()


def _render_rl_result(result: dict, task: dict = None):
    dqn = result["result_dqn"]
    dqn_best = result.get("result_dqn_best")
    bh = result["result_bh"]
    meta = result.get("meta", {})

    rows = [
        {"策略": "DQN (最终)", "最终金额": dqn["final_value"],
         "收益率%": dqn["total_return_pct"],
         "夏普比率": dqn["sharpe_ratio"],
         "最大回撤%": dqn["max_drawdown_pct"],
         "交易次数": dqn["num_trades"]},
    ]
    if dqn_best is not None:
        rows.append({"策略": "DQN (最佳episode)", "最终金额": dqn_best["final_value"],
                     "收益率%": dqn_best["total_return_pct"],
                     "夏普比率": dqn_best["sharpe_ratio"],
                     "最大回撤%": dqn_best["max_drawdown_pct"],
                     "交易次数": dqn_best["num_trades"]})
    rows.append({"策略": "买入持有(BH)", "最终金额": bh["final_value"],
                 "收益率%": bh["total_return_pct"],
                 "夏普比率": bh["sharpe_ratio"],
                 "最大回撤%": bh["max_drawdown_pct"],
                 "交易次数": 0})
    comp = pd.DataFrame(rows)
    st.dataframe(comp, width='stretch', hide_index=True)

    test_idx = _ensure_dates(meta.get("df_test_index", dqn.get("dates")))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=test_idx, y=dqn["equity_curve"],
        mode="lines", name=f"DQN 最终 ({meta.get('system_version', '?')})",
        line=dict(color="#1f77b4", width=2),
    ))
    if dqn_best is not None:
        fig.add_trace(go.Scatter(
            x=test_idx, y=dqn_best["equity_curve"],
            mode="lines", name="DQN 最佳episode",
            line=dict(color="#2ca02c", width=2, dash="dot"),
        ))
    fig.add_trace(go.Scatter(
        x=test_idx, y=bh["equity_curve"],
        mode="lines", name="买入持有(BH)",
        line=dict(color="#ff7f0e", width=2, dash="dash"),
    ))
    fig.update_layout(
        xaxis_title="日期", yaxis_title="账户总值",
        hovermode="x unified", height=350,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig, width='stretch')

    dqn_final_has_trades = not _ensure_trades(dqn.get("trades")).empty
    dqn_best_has_trades = dqn_best is not None and not _ensure_trades(dqn_best.get("trades")).empty
    if dqn_final_has_trades or dqn_best_has_trades:
        st.markdown("**📝 交易记录**")
        trade_source = st.selectbox(
            "选择查看", ["最终权重", "最佳episode"],
            key="task_rl_trade_source",
            disabled=not (dqn_final_has_trades and dqn_best_has_trades),
        )
        src = dqn_best if trade_source == "最佳episode" else dqn
        trades = _ensure_trades(src.get("trades"))
        if not trades.empty:
            trades = trades.copy()
            if "日期" in trades.columns:
                trades["日期"] = trades["日期"].dt.strftime("%Y-%m-%d")
            st.dataframe(trades, width='stretch', hide_index=True)

    _render_rl_save(result, task)


def _render_hrl_result(result: dict, task: dict = None):
    test = result["test_result"]
    benchmarks = result.get("benchmarks", {})
    capital = result.get("capital", 100000.0)

    rows = [{
        "策略": "HRL (PPO择时 + DQN选股)",
        "最终金额": test["final_value"],
        "收益率%": test["total_return_pct"],
        "夏普比率": test["sharpe_ratio"],
        "最大回撤%": test["max_drawdown_pct"],
    }]
    ew = benchmarks.get("equal_weight_bh")
    if ew:
        rows.append({
            "策略": "等权买入持有",
            "最终金额": ew["final_value"],
            "收益率%": ew["total_return_pct"],
            "夏普比率": ew["sharpe_ratio"],
            "最大回撤%": ew["max_drawdown_pct"],
        })
    for dca_key, dca_label in [("monthly_dca", "月定投(等权)"), ("ma_adjust_dca", "均线偏离定投(等权)")]:
        dca = benchmarks.get(dca_key)
        if dca:
            rows.append({
                "策略": dca_label,
                "最终金额": dca["final_value"],
                "收益率%": dca["total_return_pct"],
                "夏普比率": dca.get("sharpe_ratio", "N/A"),
                "最大回撤%": dca.get("max_drawdown_pct", "N/A"),
            })
    comp = pd.DataFrame(rows)
    st.dataframe(comp, width='stretch', hide_index=True)

    dates = _ensure_dates(test.get("dates", []))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=test["equity_curve"],
        mode="lines", name="HRL",
        line=dict(color="#2563eb", width=2),
    ))
    fig.add_hline(y=capital, line_dash="dot", line_color="gray", annotation_text="初始本金")
    if ew:
        fig.add_trace(go.Scatter(
            x=dates, y=ew["equity_curve"],
            mode="lines", name="等权买入持有",
            line=dict(color="#ef4444", width=2, dash="dash"),
        ))
    for dca_key, dca_label, dca_color in [
        ("monthly_dca", "月定投(等权)", "#10b981"),
        ("ma_adjust_dca", "均线偏离定投(等权)", "#f59e0b"),
    ]:
        dca = benchmarks.get(dca_key)
        if dca is not None and "total_value_series" in dca:
            tvs = dca["total_value_series"]
            dca_curve = (
                tvs.reindex(pd.DatetimeIndex(dates)).ffill().fillna(capital).values
                if isinstance(tvs, (pd.Series, pd.DataFrame))
                else tvs
            )
            fig.add_trace(go.Scatter(
                x=dates, y=dca_curve,
                mode="lines", name=dca_label,
                line=dict(color=dca_color, width=1.5, dash="dot"),
            ))
    fig.update_layout(
        xaxis_title="日期", yaxis_title=f"账户总值 (初始={capital:,.0f})",
        hovermode="x unified", height=350,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig, width='stretch')

    if "position_ratios" in test and len(test["position_ratios"]) > 0:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=dates, y=test["position_ratios"],
            mode="lines", name="仓位比例",
            line=dict(color="#2563eb", width=2),
            fill="tozeroy",
        ))
        fig2.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="半仓")
        fig2.update_layout(
            xaxis_title="日期", yaxis_title="仓位比例",
            hovermode="x unified", height=200,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig2, width='stretch')

    trade_log = _ensure_trades(test.get("trade_log"))
    if not trade_log.empty:
        with st.expander("📝 交易记录", expanded=False):
            st.dataframe(trade_log, width='stretch', hide_index=True)

    _render_hrl_save(result, task)


def _refetch_rl_data(meta: dict) -> dict:
    from data.symbol_registry import SymbolRegistry
    import pandas as pd

    symbol = meta.get("symbol", "")
    adjust = meta.get("adjust", "qfq")
    train_start = meta.get("train_start", "")
    train_end = meta.get("train_end", "")
    test_dates = meta.get("df_test_index", [])

    df = SymbolRegistry.fetch_data(symbol, adjust=adjust)
    if df is None or df.empty:
        raise ValueError(f"重新获取 {symbol} 数据失败")

    df_train = df[(df.index >= pd.Timestamp(train_start)) & (df.index <= pd.Timestamp(train_end))].copy()
    if len(test_dates) > 0:
        test_start_s = test_dates[0][:10]
        test_end_s = test_dates[-1][:10]
        df_test = df[(df.index >= pd.Timestamp(test_start_s)) & (df.index <= pd.Timestamp(test_end_s))].copy()
    else:
        df_test = df[df.index > pd.Timestamp(train_end)].copy()
    return {"df_train": df_train, "df_test": df_test, "df_test_index": df_test.index}


def _render_rl_retrain_form(task: dict, mgr: TaskManager):
    tid = task.get("_id", "")

    done_key = f"retrain_rl_done_{tid}"
    if done_key in st.session_state:
        new_tid = st.session_state[done_key]
        st.success(f"✅ 新训练任务已提交 (ID: {new_tid[:8]}...)")
        st.caption("可关闭此面板或在「任务列表」中查看进度")
        if st.button("📋 查看任务详情"):
            st.session_state.selected_task_id = new_tid
            st.rerun()
        return

    params = task.get("params")
    params_display = task.get("params_display")
    if params is None and params_display is None:
        st.info("任务参数已过期，无法重新训练")
        return

    needs_refetch = params is None
    src = params if not needs_refetch else params_display

    dqn = src.get("dqn_params", {})
    fee = src.get("fee_params", {})
    meta = src.get("meta", {})
    symbol = meta.get("symbol", "?")
    sys_ver_default = src.get("system_version", "1.0")

    from backtest.rl.feature_engineer import FEATURE_GROUPS
    fg_keys = list(FEATURE_GROUPS.keys())
    fg_labels = [FEATURE_GROUPS[k]["label"] for k in fg_keys]
    fg_map = dict(zip(fg_labels, fg_keys))
    orig_fgs = src.get("feature_groups", [])
    orig_labels = [lb for lb, k in fg_map.items() if k in orig_fgs]

    if needs_refetch:
        st.info("🔄 数据从持久化参数重新获取中...")

    with st.form(key=f"retrain_rl_{tid}", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            n_episodes = st.number_input("Episode 数", value=int(dqn.get("n_episodes", 64)), min_value=1)
            lr = st.number_input("学习率 lr", value=float(dqn.get("lr", 1e-5)), format="%.6f")
        with col2:
            gamma = st.number_input("折扣因子 γ", value=float(dqn.get("gamma", 0.98)), min_value=0.0, max_value=1.0, format="%.3f")
            hidden = st.number_input("隐藏层大小", value=int(dqn.get("hidden", 128)), min_value=16)
        with col3:
            batch_size = st.number_input("Batch Size", value=int(dqn.get("batch_size", 200)), min_value=16)
            epsilon_decay = st.number_input("Epsilon Decay", value=int(dqn.get("epsilon_decay", 500)), min_value=50)

        col1, col2, col3 = st.columns(3)
        with col1:
            reward_window = st.number_input("波动率窗口", value=int(dqn.get("reward_window", 63)), min_value=5)
        with col2:
            vol_penalty = st.number_input("波动惩罚系数", value=float(dqn.get("vol_penalty_coef", 0.1)), format="%.2f")
        with col3:
            dd_penalty = st.number_input("回撤惩罚系数", value=float(dqn.get("dd_penalty_coef", 0.5)), format="%.2f")

        sys_ver_options = ["basic", "1.0", "2.0"]
        sys_ver_idx = sys_ver_options.index(sys_ver_default) if sys_ver_default in sys_ver_options else 1
        sys_ver = st.selectbox("系统版本", sys_ver_options, index=sys_ver_idx)
        selected_labels = st.multiselect("特征组", fg_labels, default=orig_labels)
        selected_fgs = [fg_map[lb] for lb in selected_labels]

        col1, col2 = st.columns(2)
        with col1:
            commission = st.number_input("佣金率", value=float(fee.get("commission_rate", 0.00025)), format="%.5f")
            stamp = st.number_input("印花税率", value=float(fee.get("stamp_duty", 0.001)), format="%.4f")
        with col2:
            min_comm = st.number_input("最低佣金", value=float(fee.get("min_commission", 5.0)))
            capital = st.number_input("初始资金", value=float(fee.get("initial_capital", 100000.0)))

        st.caption(f"📦 {symbol} ｜ 训练 {meta.get('train_start','?')} ~ {meta.get('train_end','?')}"
                   f" ｜ 测试 {meta.get('df_test_index',[len(meta)]*2)} 行"
                   f" ｜ 数据来源: {'内存' if not needs_refetch else '重抓'}")

        submitted = st.form_submit_button("🚀 提交训练任务", type="primary")
        if submitted:
            if needs_refetch:
                try:
                    fetched = _refetch_rl_data(meta)
                    df_train, df_test = fetched["df_train"], fetched["df_test"]
                    meta = dict(meta, df_test_index=fetched["df_test_index"])
                except Exception as e:
                    st.error(f"数据重抓失败: {e}")
                    st.stop()
            else:
                df_train = params["df_train"]
                df_test = params["df_test"]

            new_dqn = dict(dqn,
                n_episodes=n_episodes, lr=lr, gamma=gamma,
                hidden=hidden, batch_size=batch_size,
                epsilon_decay=epsilon_decay,
                reward_window=reward_window,
                vol_penalty_coef=vol_penalty,
                dd_penalty_coef=dd_penalty,
            )
            new_fee = dict(fee,
                commission_rate=commission, min_commission=min_comm,
                stamp_duty=stamp, initial_capital=capital,
            )
            new_params = dict(
                df_train=df_train, df_test=df_test,
                system_version=sys_ver,
                feature_groups=selected_fgs,
                dqn_params=new_dqn,
                fee_params=new_fee,
                meta=meta,
            )
            from ui.rl_training import _rl_train_task
            new_tid = mgr.submit("RL训练", new_params, _rl_train_task, args=(new_params,))
            st.session_state[done_key] = new_tid
            st.rerun()


def _refetch_hrl_data(params: dict) -> dict:
    from data.symbol_registry import SymbolRegistry
    import pandas as pd

    syms = params.get("selected_symbols", [])
    adjust = params.get("adjust", "qfq")
    test_start = params.get("test_start", "")
    test_end = params.get("test_end", "")

    etf_data = {}
    for sym in syms:
        df = SymbolRegistry.fetch_data(sym, adjust=adjust)
        if df is not None and not df.empty:
            etf_data[sym] = df
    if len(etf_data) < 2:
        raise ValueError(f"数据重抓失败，成功获取 {len(etf_data)} 支 ETF")

    def _align(ed):
        common = None
        for sym, df in ed.items():
            idx = set(df.index)
            if common is None:
                common = idx
            else:
                common &= idx
        return sorted(common)

    aligned = _align(etf_data)
    aligned_str = [str(d)[:10] for d in aligned]
    train_end_i = int(len(aligned) * 0.6)
    train_dates = aligned_str[:train_end_i + 1]
    test_dates = [d for d in aligned_str if d >= test_start[:10] and d <= test_end[:10]]

    train_etf_data = {}
    for sym in syms:
        train_etf_data[sym] = etf_data[sym].loc[:aligned[train_end_i]]

    return {
        "all_etf_data": etf_data,
        "train_etf_data": train_etf_data,
        "train_dates": train_dates,
        "test_dates": test_dates,
    }


def _render_hrl_retrain_form(task: dict, mgr: TaskManager):
    tid = task.get("_id", "")

    done_key = f"retrain_hrl_done_{tid}"
    if done_key in st.session_state:
        new_tid = st.session_state[done_key]
        st.success(f"✅ 新训练任务已提交 (ID: {new_tid[:8]}...)")
        st.caption("可关闭此面板或在「任务列表」中查看进度")
        if st.button("📋 查看任务详情"):
            st.session_state.selected_task_id = new_tid
            st.rerun()
        return

    params = task.get("params")
    params_display = task.get("params_display")
    if params is None and params_display is None:
        st.info("任务参数已过期，无法重新训练")
        return

    needs_refetch = params is None
    src = params if not needs_refetch else params_display

    ep = src.get("n_episodes", 64)

    if needs_refetch:
        st.info("🔄 数据从持久化参数重新获取中...")

    with st.form(key=f"retrain_hrl_{tid}", clear_on_submit=True):
        st.markdown("**🧠 PPO 参数**")
        col1, col2, col3 = st.columns(3)
        with col1:
            ppo_lr = st.number_input("PPO 学习率", value=float(src.get("ppo_lr", 3e-4)), format="%.6f")
            clip_epsilon = st.number_input("Clip ε", value=float(src.get("clip_epsilon", 0.2)), format="%.2f")
        with col2:
            ppo_gamma = st.number_input("PPO γ", value=float(src.get("ppo_gamma", 0.99)), format="%.3f")
            entropy_beta = st.number_input("熵奖励 β", value=float(src.get("entropy_beta", 0.01)), format="%.3f")
        with col3:
            ppo_hidden = st.number_input("PPO 隐藏层", value=int(src.get("ppo_hidden", 128)), min_value=16)
            n_episodes = st.number_input("Episode 数", value=int(ep), min_value=1)

        col1, col2, col3 = st.columns(3)
        with col1:
            gae_lambda = st.number_input("GAE λ", value=float(src.get("gae_lambda", 0.95)), format="%.2f")
        with col2:
            ppo_epochs = st.number_input("PPO Epochs", value=int(src.get("ppo_epochs", 10)), min_value=1)
        with col3:
            ppo_update_freq = st.number_input("PPO 更新频率", value=int(src.get("ppo_update_freq", 128)), min_value=1)

        st.markdown("**🤖 DQN 参数**")
        col1, col2, col3 = st.columns(3)
        with col1:
            dqn_lr = st.number_input("DQN 学习率", value=float(src.get("dqn_lr", 1e-5)), format="%.6f")
            dqn_hidden = st.number_input("DQN 隐藏层", value=int(src.get("dqn_hidden", 128)), min_value=16)
        with col2:
            dqn_gamma = st.number_input("DQN γ", value=float(src.get("dqn_gamma", 0.98)), format="%.3f")
            dqn_batch_size = st.number_input("DQN Batch", value=int(src.get("dqn_batch_size", 200)), min_value=16)
        with col3:
            dqn_epsilon_decay = st.number_input("DQN ε Decay", value=int(src.get("dqn_epsilon_decay", 500)), min_value=50)

        st.markdown("**💵 费用与交易参数**")
        col1, col2, col3 = st.columns(3)
        with col1:
            commission = st.number_input("佣金率", value=float(src.get("commission_rate", 0.00025)), format="%.5f")
        with col2:
            capital = st.number_input("初始资金", value=float(src.get("initial_capital", 100000.0)))
        with col3:
            trade_fraction = st.number_input("单次交易比例", value=float(src.get("trade_fraction", 0.25)), format="%.2f")

        syms = src.get("selected_symbols", [])
        st.caption(f"📦 ETF 组合: {', '.join(syms) if syms else '?'}")

        submitted = st.form_submit_button("🚀 提交训练任务", type="primary")
        if submitted:
            if needs_refetch:
                try:
                    fetched = _refetch_hrl_data(src)
                except Exception as e:
                    st.error(f"数据重抓失败: {e}")
                    st.stop()
            else:
                fetched = {
                    "all_etf_data": params["all_etf_data"],
                    "train_etf_data": params["train_etf_data"],
                    "train_dates": params.get("train_dates", []),
                    "test_dates": params.get("test_dates", []),
                }

            new_params = dict(src,
                all_etf_data=fetched["all_etf_data"],
                train_etf_data=fetched["train_etf_data"],
                train_dates=fetched["train_dates"],
                test_dates=fetched["test_dates"],
                n_episodes=n_episodes,
                ppo_lr=ppo_lr, ppo_gamma=ppo_gamma,
                clip_epsilon=clip_epsilon, entropy_beta=entropy_beta,
                gae_lambda=gae_lambda, ppo_hidden=ppo_hidden,
                ppo_epochs=ppo_epochs, ppo_update_freq=ppo_update_freq,
                dqn_lr=dqn_lr, dqn_gamma=dqn_gamma,
                dqn_hidden=dqn_hidden,
                dqn_batch_size=dqn_batch_size,
                dqn_epsilon_decay=dqn_epsilon_decay,
                commission_rate=commission,
                initial_capital=capital,
                trade_fraction=trade_fraction,
            )
            from ui.hierarchical_rl import _hrl_train_task
            new_tid = mgr.submit("HRL训练", new_params, _hrl_train_task, args=(new_params,))
            st.session_state[done_key] = new_tid
            st.rerun()


@st.fragment(run_every=1.0)
def _render_running_detail(task: dict, mgr: TaskManager, tid: str):
    live = mgr.get_task(tid)
    if live and live["status"] != TaskStatus.RUNNING.value:
        st.rerun()
        return

    pct = task.get("progress", 0) * 100
    st.progress(task.get("progress", 0))
    st.markdown(f"**进度: {pct:.0f}%**")

    cancel_requested = task.get("_cancel_requested", False)
    if cancel_requested:
        st.info("⏳ 正在停止... (等待当前 episode 完成后进行评估)")
    elif st.button("取消训练", key=f"detail_cancel_{tid}"):
        mgr.cancel(tid)
        st.session_state[f"_cancel_{tid}"] = True
        st.rerun()

    pdata = mgr.get_progress_data(tid)
    _render_progress_chart(pdata)


def _render_progress_chart(pdata):
    if not pdata or len(pdata) < 2:
        return
    df = pd.DataFrame(pdata, columns=["ep", "value"])
    best_idx = df["value"].idxmax()
    best_val = df.loc[best_idx, "value"]
    best_ep = int(df.loc[best_idx, "ep"])

    window = max(5, len(df) // 10)
    df["trend"] = df["value"].rolling(window=window, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["ep"], y=df["value"],
        mode="lines", name="原始",
        line=dict(color="#94a3b8", width=1),
        opacity=0.5,
    ))
    fig.add_trace(go.Scatter(
        x=df["ep"], y=df["trend"],
        mode="lines", name="趋势",
        line=dict(color="#ef4444", width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=[best_ep], y=[best_val],
        mode="markers+text",
        name="最优",
        marker=dict(color="#22c55e", size=12, symbol="star"),
        text=[f"<b>Reward {best_val:.4f}</b>"],
        textposition="top center",
        textfont=dict(color="#22c55e", size=12),
    ))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="Episode", yaxis_title="Reward",
                      showlegend=True)
    st.plotly_chart(fig, width='stretch')
    st.caption(f"🏆 最优 Reward: 第 {best_ep} episode, Reward={best_val:.4f}")


def _render_detail(task: dict, mgr: TaskManager):
    tid = task.get("_id", "")
    status = task["status"]
    emoji = STATUS_EMOJI.get(status, "❓")

    st.button("← 返回任务列表", on_click=lambda: st.session_state.pop("selected_task_id", None))
    st.title(f"{emoji} {task['type']}  `{tid[:8]}...`")
    st.caption(f"创建: {task['created_at']}")

    if status == TaskStatus.RUNNING.value:
        _render_running_detail(task, mgr, tid)

    elif status in (TaskStatus.COMPLETED.value, TaskStatus.EARLY_STOPPED.value):
        if status == TaskStatus.EARLY_STOPPED.value:
            st.warning("⚠️ 训练已手动停止，以下为部分训练后的评估结果")
        result = mgr.get_result(tid)
        if result:
            st.subheader(f"📊 {task['type']} 详细结果")
            if task["type"] == "RL训练":
                _render_rl_result(result, task)
            elif task["type"] == "HRL训练":
                _render_hrl_result(result, task)

        pdata = task.get("_progress_data") or mgr.get_progress_data(tid)
        if pdata:
            st.subheader("📈 训练过程")
            _render_progress_chart(pdata)

        # ── 重新训练 ──
        st.divider()
        expand_key = f"retrain_expand_{tid}"
        expand_val = st.session_state.get(expand_key, False)
        with st.expander("🔄 重新训练", expanded=expand_val):
            if task["type"] == "RL训练":
                _render_rl_retrain_form(task, mgr)
            elif task["type"] == "HRL训练":
                _render_hrl_retrain_form(task, mgr)

    elif status == TaskStatus.FAILED.value:
        st.error(f"训练失败: {task.get('error', '未知错误')}")

    elif status == TaskStatus.CANCELLED.value:
        st.warning("训练已被取消")

    elif status == TaskStatus.PENDING.value:
        st.info("等待中... (排队中，前 3 个任务完成后自动开始)")


def _render_list(tasks: list, mgr: TaskManager):
    for i, task in enumerate(tasks):
        tid = task.get("_id", "")
        status = task["status"]
        emoji = STATUS_EMOJI.get(status, "❓")

        cols = st.columns([2.5, 1, 1.5, 1, 0.6])
        with cols[0]:
            st.markdown(f"**{task['type']}** `{tid[:8]}...`")
            st.caption(f"创建: {task['created_at']}")
        with cols[1]:
            st.markdown(f"{emoji} **{status}**")
        with cols[2]:
            if status == TaskStatus.RUNNING.value:
                pct = task.get("progress", 0) * 100
                st.progress(task.get("progress", 0))
                st.caption(f"{pct:.0f}%")
            elif status == TaskStatus.FAILED.value and task.get("error"):
                st.caption(task["error"][:40])
            elif status == TaskStatus.COMPLETED.value and task.get("finished_at"):
                st.caption(f"完成: {task['finished_at']}")
            elif status == TaskStatus.PENDING.value:
                st.caption("等待中...")
        with cols[3]:
            st.button("查看详情", key=f"view_{tid}",
                      on_click=_select_task, args=(tid,))
        with cols[4]:
            cancel_requested = st.session_state.get(f"_cancel_{tid}", False)
            if status == TaskStatus.PENDING.value:
                if st.button("取消", key=f"cancel_{tid}"):
                    mgr.cancel(tid)
                    st.rerun()
            elif status == TaskStatus.RUNNING.value and not cancel_requested:
                if st.button("取消", key=f"cancel_{tid}"):
                    mgr.cancel(tid)
                    st.session_state[f"_cancel_{tid}"] = True
                    st.rerun()
            elif status == TaskStatus.RUNNING.value and cancel_requested:
                st.caption("停止中...")

        if i < len(tasks) - 1:
            st.divider()


def _render_model_browser():
    model_dir = Path("saved_models/rl")
    files = sorted(model_dir.glob("*.pt"), reverse=True) if model_dir.exists() else []
    st.sidebar.markdown("### 📂 已保存模型")
    if not files:
        st.sidebar.caption("暂无已保存的模型")
        return
    names = [m.stem for m in files]
    sel = st.sidebar.selectbox("选择模型", names, key="task_model_browser")
    c1, c2 = st.sidebar.columns(2)
    if c1.button("📥 加载", key="task_model_load"):
        p = str(model_dir / f"{sel}.pt")
        loaded = DQNAgent.load(p)
        st.session_state.rl_agent = loaded
        meta = torch.load(p, map_location="cpu", weights_only=False).get("metadata", {})
        st.session_state.rl_model_info = {"path": p, "name": sel, **meta}
        st.rerun()
    if c2.button("🗑 删除", key="task_model_del"):
        (model_dir / f"{sel}.pt").unlink()
        st.rerun()
    with st.sidebar.expander("✏️ 重命名", expanded=False):
        rename_to = st.text_input("新名称", value=sel, key="task_model_rename_input")
        if st.button("确认重命名", key="task_model_rename_btn"):
            old_p = model_dir / f"{sel}.pt"
            new_p = model_dir / f"{rename_to}.pt"
            if not new_p.exists():
                old_p.rename(new_p)
                st.rerun()
            else:
                st.sidebar.error("名称已存在")


def _select_task(tid: str):
    st.session_state.selected_task_id = tid


def render_task_manager():
    _render_model_browser()
    st.title("📋 训练任务管理")
    mgr = TaskManager()
    tasks = mgr.list_tasks()

    selected = st.session_state.get("selected_task_id")

    if selected:
        task = mgr.get_task(selected)
        if task:
            task["_id"] = selected
            _render_detail(task, mgr)
        else:
            st.session_state.pop("selected_task_id", None)
            st.rerun()
    else:
        if not tasks:
            st.info("暂无训练任务")
            return
        _render_list(tasks, mgr)
