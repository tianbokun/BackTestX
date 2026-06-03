from pathlib import Path
from datetime import datetime

import torch
from utils.i18n import t
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from data_fetcher import add_premium_rate, ensure_ohlc
from backtest.rl.trainer import (
    train_dqn, evaluate, run_bh_baseline,
)
from backtest.rl.dqn_agent import DQNAgent
from backtest.rl.feature_engineer import FEATURE_GROUPS, DEFAULT_FEATURE_GROUPS
from data.symbol_registry import SymbolRegistry
from backtest.rl.task_manager import TaskManager, TaskStatus
from ui.rl_signal import render_rl_signal


def _rl_train_task(params, task_id=None, cancel_check=None):
    from backtest.rl.trainer import train_dqn, evaluate, run_bh_baseline
    mgr = TaskManager()

    episodes_done = [0]

    def progress(ep, total, loss):
        episodes_done[0] = ep + 1
        mgr.update_progress(task_id, (ep + 1) / total, progress_data=(ep, loss))

    sym = params["meta"]["symbol"]
    agent, _, agent_best = train_dqn(
        params["df_train"],
        system_version=params["system_version"],
        feature_groups=params["feature_groups"],
        progress_callback=progress,
        cancel_check=cancel_check,
        symbol=sym,
        **params["dqn_params"],
        **params["fee_params"],
    )
    if cancel_check() and episodes_done[0] == 0:
        return None

    result_dqn = evaluate(agent, params["df_test"],
                          system_version=params["system_version"],
                          feature_groups=params["feature_groups"],
                          symbol=sym,
                          **params["fee_params"])
    result_dqn_best = evaluate(agent_best, params["df_test"],
                               system_version=params["system_version"],
                               feature_groups=params["feature_groups"],
                               symbol=sym,
                               **params["fee_params"]) if agent_best else None
    result_bh = run_bh_baseline(params["df_test"],
                                initial_capital=params["fee_params"]["initial_capital"])

    return {
        "agent": agent,
        "agent_best": agent_best,
        "result_dqn": result_dqn,
        "result_dqn_best": result_dqn_best,
        "result_bh": result_bh,
        "meta": params["meta"],
    }


def _hyperparam_search_task(params, task_id=None, cancel_check=None):
    from backtest.rl.trainer import hyperparam_search, train_dqn, evaluate, run_bh_baseline
    mgr = TaskManager()

    df_hp = params["df_hp"]
    df_train = params["df_train"]
    df_val = params["df_val"]
    sv = params["system_version"]
    fg = params["feature_groups"]
    fee = params["fee_params"]
    meta = params["meta"]

    total_combos = 324  # 4(lr) × 3(gamma) × 3(hidden) × 3(n_episodes) × 3(epsilon_decay)
    n_folds = 3
    total_folds = total_combos * n_folds

    def _hp_fold_callback(ci, total, fi, nf, p_params, fold_sharpe):
        folds_done = ci * nf + fi + 1
        mgr.update_progress(task_id, folds_done / total_folds)

    best_scores = []
    def _hp_combo_callback(ci, total, best_params, best_score):
        if best_score > -999:
            best_scores.append(best_score)
        current_best = max(best_scores) if best_scores else 0
        mgr.update_progress(task_id, min((ci + 1) * n_folds / total_folds, 0.99),
                            progress_data=(ci, current_best))

    hp_result = hyperparam_search(
        df=df_hp, system_version=sv, feature_groups=fg,
        progress_callback=None,
        combo_callback=_hp_combo_callback,
        fold_callback=_hp_fold_callback,
        cancel_check=cancel_check,
        reward_window=meta.get("reward_window", 63),
        vol_penalty_coef=meta.get("vol_penalty_coef", 0.1),
        dd_penalty_coef=meta.get("dd_penalty_coef", 0.5),
        commission_rate=fee.get("commission_rate", 0.00025),
        min_commission=fee.get("min_commission", 5.0),
        stamp_duty=fee.get("stamp_duty", 0.001),
        initial_capital=fee.get("initial_capital", 100000.0),
    )

    was_cancelled = cancel_check()
    if was_cancelled and hp_result.get("best_params") is None:
        return None

    bp = hp_result.get("best_params")
    val_result = bh_val = agent = None
    sym = meta.get("symbol", "")
    if bp is not None and not was_cancelled:
        agent, _, _ = train_dqn(
            df_train, system_version=sv, feature_groups=fg,
            n_episodes=bp["n_episodes"], lr=bp["lr"], gamma=bp["gamma"],
            hidden=bp["hidden"], epsilon_decay=bp["epsilon_decay"],
            progress_callback=None, cancel_check=cancel_check,
            symbol=sym, **fee,
        )
        val_result = evaluate(agent, df_val, system_version=sv,
                              feature_groups=fg, symbol=sym, **fee)
        bh_val = run_bh_baseline(df_val, initial_capital=fee["initial_capital"])

    return {
        "hp_result": hp_result,
        "agent": agent,
        "agent_best": None,
        "result_dqn": val_result,
        "result_bh": bh_val,
        "meta": meta,
    }


def render_rl_training(end_date, adjust):
    st.title(t("rl.title"))

    st.sidebar.markdown(t("rl.sidebar.header"))

    all_symbols = SymbolRegistry.list()
    if not all_symbols:
        st.error(t("rl.error.no_symbols"))
        st.stop()

    type_all = t("status.all")
    type_filter = st.sidebar.selectbox(
        t("sidebar.asset_type"),
        [type_all] + sorted(set(s["asset_type"] for s in all_symbols)),
        key="rl_reg_type",
    )
    filtered = all_symbols if type_filter == type_all else [s for s in all_symbols if s["asset_type"] == type_filter]
    symbol_options = {f"{s['symbol']} - {s['name']}": s["symbol"] for s in filtered}
    selected_label = st.sidebar.selectbox(
        t("rl.sidebar.symbol_select"), list(symbol_options.keys()), key="rl_reg_symbol"
    )
    symbol = symbol_options[selected_label]
    entry = SymbolRegistry.get(symbol)
    asset_type = entry["asset_type"] if entry else "stock"

    rename_map = {"开盘": "开盘价", "收盘": "收盘价", "最高": "最高价", "最低": "最低价"}

    with st.spinner(t("app.fetching")):
        df = SymbolRegistry.fetch_data(symbol, adjust=adjust)
    if df is None or df.empty:
        st.error(t("rl.error.fetch_failed", symbol=symbol))
        st.stop()
    df = df.copy()
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
    df = ensure_ohlc(df)
    df = add_premium_rate(df, symbol, asset_type)
    has_premium = "溢价率" in df.columns

    total_n = len(df)
    # 自动 60/20/20 分割
    train_end_i = min(int(total_n * 0.6), total_n - 2)
    val_end_i = min(int(total_n * 0.8), total_n - 1)

    def _d(i):
        return pd.Timestamp(str(df.index[min(i, total_n - 1)])).date()

    train_start_def = _d(0)
    train_end_def = _d(train_end_i)
    val_start_def = _d(train_end_i + 1) if train_end_i + 1 < total_n else train_end_def
    val_end_def = _d(val_end_i)
    test_start_def = _d(val_end_i + 1) if val_end_i + 1 < total_n else val_end_def
    test_end_def = _d(-1)

    # 侧边栏参数
    st.sidebar.markdown(t("rl.sidebar.header"))

    train_start = st.sidebar.date_input(t("rl.sidebar.train_start"), value=train_start_def)
    train_end = st.sidebar.date_input(t("rl.sidebar.train_end"), value=train_end_def)
    val_start = st.sidebar.date_input(t("rl.sidebar.val_start"), value=val_start_def)
    val_end = st.sidebar.date_input(t("rl.sidebar.val_end"), value=val_end_def)
    test_start = st.sidebar.date_input(t("rl.sidebar.test_start"), value=test_start_def)
    test_end = st.sidebar.date_input(t("rl.sidebar.test_end"), value=test_end_def)

    system_version = st.sidebar.selectbox(
        t("rl.sidebar.system_version"),
        options=["basic", "1.0", "2.0"],
        format_func=lambda x: {
            "basic": t("rl.sidebar.system.basic"),
            "1.0": t("rl.sidebar.system.v1"),
            "2.0": t("rl.sidebar.system.v2"),
        }[x],
        index=1,
        help=t("rl.sidebar.system.help"),
    )

    # ── 特征选择 ──
    selected_groups = list(DEFAULT_FEATURE_GROUPS)
    if system_version == "basic":
        selected_groups = []
    with st.sidebar.expander(t("rl.sidebar.features"), expanded=False):
        st.caption(t("rl.sidebar.features.desc"))
        for key, grp in FEATURE_GROUPS.items():
            is_on = st.checkbox(
                grp["label"], value=key in DEFAULT_FEATURE_GROUPS,
                key=f"fg_{key}", help=grp.get("help", ""),
                disabled=(system_version == "basic"),
            )
            if is_on:
                if key not in selected_groups:
                    selected_groups.append(key)
            else:
                if key in selected_groups:
                    selected_groups.remove(key)

    with st.sidebar.expander(t("rl.sidebar.fee"), expanded=False):
        rl_commission = st.number_input(t("rl.sidebar.commission"), min_value=0.0, value=0.000235, step=0.000005, format="%.6f",
                                        key="rl_commission", help=t("rl.sidebar.commission.help"))
        rl_min_commission = st.number_input(t("rl.sidebar.min_commission"), min_value=0.0, value=5.0, step=1.0,
                                            key="rl_min_comm", help=t("rl.sidebar.min_commission.help"))
        rl_stamp_duty = st.number_input(t("rl.sidebar.stamp"), min_value=0.0, value=0.001, step=0.0001, format="%.4f",
                                        key="rl_stamp", help=t("rl.sidebar.stamp.help"))
        rl_capital = st.number_input(t("rl.sidebar.initial_capital"), min_value=100.0, value=100000.0, step=10000.0,
                                     key="rl_capital",
                                     help=t("rl.sidebar.initial_capital.help"))

    with st.sidebar.expander(t("rl.sidebar.hyperparams"), expanded=False):
        n_episodes = st.number_input(t("rl.sidebar.episodes"), min_value=10, value=64, step=10)
        batch_size = st.number_input(t("rl.sidebar.batch_size"), min_value=32, value=200, step=32)
        lr = st.text_input(t("rl.sidebar.learning_rate"), value="1e-5")
        gamma = st.text_input(t("rl.sidebar.gamma"), value="0.98")
        hidden = st.number_input(t("rl.sidebar.hidden_dim"), min_value=32, value=128, step=32)
        epsilon_start = st.text_input(t("rl.sidebar.epsilon_start"), value="0.9")
        epsilon_end = st.text_input(t("rl.sidebar.epsilon_end"), value="0.01")
        epsilon_decay = st.number_input(t("rl.sidebar.epsilon_decay"), min_value=100, value=500, step=100)
        target_update = st.number_input(t("rl.sidebar.target_update"), min_value=10, value=50, step=10)
        buffer_capacity = st.number_input(t("rl.sidebar.replay_capacity"), min_value=1000, value=10000, step=1000)

    with st.sidebar.expander(t("rl.sidebar.reward"), expanded=False):
        rl_reward_window = st.slider(t("rl.sidebar.reward_window"), min_value=5, max_value=252, value=63, step=5,
                                     help=t("rl.sidebar.reward_window.help"))
        rl_vol_penalty = st.number_input(t("rl.sidebar.vol_penalty"), min_value=0.0, max_value=1.0, value=0.1, step=0.05,
                                         help=t("rl.sidebar.vol_penalty.help"))
        rl_dd_penalty = st.number_input(t("rl.sidebar.drawdown_penalty"), min_value=0.0, max_value=5.0, value=1.0, step=0.1,
                                        help=t("rl.sidebar.drawdown_penalty.help"))

    search_btn = st.sidebar.button(t("rl.sidebar.btn.hp_search"), width='stretch')
    run_btn = st.sidebar.button(t("rl.sidebar.btn.train"), type="primary", width='stretch')

    st.sidebar.markdown("---")
    st.sidebar.markdown(t("rl.sidebar.saved_models"))
    model_dir = Path("saved_models/rl")
    model_files = sorted(model_dir.glob("*.pt"), reverse=True) if model_dir.exists() else []
    if model_files:
        names = [m.stem for m in model_files]
        selected_name = st.sidebar.selectbox(t("rl.sidebar.model_select"), names, key="rl_model_selector")
        col_s1, col_s2 = st.sidebar.columns(2)
        if col_s1.button(t("rl.sidebar.btn.load"), width='stretch', key="rl_load_btn"):
            selected_path = str(model_dir / f"{selected_name}.pt")
            loaded = DQNAgent.load(selected_path)
            st.session_state.rl_agent = loaded
            meta = torch.load(selected_path, map_location="cpu", weights_only=False).get("metadata", {})
            st.session_state.rl_model_info = {"path": selected_path, "name": selected_name, **meta}
            st.session_state.rl_model_just_saved = False
            st.rerun()
        if col_s2.button(t("rl.sidebar.btn.delete"), width='stretch', key="rl_del_btn"):
            (model_dir / f"{selected_name}.pt").unlink()
            if st.session_state.rl_model_info and st.session_state.rl_model_info.get("name") == selected_name:
                st.session_state.rl_agent = None
                st.session_state.rl_model_info = None
            st.session_state.rl_model_just_saved = False
            st.rerun()

        with st.sidebar.expander(t("rl.sidebar.rename"), expanded=False):
            rename_to = st.text_input(t("rl.sidebar.rename_new"), value=selected_name, key="rl_rename_input")
            if st.button(t("rl.sidebar.rename_confirm"), key="rl_rename_btn"):
                if rename_to and rename_to != selected_name:
                    old_p = model_dir / f"{selected_name}.pt"
                    new_p = model_dir / f"{rename_to}.pt"
                    if not new_p.exists():
                        old_p.rename(new_p)
                        st.rerun()
                    else:
                        st.error(t("rl.sidebar.name_exists"))
    else:
        if st.session_state.rl_model_just_saved:
            st.sidebar.success(t("rl.sidebar.save_success"))
        else:
            st.sidebar.caption(t("rl.sidebar.no_models"))

    # ── 划分数据集 ──
    df_train = df[(df.index >= pd.Timestamp(train_start)) & (df.index <= pd.Timestamp(train_end))].copy()
    df_val = df[(df.index >= pd.Timestamp(val_start)) & (df.index <= pd.Timestamp(val_end))].copy()
    df_test = df[(df.index >= pd.Timestamp(test_start)) & (df.index <= pd.Timestamp(test_end))].copy()

    if len(df_train) < 50:
        st.error(t("rl.error.train_data_short", n=len(df_train)))
        st.stop()
    if len(df_test) < 20:
        st.error(t("rl.error.test_data_short", n=len(df_test)))
        st.stop()

    st.info(t("rl.info.data_split",
              start=str(train_start)[:10], end=str(train_end)[:10], n=len(df_train),
              vs=str(val_start)[:10], ve=str(val_end)[:10], vn=len(df_val),
              ts=str(test_start)[:10], te=str(test_end)[:10], tn=len(df_test)))

    rl_capital_val = float(rl_capital)
    fee_params = dict(commission_rate=float(rl_commission),
                      min_commission=float(rl_min_commission),
                      stamp_duty=float(rl_stamp_duty),
                      initial_capital=rl_capital_val)

    # ── 超参搜索 ──
    if search_btn:
        if len(df_val) < 20:
            st.error(t("rl.error.val_data_short", n=len(df_val)))
            st.stop()
        df_hp = pd.concat([df_train, df_val]).sort_index()
        total_days = len(df_hp)
        st.info(t("rl.info.hp_window", start=str(df_hp.index[0])[:10], end=str(df_hp.index[-1])[:10], n=total_days))

        sv = system_version
        fg = list(selected_groups)
        fp = dict(fee_params)

        hp_params = {
            "df_hp": df_hp,
            "df_train": df_train,
            "df_val": df_val,
            "system_version": sv,
            "feature_groups": fg,
            "fee_params": fp,
            "meta": {
                "symbol": symbol,
                "system_version": sv,
                "feature_groups": fg,
                "train_start": str(train_start),
                "train_end": str(train_end),
                "val_start": str(val_start),
                "val_end": str(val_end),
                "adjust": adjust,
                "reward_window": int(rl_reward_window),
                "vol_penalty_coef": float(rl_vol_penalty),
                "dd_penalty_coef": float(rl_dd_penalty),
            },
        }
        mgr = TaskManager()
        tid = mgr.submit("超参搜索", hp_params, _hyperparam_search_task, args=(hp_params,))
        st.success(t("rl.success.hp_submitted", id=tid[:8]))

    # ── 训练 (后台任务) ──
    if run_btn:
        try:
            dqn_params = {
                "n_episodes": int(n_episodes), "batch_size": int(batch_size),
                "lr": float(lr), "gamma": float(gamma), "hidden": int(hidden),
                "epsilon_start": float(epsilon_start), "epsilon_end": float(epsilon_end),
                "epsilon_decay": int(epsilon_decay), "target_update": int(target_update),
                "buffer_capacity": int(buffer_capacity),
                "reward_window": int(rl_reward_window),
                "vol_penalty_coef": float(rl_vol_penalty),
                "dd_penalty_coef": float(rl_dd_penalty),
            }
        except ValueError:
            st.error(t("rl.error.hp_format"))
            st.stop()

        task_params = {
            "df_train": df_train,
            "df_test": df_test,
            "system_version": system_version,
            "feature_groups": list(selected_groups),
            "dqn_params": dqn_params,
            "fee_params": dict(fee_params),
            "meta": {
                "symbol": symbol,
                "system_version": system_version,
                "train_start": str(train_start),
                "train_end": str(train_end),
                "df_test_index": df_test.index,
                "adjust": adjust,
            },
        }
        mgr = TaskManager()
        task_id = mgr.submit("RL训练", task_params, _rl_train_task, args=(task_params,))
        st.success(t("rl.success.train_submitted", id=task_id[:8]))

    # ── 从已完成任务加载结果 ──
    rl_loaded_task_id = st.session_state.get("rl_loaded_task_id")
    if rl_loaded_task_id:
        mgr = TaskManager()
        task = mgr.get_task(rl_loaded_task_id)
        if task and task["status"] in (TaskStatus.COMPLETED.value, TaskStatus.EARLY_STOPPED.value):
            result = mgr.get_result(rl_loaded_task_id)
            if result is not None:
                st.session_state.rl_trained_agent = result["agent"]
                st.session_state.rl_best_agent = result.get("agent_best")
                st.session_state.rl_dqn_result = result["result_dqn"]
                st.session_state.rl_dqn_best_result = result.get("result_dqn_best")
                st.session_state.rl_bh_result = result["result_bh"]
                st.session_state.rl_train_meta = result["meta"]

    # ── 展示训练结果 + 保存按钮 (在 run_btn 外部, 持久化) ──
    if st.session_state.rl_trained_agent is not None:
        agent = st.session_state.rl_trained_agent
        agent_best = st.session_state.get("rl_best_agent")
        result_dqn = st.session_state.rl_dqn_result
        result_dqn_best = st.session_state.get("rl_dqn_best_result")
        result_bh = st.session_state.rl_bh_result
        meta_info = st.session_state.rl_train_meta

        st.markdown("---")
        st.subheader(t("rl.result.title"))

        st.markdown(t("rl.result.compare_header"))
        col_strategy = t("rl.col.strategy")
        col_final = t("rl.col.final_amount")
        col_return = t("rl.col.return")
        col_sharpe = t("rl.col.sharpe")
        col_mdd = t("rl.col.max_drawdown")
        col_trades = t("rl.col.trade_count")
        col_final_name = t("rl.result.strategy.final")
        col_best_name = t("rl.result.strategy.best")
        col_bh_name = t("rl.result.strategy.bh")
        rows = [
            {col_strategy: col_final_name, col_final: result_dqn["final_value"],
             col_return: result_dqn["total_return_pct"],
             col_sharpe: result_dqn["sharpe_ratio"],
             col_mdd: result_dqn["max_drawdown_pct"],
             col_trades: result_dqn["num_trades"]},
        ]
        if result_dqn_best is not None:
            rows.append({col_strategy: col_best_name, col_final: result_dqn_best["final_value"],
                         col_return: result_dqn_best["total_return_pct"],
                         col_sharpe: result_dqn_best["sharpe_ratio"],
                         col_mdd: result_dqn_best["max_drawdown_pct"],
                         col_trades: result_dqn_best["num_trades"]})
        rows.append({col_strategy: col_bh_name, col_final: result_bh["final_value"],
                     col_return: result_bh["total_return_pct"],
                     col_sharpe: result_bh["sharpe_ratio"],
                     col_mdd: result_bh["max_drawdown_pct"],
                     col_trades: 0})
        comp = pd.DataFrame(rows)
        st.dataframe(comp, width='stretch', hide_index=True)

        st.markdown(t("rl.result.profit_chart"))
        test_idx = meta_info.get("df_test_index", result_dqn.get("dates"))
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=test_idx, y=result_dqn["equity_curve"],
            mode="lines", name=t("rl.chart.trace.final", version=meta_info['system_version']),
            line=dict(color="#1f77b4", width=2),
        ))
        if result_dqn_best is not None:
            fig.add_trace(go.Scatter(
                x=test_idx, y=result_dqn_best["equity_curve"],
                mode="lines", name=t("rl.chart.trace.best"),
                line=dict(color="#2ca02c", width=2, dash="dot"),
            ))
        fig.add_trace(go.Scatter(
            x=test_idx, y=result_bh["equity_curve"],
            mode="lines", name=t("rl.chart.trace.bh"),
            line=dict(color="#ff7f0e", width=2, dash="dash"),
        ))
        fig.add_hline(y=rl_capital_val, line_dash="dot", line_color="gray", annotation_text=t("rl.chart.initial_capital"))
        fig.update_layout(
            xaxis_title=t("dca.axis.date"), yaxis_title=t("rl.chart.axis.account_value", capital=rl_capital_val),
            hovermode="x unified", height=400,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig, width='stretch')

        # ── 交易记录（可切换） ──
        dqn_best_has_trades = result_dqn_best is not None and not result_dqn_best["trades"].empty
        dqn_final_has_trades = not result_dqn["trades"].empty
        if dqn_final_has_trades or dqn_best_has_trades:
            st.markdown(t("rl.result.trade_log"))
            view_final = t("rl.result.view.final")
            view_best = t("rl.result.view.best")
            trade_source = st.selectbox(
                t("rl.result.view_select"), [view_final, view_best],
                key="rl_trade_source",
                disabled=not (dqn_final_has_trades and dqn_best_has_trades),
            )
            trades_df = (result_dqn_best["trades"] if trade_source == view_best else result_dqn["trades"]).copy()
            if not trades_df.empty:
                trades_df["日期"] = trades_df["日期"].dt.strftime("%Y-%m-%d")
                st.dataframe(trades_df, width='stretch', hide_index=True)

        # ── 保存模型 ──
        sv = meta_info["system_version"]
        sym = meta_info["symbol"]
        _train_save_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        _train_save_default = f"{sym}_{sv}_{_train_save_ts}"
        _train_save_name = st.text_input(t("rl.save.model_name"), value=_train_save_default, key="rl_train_save_name")
        choice_best = t("rl.save.choice.best")
        save_choice = st.radio(t("rl.save.which_model"), [t("rl.result.view.final"), choice_best],
                               index=0, horizontal=True, key="rl_save_choice")
        save_col1, save_col2 = st.columns([1, 5])
        with save_col1:
            if st.button(t("rl.save.btn"), type="primary", key="rl_save_model_btn"):
                save_agent = agent_best if save_choice == choice_best and agent_best is not None else agent
                save_result = result_dqn_best if save_choice == choice_best and result_dqn_best is not None else result_dqn
                save_path = Path(f"saved_models/rl/{_train_save_name}.pt")
                save_path.parent.mkdir(parents=True, exist_ok=True)
                save_agent.save(str(save_path), {
                    "symbol": sym, "system_version": sv,
                    "feature_groups": selected_groups,
                    "train_start": meta_info["train_start"],
                    "train_end": meta_info["train_end"],
                    "test_return": save_result["total_return_pct"],
                    "sharpe": save_result["sharpe_ratio"],
                })
                st.session_state.rl_agent = save_agent
                st.session_state.rl_model_info = {
                    "name": save_path.stem, "path": str(save_path),
                    "symbol": sym, "system_version": sv,
                    "feature_groups": selected_groups,
                }
                st.session_state.rl_model_just_saved = True
                st.rerun()
        with save_col2:
            st.caption(t("rl.save.caption"))

    else:
        if not run_btn and not search_btn:
            st.info(t("rl.info.waiting"))

    # 实时信号面板（有加载模型时显示）
    render_rl_signal(df, symbol, asset_type)
