from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import torch

from utils.i18n import t, tt

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
        st.subheader(t("task.save.header"))
        for label, path in auto.items():
            p = Path(path)
            st.code(f"{p.name}  ({p.parent})", language="")
            st.caption(t("task.save.auto_saved", label=label))
        st.caption(t("task.save.auto_hint"))
        return

    agent = result.get("agent")
    if agent is None:
        st.info(t("task.save.hint"))
        return
    meta = result.get("meta", {})
    agent_best = result.get("agent_best")
    dqn = result["result_dqn"]
    dqn_best = result.get("result_dqn_best")
    sym = meta.get("symbol", "unknown")
    sv = meta.get("system_version", "1.0")
    _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _default = f"{sym}_{sv}_{_ts}"
    st.subheader(t("task.save.header"))
    _name = st.text_input(t("task.save.model_name"), value=_default, key="task_rl_save_name")
    _save_final = t("task.save.final")
    _save_best = t("task.save.best")
    choice = _save_final
    if agent_best is not None:
        choice = st.radio(t("task.save.which"), [_save_final, _save_best],
                          index=0, horizontal=True, key="task_rl_save_choice")
    if st.button(t("task.save.btn"), type="primary", key="task_rl_save_btn"):
        save_agent = agent_best if (choice == _save_best and agent_best is not None) else agent
        save_result = dqn_best if (choice == _save_best and dqn_best is not None) else dqn
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
        st.success(t("task.save.saved", path=p))
        st.rerun()


def _render_hrl_save(result: dict, task: dict = None):
    auto = (task or {}).get("auto_saved_models", {})
    if auto:
        st.subheader(t("task.save.header"))
        for label, path in auto.items():
            p = Path(path)
            st.code(f"{p.name}  ({p.parent})", language="")
            st.caption(t("task.save.auto_saved", label=label))
        st.caption(t("task.save.auto_hint"))
        return

    trainer = result.get("agent")
    if trainer is None:
        st.info(t("task.save.hint"))
        return
    syms = result.get("selected_symbols", [])
    label = "_".join(syms[:3]) if syms else "hrl"
    _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _default = f"{label}_{_ts}"
    st.subheader(t("task.save.header"))
    _name = st.text_input(t("task.save.model_name"), value=_default, key="task_hrl_save_name")
    if st.button(t("task.save.btn"), type="primary", key="task_hrl_save_btn"):
        p = Path(f"saved_models/rl/{_name}.pt")
        p.parent.mkdir(parents=True, exist_ok=True)
        trainer.save(str(p))
        st.success(t("task.save.hrl_saved", path=p))
        st.rerun()


def _render_rl_result(result: dict, task: dict = None):
    dqn = result["result_dqn"]
    dqn_best = result.get("result_dqn_best")
    bh = result["result_bh"]
    meta = result.get("meta", {})

    _c_strat = t("task.rl.col.strategy")
    _c_final = t("task.rl.col.final_amount")
    _c_return = t("task.rl.col.return")
    _c_sharpe = t("task.rl.col.sharpe")
    _c_drawdown = t("task.rl.col.max_drawdown")
    _c_trades = t("task.rl.col.trade_count")

    _dl_final = t("rl.result.strategy.final")
    _dl_best = t("rl.result.strategy.best")
    _dl_bh = t("rl.result.strategy.bh")

    rows = [
        {_c_strat: _dl_final, _c_final: dqn["final_value"],
         _c_return: dqn["total_return_pct"],
         _c_sharpe: dqn["sharpe_ratio"],
         _c_drawdown: dqn["max_drawdown_pct"],
         _c_trades: dqn["num_trades"]},
    ]
    if dqn_best is not None:
        rows.append({_c_strat: _dl_best, _c_final: dqn_best["final_value"],
                     _c_return: dqn_best["total_return_pct"],
                     _c_sharpe: dqn_best["sharpe_ratio"],
                     _c_drawdown: dqn_best["max_drawdown_pct"],
                     _c_trades: dqn_best["num_trades"]})
    rows.append({_c_strat: _dl_bh, _c_final: bh["final_value"],
                 _c_return: bh["total_return_pct"],
                 _c_sharpe: bh["sharpe_ratio"],
                 _c_drawdown: bh["max_drawdown_pct"],
                 _c_trades: 0})
    comp = pd.DataFrame(rows)
    st.dataframe(comp, width='stretch', hide_index=True)

    test_idx = _ensure_dates(meta.get("df_test_index", dqn.get("dates")))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=test_idx, y=dqn["equity_curve"],
        mode="lines", name=t("rl.chart.trace.final", version=meta.get('system_version', '?')),
        line=dict(color="#1f77b4", width=2),
    ))
    if dqn_best is not None:
        fig.add_trace(go.Scatter(
            x=test_idx, y=dqn_best["equity_curve"],
            mode="lines", name=t("rl.chart.trace.best"),
            line=dict(color="#2ca02c", width=2, dash="dot"),
        ))
    fig.add_trace(go.Scatter(
        x=test_idx, y=bh["equity_curve"],
        mode="lines", name=t("rl.chart.trace.bh"),
        line=dict(color="#ff7f0e", width=2, dash="dash"),
    ))
    fig.update_layout(
        xaxis_title=t("dca.axis.date"), yaxis_title=t("rl.chart.profit"),
        hovermode="x unified", height=350,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig, width='stretch')

    dqn_final_has_trades = not _ensure_trades(dqn.get("trades")).empty
    dqn_best_has_trades = dqn_best is not None and not _ensure_trades(dqn_best.get("trades")).empty
    if dqn_final_has_trades or dqn_best_has_trades:
        st.markdown(t("task.rl.trade_log"))
        _view_final = t("task.save.final")
        _view_best = t("task.save.best")
        trade_source = st.selectbox(
            t("task.rl.view_select"), [_view_final, _view_best],
            key="task_rl_trade_source",
            disabled=not (dqn_final_has_trades and dqn_best_has_trades),
        )
        src = dqn_best if trade_source == _view_best else dqn
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

    _c_strat = t("task.rl.col.strategy")
    _c_final = t("task.rl.col.final_amount")
    _c_return = t("task.rl.col.return")
    _c_sharpe = t("task.rl.col.sharpe")
    _c_drawdown = t("task.rl.col.max_drawdown")

    _dl_hrl = t("task.hrl.strategy.hrl")
    _dl_ew = t("task.hrl.strategy.equal_weight")
    _dl_dca = t("task.hrl.strategy.monthly_dca")
    _dl_ma = t("task.hrl.strategy.ma_dca")

    rows = [{
        _c_strat: _dl_hrl,
        _c_final: test["final_value"],
        _c_return: test["total_return_pct"],
        _c_sharpe: test["sharpe_ratio"],
        _c_drawdown: test["max_drawdown_pct"],
    }]
    ew = benchmarks.get("equal_weight_bh")
    if ew:
        rows.append({
            _c_strat: _dl_ew,
            _c_final: ew["final_value"],
            _c_return: ew["total_return_pct"],
            _c_sharpe: ew["sharpe_ratio"],
            _c_drawdown: ew["max_drawdown_pct"],
        })
    for dca_key, dca_label in [("monthly_dca", _dl_dca), ("ma_adjust_dca", _dl_ma)]:
        dca = benchmarks.get(dca_key)
        if dca:
            rows.append({
                _c_strat: dca_label,
                _c_final: dca["final_value"],
                _c_return: dca["total_return_pct"],
                _c_sharpe: dca.get("sharpe_ratio", "N/A"),
                _c_drawdown: dca.get("max_drawdown_pct", "N/A"),
            })
    comp = pd.DataFrame(rows)
    st.dataframe(comp, width='stretch', hide_index=True)

    dates = _ensure_dates(test.get("dates", []))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=test["equity_curve"],
        mode="lines", name=_dl_hrl,
        line=dict(color="#2563eb", width=2),
    ))
    fig.add_hline(y=capital, line_dash="dot", line_color="gray", annotation_text=t("task.hrl.chart.initial_capital"))
    if ew:
        fig.add_trace(go.Scatter(
            x=dates, y=ew["equity_curve"],
            mode="lines", name=_dl_ew,
            line=dict(color="#ef4444", width=2, dash="dash"),
        ))
    for dca_key, dca_label, dca_color in [
        ("monthly_dca", _dl_dca, "#10b981"),
        ("ma_adjust_dca", _dl_ma, "#f59e0b"),
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
        xaxis_title=t("dca.axis.date"), yaxis_title=tt("账户总值", "Account Value"),
        hovermode="x unified", height=350,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig, width='stretch')

    if "position_ratios" in test and len(test["position_ratios"]) > 0:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=dates, y=test["position_ratios"],
            mode="lines", name=t("hrl.chart.nav"),
            line=dict(color="#2563eb", width=2),
            fill="tozeroy",
        ))
        fig2.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text=t("task.hrl.chart.half_position"))
        fig2.update_layout(
            xaxis_title=t("dca.axis.date"), yaxis_title=tt("仓位比例", "Position Ratio"),
            hovermode="x unified", height=200,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig2, width='stretch')

    trade_log = _ensure_trades(test.get("trade_log"))
    if not trade_log.empty:
        with st.expander(t("task.rl.trade_log"), expanded=False):
            st.dataframe(trade_log, width='stretch', hide_index=True)

    _render_hrl_save(result, task)


def _render_hp_result(result: dict, task: dict = None):
    hp = result.get("hp_result", {})
    dqn = result.get("result_dqn")
    bh = result.get("result_bh")

    bp = hp.get("best_params")
    bs = hp.get("best_score", -999)
    tc = hp.get("total_combos", 0)
    nf = hp.get("n_folds", 3)
    elapsed = hp.get("elapsed_sec", 0)

    st.subheader(t("task.hp.title"))
    st.caption(t("task.hp.summary", n=tc, k=nf, total=tc * nf, min=elapsed / 60))

    if bp:
        st.success(t("task.hp.best", s=bs))
        p_str = ", ".join(f"{k}={v}" for k, v in sorted(bp.items()))
        st.code(p_str, language="")
    else:
        st.warning(t("task.hp.no_valid"))

    fold_details = hp.get("fold_details", [])
    if fold_details:
        with st.expander(t("task.hp.top10"), expanded=False):
            sorted_d = sorted(fold_details, key=lambda x: x["avg_score"], reverse=True)
            rows = []
            for i, fd in enumerate(sorted_d[:10]):
                p = fd["params"]
                rows.append({
                    tt("排名", "Rank"): i + 1, "lr": p["lr"], "gamma": p["gamma"],
                    "hidden": p["hidden"], "n_episodes": p["n_episodes"],
                    "epsilon_decay": p["epsilon_decay"],
                    tt("平均夏普", "Avg Sharpe"): round(fd["avg_score"], 4),
                    tt("各折夏普", "Fold Sharpes"): ", ".join(f"{s:.4f}" for s in fd["fold_scores"]),
                })
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    if dqn is not None and bh is not None:
        st.divider()
        st.subheader(t("task.hp.val_result"))
        _c_strat = t("task.rl.col.strategy")
        _c_final = t("task.rl.col.final_amount")
        _c_return = t("task.rl.col.return")
        _c_sharpe = t("task.rl.col.sharpe")
        _c_drawdown = t("task.rl.col.max_drawdown")
        _c_trades = t("task.rl.col.trade_count")
        comp = pd.DataFrame([
            {_c_strat: "DQN", _c_final: dqn["final_value"],
             _c_return: dqn["total_return_pct"], _c_sharpe: dqn["sharpe_ratio"],
             _c_drawdown: dqn["max_drawdown_pct"], _c_trades: dqn["num_trades"]},
            {_c_strat: t("rl.result.strategy.bh"), _c_final: bh["final_value"],
             _c_return: bh["total_return_pct"], _c_sharpe: bh["sharpe_ratio"],
             _c_drawdown: bh["max_drawdown_pct"], _c_trades: 0},
        ])
        st.dataframe(comp, width='stretch', hide_index=True)

        trades = _ensure_trades(dqn.get("trades"))
        if not trades.empty:
            with st.expander(t("task.rl.trade_log")):
                t_df = trades.copy()
                if "日期" in t_df.columns:
                    t_df["日期"] = t_df["日期"].dt.strftime("%Y-%m-%d")
                st.dataframe(t_df, width='stretch', hide_index=True)

        _render_rl_save(result, task)


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
        raise ValueError(t("rl.error.fetch_failed", symbol=symbol))

    df_train = df[(df.index >= pd.Timestamp(train_start)) & (df.index <= pd.Timestamp(train_end))].copy()
    if len(test_dates) > 0:
        test_start_s = test_dates[0][:10]
        test_end_s = test_dates[-1][:10]
        df_test = df[(df.index >= pd.Timestamp(test_start_s)) & (df.index <= pd.Timestamp(test_end_s))].copy()
    else:
        df_test = df[df.index > pd.Timestamp(train_end)].copy()
    return {"df_train": df_train, "df_test": df_test, "df_test_index": df_test.index}


def _refetch_hp_data(meta: dict) -> dict:
    from data.symbol_registry import SymbolRegistry
    import pandas as pd

    symbol = meta.get("symbol", "")
    adjust = meta.get("adjust", "qfq")
    train_start = meta.get("train_start", "")
    train_end = meta.get("train_end", "")
    val_start = meta.get("val_start", "")
    val_end = meta.get("val_end", "")

    df = SymbolRegistry.fetch_data(symbol, adjust=adjust)
    if df is None or df.empty:
        raise ValueError(t("rl.error.fetch_failed", symbol=symbol))

    df_train = df[(df.index >= pd.Timestamp(train_start)) & (df.index <= pd.Timestamp(train_end))].copy()
    df_val = df[(df.index >= pd.Timestamp(val_start)) & (df.index <= pd.Timestamp(val_end))].copy()
    df_hp = pd.concat([df_train, df_val]).sort_index()
    return {"df_hp": df_hp, "df_train": df_train, "df_val": df_val}


def _render_rl_retrain_form(task: dict, mgr: TaskManager):
    tid = task.get("_id", "")

    done_key = f"retrain_rl_done_{tid}"
    if done_key in st.session_state:
        new_tid = st.session_state[done_key]
        st.success(t("task.retrain.submitted", id=new_tid[:8]))
        st.caption(t("task.retrain.hint"))
        if st.button(t("task.retrain.view_btn")):
            st.session_state.selected_task_id = new_tid
            st.rerun()
        return

    params = task.get("params")
    params_display = task.get("params_display")
    if params is None and params_display is None:
        st.info(t("rl.retrain.info"))
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
        st.info(t("rl.retrain.refetch"))

    with st.form(key=f"retrain_rl_{tid}", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            n_episodes = st.number_input(t("rl.sidebar.episodes"), value=int(dqn.get("n_episodes", 64)), min_value=1)
            lr = st.number_input(t("rl.sidebar.learning_rate"), value=float(dqn.get("lr", 1e-5)), format="%.6f")
        with col2:
            gamma = st.number_input(t("rl.sidebar.gamma"), value=float(dqn.get("gamma", 0.98)), min_value=0.0, max_value=1.0, format="%.3f")
            hidden = st.number_input(t("rl.sidebar.hidden_dim"), value=int(dqn.get("hidden", 128)), min_value=16)
        with col3:
            batch_size = st.number_input(t("rl.sidebar.batch_size"), value=int(dqn.get("batch_size", 200)), min_value=16)
            epsilon_decay = st.number_input(t("rl.sidebar.epsilon_decay"), value=int(dqn.get("epsilon_decay", 500)), min_value=50)

        col1, col2, col3 = st.columns(3)
        with col1:
            reward_window = st.number_input(t("rl.sidebar.reward_window"), value=int(dqn.get("reward_window", 63)), min_value=5)
        with col2:
            vol_penalty = st.number_input(t("rl.sidebar.vol_penalty"), value=float(dqn.get("vol_penalty_coef", 0.1)), format="%.2f")
        with col3:
            dd_penalty = st.number_input(t("rl.sidebar.drawdown_penalty"), value=float(dqn.get("dd_penalty_coef", 0.5)), format="%.2f")

        sys_ver_options = ["basic", "1.0", "2.0"]
        sys_ver_idx = sys_ver_options.index(sys_ver_default) if sys_ver_default in sys_ver_options else 1
        sys_ver = st.selectbox(t("rl.sidebar.system_version"), sys_ver_options, index=sys_ver_idx)
        selected_labels = st.multiselect(t("rl.sidebar.features"), fg_labels, default=orig_labels)
        selected_fgs = [fg_map[lb] for lb in selected_labels]

        col1, col2 = st.columns(2)
        with col1:
            commission = st.number_input(t("rl.sidebar.commission"), value=float(fee.get("commission_rate", 0.00025)), format="%.5f")
            stamp = st.number_input(t("rl.sidebar.stamp"), value=float(fee.get("stamp_duty", 0.001)), format="%.4f")
        with col2:
            min_comm = st.number_input(t("rl.sidebar.min_commission"), value=float(fee.get("min_commission", 5.0)))
            capital = st.number_input(t("rl.sidebar.initial_capital"), value=float(fee.get("initial_capital", 100000.0)))

        _src_label = t("task.retrain.source.memory") if not needs_refetch else t("task.retrain.source.refetch")
        st.caption(t("task.retrain.caption.single",
                     symbol=symbol,
                     s=meta.get('train_start', '?'), e=meta.get('train_end', '?'),
                     n=len(meta.get('df_test_index', [len(meta)] * 2)),
                     source=_src_label))

        submitted = st.form_submit_button(t("task.retrain.btn_submit"), type="primary")
        if submitted:
            if needs_refetch:
                try:
                    fetched = _refetch_rl_data(meta)
                    df_train, df_test = fetched["df_train"], fetched["df_test"]
                    meta = dict(meta, df_test_index=fetched["df_test_index"])
                except Exception as e:
                    st.error(f"{tt('数据重抓失败', 'Data refetch failed')}: {e}")
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
        raise ValueError(t("hrl.error.fetch_failed", n=len(etf_data)))

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
        st.success(t("task.retrain.submitted", id=new_tid[:8]))
        st.caption(t("task.retrain.hint"))
        if st.button(t("task.retrain.view_btn")):
            st.session_state.selected_task_id = new_tid
            st.rerun()
        return

    params = task.get("params")
    params_display = task.get("params_display")
    if params is None and params_display is None:
        st.info(t("rl.retrain.info"))
        return

    needs_refetch = params is None
    src = params if not needs_refetch else params_display

    ep = src.get("n_episodes", 64)

    if needs_refetch:
        st.info(t("rl.retrain.refetch"))

    with st.form(key=f"retrain_hrl_{tid}", clear_on_submit=True):
        st.markdown(t("task.retrain.section.ppp"))
        col1, col2, col3 = st.columns(3)
        with col1:
            ppo_lr = st.number_input(t("hrl.sidebar.ppo_lr"), value=float(src.get("ppo_lr", 3e-4)), format="%.6f")
            clip_epsilon = st.number_input(t("hrl.sidebar.ppo_clip"), value=float(src.get("clip_epsilon", 0.2)), format="%.2f")
        with col2:
            ppo_gamma = st.number_input(t("hrl.sidebar.ppo_gamma"), value=float(src.get("ppo_gamma", 0.99)), format="%.3f")
            entropy_beta = st.number_input(t("hrl.sidebar.ppo_entropy"), value=float(src.get("entropy_beta", 0.01)), format="%.3f")
        with col3:
            ppo_hidden = st.number_input(t("hrl.sidebar.ppo_hidden"), value=int(src.get("ppo_hidden", 128)), min_value=16)
            n_episodes = st.number_input(t("rl.sidebar.episodes"), value=int(ep), min_value=1)

        col1, col2, col3 = st.columns(3)
        with col1:
            gae_lambda = st.number_input(t("hrl.sidebar.ppo_gae"), value=float(src.get("gae_lambda", 0.95)), format="%.2f")
        with col2:
            ppo_epochs = st.number_input(t("hrl.sidebar.ppo_epochs"), value=int(src.get("ppo_epochs", 10)), min_value=1)
        with col3:
            ppo_update_freq = st.number_input(t("hrl.sidebar.ppo_update_freq"), value=int(src.get("ppo_update_freq", 128)), min_value=1)

        st.markdown(t("task.retrain.section.dqn"))
        col1, col2, col3 = st.columns(3)
        with col1:
            dqn_lr = st.number_input(t("hrl.sidebar.dqn_lr"), value=float(src.get("dqn_lr", 1e-5)), format="%.6f")
            dqn_hidden = st.number_input(t("hrl.sidebar.dqn_hidden"), value=int(src.get("dqn_hidden", 128)), min_value=16)
        with col2:
            dqn_gamma = st.number_input(t("hrl.sidebar.dqn_gamma"), value=float(src.get("dqn_gamma", 0.98)), format="%.3f")
            dqn_batch_size = st.number_input(t("hrl.sidebar.dqn_batch"), value=int(src.get("dqn_batch_size", 200)), min_value=16)
        with col3:
            dqn_epsilon_decay = st.number_input(t("hrl.sidebar.dqn_epsilon_decay"), value=int(src.get("dqn_epsilon_decay", 500)), min_value=50)

        st.markdown(t("task.retrain.section.fee"))
        col1, col2, col3 = st.columns(3)
        with col1:
            commission = st.number_input(t("rl.sidebar.commission"), value=float(src.get("commission_rate", 0.00025)), format="%.5f")
        with col2:
            capital = st.number_input(t("rl.sidebar.initial_capital"), value=float(src.get("initial_capital", 100000.0)))
        with col3:
            trade_fraction = st.number_input(t("hrl.sidebar.trade_ratio"), value=float(src.get("trade_fraction", 0.25)), format="%.2f")

        syms = src.get("selected_symbols", [])
        _syms_str = ", ".join(syms) if syms else "?"
        st.caption(tt(f"📦 ETF 组合: {_syms_str}", f"📦 ETF Pool: {_syms_str}"))

        submitted = st.form_submit_button(t("task.retrain.btn_submit"), type="primary")
        if submitted:
            if needs_refetch:
                try:
                    fetched = _refetch_hrl_data(src)
                except Exception as e:
                    st.error(f"{tt('数据重抓失败', 'Data refetch failed')}: {e}")
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


def _render_hp_retrain_form(task: dict, mgr: TaskManager):
    tid = task.get("_id", "")

    done_key = f"retrain_hp_done_{tid}"
    if done_key in st.session_state:
        new_tid = st.session_state[done_key]
        st.success(t("task.retrain.submitted", id=new_tid[:8]))
        st.caption(t("task.retrain.hint"))
        if st.button(t("task.retrain.view_btn")):
            st.session_state.selected_task_id = new_tid
            st.rerun()
        return

    params = task.get("params")
    params_display = task.get("params_display")
    if params is None and params_display is None:
        st.info(t("rl.retrain.info"))
        return

    needs_refetch = params is None
    src = params if not needs_refetch else params_display

    result = mgr.get_result(tid)
    hp = result.get("hp_result", {}) if isinstance(result, dict) else {}
    bp = hp.get("best_params", {})
    meta = src.get("meta", {})
    symbol = meta.get("symbol", "?")
    sys_ver_default = src.get("system_version", "1.0")

    dqn = src.get("dqn_params", {})
    fee = src.get("fee_params", {})

    if needs_refetch:
        st.info(t("rl.retrain.refetch"))

    with st.form(key=f"retrain_hp_{tid}", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            n_episodes = st.number_input(t("rl.sidebar.episodes"), value=int(bp.get("n_episodes", dqn.get("n_episodes", 64))), min_value=1)
            lr = st.number_input(t("rl.sidebar.learning_rate"), value=float(bp.get("lr", dqn.get("lr", 1e-5))), format="%.6f")
        with col2:
            gamma = st.number_input(t("rl.sidebar.gamma"), value=float(bp.get("gamma", dqn.get("gamma", 0.98))), min_value=0.0, max_value=1.0, format="%.3f")
            hidden = st.number_input(t("rl.sidebar.hidden_dim"), value=int(bp.get("hidden", dqn.get("hidden", 128))), min_value=16)
        with col3:
            batch_size = st.number_input(t("rl.sidebar.batch_size"), value=int(dqn.get("batch_size", 200)), min_value=16)
            epsilon_decay = st.number_input(t("rl.sidebar.epsilon_decay"), value=int(bp.get("epsilon_decay", dqn.get("epsilon_decay", 500))), min_value=50)

        col1, col2, col3 = st.columns(3)
        with col1:
            reward_window = st.number_input(t("rl.sidebar.reward_window"), value=int(dqn.get("reward_window", 63)), min_value=5)
        with col2:
            vol_penalty = st.number_input(t("rl.sidebar.vol_penalty"), value=float(dqn.get("vol_penalty_coef", 0.1)), format="%.2f")
        with col3:
            dd_penalty = st.number_input(t("rl.sidebar.drawdown_penalty"), value=float(dqn.get("dd_penalty_coef", 0.5)), format="%.2f")

        sys_ver_options = ["basic", "1.0", "2.0"]
        sys_ver_idx = sys_ver_options.index(sys_ver_default) if sys_ver_default in sys_ver_options else 1
        sys_ver = st.selectbox(t("rl.sidebar.system_version"), sys_ver_options, index=sys_ver_idx)

        from backtest.rl.feature_engineer import FEATURE_GROUPS
        fg_keys = list(FEATURE_GROUPS.keys())
        fg_labels = [FEATURE_GROUPS[k]["label"] for k in fg_keys]
        fg_map = dict(zip(fg_labels, fg_keys))
        orig_fgs = src.get("feature_groups", [])
        orig_labels = [lb for lb, k in fg_map.items() if k in orig_fgs]
        selected_labels = st.multiselect(t("rl.sidebar.features"), fg_labels, default=orig_labels)
        selected_fgs = [fg_map[lb] for lb in selected_labels]

        col1, col2 = st.columns(2)
        with col1:
            commission = st.number_input(t("rl.sidebar.commission"), value=float(fee.get("commission_rate", 0.00025)), format="%.5f")
            stamp = st.number_input(t("rl.sidebar.stamp"), value=float(fee.get("stamp_duty", 0.001)), format="%.4f")
        with col2:
            min_comm = st.number_input(t("rl.sidebar.min_commission"), value=float(fee.get("min_commission", 5.0)))
            capital = st.number_input(t("rl.sidebar.initial_capital"), value=float(fee.get("initial_capital", 100000.0)))

        _ts = meta.get('train_start', '?')
        _te = meta.get('train_end', '?')
        _vs = meta.get('val_start', '?')
        _ve = meta.get('val_end', '?')
        st.caption(tt("📦 {sym} ｜ 训练 {s} ~ {e} ｜ 验证 {vs} ~ {ve} ｜ 最优参数: lr={lr}, gamma={gm}",
                      "📦 {sym} Train {s}~{e} Val {vs}~{ve} Best: lr={lr} gamma={gm}")
                   .format(sym=symbol, s=_ts, e=_te, vs=_vs, ve=_ve,
                           lr=bp.get('lr', '?'), gm=bp.get('gamma', '?')))

        submitted = st.form_submit_button(t("task.retrain.btn_submit"), type="primary")
        if submitted:
            if needs_refetch:
                try:
                    fetched = _refetch_hp_data(meta)
                    df_train, df_val = fetched["df_train"], fetched["df_val"]
                except Exception as e:
                    st.error(f"{tt('数据重抓失败', 'Data refetch failed')}: {e}")
                    st.stop()
            else:
                df_train = params.get("df_train")
                df_val = params.get("df_val")
                if df_train is None:
                    st.error(tt("内存数据不可用，请刷新后重试", "In-memory data unavailable, please refresh and retry"))
                    st.stop()

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

            df_test = df_val.copy()
            new_params = dict(
                df_train=df_train, df_test=df_test,
                system_version=sys_ver,
                feature_groups=selected_fgs,
                dqn_params=new_dqn,
                fee_params=new_fee,
                meta={
                    "symbol": symbol,
                    "system_version": sys_ver,
                    "feature_groups": selected_fgs,
                    "train_start": meta.get("train_start", ""),
                    "train_end": meta.get("train_end", ""),
                    "df_test_index": df_test.index,
                    "adjust": meta.get("adjust", "qfq"),
                },
            )
            from ui.rl_training import _rl_train_task
            new_tid = mgr.submit("RL训练", new_params, _rl_train_task, args=(new_params,))
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
    st.markdown(t("task.progress.label", pct=pct))

    cancel_requested = task.get("_cancel_requested", False)
    if cancel_requested:
        st.info(t("task.progress.stopping"))
    elif st.button(t("task.btn.stop"), key=f"detail_cancel_{tid}"):
        mgr.cancel(tid)
        st.session_state[f"_cancel_{tid}"] = True
        st.rerun()

    pdata = mgr.get_progress_data(tid)
    _render_progress_chart(pdata)


def _render_progress_chart(pdata, xaxis_title="Episode", yaxis_title="Reward"):
    if not pdata or len(pdata) < 2:
        return
    df = pd.DataFrame(pdata, columns=["ep", "value"])
    best_idx = df["value"].idxmax()
    best_val = df.loc[best_idx, "value"]
    best_ep = int(df.loc[best_idx, "ep"])
    ep_label = tt("组合", "combo") if xaxis_title != "Episode" else tt("episode", "episode")

    window = max(5, len(df) // 10)
    df["trend"] = df["value"].rolling(window=window, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["ep"], y=df["value"],
        mode="lines", name=t("task.chart.raw"),
        line=dict(color="#94a3b8", width=1),
        opacity=0.5,
    ))
    fig.add_trace(go.Scatter(
        x=df["ep"], y=df["trend"],
        mode="lines", name=t("task.chart.trend"),
        line=dict(color="#ef4444", width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=[best_ep], y=[best_val],
        mode="markers+text",
        name=t("task.chart.best"),
        marker=dict(color="#22c55e", size=12, symbol="star"),
        text=[f"<b>{yaxis_title} {best_val:.4f}</b>"],
        textposition="top center",
        textfont=dict(color="#22c55e", size=12),
    ))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title=xaxis_title, yaxis_title=yaxis_title,
                      showlegend=True)
    st.plotly_chart(fig, width='stretch')
    st.caption(t("task.progress.best", yaxis=yaxis_title, ep=best_ep, ep_label=ep_label, val=best_val))


def _render_detail(task: dict, mgr: TaskManager):
    tid = task.get("_id", "")
    status = task["status"]
    emoji = STATUS_EMOJI.get(status, "❓")

    st.button(t("task.btn.back"), on_click=lambda: st.session_state.pop("selected_task_id", None))
    st.title(f"{emoji} {task['type']}  `{tid[:8]}...`")
    st.caption(f"{tt('创建', 'Created')}: {task['created_at']}")

    if status == TaskStatus.RUNNING.value:
        _render_running_detail(task, mgr, tid)

    elif status in (TaskStatus.COMPLETED.value, TaskStatus.EARLY_STOPPED.value):
        if status == TaskStatus.EARLY_STOPPED.value:
            st.warning(t("task.detail.stopped"))
        result = mgr.get_result(tid)
        if result:
            st.subheader(t("task.detail.result", type=task["type"]))
            if task["type"] == "RL训练":
                _render_rl_result(result, task)
            elif task["type"] == "HRL训练":
                _render_hrl_result(result, task)
            elif task["type"] == "超参搜索":
                _render_hp_result(result, task)

        pdata = task.get("_progress_data") or mgr.get_progress_data(tid)
        if pdata:
            st.subheader(t("task.detail.progress"))
            is_hp = task["type"] == "超参搜索"
            _render_progress_chart(pdata,
                xaxis_title=tt("超参组合", "HP combo") if is_hp else "Episode",
                yaxis_title=tt("最优夏普", "Best Sharpe") if is_hp else "Reward")

        # ── 重新训练 ──
        st.divider()
        expand_key = f"retrain_expand_{tid}"
        expand_val = st.session_state.get(expand_key, False)
        with st.expander(t("task.detail.retrain"), expanded=expand_val):
            if task["type"] == "RL训练":
                _render_rl_retrain_form(task, mgr)
            elif task["type"] == "HRL训练":
                _render_hrl_retrain_form(task, mgr)
            elif task["type"] == "超参搜索":
                _render_hp_retrain_form(task, mgr)

    elif status == TaskStatus.FAILED.value:
        st.error(t("task.detail.error", error=task.get("error", tt("未知错误", "Unknown error"))))

    elif status == TaskStatus.CANCELLED.value:
        st.warning(t("task.detail.cancelled"))

    elif status == TaskStatus.PENDING.value:
        st.info(t("task.detail.waiting"))


def _render_list(tasks: list, mgr: TaskManager):
    for i, task in enumerate(tasks):
        tid = task.get("_id", "")
        status = task["status"]
        emoji = STATUS_EMOJI.get(status, "❓")

        cols = st.columns([2.5, 1, 1.5, 1, 0.6])
        with cols[0]:
            st.markdown(f"**{task['type']}** `{tid[:8]}...`")
            st.caption(f"{tt('创建', 'Created')}: {task['created_at']}")
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
                st.caption(t("task.caption.done", time=task["finished_at"]))
            elif status == TaskStatus.PENDING.value:
                st.caption(t("task.caption.waiting"))
        with cols[3]:
            st.button(t("task.btn.detail"), key=f"view_{tid}",
                      on_click=_select_task, args=(tid,))
        with cols[4]:
            cancel_requested = st.session_state.get(f"_cancel_{tid}", False)
            if status == TaskStatus.PENDING.value:
                if st.button(t("task.btn.cancel"), key=f"cancel_{tid}"):
                    mgr.cancel(tid)
                    st.rerun()
            elif status == TaskStatus.RUNNING.value and not cancel_requested:
                if st.button(t("task.btn.cancel"), key=f"cancel_{tid}"):
                    mgr.cancel(tid)
                    st.session_state[f"_cancel_{tid}"] = True
                    st.rerun()
            elif status == TaskStatus.RUNNING.value and cancel_requested:
                st.caption(t("task.caption.stopping"))

        if i < len(tasks) - 1:
            st.divider()


def _render_model_browser():
    model_dir = Path("saved_models/rl")
    files = sorted(model_dir.glob("*.pt"), reverse=True) if model_dir.exists() else []
    st.sidebar.markdown(t("task.model.title"))
    if not files:
        st.sidebar.caption(t("task.model.empty"))
        return
    names = [m.stem for m in files]
    sel = st.sidebar.selectbox(t("task.model.select"), names, key="task_model_browser")
    c1, c2 = st.sidebar.columns(2)
    if c1.button(t("task.model.load"), key="task_model_load"):
        p = str(model_dir / f"{sel}.pt")
        loaded = DQNAgent.load(p)
        st.session_state.rl_agent = loaded
        meta = torch.load(p, map_location="cpu", weights_only=False).get("metadata", {})
        st.session_state.rl_model_info = {"path": p, "name": sel, **meta}
        st.rerun()
    if c2.button(t("task.model.delete"), key="task_model_del"):
        (model_dir / f"{sel}.pt").unlink()
        st.rerun()
    with st.sidebar.expander(t("task.model.rename"), expanded=False):
        rename_to = st.text_input(t("task.model.rename_new"), value=sel, key="task_model_rename_input")
        if st.button(t("task.model.rename_confirm"), key="task_model_rename_btn"):
            old_p = model_dir / f"{sel}.pt"
            new_p = model_dir / f"{rename_to}.pt"
            if not new_p.exists():
                old_p.rename(new_p)
                st.rerun()
            else:
                st.sidebar.error(t("task.model.name_exists"))


def _select_task(tid: str):
    st.session_state.selected_task_id = tid


def render_task_manager():
    _render_model_browser()
    st.title(t("task.title"))
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
            st.info(t("task.empty"))
            return
        _render_list(tasks, mgr)
