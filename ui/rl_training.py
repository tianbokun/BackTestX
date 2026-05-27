import threading
from pathlib import Path
from datetime import datetime

import torch
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from data_fetcher import add_premium_rate, ensure_ohlc
from backtest.rl.trainer import (
    train_dqn, evaluate, run_bh_baseline,
    hyperparam_search,
)
from backtest.rl.dqn_agent import DQNAgent
from backtest.rl.feature_engineer import FEATURE_GROUPS, DEFAULT_FEATURE_GROUPS
from ui._helpers import cached_fetch
from ui.rl_signal import render_rl_signal

# ── Threading globals for cancelable HP search ──
_hp_cancel_event = threading.Event()
_hp_output: dict = {}
_hp_progress: dict = {}


def render_rl_training(df_full, end_date, symbol, asset_type, adjust):
    st.title("🤖 DQN 强化学习训练系统")

    # 归一化列名
    rename_map = {"开盘": "开盘价", "收盘": "收盘价", "最高": "最高价", "最低": "最低价"}
    df = df_full.copy()
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # 场外基金/净值型资产只有收盘价, 用同一价格填充 OHLC 四列
    df = ensure_ohlc(df)

    # 对 ETF/LOF 添加溢价率列
    df = add_premium_rate(df, symbol, asset_type)
    has_premium = "溢价率" in df.columns

    # 重新拉取全量数据 (RL 需要尽可能多的历史数据)
    rl_start_str = "20000101"
    rl_end_str = end_date.strftime("%Y%m%d")
    df_wide = cached_fetch(symbol, asset_type, rl_start_str, rl_end_str, adjust)
    if df_wide is not None and len(df_wide) > len(df):
        df = df_wide.copy()
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
    st.sidebar.markdown("### 🤖 强化学习参数")

    train_start = st.sidebar.date_input("训练集开始", value=train_start_def)
    train_end = st.sidebar.date_input("训练集结束", value=train_end_def)
    val_start = st.sidebar.date_input("验证集开始", value=val_start_def)
    val_end = st.sidebar.date_input("验证集结束", value=val_end_def)
    test_start = st.sidebar.date_input("测试集开始", value=test_start_def)
    test_end = st.sidebar.date_input("测试集结束", value=test_end_def)

    system_version = st.sidebar.selectbox(
        "系统版本",
        options=["basic", "1.0", "2.0"],
        format_func=lambda x: {
            "basic": "基础版 (仅价格)",
            "1.0": "系统 1.0 (+技术指标)",
            "2.0": "系统 2.0 (+SVM+XGBoost)",
        }[x],
        index=1,
        help="basic=仅过去30日收盘价, 1.0=加入技术指标, 2.0=加入SVM/XGBoost涨跌信号",
    )

    # ── 特征选择 ──
    selected_groups = list(DEFAULT_FEATURE_GROUPS)
    if system_version == "basic":
        selected_groups = []
    with st.sidebar.expander("📊 特征选择", expanded=False):
        st.caption("选择 DQN 的输入特征 (basic 模式忽略)")
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

    with st.sidebar.expander("💰 费率设置", expanded=False):
        rl_commission = st.number_input("佣金费率", min_value=0.0, value=0.000235, step=0.000005, format="%.6f",
                                        key="rl_commission", help="默认万2.35")
        rl_min_commission = st.number_input("最低佣金(元)", min_value=0.0, value=5.0, step=1.0,
                                            key="rl_min_comm", help="每笔最低5元")
        rl_stamp_duty = st.number_input("印花税率", min_value=0.0, value=0.001, step=0.0001, format="%.4f",
                                        key="rl_stamp", help="仅卖出收取")
        rl_capital = st.number_input("初始本金(元)", min_value=100.0, value=100000.0, step=10000.0,
                                     key="rl_capital",
                                     help="建议>=50000, 否则最低5元佣金占比过高")

    with st.sidebar.expander("⚙️ DQN 超参数", expanded=False):
        n_episodes = st.number_input("训练轮数", min_value=10, value=64, step=10)
        batch_size = st.number_input("Batch 大小", min_value=32, value=200, step=32)
        lr = st.text_input("学习率", value="1e-5")
        gamma = st.text_input("折扣因子 γ", value="0.98")
        hidden = st.number_input("隐藏层维度", min_value=32, value=128, step=32)
        epsilon_start = st.text_input("ε 初始值", value="0.9")
        epsilon_end = st.text_input("ε 终值", value="0.01")
        epsilon_decay = st.number_input("ε 衰减步数", min_value=100, value=500, step=100)
        target_update = st.number_input("目标网络更新间隔", min_value=10, value=50, step=10)
        buffer_capacity = st.number_input("经验回放容量", min_value=1000, value=10000, step=1000)

    search_btn = st.sidebar.button("🔍 超参搜索", width='stretch')
    run_btn = st.sidebar.button("🚀 开始训练", type="primary", width='stretch')

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📂 已保存模型")
    model_dir = Path("saved_models/rl")
    model_files = sorted(model_dir.glob("*.pt"), reverse=True) if model_dir.exists() else []
    if model_files:
        names = [m.stem for m in model_files]
        selected_name = st.sidebar.selectbox("选择模型", names, key="rl_model_selector")
        col_s1, col_s2 = st.sidebar.columns(2)
        if col_s1.button("📥 加载", width='stretch', key="rl_load_btn"):
            selected_path = str(model_dir / f"{selected_name}.pt")
            loaded = DQNAgent.load(selected_path)
            st.session_state.rl_agent = loaded
            meta = torch.load(selected_path, map_location="cpu", weights_only=False).get("metadata", {})
            st.session_state.rl_model_info = {"path": selected_path, "name": selected_name, **meta}
            st.session_state.rl_model_just_saved = False
            st.rerun()
        if col_s2.button("🗑 删除", width='stretch', key="rl_del_btn"):
            (model_dir / f"{selected_name}.pt").unlink()
            if st.session_state.rl_model_info and st.session_state.rl_model_info.get("name") == selected_name:
                st.session_state.rl_agent = None
                st.session_state.rl_model_info = None
            st.session_state.rl_model_just_saved = False
            st.rerun()

        with st.sidebar.expander("✏️ 重命名", expanded=False):
            rename_to = st.text_input("新名称", value=selected_name, key="rl_rename_input")
            if st.button("确认重命名", key="rl_rename_btn"):
                if rename_to and rename_to != selected_name:
                    old_p = model_dir / f"{selected_name}.pt"
                    new_p = model_dir / f"{rename_to}.pt"
                    if not new_p.exists():
                        old_p.rename(new_p)
                        st.rerun()
                    else:
                        st.error("文件名已存在")
    else:
        if st.session_state.rl_model_just_saved:
            st.sidebar.success("✅ 模型已保存！刷新页面后显示在列表中")
        else:
            st.sidebar.caption("暂无已保存的模型")

    # ── 划分数据集 ──
    df_train = df[(df.index >= pd.Timestamp(train_start)) & (df.index <= pd.Timestamp(train_end))].copy()
    df_val = df[(df.index >= pd.Timestamp(val_start)) & (df.index <= pd.Timestamp(val_end))].copy()
    df_test = df[(df.index >= pd.Timestamp(test_start)) & (df.index <= pd.Timestamp(test_end))].copy()

    if len(df_train) < 50:
        st.error(f"训练数据不足 ({len(df_train)} 行)，请扩大训练集")
        st.stop()
    if len(df_test) < 20:
        st.error(f"测试数据不足 ({len(df_test)} 行)，请扩大测试集")
        st.stop()

    st.info(
        f"训练集: {str(train_start)[:10]} ~ {str(train_end)[:10]} ({len(df_train)} 行) | "
        f"验证集: {str(val_start)[:10]} ~ {str(val_end)[:10]} ({len(df_val)} 行) | "
        f"测试集: {str(test_start)[:10]} ~ {str(test_end)[:10]} ({len(df_test)} 行)"
    )

    rl_capital_val = float(rl_capital)
    fee_params = dict(commission_rate=float(rl_commission),
                      min_commission=float(rl_min_commission),
                      stamp_duty=float(rl_stamp_duty),
                      initial_capital=rl_capital_val)

    # ── 超参搜索 ──
    if search_btn:
        if len(df_val) < 20:
            st.error(f"验证集数据不足 ({len(df_val)} 行)，至少需要 20 行")
            st.stop()
        df_hp = pd.concat([df_train, df_val]).sort_index()
        total_days = len(df_hp)
        st.info(f"超参搜索窗口: {str(df_hp.index[0])[:10]} ~ {str(df_hp.index[-1])[:10]} ({total_days} 行)")

        # Reset threading globals
        _hp_cancel_event.clear()
        _hp_output.clear()
        _hp_progress.clear()
        _hp_progress["_status"] = "running"

        sv = system_version
        fg = list(selected_groups)
        fp = dict(fee_params)

        import time as _time_mod
        _hp_start = _time_mod.time()

        def _hp_fold_callback(ci, total, fi, nf, params, fold_sharpe):
            _hp_progress["combo_idx"] = ci
            _hp_progress["total_combos"] = total
            _hp_progress["fold_idx"] = fi
            _hp_progress["n_folds"] = nf
            _hp_progress["params"] = params
            _hp_progress["fold_sharpe"] = fold_sharpe
            _hp_progress["elapsed"] = _time_mod.time() - _hp_start

        from collections import deque as _deque
        _hp_recent = _deque(maxlen=5)

        def _hp_combo_callback(ci, total, best_params, best_score):
            _hp_progress["combo_idx"] = ci
            _hp_progress["total_combos"] = total
            _hp_progress["best_params"] = best_params
            _hp_progress["best_score"] = best_score
            _hp_progress["elapsed"] = _time_mod.time() - _hp_start
            if best_params and best_score > -999:
                _hp_recent.append((ci, best_score))
            _hp_progress["recent"] = list(_hp_recent)

        def _run_hp():
            try:
                result = hyperparam_search(
                    df_hp, system_version=sv,
                    feature_groups=fg,
                    progress_callback=None,
                    combo_callback=_hp_combo_callback,
                    fold_callback=_hp_fold_callback,
                    cancel_check=_hp_cancel_event.is_set,
                    **fp,
                )
                _hp_output["_done"] = True
                _hp_output.update(result)
            except Exception as e:
                _hp_output["_done"] = True
                _hp_output["_error"] = str(e)

        _hp_thread = threading.Thread(target=_run_hp, daemon=True)
        _hp_thread.start()
        st.session_state.hp_running = True
        st.rerun()

    # ── HP 搜索进度轮询 ──
    if st.session_state.get("hp_running", False):
        nf = 3
        total_combos = 324
        total_folds = total_combos * nf
        ci = _hp_progress.get("combo_idx", 0)
        fi = _hp_progress.get("fold_idx", 0)
        folds_done = ci * nf + min(fi + 1, nf)
        elapsed = _hp_progress.get("elapsed", 0.0)

        pct = min(folds_done / max(total_folds, 1), 1.0)
        st.progress(pct, text=f"超参搜索: {folds_done}/{total_folds} 训练 ({pct*100:.1f}%)")

        stop_col1, stop_col2 = st.columns([1, 5])
        with stop_col1:
            if st.button("⏹ 停止搜索", key="hp_stop_btn"):
                _hp_cancel_event.set()
                st.info("⏳ 正在等待当前组合完成后停止...")
                st.rerun()
        with stop_col2:
            if _hp_cancel_event.is_set():
                st.warning("停止中 (当前组合完成后停止)")

        best_score = _hp_progress.get("best_score", -999.0)
        best_params = _hp_progress.get("best_params")
        if best_params:
            p_str = ", ".join(f"{k}={v}" for k, v in sorted(best_params.items()))
            st.code(f"🏆 当前最优: 夏普={best_score:.4f}\n  参数: {p_str}")

        recent = _hp_progress.get("recent", [])
        if recent:
            log_lines = ["最近 5 个最优得分 (更新时):"]
            for idx, sc in recent:
                log_lines.append(f"  #{idx+1:>3d}  夏普={sc:+.4f}")
            st.code("\n".join(log_lines))

        avg_time = elapsed / max(folds_done, 1)
        remaining = avg_time * (total_folds - folds_done)
        remaining_str = f"{remaining/60:.0f} 分" if remaining < 3600 else f"{remaining/3600:.1f} 时"
        st.caption(f"已用 {elapsed/60:.1f} 分 | 预计剩余: ~{remaining_str}")

        done = _hp_output.get("_done", False)
        error = _hp_output.get("_error")

        if error:
            st.error(f"超参搜索失败: {error}")
            st.session_state.hp_running = False
            st.rerun()

        if done:
            hp_result = {k: v for k, v in _hp_output.items() if k not in ("_done", "_error")}
            st.session_state.hp_running = False

            bp = hp_result.get("best_params")
            bs = hp_result.get("best_score", -999)

            if bp is None:
                if _hp_cancel_event.is_set():
                    st.warning("搜索被用户手动停止，显示部分结果")
                else:
                    st.error(hp_result.get("error", "搜索失败"))
                st.stop()

            elapsed_total = hp_result.get("elapsed_sec", 0)
            st.success(f"✅ 搜索完成! 耗时 {elapsed_total/60:.1f} 分 | "
                       f"最优: lr={bp['lr']}, gamma={bp['gamma']}, "
                       f"hidden={bp['hidden']}, n_episodes={bp['n_episodes']}, "
                       f"epsilon_decay={bp['epsilon_decay']}  |  验证夏普={bs:.4f}")

            st.markdown("### 📊 验证集回测结果 (最优参数)")
            with st.spinner("正在训练/回测..."):
                best_agent, _ = train_dqn(
                    df_train, system_version=system_version,
                    feature_groups=selected_groups,
                    n_episodes=bp["n_episodes"], lr=bp["lr"], gamma=bp["gamma"],
                    hidden=bp["hidden"], epsilon_decay=bp["epsilon_decay"],
                    progress_callback=None, **fee_params,
                )
                val_result = evaluate(best_agent, df_val, system_version=system_version,
                                      feature_groups=selected_groups, **fee_params)
                bh_val = run_bh_baseline(df_val, initial_capital=rl_capital_val)

            comp_val = pd.DataFrame([
                {"策略": "DQN", "最终金额": val_result["final_value"],
                 "收益率%": val_result["total_return_pct"], "夏普比率": val_result["sharpe_ratio"],
                 "最大回撤%": val_result["max_drawdown_pct"], "交易次数": val_result["num_trades"]},
                {"策略": "买入持有(BH)", "最终金额": bh_val["final_value"],
                 "收益率%": bh_val["total_return_pct"], "夏普比率": bh_val["sharpe_ratio"],
                 "最大回撤%": bh_val["max_drawdown_pct"], "交易次数": 0},
            ])
            st.dataframe(comp_val, width='stretch', hide_index=True)

            if not val_result["trades"].empty:
                with st.expander("📝 交易记录"):
                    td = val_result["trades"].copy()
                    td["日期"] = td["日期"].dt.strftime("%Y-%m-%d")
                    st.dataframe(td, width='stretch', hide_index=True)

            st.session_state.rl_hp_agent = best_agent
            st.session_state.rl_hp_params = bp
            st.session_state.rl_hp_score = bs
            st.rerun()

        else:
            import time
            time.sleep(1)
            st.rerun()

    # ── 保存超参搜索结果 (在 search_btn 块外, 确保持久化) ──
    if st.session_state.rl_hp_agent is not None:
        st.markdown("---")
        st.subheader("💾 保存超参搜索模型")
        sym = symbol
        sv = system_version
        _hp_save_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        _hp_save_default = f"{sym}_hp_{sv}_{_hp_save_ts}"
        _hp_save_name = st.text_input("模型名称", value=_hp_save_default, key="rl_hp_save_name")
        save_col1, save_col2 = st.columns([1, 5])
        with save_col1:
            if st.button("💾 保存模型", type="primary", key="rl_save_hp_model_btn"):
                save_path = Path(f"saved_models/rl/{_hp_save_name}.pt")
                save_path.parent.mkdir(parents=True, exist_ok=True)
                st.session_state.rl_hp_agent.save(str(save_path), {
                    "symbol": sym, "system_version": sv,
                    "feature_groups": selected_groups,
                    "train_start": str(train_start),
                    "train_end": str(train_end),
                    "source": "hyperparam_search",
                    "sharpe": st.session_state.rl_hp_score,
                })
                st.session_state.rl_agent = st.session_state.rl_hp_agent
                st.session_state.rl_model_info = {
                    "name": save_path.stem, "path": str(save_path),
                    "symbol": sym, "system_version": sv,
                    "feature_groups": selected_groups,
                }
                st.session_state.rl_model_just_saved = True
                st.rerun()
        with save_col2:
            st.caption("保存超参搜索得到的最优模型到磁盘，之后可在侧边栏加载使用")

    # ── 训练 ──
    if run_btn:
        st.markdown("---")
        st.subheader("📊 训练与测试")

        progress_bar = st.progress(0)
        loss_chart = st.empty()
        loss_data = []

        def _progress(ep, total, loss):
            progress_bar.progress((ep + 1) / total)
            if loss > 0:
                loss_data.append((ep, loss))
            if len(loss_data) > 1:
                import plotly.graph_objects as go
                ldf = pd.DataFrame(loss_data, columns=["ep", "loss"])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ldf["ep"], y=ldf["loss"], mode="lines", name="Loss"))
                fig.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10),
                                  xaxis_title="Step", yaxis_title="Loss")
                loss_chart.plotly_chart(fig, width='stretch')

        try:
            params = {
                "n_episodes": int(n_episodes), "batch_size": int(batch_size),
                "lr": float(lr), "gamma": float(gamma), "hidden": int(hidden),
                "epsilon_start": float(epsilon_start), "epsilon_end": float(epsilon_end),
                "epsilon_decay": int(epsilon_decay), "target_update": int(target_update),
                "buffer_capacity": int(buffer_capacity),
            }
        except ValueError:
            st.error("超参数格式错误，请检查数字格式")
            st.stop()

        with st.spinner("正在训练 DQN 智能体..."):
            agent, _ = train_dqn(
                df_train, system_version=system_version,
                feature_groups=selected_groups,
                progress_callback=_progress, **params, **fee_params,
            )

        with st.spinner("正在回测..."):
            result_dqn = evaluate(agent, df_test, system_version=system_version,
                                  feature_groups=selected_groups, **fee_params)
            result_bh = run_bh_baseline(df_test, initial_capital=rl_capital_val)

        # 存入 session_state 持久化
        st.session_state.rl_trained_agent = agent
        st.session_state.rl_dqn_result = result_dqn
        st.session_state.rl_bh_result = result_bh
        st.session_state.rl_train_meta = {
            "symbol": symbol, "system_version": system_version,
            "train_start": str(train_start), "train_end": str(train_end),
            "df_test_index": df_test.index,
        }
        st.session_state.rl_model_just_saved = True
        st.success("✅ 训练完成！")

    # ── 展示训练结果 + 保存按钮 (在 run_btn 外部, 持久化) ──
    if st.session_state.rl_trained_agent is not None:
        agent = st.session_state.rl_trained_agent
        result_dqn = st.session_state.rl_dqn_result
        result_bh = st.session_state.rl_bh_result
        meta_info = st.session_state.rl_train_meta

        st.markdown("---")
        st.subheader("📊 测试集回测结果")

        st.markdown("#### 📋 策略指标对比")
        comp = pd.DataFrame([
            {"策略": "DQN", "最终金额": result_dqn["final_value"],
             "收益率%": result_dqn["total_return_pct"],
             "夏普比率": result_dqn["sharpe_ratio"],
             "最大回撤%": result_dqn["max_drawdown_pct"],
             "交易次数": result_dqn["num_trades"]},
            {"策略": "买入持有(BH)", "最终金额": result_bh["final_value"],
             "收益率%": result_bh["total_return_pct"],
             "夏普比率": result_bh["sharpe_ratio"],
             "最大回撤%": result_bh["max_drawdown_pct"],
             "交易次数": 0},
        ])
        st.dataframe(comp, width='stretch', hide_index=True)

        st.markdown("#### 📈 累计利润对比")
        import plotly.graph_objects as go
        test_idx = meta_info.get("df_test_index", result_dqn.get("dates"))
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=test_idx, y=result_dqn["equity_curve"],
            mode="lines", name=f"DQN ({meta_info['system_version']})",
            line=dict(color="#1f77b4", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=test_idx, y=result_bh["equity_curve"],
            mode="lines", name="买入持有(BH)",
            line=dict(color="#ff7f0e", width=2, dash="dash"),
        ))
        fig.add_hline(y=rl_capital_val, line_dash="dot", line_color="gray", annotation_text="初始本金")
        fig.update_layout(
            xaxis_title="日期", yaxis_title=f"账户总值 (初始={rl_capital_val:,.0f})",
            hovermode="x unified", height=400,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig, width='stretch')

        if not result_dqn["trades"].empty:
            st.markdown("#### 📝 交易记录")
            trades_df = result_dqn["trades"].copy()
            trades_df["日期"] = trades_df["日期"].dt.strftime("%Y-%m-%d")
            st.dataframe(trades_df, width='stretch', hide_index=True)

        sv = meta_info["system_version"]
        sym = meta_info["symbol"]
        _train_save_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        _train_save_default = f"{sym}_{sv}_{_train_save_ts}"
        _train_save_name = st.text_input("模型名称", value=_train_save_default, key="rl_train_save_name")
        save_col1, save_col2 = st.columns([1, 5])
        with save_col1:
            if st.button("💾 保存模型", type="primary", key="rl_save_model_btn"):
                save_path = Path(f"saved_models/rl/{_train_save_name}.pt")
                save_path.parent.mkdir(parents=True, exist_ok=True)
                agent.save(str(save_path), {
                    "symbol": sym, "system_version": sv,
                    "feature_groups": selected_groups,
                    "train_start": meta_info["train_start"],
                    "train_end": meta_info["train_end"],
                    "test_return": result_dqn["total_return_pct"],
                    "sharpe": result_dqn["sharpe_ratio"],
                })
                st.session_state.rl_agent = agent
                st.session_state.rl_model_info = {
                    "name": save_path.stem, "path": str(save_path),
                    "symbol": sym, "system_version": sv,
                    "feature_groups": selected_groups,
                }
                st.session_state.rl_model_just_saved = True
                st.rerun()
        with save_col2:
            st.caption("保存当前训练的 DQN 模型到磁盘，之后可在侧边栏加载使用")

    else:
        if not run_btn and not search_btn:
            st.info("👈 在侧边栏设置好参数后，点击「开始训练」或「超参搜索」")

    # 实时信号面板（有加载模型时显示）
    render_rl_signal(df, symbol, asset_type)
