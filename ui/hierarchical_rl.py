from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from backtest.rl.hierarchical_trainer import (
    HierarchicalTrainer,
    fetch_multi_etf_data,
    _align_dates,
    DEFAULT_ETF_POOL,
)
from data.symbol_registry import SymbolRegistry


def _init_session_keys():
    for k in ("hrl_trainer", "hrl_train_result", "hrl_val_result", "hrl_test_result",
              "hrl_capital", "hrl_test_dates", "hrl_benchmarks"):
        if k not in st.session_state:
            st.session_state[k] = None


def _etf_name(sym: str) -> str:
    entry = SymbolRegistry.get(sym)
    return entry["name"] if entry else sym


def render_hierarchical_rl(end_date, adjust):
    _init_session_keys()

    st.title("🧠 分层强化学习 (PPO + DQN)")

    st.markdown("""
    <div style="background:#f0f4ff;border-radius:10px;padding:1rem 1.25rem;margin-bottom:1.5rem;font-size:0.9rem;color:#1e293b">
    <b>架构说明：</b><br>
    <b>上层 (PPO)</b> — 择时：根据市场状态决定总仓位比例 (0%~100%)<br>
    <b>下层 (DQN)</b> — 选股：在 ETF 池中独立决策每支的买入/持有/卖出，受 PPO 仓位约束
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("### 🧠 分层强化学习参数")

    all_registered_etfs = SymbolRegistry.list(asset_type="etf")
    if not all_registered_etfs:
        st.error("尚未添加任何 ETF 代码。请先在「📋 代码管理」中添加。")
        st.stop()

    all_tags = sorted({t for s in all_registered_etfs for t in s.get("tags", [])})
    selected_tags = st.sidebar.multiselect("标签过滤", all_tags, default=all_tags, key="hrl_tag_filter")
    filtered_etfs = [s for s in all_registered_etfs if any(t in s.get("tags", []) for t in selected_tags)]

    selected_symbols = []
    with st.sidebar.expander("📋 ETF 池选择", expanded=True):
        all_on = st.checkbox("全选", value=True, key="hrl_select_all")
        for s in filtered_etfs:
            sym = s["symbol"]
            label = s["name"]
            on = st.checkbox(f"{sym} {label}", value=all_on, key=f"hrl_etf_{sym}")
            if on:
                selected_symbols.append(sym)

    if not selected_symbols:
        st.error("请至少选择一支 ETF")
        st.stop()

    rl_end_str = end_date.strftime("%Y%m%d")

    with st.spinner(f"正在获取 {len(selected_symbols)} 支 ETF 数据..."):
        etf_data = {}
        for sym in selected_symbols:
            df = SymbolRegistry.fetch_data(sym, adjust=adjust)
            if df is not None and not df.empty:
                etf_data[sym] = df

    if len(etf_data) < 2:
        st.error(f"数据获取失败，成功获取 {len(etf_data)} 支，需要至少 2 支 ETF")
        st.stop()

    aligned_dates = _align_dates(etf_data)
    st.success(f"成功获取 {len(etf_data)} 支 ETF 数据，对齐后共 {len(aligned_dates)} 个交易日")
    st.caption(f"ETF: {', '.join(f'{s}({_etf_name(s)})' for s in sorted(etf_data.keys()))}")
    st.caption(f"日期范围: {aligned_dates[0][:10]} ~ {aligned_dates[-1][:10]}")

    n_total = len(aligned_dates)
    train_end_i = int(n_total * 0.6)
    val_end_i = int(n_total * 0.8)

    train_start = aligned_dates[0][:10]
    train_end = aligned_dates[min(train_end_i, n_total - 1)][:10]

    val_start = aligned_dates[min(train_end_i + 1, n_total - 1)][:10]
    val_end = aligned_dates[min(val_end_i, n_total - 1)][:10]

    test_start = aligned_dates[min(val_end_i + 1, n_total - 1)][:10]
    test_end = aligned_dates[-1][:10]

    st.info(f"训练集: {train_start} ~ {train_end} ({train_end_i+1} 天) | "
            f"验证集: {val_start} ~ {val_end} ({val_end_i - train_end_i} 天) | "
            f"测试集: {test_start} ~ {test_end} ({n_total - val_end_i - 1} 天)")

    with st.sidebar.expander("💰 费率设置", expanded=False):
        hrl_commission = st.number_input("佣金费率", min_value=0.0, value=0.000235, step=0.000005, format="%.6f", key="hrl_commission")
        hrl_min_commission = st.number_input("最低佣金(元)", min_value=0.0, value=5.0, step=1.0, key="hrl_min_comm")
        hrl_stamp_duty = st.number_input("印花税率", min_value=0.0, value=0.001, step=0.0001, format="%.4f", key="hrl_stamp")
        hrl_capital = st.number_input("初始本金(元)", min_value=1000.0, value=100000.0, step=10000.0, key="hrl_capital_input")

    with st.sidebar.expander("⚙️ PPO 超参数", expanded=False):
        ppo_lr = st.text_input("PPO 学习率", value="3e-4", key="hrl_ppo_lr")
        ppo_gamma = st.text_input("PPO γ", value="0.99", key="hrl_ppo_gamma")
        ppo_clip = st.text_input("PPO Clip ε", value="0.2", key="hrl_ppo_clip")
        ppo_entropy = st.text_input("熵奖励 β", value="0.01", key="hrl_ppo_entropy")
        ppo_epochs = st.number_input("PPO Epochs", min_value=1, value=4, key="hrl_ppo_epochs")
        ppo_hidden = st.number_input("PPO 隐藏层", min_value=32, value=64, step=16, key="hrl_ppo_hidden")
        gae_lambda = st.text_input("GAE λ", value="0.95", key="hrl_gae_lambda")

    with st.sidebar.expander("⚙️ DQN 超参数", expanded=False):
        dqn_lr = st.text_input("DQN 学习率", value="1e-5", key="hrl_dqn_lr")
        dqn_gamma = st.text_input("DQN γ", value="0.98", key="hrl_dqn_gamma")
        dqn_hidden = st.number_input("DQN 隐藏层", min_value=32, value=128, step=32, key="hrl_dqn_hidden")
        dqn_epsilon_start = st.text_input("ε 初始值", value="0.9", key="hrl_dqn_eps_start")
        dqn_epsilon_end = st.text_input("ε 终值", value="0.01", key="hrl_dqn_eps_end")
        dqn_epsilon_decay = st.number_input("ε 衰减步数", min_value=100, value=500, step=100, key="hrl_dqn_eps_decay")
        dqn_buffer = st.number_input("回放容量", min_value=1000, value=10000, step=1000, key="hrl_dqn_buffer")
        dqn_batch = st.number_input("Batch 大小", min_value=32, value=200, step=32, key="hrl_dqn_batch")

    with st.sidebar.expander("⚙️ 训练参数", expanded=False):
        n_episodes = st.number_input("训练轮数", min_value=5, value=32, step=5, key="hrl_n_episodes")
        ppo_update_freq = st.number_input("PPO 更新频率(步)", min_value=5, value=20, step=5, key="hrl_ppo_update_freq")
        trade_fraction = st.text_input("每笔交易比例", value="0.2", key="hrl_trade_fraction",
                                       help="单支ETF每次最多交易占总资金的比例")

    run_btn = st.sidebar.button("🚀 开始分层训练", type="primary", width='stretch', key="hrl_run_btn")

    total_dates = aligned_dates

    if run_btn:
        st.markdown("---")
        st.subheader("📊 分层训练与回测")

        progress_bar = st.progress(0)
        reward_chart = st.empty()
        ep_rewards = []

        try:
            ppo_lr_f = float(ppo_lr)
            ppo_gamma_f = float(ppo_gamma)
            ppo_clip_f = float(ppo_clip)
            ppo_entropy_f = float(ppo_entropy)
            gae_lambda_f = float(gae_lambda)
            dqn_lr_f = float(dqn_lr)
            dqn_gamma_f = float(dqn_gamma)
            trade_fraction_f = float(trade_fraction)
        except ValueError:
            st.error("超参数格式错误")
            st.stop()

        def _progress(ep, total, reward):
            progress_bar.progress((ep + 1) / total)
            ep_rewards.append((ep, reward))
            if len(ep_rewards) > 1:
                rdf = pd.DataFrame(ep_rewards, columns=["ep", "reward"])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=rdf["ep"], y=rdf["reward"], mode="lines", name="Episode Reward"))
                fig.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10),
                                  xaxis_title="Episode", yaxis_title="Reward")
                reward_chart.plotly_chart(fig, width='stretch')

        train_dates = total_dates[:train_end_i + 1]
        train_etf_data = {}
        for sym, df in etf_data.items():
            train_etf_data[sym] = df.loc[:pd.Timestamp(train_end)]

        with st.spinner("正在分层训练 (PPO+DQN)..."):
            trainer = HierarchicalTrainer(
                etf_data=train_etf_data,
                aligned_dates=train_dates,
                n_episodes=n_episodes,
                ppo_lr=ppo_lr_f, ppo_gamma=ppo_gamma_f,
                clip_epsilon=ppo_clip_f, entropy_beta=ppo_entropy_f,
                gae_lambda=gae_lambda_f, ppo_hidden=int(ppo_hidden),
                ppo_epochs=int(ppo_epochs), ppo_update_freq=int(ppo_update_freq),
                dqn_lr=dqn_lr_f, dqn_gamma=dqn_gamma_f,
                dqn_hidden=int(dqn_hidden),
                dqn_epsilon_start=float(dqn_epsilon_start),
                dqn_epsilon_end=float(dqn_epsilon_end),
                dqn_epsilon_decay=int(dqn_epsilon_decay),
                dqn_buffer_capacity=int(dqn_buffer),
                dqn_batch_size=int(dqn_batch),
                commission_rate=float(hrl_commission),
                min_commission=float(hrl_min_commission),
                stamp_duty=float(hrl_stamp_duty),
                initial_capital=float(hrl_capital),
                trade_fraction=trade_fraction_f,
            )
            train_result = trainer.train(progress_callback=_progress)

        test_dates = total_dates[val_end_i + 1:]
        test_etf_data = {}
        for sym, df in etf_data.items():
            test_etf_data[sym] = df.loc[pd.Timestamp(test_start):pd.Timestamp(test_end)]

        if test_etf_data and test_dates:
            test_trainer = HierarchicalTrainer(
                etf_data=test_etf_data,
                aligned_dates=test_dates,
                n_episodes=1,
                ppo_lr=ppo_lr_f, ppo_gamma=ppo_gamma_f,
                clip_epsilon=ppo_clip_f, entropy_beta=ppo_entropy_f,
                gae_lambda=gae_lambda_f, ppo_hidden=int(ppo_hidden),
                ppo_epochs=int(ppo_epochs), ppo_update_freq=int(ppo_update_freq),
                dqn_lr=dqn_lr_f, dqn_gamma=dqn_gamma_f,
                dqn_hidden=int(dqn_hidden),
                dqn_epsilon_start=float(dqn_epsilon_start),
                dqn_epsilon_end=float(dqn_epsilon_end),
                dqn_epsilon_decay=int(dqn_epsilon_decay),
                dqn_buffer_capacity=int(dqn_buffer),
                dqn_batch_size=int(dqn_batch),
                commission_rate=float(hrl_commission),
                min_commission=float(hrl_min_commission),
                stamp_duty=float(hrl_stamp_duty),
                initial_capital=float(hrl_capital),
                trade_fraction=trade_fraction_f,
            )
            test_trainer.ppo = trainer.ppo
            test_trainer.dqn = trainer.dqn
            test_result = test_trainer.evaluate()

            st.session_state.hrl_test_result = test_result
            st.session_state.hrl_test_dates = test_dates

            benchmarks = test_trainer.compute_benchmarks()
            st.session_state.hrl_benchmarks = benchmarks

        st.session_state.hrl_trainer = trainer
        st.session_state.hrl_train_result = train_result
        st.session_state.hrl_capital = float(hrl_capital)
        st.success("✅ 训练完成！")

    test_result = st.session_state.hrl_test_result
    capital = st.session_state.hrl_capital or 100000.0

    if test_result is not None:
        st.markdown("---")
        st.subheader("📊 分层 RL 回测结果")

        benchmarks = st.session_state.hrl_benchmarks

        st.markdown("#### 📋 测试集策略指标")
        rows = [{
            "策略": "HRL (PPO择时 + DQN选股)",
            "最终金额": test_result["final_value"],
            "收益率%": test_result["total_return_pct"],
            "夏普比率": test_result["sharpe_ratio"],
            "最大回撤%": test_result["max_drawdown_pct"],
        }]
        if benchmarks:
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

        if test_result.get("equity_curve") is not None:
            st.markdown("#### 📈 累计净值曲线")

            single_bh = (benchmarks or {}).get("single_etf_bh", {})
            if single_bh:
                selected_syms = st.multiselect(
                    "显示单只ETF全仓持有对比",
                    options=list(single_bh.keys()),
                    default=[],
                    key="hrl_bh_selector",
                )
            else:
                selected_syms = []

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=test_result["dates"],
                y=test_result["equity_curve"],
                mode="lines", name="HRL",
                line=dict(color="#2563eb", width=2),
            ))
            fig.add_hline(y=float(capital), line_dash="dot", line_color="gray", annotation_text="初始本金")

            if benchmarks:
                ew = benchmarks.get("equal_weight_bh")
                if ew:
                    fig.add_trace(go.Scatter(
                        x=test_result["dates"],
                        y=ew["equity_curve"],
                        mode="lines", name="等权买入持有",
                        line=dict(color="#ef4444", width=2, dash="dash"),
                    ))
                for dca_key, dca_label, dca_color in [
                    ("monthly_dca", "月定投(等权)", "#10b981"),
                    ("ma_adjust_dca", "均线偏离定投(等权)", "#f59e0b"),
                ]:
                    dca = benchmarks.get(dca_key)
                    if dca is not None and "total_value_series" in dca:
                        dca_curve = dca["total_value_series"].reindex(
                            pd.DatetimeIndex(test_result["dates"])
                        ).ffill().fillna(float(capital)).values
                        fig.add_trace(go.Scatter(
                            x=test_result["dates"],
                            y=dca_curve,
                            mode="lines", name=dca_label,
                            line=dict(color=dca_color, width=1.5, dash="dot"),
                        ))
                bh_colors = ["#8b5cf6", "#ec4899", "#06b6d4", "#84cc16", "#f97316", "#a855f7"]
                for i, sym in enumerate(selected_syms):
                    bh = single_bh.get(sym)
                    if bh is not None:
                        fig.add_trace(go.Scatter(
                            x=test_result["dates"],
                            y=bh["equity_curve"],
                            mode="lines", name=f"{sym}全仓持有",
                            line=dict(color=bh_colors[i % len(bh_colors)], width=1.5, dash="dot"),
                        ))

            fig.update_layout(
                xaxis_title="日期", yaxis_title=f"账户总值 (初始={capital:,.0f})",
                hovermode="x unified", height=450,
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
            )
            st.plotly_chart(fig, width='stretch')

            st.markdown("#### 📊 仓位比例时序 (PPO 决策)")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=test_result["dates"],
                y=test_result["position_ratios"],
                mode="lines", name="仓位比例",
                line=dict(color="#2563eb", width=2),
                fill="tozeroy",
            ))
            fig2.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="半仓")
            fig2.update_layout(
                xaxis_title="日期", yaxis_title="仓位比例",
                hovermode="x unified", height=250,
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig2, width='stretch')

        if "trade_log" in test_result and not test_result["trade_log"].empty:
            with st.expander("📝 交易记录", expanded=False):
                st.dataframe(test_result["trade_log"], width='stretch', hide_index=True)
                if "trade_events" in test_result and not test_result["trade_events"].empty:
                    st.caption("交易事件（仅操作变化日）:")
                    st.dataframe(test_result["trade_events"], width='stretch', hide_index=True)

        st.markdown("---")
        st.markdown("#### 💾 保存模型")
        _save_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        _save_default = f"hrl_{_save_ts}"
        _save_name = st.text_input("模型名称", value=_save_default, key="hrl_save_name")
        if st.button("💾 保存 HRL 模型", type="primary", key="hrl_save_btn"):
            trainer = st.session_state.get("hrl_trainer")
            if trainer:
                save_path = f"saved_models/rl/{_save_name}"
                trainer.save(save_path)
                st.success(f"✅ 模型已保存: {save_path}")
    else:
        if not run_btn:
            st.info("👈 在侧边栏设置好参数后，点击「开始分层训练」")
