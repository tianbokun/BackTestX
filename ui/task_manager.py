from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from backtest.rl.task_manager import TaskManager, TaskStatus


STATUS_EMOJI = {
    TaskStatus.PENDING.value: "⏳",
    TaskStatus.RUNNING.value: "🔄",
    TaskStatus.COMPLETED.value: "✅",
    TaskStatus.FAILED.value: "❌",
    TaskStatus.CANCELLED.value: "🚫",
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


def _render_rl_result(result: dict):
    dqn = result["result_dqn"]
    bh = result["result_bh"]
    meta = result.get("meta", {})

    comp = pd.DataFrame([
        {"策略": "DQN", "最终金额": dqn["final_value"],
         "收益率%": dqn["total_return_pct"],
         "夏普比率": dqn["sharpe_ratio"],
         "最大回撤%": dqn["max_drawdown_pct"],
         "交易次数": dqn["num_trades"]},
        {"策略": "买入持有(BH)", "最终金额": bh["final_value"],
         "收益率%": bh["total_return_pct"],
         "夏普比率": bh["sharpe_ratio"],
         "最大回撤%": bh["max_drawdown_pct"],
         "交易次数": 0},
    ])
    st.dataframe(comp, width='stretch', hide_index=True)

    test_idx = _ensure_dates(meta.get("df_test_index", dqn.get("dates")))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=test_idx, y=dqn["equity_curve"],
        mode="lines", name=f"DQN ({meta.get('system_version', '?')})",
        line=dict(color="#1f77b4", width=2),
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

    trades = _ensure_trades(dqn.get("trades"))
    if not trades.empty:
        trades = trades.copy()
        if "日期" in trades.columns:
            trades["日期"] = trades["日期"].dt.strftime("%Y-%m-%d")
        st.markdown("**📝 交易记录**")
        st.dataframe(trades, width='stretch', hide_index=True)


def _render_hrl_result(result: dict):
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


@st.fragment(run_every=1.0)
def _render_running_detail(task: dict, mgr: TaskManager, tid: str):
    live = mgr.get_task(tid)
    if live and live["status"] != TaskStatus.RUNNING.value:
        st.rerun()
        return

    pct = task.get("progress", 0) * 100
    st.progress(task.get("progress", 0))
    st.markdown(f"**进度: {pct:.0f}%**")

    if st.button("取消训练", key=f"detail_cancel_{tid}"):
        mgr.cancel(tid)
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

    elif status == TaskStatus.COMPLETED.value:
        result = mgr.get_result(tid)
        if result:
            st.subheader(f"📊 {task['type']} 详细结果")
            if task["type"] == "RL训练":
                _render_rl_result(result)
            elif task["type"] == "HRL训练":
                _render_hrl_result(result)

        pdata = task.get("_progress_data") or mgr.get_progress_data(tid)
        if pdata:
            st.subheader("📈 训练过程")
            _render_progress_chart(pdata)

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
            if status in (TaskStatus.PENDING.value, TaskStatus.RUNNING.value):
                if st.button("取消", key=f"cancel_{tid}"):
                    mgr.cancel(tid)
                    st.rerun()

        if i < len(tasks) - 1:
            st.divider()


def _select_task(tid: str):
    st.session_state.selected_task_id = tid


def render_task_manager():
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
