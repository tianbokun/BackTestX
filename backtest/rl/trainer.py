import numpy as np
import pandas as pd
import time

from .environment import StockTradingEnv
from .dqn_agent import DQNAgent
from .feature_engineer import (
    compute_technical_indicators,
    get_state_vector,
    get_state_dim,
    compute_svm_xgb_signals,
)
from .metrics import sharpe_ratio, max_drawdown


def _close_col(df):
    for n in ["收盘", "收盘价", "close"]:
        if n in df.columns:
            return df[n].values.astype(float)
    raise KeyError(f"找不到收盘价列，可用列: {list(df.columns)}")


def train_dqn(
    df_train: pd.DataFrame,
    system_version: str = "1.0",
    n_episodes: int = 64,
    batch_size: int = 200,
    lr: float = 1e-5,
    gamma: float = 0.98,
    hidden: int = 128,
    target_update: int = 50,
    buffer_capacity: int = 10000,
    epsilon_start: float = 0.9,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 500,
    commission_rate: float = 0.00025,
    min_commission: float = 5.0,
    stamp_duty: float = 0.001,
    initial_capital: float = 1.0,
    progress_callback=None,
) -> tuple:
    close = _close_col(df_train)
    indicators = compute_technical_indicators(df_train)

    svm_sig, xgb_sig = None, None
    if system_version == "2.0":
        svm_sig, xgb_sig = compute_svm_xgb_signals(df_train)

    state_vectors = []
    for t in range(len(close)):
        sv = get_state_vector(indicators, t, system_version, svm_sig, xgb_sig)
        state_vectors.append(sv)
    state_vectors = np.array(state_vectors)

    state_dim = get_state_dim(system_version)
    dates = df_train.index.tolist()

    agent = DQNAgent(
        state_dim=state_dim,
        n_actions=3,
        hidden=hidden,
        lr=lr,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        buffer_capacity=buffer_capacity,
        batch_size=batch_size,
        target_update=target_update,
    )

    for ep in range(n_episodes):
        env = StockTradingEnv(
            state_vectors, close, dates,
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            min_commission=min_commission,
            stamp_duty=stamp_duty,
        )
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            agent.learn()

        if progress_callback:
            progress_callback(ep, n_episodes, agent.losses[-1] if agent.losses else 0)

    return agent, state_vectors


def evaluate(
    agent: DQNAgent,
    df_test: pd.DataFrame,
    system_version: str = "1.0",
    initial_capital: float = 1.0,
    commission_rate: float = 0.00025,
    min_commission: float = 5.0,
    stamp_duty: float = 0.001,
) -> dict:
    close = _close_col(df_test)
    indicators = compute_technical_indicators(df_test)

    svm_sig, xgb_sig = None, None
    if system_version == "2.0":
        svm_sig, xgb_sig = compute_svm_xgb_signals(df_test)

    state_vectors = []
    for t in range(len(close)):
        sv = get_state_vector(indicators, t, system_version, svm_sig, xgb_sig)
        state_vectors.append(sv)
    state_vectors = np.array(state_vectors)

    env = StockTradingEnv(
        state_vectors, close, df_test.index.tolist(),
        initial_capital=initial_capital,
        commission_rate=commission_rate,
        min_commission=min_commission,
        stamp_duty=stamp_duty,
    )
    state = env.reset()
    done = False
    while not done:
        action = agent.act(state, eval_mode=True)
        state, reward, done = env.step(action)

    pv = np.array(env.portfolio_values)
    actions = np.array(env.actions_taken)

    final_value = float(pv[-1]) if len(pv) > 0 else initial_capital
    total_ret = (final_value - initial_capital) / initial_capital * 100

    daily_returns = np.diff(pv) / pv[:-1] if len(pv) > 1 else np.array([0])
    sharpe = sharpe_ratio(daily_returns)
    mdd = max_drawdown(pv)

    trades = []
    prev_action = 0
    for i, a in enumerate(actions):
        if a != prev_action and a != 0:
            trades.append({
                "日期": df_test.index[i],
                "动作": "买入" if a == 1 else "卖出",
                "价格": round(float(close[i]), 4),
                "持仓市值": round(float(pv[i]), 4),
            })
        prev_action = a

    return {
        "final_value": round(final_value, 4),
        "total_return_pct": round(total_ret, 2),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown_pct": round(mdd, 2),
        "num_trades": len(trades),
        "trades": pd.DataFrame(trades) if trades else pd.DataFrame(),
        "equity_curve": pv,
        "dates": df_test.index,
        "actions": actions,
    }


def run_bh_baseline(
    df_test: pd.DataFrame,
    initial_capital: float = 1.0,
) -> dict:
    close = _close_col(df_test)
    n = len(close)
    if n < 2:
        return {"final_value": initial_capital, "total_return_pct": 0, "sharpe_ratio": 0, "max_drawdown_pct": 0}
    buy_price = close[0]
    shares = initial_capital / buy_price
    pv = shares * close
    final_value = float(pv[-1])
    total_ret = (final_value - initial_capital) / initial_capital * 100
    daily_returns = np.diff(pv) / pv[:-1]
    sharpe = sharpe_ratio(daily_returns)
    mdd = max_drawdown(pv)
    return {
        "final_value": round(final_value, 4),
        "total_return_pct": round(total_ret, 2),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown_pct": round(mdd, 2),
        "equity_curve": pv,
        "dates": df_test.index,
    }


def predict_signal(
    agent: DQNAgent,
    df: pd.DataFrame,
    system_version: str = "1.0",
) -> int:
    close = _close_col(df)
    indicators = compute_technical_indicators(df)
    svm_sig, xgb_sig = None, None
    if system_version == "2.0":
        svm_sig, xgb_sig = compute_svm_xgb_signals(df)
    t = len(close) - 1
    state = get_state_vector(indicators, t, system_version, svm_sig, xgb_sig)
    return int(agent.act(state, eval_mode=True))


def compute_signal_history(
    agent: DQNAgent,
    df: pd.DataFrame,
    system_version: str = "1.0",
) -> list:
    close = _close_col(df)
    indicators = compute_technical_indicators(df)
    svm_sig, xgb_sig = None, None
    if system_version == "2.0":
        svm_sig, xgb_sig = compute_svm_xgb_signals(df)
    signals = []
    for t in range(len(close)):
        state = get_state_vector(indicators, t, system_version, svm_sig, xgb_sig)
        action = int(agent.act(state, eval_mode=True))
        signals.append(action)
    return signals
