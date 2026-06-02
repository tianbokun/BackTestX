import copy
import itertools
import numpy as np
import pandas as pd
import time

from .environment import StockTradingEnv
from .dqn_agent import DQNAgent
from .feature_engineer import (
    compute_technical_indicators,
    normalize_indicators,
    get_state_vector,
    get_state_dim,
    compute_svm_xgb_signals,
    get_selected_columns,
    DEFAULT_FEATURE_GROUPS,
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
    feature_groups: list[str] = None,
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
    reward_window: int = 63,
    vol_penalty_coef: float = 0.1,
    dd_penalty_coef: float = 0.5,
    progress_callback=None,
    cancel_check=None,
    symbol: str = "",
) -> tuple:
    if feature_groups is None:
        feature_groups = DEFAULT_FEATURE_GROUPS
    close = _close_col(df_train)
    indicators = compute_technical_indicators(df_train)
    if "sentiment" in feature_groups:
        from data.fetcher import fetch_sentiment_data
        df_sent = fetch_sentiment_data(symbol)
        if df_sent is not None and not df_sent.empty:
            indicators = indicators.merge(df_sent, left_index=True, right_index=True, how="left")
            indicators = indicators.ffill().fillna(0)
    _, raw_cols = get_selected_columns(feature_groups)
    indicators = normalize_indicators(indicators, raw_columns=raw_cols)

    svm_sig, xgb_sig = None, None
    if system_version == "2.0":
        svm_sig, xgb_sig = compute_svm_xgb_signals(df_train)

    state_vectors = []
    for t in range(len(close)):
        sv = get_state_vector(indicators, t, system_version, feature_groups, svm_sig, xgb_sig)
        state_vectors.append(sv)
    state_vectors = np.array(state_vectors)

    state_dim = get_state_dim(system_version, feature_groups)
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

    best_reward = -float("inf")
    best_q_state = None
    best_target_state = None

    if cancel_check is None:
        cancel_check = lambda: False
    for ep in range(n_episodes):
        if cancel_check():
            break
        env = StockTradingEnv(
            state_vectors, close, dates,
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            min_commission=min_commission,
            stamp_duty=stamp_duty,
            reward_window=reward_window,
            vol_penalty_coef=vol_penalty_coef,
            dd_penalty_coef=dd_penalty_coef,
        )
        state = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            ep_reward += reward
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            agent.learn()

        if ep_reward > best_reward:
            best_reward = ep_reward
            best_q_state = copy.deepcopy(agent.q_net.state_dict())
            best_target_state = copy.deepcopy(agent.target_net.state_dict())

        if progress_callback:
            progress_callback(ep, n_episodes, ep_reward)

    agent_best = None
    if best_q_state is not None:
        agent_best = DQNAgent(
            state_dim=state_dim, n_actions=3,
            hidden=hidden, lr=lr, gamma=gamma,
            epsilon_start=epsilon_start, epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            buffer_capacity=buffer_capacity,
            batch_size=batch_size,
            target_update=target_update,
        )
        agent_best.q_net.load_state_dict(best_q_state)
        agent_best.target_net.load_state_dict(best_target_state)

    return agent, state_vectors, agent_best


def evaluate(
    agent: DQNAgent,
    df_test: pd.DataFrame,
    system_version: str = "1.0",
    feature_groups: list[str] = None,
    initial_capital: float = 1.0,
    commission_rate: float = 0.00025,
    min_commission: float = 5.0,
    stamp_duty: float = 0.001,
    symbol: str = "",
) -> dict:
    if feature_groups is None:
        feature_groups = DEFAULT_FEATURE_GROUPS
    close = _close_col(df_test)
    indicators = compute_technical_indicators(df_test)
    if "sentiment" in feature_groups:
        from data.fetcher import fetch_sentiment_data
        df_sent = fetch_sentiment_data(symbol)
        if df_sent is not None and not df_sent.empty:
            indicators = indicators.merge(df_sent, left_index=True, right_index=True, how="left")
            indicators = indicators.ffill().fillna(0)
    _, raw_cols = get_selected_columns(feature_groups)
    indicators = normalize_indicators(indicators, raw_columns=raw_cols)

    svm_sig, xgb_sig = None, None
    if system_version == "2.0":
        svm_sig, xgb_sig = compute_svm_xgb_signals(df_test)

    state_vectors = []
    for t in range(len(close)):
        sv = get_state_vector(indicators, t, system_version, feature_groups, svm_sig, xgb_sig)
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
    in_position = False
    for i, a in enumerate(actions):
        if a == 1 and not in_position:
            trades.append({
                "日期": df_test.index[i],
                "动作": "买入",
                "价格": round(float(close[i]), 4),
                "持仓市值": round(float(pv[i]), 4),
            })
            in_position = True
        elif a in (-1, 2) and in_position:
            trades.append({
                "日期": df_test.index[i],
                "动作": "卖出",
                "价格": round(float(close[i]), 4),
                "持仓市值": round(float(pv[i]), 4),
            })
            in_position = False

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
    feature_groups: list[str] = None,
) -> int:
    if feature_groups is None:
        feature_groups = DEFAULT_FEATURE_GROUPS
    close = _close_col(df)
    indicators = compute_technical_indicators(df)
    _, raw_cols = get_selected_columns(feature_groups)
    indicators = normalize_indicators(indicators, raw_columns=raw_cols)
    svm_sig, xgb_sig = None, None
    if system_version == "2.0":
        svm_sig, xgb_sig = compute_svm_xgb_signals(df)
    t = len(close) - 1
    state = get_state_vector(indicators, t, system_version, feature_groups, svm_sig, xgb_sig)
    return int(agent.act(state, eval_mode=True))


_HPARAM_GRID = {
    "lr": [1e-6, 5e-6, 1e-5, 5e-5],
    "gamma": [0.95, 0.98, 0.99],
    "hidden": [64, 128, 256],
    "n_episodes": [32, 64, 128],
    "epsilon_decay": [200, 500, 1000],
}


def hyperparam_search(
    df: pd.DataFrame,
    system_version: str = "1.0",
    feature_groups: list[str] = None,
    commission_rate: float = 0.00025,
    min_commission: float = 5.0,
    stamp_duty: float = 0.001,
    initial_capital: float = 1.0,
    reward_window: int = 63,
    vol_penalty_coef: float = 0.1,
    dd_penalty_coef: float = 0.5,
    n_folds: int = 3,
    progress_callback=None,
    combo_callback=None,
    fold_callback=None,
    cancel_check: callable = None,
) -> dict:
    """超参数网格搜索 + walk-forward 交叉验证.

    combo_callback(combo_idx, total_combos, best_params, best_score)
        — 每组合所有折完成后调用
    fold_callback(combo_idx, total_combos, fold_idx, n_folds, params_dict, fold_sharpe)
        — 每折完成后立即调用

    返回 best_params, best_score, 每折详情, 总耗时.
    """
    import time as _time
    t0 = _time.time()

    close = _close_col(df)
    n = len(close)
    fold_size = n // (n_folds + 1)
    if fold_size < 30:
        return {"best_params": None, "best_score": -999, "error": f"数据太少 ({n}行), 至少需要 {(n_folds+1)*30} 行"}

    keys, values = zip(*_HPARAM_GRID.items())
    combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
    total = len(combos)

    best_score = -999.0
    best_params = None
    fold_details = []

    for ci, base_params in enumerate(combos):
        if cancel_check is not None and cancel_check():
            break
        fold_scores = []
        for fold in range(n_folds):
            train_end = (fold + 1) * fold_size
            val_start = train_end
            val_end = min(val_start + fold_size, n)
            if val_end - val_start < 20:
                continue

            df_fold_train = df.iloc[:train_end]
            df_fold_val = df.iloc[val_start:val_end]

            params = {
                **base_params,
                "commission_rate": commission_rate,
                "min_commission": min_commission,
                "stamp_duty": stamp_duty,
                "initial_capital": initial_capital,
                "reward_window": reward_window,
                "vol_penalty_coef": vol_penalty_coef,
                "dd_penalty_coef": dd_penalty_coef,
            }
            try:
                agent, _, _ = train_dqn(df_fold_train, system_version, feature_groups, **params)
                result = evaluate(agent, df_fold_val, system_version, feature_groups,
                                  initial_capital=initial_capital,
                                  commission_rate=commission_rate,
                                  min_commission=min_commission,
                                  stamp_duty=stamp_duty)
                fold_score = float(result["sharpe_ratio"])
                fold_scores.append(fold_score)
            except Exception:
                fold_score = -999.0
                fold_scores.append(-999)

            if fold_callback:
                fold_callback(ci, total, fold, n_folds, dict(base_params), fold_score)

        avg_score = float(np.mean(fold_scores)) if fold_scores else -999.0
        if avg_score > best_score:
            best_score = avg_score
            best_params = dict(base_params)

        fold_details.append({
            "params": dict(base_params),
            "fold_scores": fold_scores,
            "avg_score": avg_score,
        })

        if combo_callback:
            combo_callback(ci, total, best_params, best_score)

    elapsed = _time.time() - t0
    return {
        "best_params": best_params,
        "best_score": best_score,
        "total_combos": total,
        "fold_details": fold_details,
        "n_folds": n_folds,
        "elapsed_sec": elapsed,
    }


def compute_signal_history(
    agent: DQNAgent,
    df: pd.DataFrame,
    system_version: str = "1.0",
    feature_groups: list[str] = None,
) -> list:
    if feature_groups is None:
        feature_groups = DEFAULT_FEATURE_GROUPS
    close = _close_col(df)
    indicators = compute_technical_indicators(df)
    _, raw_cols = get_selected_columns(feature_groups)
    indicators = normalize_indicators(indicators, raw_columns=raw_cols)
    svm_sig, xgb_sig = None, None
    if system_version == "2.0":
        svm_sig, xgb_sig = compute_svm_xgb_signals(df)
    signals = []
    for t in range(len(close)):
        state = get_state_vector(indicators, t, system_version, feature_groups, svm_sig, xgb_sig)
        action = int(agent.act(state, eval_mode=True))
        signals.append(action)
    return signals
