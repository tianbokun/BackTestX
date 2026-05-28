import numpy as np
import pandas as pd

from .ppo_agent import PPOAgent
from .dqn_agent import DQNAgent
from .multi_asset_env import MultiAssetTradingEnv
from .feature_engineer import (
    compute_technical_indicators,
    normalize_indicators,
    get_state_vector,
    get_state_dim,
    DEFAULT_FEATURE_GROUPS,
)
from .metrics import sharpe_ratio, max_drawdown


DEFAULT_ETF_POOL = [
    "515790", "159790", "512660", "588000", "515070",
    "515030", "512480", "159995", "515050", "159819",
    "515900", "512010", "512580", "510880", "159915",
]


def _close_col(df):
    for n in ["收盘", "收盘价", "close"]:
        if n in df.columns:
            return df[n].values.astype(float)
    raise KeyError(f"找不到收盘价列，可用列: {list(df.columns)}")


def _align_dates(etf_data: dict[str, pd.DataFrame]) -> list:
    common_idx = None
    for sym, df in etf_data.items():
        if common_idx is None:
            common_idx = set(df.index)
        else:
            common_idx &= set(df.index)
    common_idx = sorted(common_idx)
    return [str(d) for d in common_idx]


def _close_col_safe(df):
    for n in ["收盘", "收盘价", "close"]:
        if n in df.columns:
            return df[n]
    return None


def fetch_multi_etf_data(etf_symbols: list[str], start_date: str, end_date: str, adjust: str = "qfq") -> dict[str, pd.DataFrame]:
    from data_fetcher import fetch_history
    result = {}
    for sym in etf_symbols:
        try:
            df = fetch_history(asset_type="etf", symbol=sym, start_date=start_date, end_date=end_date, adjust=adjust)
            if df is not None and not df.empty:
                result[sym] = df
        except Exception:
            pass
    return result


class HierarchicalTrainer:
    def __init__(
        self,
        etf_data: dict[str, pd.DataFrame],
        aligned_dates: list,
        ppo_state_dim: int = 12,
        dqn_state_dim: int = 6,
        ppo_hidden: int = 64,
        dqn_hidden: int = 128,
        ppo_lr: float = 3e-4,
        dqn_lr: float = 1e-5,
        ppo_gamma: float = 0.99,
        dqn_gamma: float = 0.98,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_beta: float = 0.01,
        value_coef: float = 0.5,
        ppo_epochs: int = 4,
        dqn_batch_size: int = 200,
        dqn_target_update: int = 50,
        dqn_buffer_capacity: int = 10000,
        dqn_epsilon_start: float = 0.9,
        dqn_epsilon_end: float = 0.01,
        dqn_epsilon_decay: float = 500,
        n_episodes: int = 64,
        ppo_update_freq: int = 20,
        commission_rate: float = 0.000235,
        min_commission: float = 5.0,
        stamp_duty: float = 0.001,
        initial_capital: float = 100000.0,
        trade_fraction: float = 0.2,
    ):
        self.etf_data = etf_data
        self.aligned_dates = aligned_dates
        self.symbols = sorted(etf_data.keys())
        self.n_etfs = len(self.symbols)
        self.n_episodes = n_episodes
        self.ppo_update_freq = ppo_update_freq

        self.ppo = PPOAgent(
            state_dim=ppo_state_dim,
            hidden=ppo_hidden,
            lr=ppo_lr,
            gamma=ppo_gamma,
            gae_lambda=gae_lambda,
            clip_epsilon=clip_epsilon,
            entropy_beta=entropy_beta,
            value_coef=value_coef,
            ppo_epochs=ppo_epochs,
        )

        self.dqn = DQNAgent(
            state_dim=dqn_state_dim,
            n_actions=3,
            hidden=dqn_hidden,
            lr=dqn_lr,
            gamma=dqn_gamma,
            epsilon_start=dqn_epsilon_start,
            epsilon_end=dqn_epsilon_end,
            epsilon_decay=dqn_epsilon_decay,
            buffer_capacity=dqn_buffer_capacity,
            batch_size=dqn_batch_size,
            target_update=dqn_target_update,
        )

        self.fee_params = dict(
            commission_rate=commission_rate,
            min_commission=min_commission,
            stamp_duty=stamp_duty,
            initial_capital=initial_capital,
            trade_fraction=trade_fraction,
        )

        self.env = None
        self.ppo_losses = []
        self.dqn_losses = []
        self.episode_rewards = []

    def train(self, progress_callback=None):
        for ep in range(self.n_episodes):
            self.env = MultiAssetTradingEnv(
                etf_data=self.etf_data,
                aligned_dates=self.aligned_dates,
                **self.fee_params,
            )
            market_state, etf_states = self.env.reset()
            done = False
            ep_reward = 0
            step = 0

            while not done:
                position_ratio, log_prob, value = self.ppo.act(market_state)
                actions = {}
                for sym in self.symbols:
                    s = etf_states.get(sym, np.zeros(6, dtype=np.float32))
                    action = self.dqn.act(s)
                    actions[sym] = action

                next_market, next_etf_states, reward, done, info = self.env.step(position_ratio, actions)

                self.ppo.memory.push(market_state, position_ratio, log_prob, reward, done, value)

                for sym in self.symbols:
                    s = etf_states.get(sym, np.zeros(6, dtype=np.float32))
                    ns = next_etf_states.get(sym, np.zeros(6, dtype=np.float32))
                    a = actions[sym]
                    self.dqn.memory.push(s, a, reward, ns, done)

                self.dqn.learn()

                market_state = next_market
                etf_states = next_etf_states
                ep_reward += reward
                step += 1

                if step % self.ppo_update_freq == 0:
                    self.ppo.learn()

            if self.ppo.memory:
                self.ppo.learn()

            self.episode_rewards.append(ep_reward)
            if self.ppo.losses:
                self.ppo_losses.append(float(np.mean(self.ppo.losses[-10:])))
            if self.dqn.losses:
                self.dqn_losses.append(float(np.mean(self.dqn.losses[-10:])))

            if progress_callback:
                progress_callback(ep, self.n_episodes, ep_reward)

        return {
            "ppo_losses": self.ppo_losses,
            "dqn_losses": self.dqn_losses,
            "episode_rewards": self.episode_rewards,
        }

    def evaluate(self):
        env = MultiAssetTradingEnv(
            etf_data=self.etf_data,
            aligned_dates=self.aligned_dates,
            **self.fee_params,
        )
        market_state, etf_states = env.reset()
        done = False

        ACTION_LABELS = {0: "持有", 1: "买入", 2: "卖出"}

        portfolio_values = [env.initial_capital]
        position_ratio_history = []
        actions_history = []
        dates = []
        trade_log = []

        while not done:
            position_ratio, _, _ = self.ppo.act(market_state, eval_mode=True)
            actions = {}
            for sym in self.symbols:
                s = etf_states.get(sym, np.zeros(6, dtype=np.float32))
                action = self.dqn.act(s, eval_mode=True)
                actions[sym] = action

            next_market, next_etf_states, reward, done, info = env.step(position_ratio, actions)

            pv = info["portfolio_value"]
            portfolio_values.append(pv)
            position_ratio_history.append(position_ratio)
            actions_history.append(actions)
            date = env.dates[min(env.t, len(env.dates) - 1)] if env.t < len(env.dates) else env.dates[-1]
            dates.append(date)

            entry = {
                "日期": date,
                "PPO仓位%": round(float(position_ratio) * 100, 1),
                "现金": info["cash"],
                "总资产": info["portfolio_value"],
            }
            for sym in self.symbols:
                pos = info["positions"].get(sym, {})
                entry[f"{sym}_操作"] = ACTION_LABELS.get(actions.get(sym, 0), "?")
                entry[f"{sym}_市值"] = pos.get("value", 0)
                entry[f"{sym}_比例%"] = pos.get("ratio", 0)
            trade_log.append(entry)

            market_state = next_market
            etf_states = next_etf_states

        pv_array = np.array(portfolio_values)
        total_ret = (pv_array[-1] - env.initial_capital) / env.initial_capital * 100

        daily_returns = np.diff(pv_array) / pv_array[:-1] if len(pv_array) > 1 else np.array([0])
        sharpe = sharpe_ratio(daily_returns)
        mdd = max_drawdown(pv_array)

        trade_log_df = pd.DataFrame(trade_log)
        op_cols = [c for c in trade_log_df.columns if c.endswith("_操作")]
        if len(trade_log_df) > 0 and len(op_cols) > 0:
            prev_ops = None
            event_rows = []
            for _, row in trade_log_df.iterrows():
                cur_ops = tuple(row[c] for c in op_cols)
                if prev_ops is None or cur_ops != prev_ops:
                    event_rows.append(row)
                prev_ops = cur_ops
            trade_events_df = pd.DataFrame(event_rows) if event_rows else pd.DataFrame()
        else:
            trade_events_df = pd.DataFrame()

        return {
            "final_value": round(float(pv_array[-1]), 2),
            "total_return_pct": round(float(total_ret), 2),
            "sharpe_ratio": round(float(sharpe), 4),
            "max_drawdown_pct": round(float(mdd), 2),
            "equity_curve": pv_array,
            "dates": dates,
            "position_ratios": position_ratio_history,
            "actions": actions_history,
            "trade_log": trade_log_df,
            "trade_events": trade_events_df,
        }

    def run_bh_baseline(self):
        close_series = None
        for sym in self.symbols:
            cs = _close_col_safe(self.etf_data[sym])
            if cs is not None:
                close_series = cs
                break
        if close_series is None:
            return {"final_value": self.fee_params["initial_capital"], "total_return_pct": 0, "sharpe_ratio": 0, "max_drawdown_pct": 0}
        close_arr = close_series.values.astype(float)
        aligned_close = close_series.loc[[pd.Timestamp(d) for d in self.aligned_dates]].values.astype(float)
        n = len(aligned_close)
        if n < 2:
            return {"final_value": self.fee_params["initial_capital"], "total_return_pct": 0, "sharpe_ratio": 0, "max_drawdown_pct": 0}
        cap = self.fee_params["initial_capital"] / self.n_etfs
        total_pv = np.zeros(n)
        for sym in self.symbols:
            cs = _close_col_safe(self.etf_data[sym])
            if cs is None:
                total_pv += cap
                continue
            arr = cs.reindex(pd.DatetimeIndex(self.aligned_dates)).values.astype(float)
            arr = np.nan_to_num(arr, nan=arr[~np.isnan(arr)][0] if len(arr[~np.isnan(arr)]) > 0 else 1.0)
            shares = cap / arr[0]
            total_pv += shares * arr
        final_value = float(total_pv[-1])
        total_ret = (final_value - self.fee_params["initial_capital"]) / self.fee_params["initial_capital"] * 100
        daily_returns = np.diff(total_pv) / total_pv[:-1]
        sharpe = sharpe_ratio(daily_returns)
        mdd = max_drawdown(total_pv)
        return {
            "final_value": round(final_value, 2),
            "total_return_pct": round(total_ret, 2),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown_pct": round(mdd, 2),
            "equity_curve": total_pv,
        }

    def _build_composite_nav(self):
        prices = {}
        for sym in self.symbols:
            cs = _close_col_safe(self.etf_data.get(sym))
            if cs is not None:
                prices[sym] = cs
        if not prices:
            return None
        combined = pd.concat(prices, axis=1)
        combined = combined.dropna()
        normalized = combined / combined.iloc[0]
        composite = normalized.mean(axis=1)
        return composite

    def _run_single_etf_bh(self, symbol):
        cs = _close_col_safe(self.etf_data.get(symbol))
        if cs is None:
            return None
        arr = cs.reindex(pd.DatetimeIndex(self.aligned_dates)).values.astype(float)
        arr = np.nan_to_num(arr, nan=arr[~np.isnan(arr)][0] if len(arr[~np.isnan(arr)]) > 0 else 1.0)
        capital = self.fee_params["initial_capital"]
        shares = capital / arr[0]
        pv = shares * arr
        final = float(pv[-1])
        ret = (final - capital) / capital * 100
        dr = np.diff(pv) / pv[:-1] if len(pv) > 1 else np.array([0])
        sharpe = sharpe_ratio(dr)
        mdd = max_drawdown(pv)
        return {
            "symbol": symbol,
            "final_value": round(final, 2),
            "total_return_pct": round(ret, 2),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown_pct": round(mdd, 2),
            "equity_curve": pv,
        }

    def compute_benchmarks(self):
        capital = self.fee_params["initial_capital"]
        results = {}

        results["equal_weight_bh"] = self.run_bh_baseline()

        results["single_etf_bh"] = {}
        for sym in self.symbols:
            bh = self._run_single_etf_bh(sym)
            if bh is not None:
                results["single_etf_bh"][sym] = bh

        composite = self._build_composite_nav()
        if composite is not None and len(composite) >= 22:
            n_periods = max(1, len(composite) // 22)
            base_amount = capital / n_periods / 22
            start = str(composite.index[0])[:10]
            end = str(composite.index[-1])[:10]
            try:
                from backtest.dca import run_dca_backtest
                results["monthly_dca"] = run_dca_backtest(
                    composite, start, end, "monthly", base_amount,
                    commission_rate=self.fee_params["commission_rate"],
                    min_commission=self.fee_params["min_commission"],
                    stamp_duty=self.fee_params["stamp_duty"],
                )
                if results["monthly_dca"] and "portfolio_series" in results["monthly_dca"] and "invested_series" in results["monthly_dca"]:
                    ps = results["monthly_dca"]["portfolio_series"]
                    inv = results["monthly_dca"]["invested_series"]
                    total_val = ps + (capital - inv.reindex(ps.index).fillna(0))
                    results["monthly_dca"]["total_value_series"] = total_val
                    pv = total_val.values
                    if len(pv) > 1:
                        dr = np.diff(pv) / pv[:-1]
                        results["monthly_dca"]["sharpe_ratio"] = round(float(sharpe_ratio(dr)), 4)
                        results["monthly_dca"]["max_drawdown_pct"] = round(float(max_drawdown(pv)), 2)
                        results["monthly_dca"]["total_return_pct"] = round(float((pv[-1] - capital) / capital * 100), 2)
                        results["monthly_dca"]["final_value"] = round(float(pv[-1]), 2)
            except Exception:
                results["monthly_dca"] = None
            try:
                from backtest.strategies import run_ma_adjust_dca
                results["ma_adjust_dca"] = run_ma_adjust_dca(
                    composite, start, end, base_amount,
                    ma_period=250,
                )
                if results["ma_adjust_dca"] and "portfolio_series" in results["ma_adjust_dca"] and "invested_series" in results["ma_adjust_dca"]:
                    ps = results["ma_adjust_dca"]["portfolio_series"]
                    inv = results["ma_adjust_dca"]["invested_series"]
                    total_val = ps + (capital - inv.reindex(ps.index).fillna(0))
                    results["ma_adjust_dca"]["total_value_series"] = total_val
                    pv = total_val.values
                    if len(pv) > 1:
                        dr = np.diff(pv) / pv[:-1]
                        results["ma_adjust_dca"]["sharpe_ratio"] = round(float(sharpe_ratio(dr)), 4)
                        results["ma_adjust_dca"]["max_drawdown_pct"] = round(float(max_drawdown(pv)), 2)
                        results["ma_adjust_dca"]["total_return_pct"] = round(float((pv[-1] - capital) / capital * 100), 2)
                        results["ma_adjust_dca"]["final_value"] = round(float(pv[-1]), 2)
            except Exception:
                results["ma_adjust_dca"] = None

        return results

    def save(self, path: str):
        import torch
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        ppo_path = path.replace(".pt", "_ppo.pt")
        dqn_path = path.replace(".pt", "_dqn.pt")
        self.ppo.save(ppo_path)
        self.dqn.save(dqn_path)
        meta = {
            "ppo_state_dim": self.ppo.network.actor_base[0].in_features,
            "dqn_state_dim": self.dqn.q_net.net[0].in_features,
            "ppo_hidden": self.ppo.network.actor_base[0].out_features,
            "dqn_hidden": self.dqn.q_net.net[0].out_features,
            "n_episodes": self.n_episodes,
            "symbols": self.symbols,
        }
        torch.save({"metadata": meta}, path)

    @classmethod
    def load(cls, path: str):
        import torch
        data = torch.load(path, map_location="cpu")
        meta = data.get("metadata", {})
        inst = cls(
            etf_data={},
            aligned_dates=[],
            ppo_state_dim=meta.get("ppo_state_dim", 12),
            dqn_state_dim=meta.get("dqn_state_dim", 6),
            ppo_hidden=meta.get("ppo_hidden", 64),
            dqn_hidden=meta.get("dqn_hidden", 128),
            n_episodes=meta.get("n_episodes", 64),
        )
        ppo_path = path.replace(".pt", "_ppo.pt")
        dqn_path = path.replace(".pt", "_dqn.pt")
        inst.ppo = PPOAgent.load(ppo_path)
        inst.dqn = DQNAgent.load(dqn_path)
        return inst
