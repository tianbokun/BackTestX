import numpy as np
import pandas as pd


class MultiAssetTradingEnv:
    def __init__(
        self,
        etf_data: dict[str, pd.DataFrame],
        aligned_dates: list,
        feature_engineer_fn=None,
        initial_capital: float = 100000.0,
        commission_rate: float = 0.000235,
        min_commission: float = 5.0,
        stamp_duty: float = 0.001,
        reward_window: int = 22,
        trade_fraction: float = 0.2,
    ):
        self.etf_data = etf_data
        self.symbols = sorted(etf_data.keys())
        self.n_etfs = len(self.symbols)
        self.dates = aligned_dates
        self.n_steps = len(aligned_dates)
        self.initial_capital = initial_capital
        self.cr = commission_rate
        self.min_c = min_commission
        self.sd = stamp_duty
        self.reward_window = reward_window
        self.trade_fraction = trade_fraction

        self.close_cache = {}
        for sym in self.symbols:
            df = etf_data[sym]
            for col in ["收盘", "收盘价", "close"]:
                if col in df.columns:
                    self.close_cache[sym] = df[col].values.astype(float)
                    break

        self._last_trade_cost = 0.0
        self.t = 0
        self.cash = initial_capital
        self.positions = {sym: 0.0 for sym in self.symbols}
        self.portfolio_values = []
        self.position_ratios = []
        self.actions_taken = []
        self.done = False

    def reset(self):
        self.t = 0
        self.cash = self.initial_capital
        self.positions = {sym: 0.0 for sym in self.symbols}
        self.portfolio_values = []
        self.position_ratios = []
        self.actions_taken = []
        self._last_trade_cost = 0.0
        self.done = False
        return self._get_state()

    def _get_close(self, sym: str, t: int = None):
        t = t if t is not None else self.t
        arr = self.close_cache.get(sym)
        if arr is not None and t < len(arr):
            return arr[t]
        return 0.0

    def _portfolio_value(self):
        pv = self.cash
        for sym in self.symbols:
            pv += self.positions[sym] * self._get_close(sym)
        return pv

    def _get_state(self):
        pv = self._portfolio_value()
        pv_hist = self.portfolio_values[-self.reward_window:] if self.portfolio_values else [self.initial_capital]

        market_state = self._build_market_state(pv, pv_hist)
        etf_states = self._build_etf_states(pv)
        return market_state, etf_states

    def _build_market_state(self, pv, pv_hist):
        dim = 12
        state = np.zeros(dim, dtype=np.float32)
        state[0] = self.cash / max(pv, 1)
        total_pos = sum(self.positions[sym] * self._get_close(sym) for sym in self.symbols)
        state[1] = total_pos / max(pv, 1)

        if len(pv_hist) >= 2:
            rets = np.diff(pv_hist) / np.maximum(np.array(pv_hist[:-1]), 1e-9)
            state[2] = float(np.mean(rets))
            state[3] = float(np.std(rets)) * np.sqrt(252)
            state[4] = float(pv_hist[-1] / max(pv_hist[0], 1) - 1)
        else:
            state[2:5] = 0

        etf_rets = []
        etf_vols = []
        up_count = 0
        for sym in self.symbols:
            c = self.close_cache.get(sym)
            if c is not None and self.t > 0:
                r = (c[self.t] - c[self.t - 1]) / max(c[self.t - 1], 1e-9)
                etf_rets.append(r)
                if r > 0:
                    up_count += 1
                window = max(0, self.t - 20)
                prices = c[window:self.t + 1]
                if len(prices) > 1:
                    dret = np.diff(prices) / np.maximum(prices[:-1], 1e-9)
                    etf_vols.append(float(np.std(dret)))
        state[5] = float(np.mean(etf_rets)) if etf_rets else 0
        state[6] = float(np.mean(etf_vols)) if etf_vols else 0
        state[7] = up_count / max(self.n_etfs, 1)

        if etf_rets:
            state[8] = float(max(etf_rets) - min(etf_rets))

        if len(pv_hist) > 1:
            state[9] = max(pv_hist) / max(pv_hist[-1], 1) - 1

        state[10] = self.t / max(self.n_steps, 1)

        return state

    def _build_etf_states(self, pv):
        etf_states = {}
        for sym in self.symbols:
            if sym in self.etf_data:
                df = self.etf_data[sym]
            else:
                etf_states[sym] = np.zeros(6, dtype=np.float32)
                continue
            close_arr = self.close_cache.get(sym, np.zeros(self.n_steps))
            t = self.t
            c = close_arr[t] if t < len(close_arr) else 1.0
            p = close_arr[t - 1] if t > 0 else c
            ret_1d = (c - p) / max(p, 1e-9)
            window = max(0, t - 20)
            prices = close_arr[window:t + 1]
            vol = float(np.std(np.diff(prices) / np.maximum(prices[:-1], 1e-9))) if len(prices) > 2 else 0
            pos = self.positions.get(sym, 0)
            pos_value = pos * c
            pos_ratio = pos_value / max(pv, 1)
            max_pos = pv * self.trade_fraction / max(c, 1)
            pos_pct = pos / max(max_pos, 1)
            n_pos = sum(1 for s in self.symbols if self.positions.get(s, 0) > 0)
            state = np.array([
                ret_1d,
                vol * np.sqrt(252),
                pos_ratio,
                pos_pct,
                n_pos / max(self.n_etfs, 1),
                self.cash / max(pv, 1),
            ], dtype=np.float32)
            etf_states[sym] = state
        return etf_states

    def step(self, position_ratio: float, per_etf_actions: dict[str, int]):
        self._last_trade_cost = 0.0
        self.position_ratios.append(position_ratio)

        pv_before = self._portfolio_value()
        target_exposure = position_ratio * pv_before
        current_exposure = sum(self.positions[sym] * self._get_close(sym) for sym in self.symbols)
        target_cash = pv_before - target_exposure

        if target_cash > self.cash:
            sell_amount = target_cash - self.cash
            self._rebalance_sell(sell_amount, per_etf_actions)
        elif target_cash < self.cash:
            buy_amount = self.cash - target_cash
            self._rebalance_buy(buy_amount, per_etf_actions)

        pv = self._portfolio_value()
        self.portfolio_values.append(pv)
        actions_flat = [per_etf_actions.get(sym, 0) for sym in self.symbols]
        self.actions_taken.append(actions_flat)

        self.t += 1
        if self.t >= self.n_steps - 1:
            self.done = True

        market_state, etf_states = self._get_state()
        reward = self._calc_reward()
        return market_state, etf_states, reward, self.done, self._get_info()

    def _rebalance_sell(self, sell_amount, actions):
        for sym in self.symbols:
            if sell_amount <= 0:
                break
            if actions.get(sym) == 0 and self.positions.get(sym, 0) > 0:
                price = self._get_close(sym)
                shares_to_sell = min(self.positions[sym], sell_amount * 0.5 / max(price, 1))
                if shares_to_sell <= 0:
                    continue
                proceeds = shares_to_sell * price
                commission = max(proceeds * self.cr, self.min_c)
                tax = proceeds * self.sd
                net_proceeds = proceeds - commission - tax
                self.cash += net_proceeds
                self.positions[sym] -= shares_to_sell
                self._last_trade_cost += commission + tax
                sell_amount -= net_proceeds

        for sym in self.symbols:
            if sell_amount <= 0:
                break
            if actions.get(sym) == 2 and self.positions.get(sym, 0) > 0:
                price = self._get_close(sym)
                shares_to_sell = min(self.positions[sym], sell_amount / max(price, 1))
                if shares_to_sell <= 0:
                    continue
                proceeds = shares_to_sell * price
                commission = max(proceeds * self.cr, self.min_c)
                tax = proceeds * self.sd
                net_proceeds = proceeds - commission - tax
                self.cash += net_proceeds
                self.positions[sym] -= shares_to_sell
                self._last_trade_cost += commission + tax
                sell_amount -= net_proceeds

    def _rebalance_buy(self, buy_amount, actions):
        buy_actions = [sym for sym in self.symbols if actions.get(sym) == 1]
        if not buy_actions:
            buy_actions = self.symbols
        per_budget = buy_amount / max(len(buy_actions), 1)
        for sym in buy_actions:
            price = self._get_close(sym)
            if price <= 0:
                continue
            cost = min(per_budget, self.cash)
            max_cost = cost / (1 + self.cr)
            commission = max(max_cost * self.cr, self.min_c)
            total = max_cost + commission
            if total <= self.cash + 1e-9 and max_cost > 0:
                    total = min(total, self.cash)
                    shares_bought = max_cost / price
                    self.cash -= total
                    self.positions[sym] += shares_bought
                    self._last_trade_cost += commission

    def _calc_reward(self):
        n = min(self.reward_window, self.t)
        if n < 2:
            return 0.0
        pv_window = self.portfolio_values[-n:]
        pv_start = pv_window[0]
        base_ret = float((pv_window[-1] - pv_start) / max(pv_start, 1e-9))

        peak = max(pv_window)
        drawdown = (peak - pv_window[-1]) / max(peak, 1e-9)
        dd_penalty = 0.5 * drawdown

        daily_rets = np.diff(pv_window) / np.maximum(np.array(pv_window[:-1]), 1e-9)
        vol = np.std(daily_rets) if len(daily_rets) > 0 else 0
        vol_penalty = 0.3 * vol * np.sqrt(252)

        cost_penalty = self._last_trade_cost / self.initial_capital
        return base_ret - dd_penalty - vol_penalty - cost_penalty

    def _get_info(self):
        pv = self._portfolio_value()
        ret = (pv - self.initial_capital) / self.initial_capital * 100
        positions = {
            sym: {
                "shares": round(float(self.positions.get(sym, 0)), 4),
                "value": round(float(self.positions.get(sym, 0) * self._get_close(sym)), 2),
                "ratio": round(float(self.positions.get(sym, 0) * self._get_close(sym) / max(pv, 1) * 100), 2),
            }
            for sym in self.symbols
        }
        return {
            "portfolio_value": round(float(pv), 2),
            "return_pct": round(float(ret), 2),
            "cash": round(float(self.cash), 2),
            "positions": positions,
        }
