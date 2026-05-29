import numpy as np


class StockTradingEnv:
    def __init__(
        self,
        state_vectors: np.ndarray,
        close_prices: np.ndarray,
        dates: list,
        initial_capital: float = 100000.0,
        commission_rate: float = 0.000235,
        min_commission: float = 5.0,
        stamp_duty: float = 0.001,
        reward_window: int = 63,
        vol_penalty_coef: float = 0.1,
        dd_penalty_coef: float = 1.0,
    ):
        self.state_vectors = state_vectors
        self.close = close_prices
        self.dates = dates
        self.initial_capital = initial_capital
        self.cr = commission_rate
        self.sd = stamp_duty
        self.min_c = min_commission
        self.reward_window = reward_window
        self.vol_penalty_coef = vol_penalty_coef
        self.dd_penalty_coef = dd_penalty_coef
        self.n_steps = len(close_prices)

        self._last_trade_cost = 0.0
        self.reset()

    def reset(self):
        self.t = 0
        self.cash = self.initial_capital
        self.shares = 0.0
        self.portfolio_values = []
        self.actions_taken = []
        self._last_trade_cost = 0.0
        self.done = False
        return self._get_state()

    def _get_state(self):
        return self.state_vectors[self.t]

    def _portfolio_value(self):
        return self.cash + self.shares * self.close[self.t]

    def step(self, action: int):
        price = self.close[self.t]
        self._last_trade_cost = 0.0

        if action == 1:
            if self.cash > self.min_c:
                max_cost = self.cash / (1 + self.cr)
                commission = max_cost * self.cr
                if commission < self.min_c:
                    cost = self.cash - self.min_c
                    commission = self.min_c
                else:
                    cost = max_cost
                shares_bought = cost / price
                total = cost + commission
                if total <= self.cash + 1e-9 and cost > 0:
                    total = min(total, self.cash)
                    self.cash -= total
                    self.shares += shares_bought
                    self._last_trade_cost = commission
        elif action == -1 or action == 2:
            if self.shares > 0:
                proceeds = self.shares * price
                commission = max(proceeds * self.cr, self.min_c)
                tax = proceeds * self.sd
                self.cash += proceeds - commission - tax
                self.shares = 0.0
                self._last_trade_cost = commission + tax

        pv = self._portfolio_value()
        self.portfolio_values.append(pv)
        self.actions_taken.append(action)

        self.t += 1

        if self.t >= self.n_steps - 1:
            self.done = True

        return self._get_state(), self._calc_reward(), self.done

    def _calc_reward(self):
        n = min(self.reward_window, self.t)
        if n < 2:
            return 0.0
        pv_window = self.portfolio_values[-n:]
        pv_start = pv_window[0]
        base_ret = float((pv_window[-1] - pv_start) / max(pv_start, 1e-9))

        peak = max(pv_window)
        drawdown = (peak - pv_window[-1]) / max(peak, 1e-9)
        dd_penalty = self.dd_penalty_coef * drawdown

        daily_rets = np.diff(pv_window) / np.maximum(np.array(pv_window[:-1]), 1e-9)
        vol = np.std(daily_rets)
        vol_penalty = self.vol_penalty_coef * vol * np.sqrt(252)

        cost_penalty = self._last_trade_cost / self.initial_capital

        return base_ret - dd_penalty - vol_penalty - cost_penalty
