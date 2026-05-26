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
        reward_window: int = 10,
    ):
        self.state_vectors = state_vectors
        self.close = close_prices
        self.dates = dates
        self.initial_capital = initial_capital
        self.cr = commission_rate
        self.sd = stamp_duty
        self.min_c = min_commission
        self.reward_window = reward_window
        self.n_steps = len(close_prices)

        self.reset()

    def reset(self):
        self.t = 0
        self.cash = self.initial_capital
        self.shares = 0.0
        self.portfolio_values = []
        self.actions_taken = []
        self.done = False
        return self._get_state()

    def _get_state(self):
        return self.state_vectors[self.t]

    def _portfolio_value(self):
        return self.cash + self.shares * self.close[self.t]

    def step(self, action: int):
        price = self.close[self.t]

        if action == 1:
            if self.cash > self.min_c:
                effective_cash = self.cash - self.min_c
                shares_bought = effective_cash / price
                cost = shares_bought * price
                commission = max(cost * self.cr, self.min_c)
                total = cost + commission
                if total <= self.cash and cost > 0:
                    self.cash -= total
                    self.shares += shares_bought
        elif action == -1:
            if self.shares > 0:
                proceeds = self.shares * price
                commission = max(proceeds * self.cr, self.min_c)
                tax = proceeds * self.sd
                self.cash += proceeds - commission - tax
                self.shares = 0.0

        self.t += 1
        pv = self._portfolio_value()
        self.portfolio_values.append(pv)
        self.actions_taken.append(action)

        if self.t >= self.n_steps - 1:
            self.done = True
            reward = (pv - self.initial_capital) / self.initial_capital * 100
            return self._get_state(), reward, self.done

        reward = self._calc_reward()
        return self._get_state(), reward, self.done

    def _calc_reward(self):
        n = min(self.reward_window, self.t - 1)
        if n < 1:
            return 0.0
        pv_now = self.portfolio_values[-1]
        pv_start = self.portfolio_values[-n] if n <= len(self.portfolio_values) else self.initial_capital
        return float((pv_now - pv_start) / max(pv_start, 1e-9))
