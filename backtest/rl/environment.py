import numpy as np


class StockTradingEnv:
    def __init__(
        self,
        state_vectors: np.ndarray,
        close_prices: np.ndarray,
        dates: list,
        initial_capital: float = 1.0,
        transaction_cost: float = 0.0001,
        reward_window: int = 10,
    ):
        self.state_vectors = state_vectors
        self.close = close_prices
        self.dates = dates
        self.initial_capital = initial_capital
        self.tc = transaction_cost
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
        pv_before = self._portfolio_value()

        if action == 1:
            affordable = self.cash * (1 - self.tc) / price
            cost = affordable * price
            fee = cost * self.tc
            actual_spend = cost + fee
            if actual_spend <= self.cash:
                self.cash -= actual_spend
                self.shares += affordable
        elif action == -1:
            if self.shares > 0:
                proceeds = self.shares * price
                fee = proceeds * self.tc
                self.cash += proceeds - fee
                self.shares = 0.0

        self.t += 1
        pv_after = self._portfolio_value()
        self.portfolio_values.append(pv_after)
        self.actions_taken.append(action)

        if self.t >= self.n_steps - 1:
            self.done = True
            pv_final = self._portfolio_value()
            reward = (pv_final - self.initial_capital) / self.initial_capital * 100
            return self._get_state(), reward, self.done

        reward = self._calc_reward(pv_before, pv_after, action)
        return self._get_state(), reward, self.done

    def _calc_reward(self, pv_before, pv_after, action):
        n = min(self.reward_window, self.t - 1)
        if n < 1:
            return 0.0
        pv_window_start = self.portfolio_values[-n] if n <= len(self.portfolio_values) else self.initial_capital
        ret = (pv_after - pv_window_start) / max(pv_window_start, 1e-9)
        trade_cost = 0.0
        if action != 0:
            trade_value = abs(pv_after - pv_before)
            trade_cost = trade_value * self.tc
        return float(ret - trade_cost)
