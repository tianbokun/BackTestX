import numpy as np

def total_return(final_value: float, initial_capital: float = 1.0) -> float:
    return (final_value - initial_capital) / initial_capital * 100

def sharpe_ratio(daily_returns: np.ndarray, risk_free: float = 0.0) -> float:
    if len(daily_returns) < 2 or np.std(daily_returns) == 0:
        return 0.0
    excess = daily_returns - risk_free / 252
    return float(np.sqrt(252) * np.mean(excess) / np.std(daily_returns))

def max_drawdown(equity_curve: np.ndarray) -> float:
    if len(equity_curve) < 2:
        return 0.0
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / peak
    return float(abs(np.min(dd)) * 100)
