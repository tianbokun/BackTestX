"""XIRR (年化内部收益率) 计算

使用牛顿法求解: NPV = sum(cf_i / (1+r)^((d_i - d_0)/365)) = 0
"""

import numpy as np


def xirr(transactions):
    """计算 XIRR 年化收益率

    Parameters
    ----------
    transactions : list of (date, amount)
        现金流, 买入为负, 卖出为 (最终市值) 正

    Returns
    -------
    float
        年化收益率 (如 0.10 表示 10%)
    """
    if len(transactions) < 2:
        return 0.0

    amounts = np.array([t[1] for t in transactions])
    days = np.array([(t[0] - transactions[0][0]).days for t in transactions], dtype=float)

    rate = 0.1
    for _ in range(1000):
        exp = days / 365.0
        try:
            npv = np.sum(amounts / (1 + rate) ** exp)
        except (FloatingPointError, OverflowError):
            break
        if abs(npv) < 1e-7:
            break
        deriv = np.sum(-amounts * exp / (1 + rate) ** (exp + 1))
        if abs(deriv) < 1e-12:
            break
        rate -= npv / deriv

    return rate
