"""
定投 (DCA) 回测引擎

支持:
  - 定期定额投资回测
  - 多种频率: 每周/每两周/每月/每季/每年
  - 收益率计算: 总收益率, 年化收益率 (XIRR 近似)
  - 可视化数据: 每笔买入记录, 市值曲线
"""

from typing import Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from core.xirr import xirr
from core.config import DAILY_MULTIPLIER, freq_map, DEFAULT_COMMISSION_RATE, DEFAULT_MIN_COMMISSION, DEFAULT_STAMP_DUTY


def _get_invest_dates(
    index: pd.DatetimeIndex,
    start_date: str,
    end_date: str,
    frequency: str,
    day: int = 1,
) -> pd.DatetimeIndex:
    """
    根据定投频率生成投资日期

    Parameters
    ----------
    index : pd.DatetimeIndex
        交易日历 (实际有价格数据的日期)
    start_date, end_date : str
        定投区间 YYYY-MM-DD
    frequency : str
        "daily" / "weekly" / "biweekly" / "monthly" / "quarterly" / "yearly"
    day : int
        每月/每季的哪一天执行定投 (默认 1 号); 仅 monthly/quarterly/yearly 生效

    Returns
    -------
    pd.DatetimeIndex
        实际执行定投的日期 (均在 index 内, 若当天非交易日则顺延)
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    # 生成计划定投日期
    if frequency == "daily":
        # 每个交易日: 直接用价格序列中所有日期 (已过滤到只有交易日)
        plan_dates = index[(index >= start) & (index <= end)]
        return pd.DatetimeIndex(sorted(set(plan_dates)))
    elif frequency == "weekly":
        plan_dates = pd.date_range(start=start, end=end, freq="W-" + start.strftime("%a"))
    elif frequency == "biweekly":
        plan_dates = pd.date_range(start=start, end=end, freq="W-" + start.strftime("%a"))
        plan_dates = plan_dates[::2]
    elif frequency == "monthly":
        plan_dates = pd.date_range(
            start=start, end=end, freq="MS"
        ) + pd.Timedelta(days=day - 1)
        plan_dates = plan_dates[(plan_dates >= start) & (plan_dates <= end)]
    elif frequency == "quarterly":
        plan_dates = pd.date_range(
            start=start, end=end, freq="QS"
        ) + pd.Timedelta(days=day - 1)
        plan_dates = plan_dates[(plan_dates >= start) & (plan_dates <= end)]
    elif frequency == "yearly":
        plan_dates = pd.date_range(
            start=start, end=end, freq="YS"
        ) + pd.Timedelta(days=day - 1)
        plan_dates = plan_dates[(plan_dates >= start) & (plan_dates <= end)]
    else:
        raise ValueError(f"不支持的定投频率: {frequency}")

    if len(plan_dates) == 0:
        return pd.DatetimeIndex([])

    invested = []
    for plan_date in plan_dates:
        # 寻找计划日期当天或之后的第一个交易日
        mask = index >= plan_date
        if mask.any():
            actual_date = index[mask][0]
            invested.append(actual_date)
        else:
            break

    return pd.DatetimeIndex(sorted(set(invested)))


def _xirr(transactions):
    """
    计算 XIRR (年化内部收益率)
    使用牛顿法求解: NPV = sum(cf_i / (1+r)^((d_i - d_0)/365)) = 0

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

    # 初始猜测
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


# ── 日均乘数: amount 理解为"日均投入", 各频率按交易日天数放大 ──
DAILY_MULTIPLIER = {
    "daily": 1,
    "weekly": 5,
    "biweekly": 10,
    "monthly": 22,
    "quarterly": 66,
    "yearly": 252,
}


def run_dca_backtest(
    price_series: pd.Series,
    start_date: str,
    end_date: str,
    frequency: str = "monthly",
    amount: float = 1000,
    day: int = 1,
    max_total: float = 0,
    commission_rate: float = DEFAULT_COMMISSION_RATE,
    min_commission: float = DEFAULT_MIN_COMMISSION,
    stamp_duty: float = DEFAULT_STAMP_DUTY,
) -> dict:
    """
    执行定投回测

    Parameters
    ----------
    price_series : pd.Series
        date-indexed price series (收盘价/单位净值)
    start_date : str
        定投开始日期 YYYY-MM-DD 或 YYYYMMDD
    end_date : str
        定投结束日期 (持有到这一天)
    frequency : str
        "daily" / "weekly" / "biweekly" / "monthly" / "quarterly" / "yearly"
    amount : float
        平均每日投入金额 (实际每期投入 = amount × 对应频率的交易日乘数)
    day : int
        每月/每季的哪一天 (1-28); 仅 monthly/quarterly/yearly 生效
    max_total : float
        总投资上限, 达到后停止定投; 0 表示不设上限

    Returns
    -------
    dict
        {
            "total_invested": 总投入,
            "final_value": 终值,
            "total_return_pct": 总收益率%,
            "annualized_return_pct": 年化收益率%,
            "records": pd.DataFrame,   # 每笔定投明细
            "nav_series": pd.Series,   # 净值曲线
            "portfolio_series": pd.Series,  # 持仓市值曲线
            "invested_series": pd.Series,    # 累计投入曲线
        }
    """
    start_date = start_date.replace("-", "")
    end_date = end_date.replace("-", "")
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    period_amount = amount * DAILY_MULTIPLIER.get(frequency, 22)
    cap_hit = max_total > 0

    prices = price_series.sort_index().dropna()
    prices = prices[(prices.index >= start_ts) & (prices.index <= end_ts)]

    if len(prices) == 0:
        return {
            "max_total": max_total, "strategy": "", "total_invested": 0,
            "final_value": 0, "total_return_pct": 0, "annualized_return_pct": 0,
            "records": pd.DataFrame(), "nav_series": prices,
            "portfolio_series": pd.Series(dtype=float),
            "invested_series": pd.Series(dtype=float),
            "final_price": 0, "total_shares": 0, "num_investments": 0,
            "cap_hit": False,
        }

    invest_dates = _get_invest_dates(
        prices.index, start_date, end_date, frequency, day
    )

    if len(invest_dates) == 0:
        return {
            "max_total": max_total, "strategy": "", "total_invested": 0,
            "final_value": 0, "total_return_pct": 0, "annualized_return_pct": 0,
            "records": pd.DataFrame(), "nav_series": prices,
            "portfolio_series": pd.Series(dtype=float),
            "invested_series": pd.Series(dtype=float),
            "final_price": 0, "total_shares": 0, "num_investments": 0,
            "cap_hit": False,
        }

    invest_entries = []
    total_invested = 0.0
    total_shares = 0.0
    total_commissions = 0.0

    for inv_date in invest_dates:
        price = prices.loc[inv_date]
        actual = period_amount
        if cap_hit and total_invested + actual > max_total:
            actual = max_total - total_invested
            if actual <= 0:
                break
        buy_commission = max(actual * commission_rate, min_commission)
        buy_commission = min(buy_commission, actual)
        total_commissions += buy_commission
        shares = (actual - buy_commission) / price
        total_shares += shares
        total_invested += actual
        invest_entries.append((inv_date, actual, price, shares))

    records = []
    cash_flows = []
    cum_shares = 0.0
    cum_invested = 0.0
    for inv_date, actual, price, shares in invest_entries:
        cum_shares += shares
        cum_invested += actual
        records.append({
            "日期": inv_date,
            "价格": round(price, 4),
            "买入份额": round(shares, 4),
            "累计份额": round(cum_shares, 4),
            "投入金额": round(actual, 2),
            "累计投入": round(cum_invested, 2),
            "佣金": round(actual - shares * price, 4),
        })
        cash_flows.append((inv_date.to_pydatetime(), -actual))

    final_price = prices.iloc[-1]
    final_value_before = total_shares * final_price
    sell_commission = max(final_value_before * commission_rate, min_commission)
    sell_stamp = final_value_before * stamp_duty
    final_value = final_value_before - sell_commission - sell_stamp
    total_return = (final_value - total_invested) / total_invested * 100 if total_invested > 0 else 0
    cash_flows.append((prices.index[-1].to_pydatetime(), final_value))
    annualized = xirr(cash_flows) * 100
    records_df = pd.DataFrame(records)

    portfolio_values = []
    invested_values = []
    cum_shares = 0.0
    cum_invested = 0.0
    inv_idx = 0
    inv_count = len(invest_entries)

    for date_idx, price in prices.items():
        if inv_idx < inv_count and date_idx >= invest_entries[inv_idx][0]:
            while inv_idx < inv_count and date_idx >= invest_entries[inv_idx][0]:
                cum_shares += invest_entries[inv_idx][3]
                cum_invested += invest_entries[inv_idx][1]
                inv_idx += 1
        portfolio_values.append(cum_shares * price)
        invested_values.append(cum_invested)

    portfolio_series = pd.Series(portfolio_values, index=prices.index)
    invested_series = pd.Series(invested_values, index=prices.index)

    freq_label = freq_map.get(frequency, frequency)
    strategy_name = f"{freq_label}定额({period_amount:.0f}元/期)"
    if cap_hit and max_total > 0:
        strategy_name += f"·上限{max_total:.0f}元"

    return {
        "max_total": max_total,
        "cap_hit": cap_hit and total_invested >= max_total,
        "strategy": strategy_name,
        "total_invested": round(total_invested, 2),
        "final_value": round(final_value, 2),
        "total_return_pct": round(total_return, 2),
        "annualized_return_pct": round(annualized, 2),
        "records": records_df,
        "nav_series": prices,
        "portfolio_series": portfolio_series,
        "invested_series": invested_series,
        "final_price": round(final_price, 4),
        "total_shares": round(total_shares, 4),
        "num_investments": len(invest_entries),
        "total_commissions": round(total_commissions, 2),
        "sell_commission": round(sell_commission, 2),
        "stamp_duty_paid": round(sell_stamp, 2),
    }


# ══════════════════════════════════════════
#  一次性投入 (Lump Sum) 对比
# ══════════════════════════════════════════

freq_map = {
    "daily": "每日",
    "weekly": "每周",
    "biweekly": "每两周",
    "monthly": "每月",
    "quarterly": "每季度",
    "yearly": "每年",
}


def run_lump_sum_backtest(
    price_series: pd.Series,
    start_date: str,
    end_date: str,
    total_amount: float,
    commission_rate: float = DEFAULT_COMMISSION_RATE,
    min_commission: float = DEFAULT_MIN_COMMISSION,
    stamp_duty: float = DEFAULT_STAMP_DUTY,
) -> dict:
    """
    一次性投入回测 (用于对比定投)

    在回测区间第一个交易日一次性买入, 持有到 end_date

    Returns
    -------
    dict with same structure as run_dca_backtest
    """
    start_date = start_date.replace("-", "")
    end_date = end_date.replace("-", "")
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    prices = price_series.sort_index().dropna()
    prices = prices[(prices.index >= start_ts) & (prices.index <= end_ts)]

    if len(prices) == 0:
        return {
            "strategy": "一次性投入",
            "total_invested": 0,
            "final_value": 0,
            "total_return_pct": 0,
            "annualized_return_pct": 0,
            "records": pd.DataFrame(),
            "nav_series": pd.Series(dtype=float),
            "portfolio_series": pd.Series(dtype=float),
            "invested_series": pd.Series(dtype=float),
            "final_price": 0,
            "total_shares": 0,
            "num_investments": 0,
        }

    buy_price = prices.iloc[0]
    buy_date = prices.index[0]
    buy_commission = max(total_amount * commission_rate, min_commission)
    buy_commission = min(buy_commission, total_amount)
    shares = (total_amount - buy_commission) / buy_price
    final_price = prices.iloc[-1]
    final_value_before = shares * final_price
    sell_commission = max(final_value_before * commission_rate, min_commission)
    sell_stamp = final_value_before * stamp_duty
    final_value = final_value_before - sell_commission - sell_stamp
    total_return = (final_value - total_amount) / total_amount * 100

    # 年化: (final/total)^(1/years) - 1
    years = (prices.index[-1] - prices.index[0]).days / 365.0
    if years > 0:
        annualized = ((final_value / total_amount) ** (1 / years) - 1) * 100
    else:
        annualized = 0.0

    # 记录
    records = pd.DataFrame([{
        "日期": buy_date,
        "价格": round(buy_price, 4),
        "买入份额": round(shares, 4),
        "累计份额": round(shares, 4),
        "投入金额": total_amount,
        "累计投入": round(total_amount, 2),
        "佣金": round(buy_commission, 2),
    }])

    # 市值曲线
    portfolio_series = shares * prices
    invested_series = pd.Series(total_amount, index=prices.index)

    return {
        "strategy": "一次性投入",
        "total_invested": round(total_amount, 2),
        "final_value": round(final_value, 2),
        "total_return_pct": round(total_return, 2),
        "annualized_return_pct": round(annualized, 2),
        "records": records,
        "nav_series": prices,
        "portfolio_series": portfolio_series,
        "invested_series": invested_series,
        "final_price": round(final_price, 4),
        "total_shares": round(shares, 4),
        "num_investments": 1,
        "total_commissions": round(buy_commission + sell_commission + sell_stamp, 2),
        "sell_commission": round(sell_commission, 2),
        "stamp_duty_paid": round(sell_stamp, 2),
    }
