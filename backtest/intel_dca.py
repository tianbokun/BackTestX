"""智能定投策略实时测算引擎.

7 种策略的当前时点计算, 输入历史价格 + 策略参数, 返回建议金额 + 解释.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd


@dataclass
class StrategyResult:
    name: str
    signal: str
    amount: float
    amount_pct: float
    key_metrics: dict[str, Any] = field(default_factory=dict)
    explanation: str = ""
    detail: str = ""


def _safe_series(price_series: pd.Series, min_len: int = 2) -> bool:
    return price_series is not None and len(price_series.dropna()) >= min_len


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


# ──────────────────────────────────────────────
# 2. 均线偏离法 (慧定投)
# ──────────────────────────────────────────────

def calc_ma_deviation(
    price_series: pd.Series,
    current_price: float,
    base_amount: float,
    min_amount: float,
    max_amount: float,
    ma_period: int = 250,
    adjustment_factor: float = 2.0,
) -> StrategyResult:
    if not _safe_series(price_series, ma_period + 1):
        return StrategyResult(
            name="均线偏离法", signal="数据不足", amount=base_amount,
            amount_pct=100, explanation="历史数据不足以计算均线",
        )
    ma = price_series.rolling(ma_period).mean().iloc[-1]
    deviation_pct = (current_price - ma) / ma * 100
    ratio = 1 - deviation_pct / 100 * adjustment_factor
    ratio = _clamp(ratio, min_amount / base_amount, max_amount / base_amount)
    amount = round(base_amount * ratio, 2)

    if deviation_pct > 0:
        direction = "高于"
        signal = "减少"
    else:
        direction = "低于"
        signal = "增加"
    explanation = (
        f"当前价格 {current_price:.2f} {direction} {ma_period}日均线 {ma:.2f} "
        f"({deviation_pct:+.2f}%), 建议{signal}定投至 {amount:.0f} 元"
    )
    detail = (
        f"MA{ma_period} = {ma:.4f}\n"
        f"偏离度 = ({current_price:.4f} - {ma:.4f}) / {ma:.4f} = {deviation_pct:+.2f}%\n"
        f"调整系数 = 1 - ({deviation_pct:+.2f}% / 100 × {adjustment_factor}) = {ratio:.4f}\n"
        f"建议金额 = {base_amount:.0f} × {ratio:.4f} = {amount:.0f} 元\n"
        f"(约束范围: {min_amount:.0f} ~ {max_amount:.0f})"
    )
    return StrategyResult(
        name="均线偏离法",
        signal="买入" if amount > 0 else "暂停",
        amount=amount,
        amount_pct=round(ratio * 100, 1),
        key_metrics={
            "均线(MA)": f"{ma:.2f}",
            "偏离度": f"{deviation_pct:+.2f}%",
            "调整系数": f"{ratio:.2f}",
        },
        explanation=explanation,
        detail=detail,
    )


# ──────────────────────────────────────────────
# 3. 估值定投法 (PE/PB百分位) — 仅指数/ETF
# ──────────────────────────────────────────────

def _fetch_current_pe(symbol: str, price_series: pd.Series) -> Optional[float]:
    """尝试获取当前 PE (市盈率)."""
    try:
        import akshare as ak
        if symbol.isdigit() and len(symbol) == 6:
            df = ak.stock_a_lg_indicator(symbol=symbol)
            if df is not None and not df.empty and "市盈率-动态" in df.columns:
                val = df["市盈率-动态"].dropna().iloc[-1]
                return float(val)
        df_idx = ak.stock_index_pe_pb()
        if df_idx is not None and not df_idx.empty:
            for col in ("pe", "市盈率", "滚动市盈率"):
                if col in df_idx.columns:
                    val = df_idx[col].dropna().iloc[-1]
                    return float(val)
    except Exception:
        pass
    return None


def calc_valuation(
    price_series: pd.Series,
    current_price: float,
    base_amount: float,
    min_amount: float,
    max_amount: float,
    symbol: str = "",
    asset_type: str = "etf",
    low_percentile: float = 30,
    high_percentile: float = 70,
) -> StrategyResult:
    if asset_type not in ("index", "etf", "lof"):
        return StrategyResult(
            name="估值定投法", signal="不支持", amount=base_amount,
            amount_pct=100,
            explanation="估值策略仅适用于指数/ETF, 当前资产类型不支持",
        )
    current_pe = _fetch_current_pe(symbol, price_series)
    if current_pe is None or current_pe <= 0:
        if price_series is not None and len(price_series) > 60:
            lookback = min(252, len(price_series))
            low = price_series.iloc[-lookback:].min()
            high = price_series.iloc[-lookback:].max()
            pseudo_pct = (current_price - low) / (high - low) * 100
            explanation = (
                f"PE数据不可用, 以近{lookback}日价格分位 {pseudo_pct:.0f}% "
                "作为替代参考"
            )
            signal = "买入" if pseudo_pct < low_percentile else ("正常" if pseudo_pct < high_percentile else "减少")
            if pseudo_pct < low_percentile:
                amount = max_amount
                ratio = max_amount / base_amount
            elif pseudo_pct > high_percentile:
                amount = min_amount
                ratio = min_amount / base_amount
            else:
                amount = base_amount
                ratio = 1.0
            return StrategyResult(
                name="估值定投法", signal=signal, amount=round(amount, 2),
                amount_pct=round(ratio * 100, 1),
                key_metrics={"估值参考": f"价格分位 {pseudo_pct:.0f}%"},
                explanation=explanation,
            )
        return StrategyResult(
            name="估值定投法", signal="数据不足", amount=base_amount,
            amount_pct=100, explanation="无法获取PE/PB数据",
        )

    start = price_series.index[0]
    end = price_series.index[-1]

    try:
        import akshare as ak
        pe_df = ak.stock_a_lg_indicator(symbol=symbol)
        if pe_df is None or pe_df.empty:
            raise ValueError("no data")
        pe_vals = pd.to_numeric(pe_df["市盈率-动态"], errors="coerce").dropna()
        pe_pct = (pe_vals < current_pe).sum() / len(pe_vals) * 100
        pe_label = f"{pe_pct:.1f}%"
    except Exception:
        pe_pct = 50.0
        pe_label = "N/A"

    if pe_pct < low_percentile:
        amount = max_amount
        ratio = max_amount / base_amount
        signal = "加仓"
        level = "低估"
    elif pe_pct > high_percentile:
        amount = min_amount
        ratio = min_amount / base_amount
        signal = "减仓"
        level = "高估"
    else:
        amount = base_amount
        ratio = 1.0
        signal = "正常"
        level = "正常"

    explanation = (
        f"当前PE百分位 {pe_label}, 处于{level}区间, "
        f"建议定投 {amount:.0f} 元"
    )
    return StrategyResult(
        name="估值定投法", signal=signal, amount=round(amount, 2),
        amount_pct=round(ratio * 100, 1),
        key_metrics={"PE百分位": pe_label, "估值状态": level},
        explanation=explanation,
    )


# ──────────────────────────────────────────────
# 4. 成本定投法 (移动平均成本)
# ──────────────────────────────────────────────

def calc_cost_average(
    price_series: pd.Series,
    current_price: float,
    base_amount: float,
    min_amount: float,
    max_amount: float,
    avg_cost: Optional[float] = None,
    min_rate: float = 0.5,
    max_rate: float = 2.0,
    simulated_periods: int = 24,
) -> StrategyResult:
    if avg_cost is None or avg_cost <= 0:
        if _safe_series(price_series, simulated_periods + 1):
            invest_dates = price_series.iloc[-simulated_periods:].index
            total_shares = 0.0
            total_cost = 0.0
            for dt in invest_dates:
                p = price_series.loc[dt]
                shares = base_amount / p
                total_shares += shares
                total_cost += base_amount
            avg_cost = total_cost / total_shares if total_shares > 0 else current_price
            cost_note = f"(模拟近{simulated_periods}期定投)"
        else:
            avg_cost = current_price
            cost_note = "(数据不足, 以现价代替)"

    cost_deviation = (current_price - avg_cost) / avg_cost * 100
    raw_ratio = 1 - cost_deviation / 100
    ratio = _clamp(raw_ratio, min_rate, max_rate)
    amount = round(base_amount * ratio, 2)

    if cost_deviation > 0:
        direction = "高于"
        action = "减少"
    else:
        direction = "低于"
        action = "增加"

    explanation = (
        f"当前价格 {current_price:.2f} {direction} 平均成本 {avg_cost:.2f} "
        f"({cost_deviation:+.2f}%), 建议{action}定投至 {amount:.0f} 元{cost_note}"
    )
    return StrategyResult(
        name="成本定投法",
        signal="买入" if amount > 0 else "暂停",
        amount=amount,
        amount_pct=round(ratio * 100, 1),
        key_metrics={
            "平均成本": f"{avg_cost:.2f}",
            "偏离度": f"{cost_deviation:+.2f}%",
            "调整比率": f"{ratio:.2f}",
        },
        explanation=explanation,
    )


# ──────────────────────────────────────────────
# 5. 价值平均法 (市值定投)
# ──────────────────────────────────────────────

def calc_value_averaging(
    price_series: pd.Series,
    current_price: float,
    base_amount: float,
    min_amount: float,
    max_amount: float,
    target_increment: float = 1000,
    existing_shares: float = 0,
    periods_elapsed: int = 1,
    initial_value: float = 0,
) -> StrategyResult:
    if existing_shares <= 0 and initial_value <= 0:
        amount = target_increment
        ratio = target_increment / base_amount if base_amount > 0 else 1
        amount = _clamp(amount, min_amount, max_amount)
        explanation = (
            f"首次定投, 按目标每期增值 {target_increment:.0f} 元, "
            f"建议投入 {amount:.0f} 元"
        )
        return StrategyResult(
            name="价值平均法", signal="首次定投", amount=round(amount, 2),
            amount_pct=round(ratio * 100, 1),
            key_metrics={"每期目标": f"{target_increment:.0f}"},
            explanation=explanation,
        )

    current_value = existing_shares * current_price
    target_value = initial_value + target_increment * periods_elapsed
    gap = target_value - current_value

    if gap > 0:
        amount = _clamp(gap, min_amount, max_amount)
        ratio = amount / base_amount if base_amount > 0 else 1
        signal = "补仓"
        explanation = (
            f"当前市值 {current_value:.0f} < 目标 {target_value:.0f}, "
            f"差额 {gap:.0f}, 建议买入 {amount:.0f} 元"
        )
    else:
        amount = _clamp(abs(gap), min_amount, max_amount)
        ratio = -amount / base_amount if base_amount > 0 else -1
        signal = "减仓/卖出"
        explanation = (
            f"当前市值 {current_value:.0f} ≥ 目标 {target_value:.0f}, "
            f"超出 {-gap:.0f}, 建议卖出 {amount:.0f} 元等值份额"
        )

    return StrategyResult(
        name="价值平均法", signal=signal, amount=round(amount, 2),
        amount_pct=round(ratio * 100, 1),
        key_metrics={
            "当前市值": f"{current_value:.0f}",
            "目标市值": f"{target_value:.0f}",
            "差额": f"{gap:+.0f}",
        },
        explanation=explanation,
    )


# ──────────────────────────────────────────────
# 6. 下跌加仓法 (跌幅触发)
# ──────────────────────────────────────────────

def calc_drop_trigger(
    price_series: pd.Series,
    current_price: float,
    base_amount: float,
    min_amount: float,
    max_amount: float,
    drop_threshold: float = 3.0,
    buy_base: float = 1000,
    cooldown_days: int = 1,
    last_trigger_idx: Optional[int] = None,
) -> StrategyResult:
    if not _safe_series(price_series, 3):
        return StrategyResult(
            name="下跌加仓法", signal="数据不足", amount=0,
            amount_pct=0, explanation="历史数据不足3日",
        )
    prev_close = price_series.iloc[-2]
    drop_pct = (prev_close - current_price) / prev_close * 100

    if drop_pct <= 0:
        return StrategyResult(
            name="下跌加仓法", signal="未触发", amount=0,
            amount_pct=0,
            key_metrics={"当日涨跌": f"{drop_pct:+.2f}%", "阈值": f"{drop_threshold}%"},
            explanation=f"今日上涨 {abs(drop_pct):.2f}%, 未触发跌幅阈值",
        )
    if drop_pct < drop_threshold:
        return StrategyResult(
            name="下跌加仓法", signal="未触发", amount=0,
            amount_pct=0,
            key_metrics={"当日跌幅": f"{drop_pct:.2f}%", "阈值": f"{drop_threshold}%"},
            explanation=f"今日下跌 {drop_pct:.2f}%, 未达阈值 {drop_threshold}%",
        )

    severity_ratio = _clamp(drop_pct / drop_threshold, 1.0, 3.0)
    amount = _clamp(buy_base * severity_ratio, min_amount, max_amount)
    ratio = amount / base_amount if base_amount > 0 else 0
    explanation = (
        f"今日下跌 {drop_pct:.2f}%, 触发阈值 {drop_threshold}%, "
        f"严重度 {severity_ratio:.1f}x, 建议买入 {amount:.0f} 元"
    )
    return StrategyResult(
        name="下跌加仓法", signal="触发买入", amount=round(amount, 2),
        amount_pct=round(ratio * 100, 1),
        key_metrics={
            "当日跌幅": f"{drop_pct:.2f}%",
            "阈值": f"{drop_threshold}%",
            "严重度": f"{severity_ratio:.1f}x",
        },
        explanation=explanation,
    )


# ──────────────────────────────────────────────
# 7. 网格交易法
# ──────────────────────────────────────────────

def calc_grid_trading(
    price_series: pd.Series,
    current_price: float,
    base_amount: float,
    min_amount: float,
    max_amount: float,
    grid_lower: float,
    grid_upper: float,
    grid_count: int = 10,
    amount_per_grid: float = 1000,
    prev_price: Optional[float] = None,
) -> StrategyResult:
    if grid_upper <= grid_lower or grid_count < 2:
        return StrategyResult(
            name="网格交易法", signal="参数错误", amount=0,
            amount_pct=0, explanation="网格上下限或层数设置无效",
        )
    grid_step = (grid_upper - grid_lower) / grid_count
    grid_levels = [grid_lower + i * grid_step for i in range(grid_count + 1)]

    def _find_level(price: float) -> int:
        for i in range(grid_count):
            if grid_levels[i] <= price < grid_levels[i + 1]:
                return i
        return 0 if price < grid_lower else grid_count - 1

    current_level = _find_level(current_price)
    if prev_price is None and _safe_series(price_series, 2):
        prev_price = float(price_series.iloc[-2])
    prev_level = _find_level(prev_price) if prev_price is not None else current_level

    level_diff = current_level - prev_level

    if level_diff < -0.5:
        buy_count = int(abs(level_diff))
        amount = _clamp(amount_per_grid * buy_count, min_amount, max_amount)
        ratio = amount / base_amount if base_amount > 0 else 0
        signal = "买入"
        explanation = (
            f"价格从网格{prev_level+1}跌至{current_level+1}层 "
            f"(下跌{buy_count}格), 建议买入 {amount:.0f} 元"
        )
    elif level_diff > 0.5:
        sell_count = int(level_diff)
        amount = _clamp(amount_per_grid * sell_count, min_amount, max_amount)
        ratio = -amount / base_amount if base_amount > 0 else 0
        signal = "卖出"
        explanation = (
            f"价格从网格{prev_level+1}涨至{current_level+1}层 "
            f"(上涨{sell_count}格), 建议卖出 {amount:.0f} 元等值份额"
        )
    else:
        amount = 0
        ratio = 0
        signal = "观望"
        explanation = (
            f"价格位于网格第{current_level+1}/{grid_count}层, "
            "未触发网格边界, 建议持有不动"
        )

    return StrategyResult(
        name="网格交易法", signal=signal, amount=round(amount, 2),
        amount_pct=round(ratio * 100, 1),
        key_metrics={
            "网格层": f"{current_level+1}/{grid_count}",
            "价格区间": f"[{grid_lower:.2f}, {grid_upper:.2f}]",
            "格差": f"{level_diff:+.0f}格",
        },
        explanation=explanation,
    )


# ──────────────────────────────────────────────
# 8. 趋势定投法 (均线金叉/死叉)
# ──────────────────────────────────────────────

def calc_trend_following(
    price_series: pd.Series,
    current_price: float,
    base_amount: float,
    min_amount: float,
    max_amount: float,
    short_period: int = 20,
    long_period: int = 120,
) -> StrategyResult:
    if not _safe_series(price_series, long_period + 2):
        return StrategyResult(
            name="趋势定投法", signal="数据不足", amount=base_amount,
            amount_pct=100, explanation=f"历史数据不足 {long_period+1} 期",
        )

    short_ma = price_series.rolling(short_period).mean()
    long_ma = price_series.rolling(long_period).mean()

    if len(short_ma.dropna()) < 2 or len(long_ma.dropna()) < 2:
        return StrategyResult(
            name="趋势定投法", signal="数据不足", amount=base_amount,
            amount_pct=100, explanation="均线数据不足",
        )

    cur_short = short_ma.iloc[-1]
    cur_long = long_ma.iloc[-1]
    prev_short = short_ma.iloc[-2]
    prev_long = long_ma.iloc[-2]

    is_golden_cross = prev_short <= prev_long and cur_short > cur_long
    is_death_cross = prev_short >= prev_long and cur_short < cur_long

    if is_golden_cross:
        amount = max_amount
        ratio = max_amount / base_amount
        signal = "金叉加仓"
        status = "金叉"
        explanation = (
            f"短均线({short_period}日)上穿长均线({long_period}日)形成金叉, "
            f"趋势转多, 建议加大投入 {amount:.0f} 元"
        )
    elif is_death_cross:
        amount = min_amount
        ratio = min_amount / base_amount
        signal = "死叉减仓"
        status = "死叉"
        explanation = (
            f"短均线({short_period}日)下穿长均线({long_period}日)形成死叉, "
            f"趋势转空, 建议减少投入至 {amount:.0f} 元"
        )
    elif cur_short > cur_long:
        spread_pct = (cur_short - cur_long) / cur_long * 100
        amount = base_amount
        ratio = 1.0
        signal = "多头持有"
        status = "多头"
        explanation = (
            f"短均线({short_period}={cur_short:.2f}) > "
            f"长均线({long_period}={cur_long:.2f}) 多头排列, "
            f"按计划定投 {amount:.0f} 元"
        )
    else:
        amount = min_amount
        ratio = min_amount / base_amount
        signal = "空头减仓"
        status = "空头"
        explanation = (
            f"短均线({short_period}={cur_short:.2f}) < "
            f"长均线({long_period}={cur_long:.2f}) 空头排列, "
            f"建议减少投入至 {amount:.0f} 元"
        )

    return StrategyResult(
        name="趋势定投法", signal=signal, amount=round(amount, 2),
        amount_pct=round(ratio * 100, 1),
        key_metrics={
            f"MA{short_period}": f"{cur_short:.2f}",
            f"MA{long_period}": f"{cur_long:.2f}",
            "趋势状态": status,
        },
        explanation=explanation,
    )


# ──────────────────────────────────────────────
# 统一入口
# ──────────────────────────────────────────────


def calc_all_strategies(
    price_series: pd.Series,
    base_amount: float,
    min_amount: float,
    max_amount: float,
    params: Optional[dict] = None,
    symbol: str = "",
    asset_type: str = "etf",
    avg_cost: Optional[float] = None,
    existing_shares: float = 0,
    periods_elapsed: int = 1,
    initial_value: float = 0,
    prev_price: Optional[float] = None,
    last_trigger_idx: Optional[int] = None,
) -> dict[str, StrategyResult]:
    """
    统一入口, 运行全部 7 个策略并返回 {策略名: StrategyResult}.
    """
    if params is None:
        params = {}
    if not _safe_series(price_series, 5):
        return {}

    current_price = float(price_series.iloc[-1])

    results = {}
    results["ma_deviation"] = calc_ma_deviation(
        price_series, current_price, base_amount, min_amount, max_amount,
        ma_period=params.get("ma_period", 250),
        adjustment_factor=params.get("ma_adjustment", 2.0),
    )
    results["valuation"] = calc_valuation(
        price_series, current_price, base_amount, min_amount, max_amount,
        symbol=symbol, asset_type=asset_type,
        low_percentile=params.get("low_percentile", 30),
        high_percentile=params.get("high_percentile", 70),
    )
    results["cost_average"] = calc_cost_average(
        price_series, current_price, base_amount, min_amount, max_amount,
        avg_cost=avg_cost,
        min_rate=params.get("cost_min_rate", 0.5),
        max_rate=params.get("cost_max_rate", 2.0),
        simulated_periods=params.get("simulated_periods", 24),
    )
    results["value_averaging"] = calc_value_averaging(
        price_series, current_price, base_amount, min_amount, max_amount,
        target_increment=params.get("target_increment", 1000),
        existing_shares=existing_shares,
        periods_elapsed=periods_elapsed,
        initial_value=initial_value,
    )
    results["drop_trigger"] = calc_drop_trigger(
        price_series, current_price, base_amount, min_amount, max_amount,
        drop_threshold=params.get("drop_threshold", 3.0),
        buy_base=params.get("drop_buy_base", 1000),
        cooldown_days=params.get("cooldown_days", 1),
        last_trigger_idx=last_trigger_idx,
    )
    grid_lower = params.get("grid_lower", current_price * 0.8)
    grid_upper = params.get("grid_upper", current_price * 1.2)
    results["grid_trading"] = calc_grid_trading(
        price_series, current_price, base_amount, min_amount, max_amount,
        grid_lower=grid_lower, grid_upper=grid_upper,
        grid_count=params.get("grid_count", 10),
        amount_per_grid=params.get("amount_per_grid", 1000),
        prev_price=prev_price,
    )
    results["trend_following"] = calc_trend_following(
        price_series, current_price, base_amount, min_amount, max_amount,
        short_period=params.get("short_period", 20),
        long_period=params.get("long_period", 120),
    )
    return results


# ──────────────────────────────────────────────
# 历史回测 — 均线偏离法
# ──────────────────────────────────────────────

def backtest_ma_deviation(
    price_series: pd.Series,
    base_daily: float,
    min_daily: float,
    max_daily: float,
    ma_period: int = 250,
    adjustment_factor: float = 2.0,
    lookback_weeks: int = 52,
    trade_days_per_week: int = 5,
    total_budget: float = 0,
) -> pd.DataFrame:
    """逐周回测均线偏离法, 无未来函数.

    每个时间点仅使用截止到该周的日线数据计算建议金额.
    total_budget > 0 时限制总投入不超出此上限.
    """
    if not _safe_series(price_series, ma_period + 5):
        return pd.DataFrame()

    weekly = price_series.resample("W").last().dropna()

    base_weekly = base_daily * trade_days_per_week
    min_weekly = min_daily * trade_days_per_week
    max_weekly = max_daily * trade_days_per_week

    rows = []
    for dt, week_price in reversed(list(weekly.items())):
        if len(rows) >= lookback_weeks:
            break
        data_slice = price_series[price_series.index <= dt]
        if len(data_slice) < ma_period:
            continue
        r = calc_ma_deviation(
            data_slice, float(week_price),
            base_weekly, min_weekly, max_weekly,
            ma_period=ma_period,
            adjustment_factor=adjustment_factor,
        )
        deviation_pct = float(r.key_metrics.get("偏离度", "0%").replace("%", ""))
        rows.append({
            "date": dt,
            "price": float(week_price),
            "ma": float(data_slice.rolling(ma_period).mean().iloc[-1]),
            "deviation_pct": round(deviation_pct, 2),
            "amount": r.amount,
            "baseline": base_weekly,
            "signal": r.signal,
            "amount_pct": r.amount_pct,
        })
    rows.reverse()

    if total_budget > 0:
        cumulative = 0.0
        for i in range(len(rows)):
            remaining = total_budget - cumulative
            if remaining <= 0:
                rows[i]["amount"] = 0.0
                rows[i]["signal"] = "暂停"
            else:
                rows[i]["amount"] = min(rows[i]["amount"], remaining)
            cumulative += rows[i]["amount"]

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def simulate_portfolios(
    bt_df: pd.DataFrame,
    max_daily: float = 0,
    trade_days_per_week: int = 5,
    total_budget: float = 0,
) -> pd.DataFrame:
    """模拟四种策略的逐周收益率对比, 总投入本金一致.

    输入: backtest_ma_deviation 的返回 (含 date, price, amount 列).
    total_budget > 0 时作为统一本金上限, 否则取智能定投实际总投入.
    返回: 每行 date / four portfolios' value, invested, return.
    """
    if bt_df.empty or len(bt_df) < 2:
        return pd.DataFrame()

    dates = bt_df["date"].values
    prices = bt_df["price"].values
    smart_amounts = bt_df["amount"].values
    n = len(dates)
    total_principal = total_budget if total_budget > 0 else smart_amounts.sum()
    fixed_amount = total_principal / n
    max_weekly = max_daily * trade_days_per_week

    max_budget = total_principal

    history = []
    smart_shares = 0.0
    smart_invested = 0.0
    fixed_shares = 0.0
    fixed_invested = 0.0

    for i in range(n):
        dt = dates[i]
        p = prices[i]

        # 智能定投 — 当期实际投入
        smart_shares += smart_amounts[i] / p
        smart_invested += smart_amounts[i]
        smart_value = smart_shares * p

        # 固定定投
        fixed_shares += fixed_amount / p
        fixed_invested += fixed_amount
        fixed_value = fixed_shares * p

        # 一次性梭哈 — 第一期全仓买入
        if i == 0:
            lump_shares = total_principal / p
        lump_value = lump_shares * p

        # 最大值定投: 每期投入 max_weekly 直到总预算耗尽
        max_amount_i = min(max_weekly, max_budget) if i == n - 1 else min(max_weekly, max_budget)
        if i == 0:
            max_shares = 0.0
            max_invested = 0.0
        actual = min(max_weekly, max_budget)
        max_shares += actual / p
        max_invested += actual
        max_budget -= actual
        max_value = max_shares * p

        history.append({
            "date": dt,
            "price": p,
            "smart_invested": smart_invested,
            "smart_value": smart_value,
            "smart_return": (smart_value / smart_invested - 1) * 100,
            "fixed_invested": fixed_invested,
            "fixed_value": fixed_value,
            "fixed_return": (fixed_value / fixed_invested - 1) * 100,
            "lump_invested": total_principal,
            "lump_value": lump_value,
            "lump_return": (lump_value / total_principal - 1) * 100,
            "max_invested": max_invested,
            "max_value": max_value,
            "max_return": (max_value / max_invested - 1) * 100 if max_invested > 0 else 0,
        })

    result = pd.DataFrame(history)
    result["date"] = pd.to_datetime(result["date"])
    return result


def calc_cagr(final_value: float, total_invested: float, days: float) -> float:
    """计算年化收益率."""
    if total_invested <= 0 or days <= 0:
        return 0.0
    years = days / 365.25
    if years <= 0:
        return 0.0
    return (final_value / total_invested) ** (1 / years) - 1
