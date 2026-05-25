"""
策略定义模块
包含多种智能定投策略的实现, 以及策略注册表
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field

import pandas as pd
import numpy as np


# ══════════════════════════════════════════════════════════════
#  策略信息 (用于展示)
# ══════════════════════════════════════════════════════════════

STRATEGY_CATALOG = [
    {
        "name": "定期定额 (普通定投)",
        "category": "基础策略",
        "description": "每期固定时间、固定金额买入。最基础的定投方式, 无需任何判断, 适合所有投资者。",
        "source": "通用",
        "pros": "简单省心, 强制储蓄, 无需择时能力。",
        "cons": "牛市高位和熊市低位买入同样金额, 成本摊薄效率较低。",
    },
    {
        "name": "均线偏离法 (慧定投)",
        "category": "均线策略",
        "description": (
            "以某指数均线（如250日/500日均线）为基准, 当前价格高于均线时减少定投金额, "
            "低于均线时增加定投金额。偏离度越大, 调整比例越高。"
            "支付宝慧定投、天天基金慧定投均采用此策略。"
        ),
        "source": "支付宝/天天基金/各银行",
        "pros": "低位多买、高位少买, 摊薄成本效果显著; 规则透明, 易于执行。",
        "cons": (
            "依赖均线有效性; 在长期单边上涨行情中可能减少投入而错过收益; "
            "均线选择（250日 vs 500日）对结果影响较大。"
        ),
    },
    {
        "name": "估值定投法 (PE/PB百分位)",
        "category": "估值策略",
        "description": (
            "参考跟踪指数的市盈率(PE)或市净率(PB)历史百分位。"
            "低估区间（如PE百分位<30%）加大投入; 正常区间按计划投; "
            "高估区间（如PE百分位>70%）减少或暂停投入。"
        ),
        "source": "天天基金/支付宝/各基金公司",
        "pros": "基于价值判断, 安全边际高; 适合宽基指数基金。",
        "cons": "仅适用于指数基金; 低估可能持续很久, 需要耐心; 主动管理型基金不适用。",
    },
    {
        "name": "移动平均成本法 (成本定投)",
        "category": "成本策略",
        "description": (
            "根据当前基金净值与投资者平均持仓成本的偏离度来调整定投金额。"
            "净值低于平均成本时多投, 高于平均成本时少投。"
            "偏离度越大, 调整幅度越大（通常50%~200%）。"
        ),
        "source": "支付宝涨跌幅模式/各平台",
        "pros": "直接以投资者自身成本为锚, 个性化强; 越跌越买, 摊薄效果明显。",
        "cons": "初始阶段成本不稳定; 极端行情下调整幅度可能过大; 需要预留足够资金。",
    },
    {
        "name": "价值平均策略 (市值定投)",
        "category": "目标策略",
        "description": (
            "设定每期目标市值增长路径（如每月市值增长1000元）。"
            "当期市值低于目标时补足差额买入; 高于目标时卖出超出部分。"
            "由Michael Edelson提出。"
        ),
        "source": "《Value Averaging》/学术经典",
        "pros": "强制实现市值增长路径, 低买高卖效果明显。",
        "cons": "大幅下跌后需投入巨额资金补仓; 上涨行情中需卖出, 可能产生税务成本。",
    },
    {
        "name": "下跌加仓法 (跌幅触发)",
        "category": "事件驱动",
        "description": (
            "设定跌幅阈值X%和买入金额Y。每当某日价格较前一日下跌超过X%时, "
            "触发买入Y元。跌幅越大、越频繁, 投入越多。"
            "本质是「越跌越买」的极端形式。"
        ),
        "source": "常见量化策略/本系统",
        "pros": "精准捕捉恐慌性下跌; 资金使用效率高, 不在平稳行情中浪费子弹。",
        "cons": "长期横盘或微跌行情中可能长时间不触发; 参数选择对结果敏感。",
    },
    {
        "name": "网格交易法",
        "category": "震荡策略",
        "description": (
            "在预设价格区间内划分多个等距网格。价格每跌一格买入固定金额/数量, "
            "每涨一格卖出。反复收割震荡收益。可叠加底仓长期持有。"
        ),
        "source": "量化交易/数字货币/ETF",
        "pros": "震荡市中持续盈利; 自动化程度高; 可与定投组合使用。",
        "cons": "单边牛市容易卖飞; 单边熊市持续买入加大亏损; 参数设置复杂。",
    },
    {
        "name": "趋势定投 (均线金叉/死叉)",
        "category": "趋势策略",
        "description": (
            "利用短期均线和长期均线的交叉判断趋势。"
            "短期均线上穿长期均线（金叉）时买入高风险品种; "
            "下穿（死叉）时转入低风险品种或暂停定投。可实现基金转换。"
        ),
        "source": "华安基金/各银行",
        "pros": "顺势而为, 牛市多赚, 熊市少亏; 可通过基金转换降低摩擦成本。",
        "cons": "均线信号滞后; 震荡市中频繁交叉导致错误信号。",
    },
    {
        "name": "目标止盈定投",
        "category": "止盈策略",
        "description": (
            "设定收益率目标（如20%）, 达到目标后全部或部分赎回, 继续下一轮定投。"
            "可配合最大回撤止盈（达到目标后不回撤到阈值不止盈）。"
        ),
        "source": "各平台通用",
        "pros": "及时锁定收益, 避免坐过山车; 简单可执行。",
        "cons": "可能过早止盈错过牛市后半段; 需配合再投入计划。",
    },
    {
        "name": "均线偏离 + 振幅调节 (支付宝)",
        "category": "综合策略",
        "description": (
            "支付宝慧定投的完整算法: 以500日均线为基准, 计算当前价格偏离均线的百分比; "
            "同时计算近10日振幅, 振幅过大时降低扣款率。"
            "扣款率范围60%~210%, 偏离+振幅共同决定。参考指数可选沪深300/中证500/创业板指。"
        ),
        "source": "支付宝-慧定投",
        "pros": "多因子综合判断, 比单一均线更稳健; 振幅过滤避免极端波动时追高。",
        "cons": "算法黑箱, 具体参数不透明; 仅支持中证/沪深/创业板三大指数相关基金。",
    },
    {
        "name": "微信理财通智能定投",
        "category": "综合策略",
        "description": (
            "以跟踪指数的20日/250日均线为基准, 低于均线最高可投2倍, 高于均线最低投0.5倍。"
            "部分基金联动PE/PB估值辅助判断, 低估加码、高估减码, 浮动区间0.5~2倍。"
        ),
        "source": "微信理财通",
        "pros": "均线+估值双因子; 灵活性高, 支持手动切换。",
        "cons": "策略覆盖基金范围有限; 需搭配腾讯理财通生态使用。",
    },
    {
        "name": "天天基金慧定投",
        "category": "综合策略",
        "description": (
            "策略最为丰富: 支持估值策略（PE/PB分位<30%多投, >70%少投, 最高3倍）; "
            "均线策略（自定义指数与均线周期50/120/250日, 偏离度50%~200%）; "
            "成本策略（净值与持仓成本偏离调仓）; 目标止盈（收益率或回撤阈值）。"
        ),
        "source": "天天基金",
        "pros": "策略最全, 可自定义程度高; 支持止盈设置。",
        "cons": "参数复杂, 新手门槛高; 需天天基金平台支持。",
    },
    {
        "name": "银行智能定投 (招商/中行)",
        "category": "综合策略",
        "description": (
            "招商银行: 基于均线偏离的高低位智能定投; 中国银行: BOC smart模型遴选适配基金, "
            "自动匹配指数和均线参数, 懒人投资者无需手动设置。"
            "各银行普遍在买入策略+止盈策略两方面发力。"
        ),
        "source": "招商银行/中国银行/各商业银行",
        "pros": "无需手动设置指数和均线; 结合银行理财经理服务。",
        "cons": "策略相对保守; 仅限本行代销基金。",
    },
    {
        "name": "低估值指数策略",
        "category": "指数策略",
        "description": (
            "综合运用PE、PB、ROE、估值历史分位等指标, 寻找相对低估的宽基/行业指数进行定投。"
            "在估值达到历史高位时一次性卖出。本质上是「定投买入 + 择时卖出」。"
        ),
        "source": "自媒体大V/二鸟说/各平台",
        "pros": "基于价值判断, 买入便宜指数; 卖出纪律清晰。",
        "cons": "需要持续跟踪估值; 低估后可能继续低估; 行业指数估值可能长期失真。",
    },
    {
        "name": "量化增强策略",
        "category": "增强策略",
        "description": (
            "选择市场上最强的量化增强基金进行定投。"
            "核心逻辑: ①主流指数长期上涨; ②量化增强基金能大幅超越对标指数。"
            "如华泰柏瑞量化增强(对标沪深300)、建信中证500增强。"
        ),
        "source": "二鸟说/基金研究",
        "pros": "享受量化模型的超额收益; 简单只需选择强基金。",
        "cons": "过去超额不代表未来; 量化模型失效风险; 增强基金费率较高。",
    },
]


def print_strategy_catalog():
    """打印完整策略清单"""
    for i, s in enumerate(STRATEGY_CATALOG, 1):
        print(f"{i:2d}. [{s['category']}] {s['name']}")
        print(f"    {s['description'][:80]}...")
        print()


# ══════════════════════════════════════════════════════════════
#  DropBuy 策略: 前一日跌幅超过 X% 时买入 Y 元
# ══════════════════════════════════════════════════════════════

@dataclass
class DropBuyResult:
    """单次回测的结果"""
    total_invested: float = 0.0
    final_value: float = 0.0
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    num_investments: int = 0
    records: pd.DataFrame = field(default_factory=pd.DataFrame)
    portfolio_series: pd.Series = field(default_factory=pd.Series)
    invested_series: pd.Series = field(default_factory=pd.Series)
    nav_series: pd.Series = field(default_factory=pd.Series)


def run_dropbuy_backtest(
    price_series: pd.Series,
    X: float = 1.0,
    Y: float = 1000.0,
    max_total: float = 0.0,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> DropBuyResult:
    """
    下跌加仓策略回测

    逻辑: 每日检查, 若当日涨跌幅 < -X% (即跌幅超过 X%), 则在当日买入 Y 元。
    若设置了 max_total, 累计投入达到后停止。

    Parameters
    ----------
    price_series : pd.Series
        DatetimeIndex 价格序列
    X : float
        跌幅阈值 (百分比), e.g. 1.5 表示跌幅超过 1.5% 时触发
    Y : float
        每期买入金额 (元)
    max_total : float
        总投资上限, 0 表示不设限
    start_date, end_date : str, optional
        回测区间
    """
    prices = price_series.sort_index().dropna()

    if start_date:
        prices = prices[prices.index >= pd.Timestamp(start_date)]
    if end_date:
        prices = prices[prices.index <= pd.Timestamp(end_date)]

    if len(prices) < 2:
        return DropBuyResult(nav_series=prices)

    # 计算每日涨跌幅 (%)
    daily_ret = prices.pct_change() * 100  # 第一个值为 NaN

    # 寻找触发日期: 当日涨跌幅 < -X (即跌幅超过 X%)
    trigger_mask = daily_ret < -X
    trigger_dates = prices.index[trigger_mask]

    # 执行买入
    total_invested = 0.0
    total_shares = 0.0
    records = []
    cash_flows: List[Tuple] = []

    for d in trigger_dates:
        price = prices.loc[d]
        actual = Y
        if max_total > 0 and total_invested + actual > max_total:
            actual = max_total - total_invested
            if actual <= 0:
                break
        shares = actual / price
        total_shares += shares
        total_invested += actual
        records.append({
            "日期": d,
            "价格": round(price, 4),
            "买入份额": round(shares, 4),
            "累计份额": round(total_shares, 4),
            "投入金额": round(actual, 2),
            "累计投入": round(total_invested, 2),
            "当日涨跌幅%": round(daily_ret.loc[d], 2),
        })
        cash_flows.append((d.to_pydatetime(), -actual))

    if total_invested == 0:
        return DropBuyResult(nav_series=prices)

    final_price = prices.iloc[-1]
    final_value = total_shares * final_price
    total_return = (final_value - total_invested) / total_invested * 100

    cash_flows.append((prices.index[-1].to_pydatetime(), final_value))
    annualized = _xirr(cash_flows) * 100

    records_df = pd.DataFrame(records)

    # 市值曲线
    portfolio_values = []
    invested_values = []
    cum_shares = 0.0
    cum_invested = 0.0
    inv_idx = 0

    for date_idx, price in prices.items():
        if inv_idx < len(records):
            rec_date = records[inv_idx]["日期"]
            if date_idx >= rec_date:
                while inv_idx < len(records) and date_idx >= records[inv_idx]["日期"]:
                    cum_shares += records[inv_idx]["买入份额"]
                    cum_invested += records[inv_idx]["投入金额"]
                    inv_idx += 1
        portfolio_values.append(cum_shares * price)
        invested_values.append(cum_invested)

    return DropBuyResult(
        total_invested=round(total_invested, 2),
        final_value=round(final_value, 2),
        total_return_pct=round(total_return, 2),
        annualized_return_pct=round(annualized, 2),
        num_investments=len(records),
        records=records_df,
        portfolio_series=pd.Series(portfolio_values, index=prices.index),
        invested_series=pd.Series(invested_values, index=prices.index),
        nav_series=prices,
    )


# ══════════════════════════════════════════════════════════════
#  内部工具
# ══════════════════════════════════════════════════════════════

def _xirr(transactions):
    """XIRR 计算 (与 dca.py 相同实现)"""
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


# ══════════════════════════════════════════════════════════════
#  智能策略回测引擎
# ══════════════════════════════════════════════════════════════

def _build_smart_result(
    strategy_name: str,
    invest_entries: List[Tuple],  # [(date, amount, price, extra_info_dict?), ...]
    prices: pd.Series,
    max_total: float = 0,
) -> dict:
    """
    通用的策略结果构建器

    将 [(date, amount, price), ...] 列表
    转换为标准的结果 dict (与 run_dca_backtest 输出兼容)。

    amount > 0 = 买入, amount < 0 = 卖出 (目标止盈策略), 均被正确处理。
    """
    cleaned = [(d, amt, p) for d, amt, p, *_ in invest_entries]

    buy_total = sum(amt for _, amt, _ in cleaned if amt > 0)
    sell_total = sum(abs(amt) for _, amt, _ in cleaned if amt < 0)
    total_shares = sum(amt / p for _, amt, p in cleaned)

    final_price = float(prices.iloc[-1])
    final_value = total_shares * final_price
    net_invested = buy_total - sell_total
    total_return = (final_value - buy_total) / max(buy_total, 1e-9) * 100

    cash_flows = [(d.to_pydatetime(), -amt) for d, amt, _ in cleaned]
    cash_flows.append((prices.index[-1].to_pydatetime(), final_value))
    annualized = _xirr(cash_flows) * 100

    records = []
    cs, ci = 0.0, 0.0
    for d, amt, price in cleaned:
        is_buy = amt > 0
        shares = amt / price
        cs += shares
        ci += amt
        label = "买入" if is_buy else "卖出"
        records.append({
            "日期": d,
            "价格": round(price, 4),
            label: round(abs(shares), 4),
            "累计份额": round(cs, 4),
            "交易金额": round(abs(amt), 2),
            "累计净投入": round(ci, 2),
        })
    records_df = pd.DataFrame(records)

    portfolio_vals, invested_vals = [], []
    cs, ci = 0.0, 0.0
    idx = 0
    n = len(cleaned)
    for dt, price in prices.items():
        if idx < n and dt >= cleaned[idx][0]:
            while idx < n and dt >= cleaned[idx][0]:
                cs += cleaned[idx][1] / cleaned[idx][2]
                ci += cleaned[idx][1]
                idx += 1
        portfolio_vals.append(cs * price)
        invested_vals.append(ci)

    return {
        "strategy": strategy_name,
        "total_invested": round(buy_total, 2),
        "final_value": round(final_value, 2),
        "total_return_pct": round(total_return, 2),
        "annualized_return_pct": round(annualized, 2),
        "num_investments": len(cleaned),
        "records": records_df,
        "nav_series": prices,
        "portfolio_series": pd.Series(portfolio_vals, index=prices.index),
        "invested_series": pd.Series(invested_vals, index=prices.index),
    }


# ── 1. 均线偏离法 (慧定投) ──

def _ma_adjust_factor(price: float, ma: float, min_r: float, max_r: float) -> float:
    """计算均线偏离调整系数"""
    if np.isnan(ma) or ma <= 0:
        return 1.0
    ratio = ma / price  # below MA → ratio > 1 → increase; above → ratio < 1 → decrease
    return max(min_r, min(max_r, ratio))


def run_ma_adjust_dca(
    price_series: pd.Series,
    start_date: str = None,
    end_date: str = None,
    amount: float = 1000,
    ma_period: int = 250,
    min_ratio: float = 0.5,
    max_ratio: float = 2.0,
    max_total: float = 0,
) -> dict:
    """
    均线偏离法 (慧定投)

    若当前价格低于均线 → 增加金额 (最多 2 倍)
    若当前价格高于均线 → 减少金额 (最少 0.5 倍)
    """
    prices = price_series.sort_index().dropna()
    if start_date: prices = prices[prices.index >= pd.Timestamp(start_date)]
    if end_date: prices = prices[prices.index <= pd.Timestamp(end_date)]
    if len(prices) < 2:
        return _build_smart_result("均线偏离(慧定投)", [], prices)

    ma = prices.rolling(ma_period, min_periods=ma_period).mean()

    # 每月 1 号检查
    invest_dates = []
    for dt, price in prices.items():
        if dt.day == 1 or (len(invest_dates) == 0 and dt == prices.index[0]):
            factor = _ma_adjust_factor(price, ma.loc[dt], min_ratio, max_ratio)
            actual = amount * factor
            invest_dates.append((dt, actual, price))

    # Apply max_total cap
    capped = []
    cum = 0.0
    for dt, amt, price in invest_dates:
        if max_total > 0 and cum + amt > max_total:
            amt = max(0, max_total - cum)
        if amt > 0:
            capped.append((dt, amt, price))
            cum += amt
        if max_total > 0 and cum >= max_total:
            break

    return _build_smart_result(
        f"均线偏离(MA{ma_period}, {min_ratio}~{max_ratio}x)",
        capped, prices, max_total,
    )


# ── 2. 移动平均成本法 (成本定投) ──

def run_cost_average_dca(
    price_series: pd.Series,
    start_date: str = None,
    end_date: str = None,
    amount: float = 1000,
    min_ratio: float = 0.5,
    max_ratio: float = 2.0,
    max_total: float = 0,
) -> dict:
    """
    移动平均成本法 (成本定投)

    净值低于平均成本 → 多投; 净值高于平均成本 → 少投
    扣款率 = avg_cost / price, 范围 0.5~2.0
    """
    prices = price_series.sort_index().dropna()
    if start_date: prices = prices[prices.index >= pd.Timestamp(start_date)]
    if end_date: prices = prices[prices.index <= pd.Timestamp(end_date)]
    if len(prices) < 2:
        return _build_smart_result("成本定投", [], prices)

    entries = []
    total_shares, total_invested = 0.0, 0.0

    for dt, price in prices.items():
        if dt.day == 1 or (len(entries) == 0 and dt == prices.index[0]):
            avg_cost = total_invested / total_shares if total_shares > 0 else price
            factor = avg_cost / price  # below cost → >1, above cost → <1
            factor = max(min_ratio, min(max_ratio, factor))
            actual = amount * factor

            if max_total > 0 and total_invested + actual > max_total:
                actual = max(0, max_total - total_invested)
            if actual > 0:
                entries.append((dt, actual, price))
                total_shares += actual / price
                total_invested += actual
            if max_total > 0 and total_invested >= max_total:
                break

    return _build_smart_result(
        f"成本定投({min_ratio}~{max_ratio}x)",
        entries, prices, max_total,
    )


# ── 3. 价值平均策略 (市值定投) ──

def run_value_averaging(
    price_series: pd.Series,
    start_date: str = None,
    end_date: str = None,
    amount: float = 1000,
    max_total: float = 0,
) -> dict:
    """
    价值平均策略 (市值定投)

    每月设定目标市值 = 当月期数 × amount
    若当前市值 < 目标 → 补足差额 (买入)
    若当前市值 > 目标 → 不减仓 (长期持有, 不卖出)
    """
    prices = price_series.sort_index().dropna()
    if start_date: prices = prices[prices.index >= pd.Timestamp(start_date)]
    if end_date: prices = prices[prices.index <= pd.Timestamp(end_date)]
    if len(prices) < 2:
        return _build_smart_result("价值平均", [], prices)

    entries = []
    total_shares, total_invested = 0.0, 0.0
    period = 0

    for dt, price in prices.items():
        if dt.day == 1 or (len(entries) == 0 and dt == prices.index[0]):
            period += 1
            target = amount * period
            current_value = total_shares * price

            diff = target - current_value
            if diff > 100:  # 微小差额忽略
                actual = min(diff, max_total - total_invested) if max_total > 0 else diff
                if actual > 0:
                    entries.append((dt, actual, price))
                    total_shares += actual / price
                    total_invested += actual
            # diff <= 0 → 不操作 (长期持有)
            if max_total > 0 and total_invested >= max_total:
                break

    return _build_smart_result(
        f"价值平均({amount:.0f}元/期)",
        entries, prices, max_total,
    )


# ── 4. 趋势定投 (均线金叉/死叉) ──

def run_trend_dca(
    price_series: pd.Series,
    start_date: str = None,
    end_date: str = None,
    amount: float = 1000,
    short_period: int = 20,
    long_period: int = 120,
    aggressive_mult: float = 1.5,
    conservative_mult: float = 0.5,
    max_total: float = 0,
) -> dict:
    """
    趋势定投 (均线金叉/死叉)

    短期均线 > 长期均线 (金叉) → 牛市信号, 多投 1.5 倍
    短期均线 < 长期均线 (死叉) → 熊市信号, 少投 0.5 倍
    """
    prices = price_series.sort_index().dropna()
    if start_date: prices = prices[prices.index >= pd.Timestamp(start_date)]
    if end_date: prices = prices[prices.index <= pd.Timestamp(end_date)]
    if len(prices) < long_period + 1:
        return _build_smart_result("趋势定投(金叉/死叉)", [], prices)

    ma_short = prices.rolling(short_period, min_periods=short_period).mean()
    ma_long = prices.rolling(long_period, min_periods=long_period).mean()

    entries = []
    cum = 0.0

    for dt, price in prices.items():
        if dt.day != 1 and dt != prices.index[0]:
            continue
        s, l = ma_short.loc[dt], ma_long.loc[dt]
        if np.isnan(s) or np.isnan(l):
            continue
        mult = aggressive_mult if s > l else conservative_mult
        actual = amount * mult
        if max_total > 0 and cum + actual > max_total:
            actual = max(0, max_total - cum)
        if actual > 0:
            entries.append((dt, actual, price))
            cum += actual
        if max_total > 0 and cum >= max_total:
            break

    return _build_smart_result(
        f"趋势定投(MA{short_period}/{long_period}, {conservative_mult}~{aggressive_mult}x)",
        entries, prices, max_total,
    )


# ── 5. 支付宝慧定投 (均线+振幅) ──

def run_alipay_smart_dca(
    price_series: pd.Series,
    start_date: str = None,
    end_date: str = None,
    amount: float = 1000,
    ma_period: int = 500,
    min_rate: float = 0.6,
    max_rate: float = 2.1,
    max_total: float = 0,
) -> dict:
    """
    支付宝慧定投 (均线偏离 + 振幅调节)

    1. 计算当前价格相对 500 日均线的偏离度
    2. 偏离度决定基础扣款率 (0.6~2.1)
    3. 近 10 日振幅过大时进一步降低扣款率
    """
    prices = price_series.sort_index().dropna()
    if start_date: prices = prices[prices.index >= pd.Timestamp(start_date)]
    if end_date: prices = prices[prices.index <= pd.Timestamp(end_date)]
    if len(prices) < ma_period:
        return _build_smart_result("支付宝慧定投", [], prices)

    ma = prices.rolling(ma_period, min_periods=ma_period).mean()
    # 10 日振幅: (high - low) / close, rolling 10
    amp_10 = prices.rolling(10).max() / prices.rolling(10).min() - 1

    entries = []
    cum = 0.0

    for dt, price in prices.items():
        if dt.day != 1 and dt != prices.index[0]:
            continue
        mav = ma.loc[dt]
        if np.isnan(mav) or mav <= 0:
            continue

        # 偏离度: 正=高于均线(应少投), 负=低于均线(应多投)
        deviation = price / mav - 1
        base_rate = 1.0 - deviation  # below MA → >1, above → <1

        # 振幅调节: 10日振幅超 5% 时降低 0.1
        amp = amp_10.loc[dt] if not np.isnan(amp_10.loc[dt]) else 0
        if amp > 0.05:
            base_rate -= 0.1 * min(amp / 0.05, 2.0)  # 最高降 0.2

        rate = max(min_rate, min(max_rate, base_rate))
        actual = amount * rate
        if max_total > 0 and cum + actual > max_total:
            actual = max(0, max_total - cum)
        if actual > 0:
            entries.append((dt, actual, price))
            cum += actual
        if max_total > 0 and cum >= max_total:
            break

    return _build_smart_result(
        f"支付宝慧定投(MA{ma_period}, {min_rate}~{max_rate}x)",
        entries, prices, max_total,
    )
