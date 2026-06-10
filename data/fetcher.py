"""数据获取模块 - 基于 AKShare / 东方财富 REST API

支持: A股个股, ETF基金, LOF基金, 开放式基金, 指数, 境外资产(美股/ETF/商品)
"""

import time
from typing import Optional, Literal
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import akshare as ak

from data.cache import cache_key, read_cache, write_cache, filter_by_date, clear_expired
from data.asset_config import ASSET_TYPE_CONFIG

AssetType = Literal["stock", "etf", "lof", "open_fund", "index", "us"]

# ── HTTP 会话 ──
_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://quote.eastmoney.com/",
})

_CONNECT_ERRORS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)

_FULL_START = "19900101"
_FULL_END   = "20500101"


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(_CONNECT_ERRORS),
    reraise=True,
)
def _req_json(url: str, params: dict) -> dict:
    r = _SESSION.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def _parse_em_klines(data_json: dict, symbol: str) -> pd.DataFrame:
    if not (data_json.get("data") and data_json["data"].get("klines")):
        return pd.DataFrame()
    rows = [item.split(",") for item in data_json["data"]["klines"]]
    cols = ["日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅", "涨跌幅", "涨跌额", "换手率"]
    df = pd.DataFrame(rows, columns=cols[:len(rows[0])])
    df["股票代码"] = symbol
    for col in df.columns:
        if col in ("日期", "股票代码"):
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["日期"] = pd.to_datetime(df["日期"])
    df.set_index("日期", inplace=True)
    return df


def _em_kline(symbol: str, market_code: int, period: str, adjust: str,
              start: str = _FULL_START, end: str = _FULL_END) -> pd.DataFrame:
    period_map = {"daily": "101", "weekly": "102", "monthly": "103"}
    adjust_map = {"qfq": "1", "hfq": "2", "": "0"}
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
        "klt": period_map.get(period, "101"),
        "fqt": adjust_map.get(adjust, "0"),
        "secid": f"{market_code}.{symbol}",
        "beg": start,
        "end": end,
    }
    data = _req_json(url, params)
    return _parse_em_klines(data, symbol)


# ══════════════════════════════════════════
#  1.  A股个股
# ══════════════════════════════════════════

def fetch_stock_history(
    symbol: str,
    start_date: str = "20000101",
    end_date: Optional[str] = None,
    period: Literal["daily", "weekly", "monthly"] = "daily",
    adjust: Literal["", "qfq", "hfq"] = "qfq",
) -> pd.DataFrame:
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")

    ck = cache_key("stock", symbol, period, adjust)
    cached = read_cache(ck)
    if cached is not None:
        return filter_by_date(cached, start_date, end_date)

    mc = 1 if symbol.startswith("6") else 0
    err = None
    for _ in range(3):
        try:
            full = _em_kline(symbol, mc, period, adjust)
            if not full.empty:
                write_cache(ck, full)
                return filter_by_date(full, start_date, end_date)
        except _CONNECT_ERRORS as e:
            err = e
            time.sleep(2)
            continue

    try:
        df = ak.stock_zh_a_hist(symbol=symbol, period=period,
                                start_date=_validate_date(start_date),
                                end_date=_validate_date(end_date), adjust=adjust)
        if not df.empty:
            df["日期"] = pd.to_datetime(df["日期"])
            df.set_index("日期", inplace=True)
            write_cache(ck, df)
            return df
    except Exception as e:
        err = e

    raise ConnectionError(
        f"无法获取股票 {symbol} 的数据。可能是代码不存在、类型选择有误或网络波动。"
        f"建议: 确认代码正确, 或尝试切换资产类型。({err})"
    )


# ══════════════════════════════════════════
#  2.  ETF 基金
# ══════════════════════════════════════════

def fetch_etf_history(
    symbol: str,
    start_date: str = "20000101",
    end_date: Optional[str] = None,
    period: Literal["daily", "weekly", "monthly"] = "daily",
    adjust: Literal["", "qfq", "hfq"] = "qfq",
) -> pd.DataFrame:
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")

    ck = cache_key("etf", symbol, period, adjust)
    cached = read_cache(ck)
    if cached is not None:
        return filter_by_date(cached, start_date, end_date)

    err = None
    for _ in range(3):
        try:
            full = _em_kline(symbol, 0, period, adjust)
            if not full.empty:
                write_cache(ck, full)
                return filter_by_date(full, start_date, end_date)
        except _CONNECT_ERRORS as e:
            err = e
            time.sleep(2)
            continue

    try:
        df = ak.fund_etf_hist_em(symbol=symbol, period=period,
                                 start_date=_validate_date(start_date),
                                 end_date=_validate_date(end_date), adjust=adjust)
        if not df.empty:
            df["日期"] = pd.to_datetime(df["日期"])
            df.set_index("日期", inplace=True)
            write_cache(ck, df)
            return df
    except Exception as e:
        err = e

    try:
        df = ak.fund_lof_hist_em(symbol=symbol, period=period,
                                 start_date=_validate_date(start_date),
                                 end_date=_validate_date(end_date), adjust=adjust)
        if not df.empty:
            df["日期"] = pd.to_datetime(df["日期"])
            df.set_index("日期", inplace=True)
            write_cache(ck, df)
            return df
    except Exception as e:
        err = e

    try:
        nav_df = fetch_open_fund_nav(symbol=symbol, adjust=adjust)
        if not nav_df.empty:
            nav_df = nav_df.rename(columns={"单位净值": "收盘"})
            write_cache(ck, nav_df)
            return filter_by_date(nav_df, start_date, end_date)
    except Exception as nav_err:
        err = nav_err

    raise ConnectionError(
        f"无法获取ETF {symbol} 的数据。可能是代码不存在、类型选择有误或网络波动。"
        f"已尝试 kline、LOF 接口和基金净值三种路径均失败。({err})"
    )


def fetch_etf_nav_history(
    symbol: str,
    start_date: str = "20000101",
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")
    ck = cache_key("etf_nav", symbol)
    cached = read_cache(ck)
    if cached is not None:
        return filter_by_date(cached, start_date, end_date)

    df = ak.fund_etf_fund_info_em(
        fund=symbol,
        start_date=_validate_date(_FULL_START),
        end_date=_validate_date(_FULL_END),
    )
    if not df.empty:
        df["净值日期"] = pd.to_datetime(df["净值日期"])
        df.set_index("净值日期", inplace=True)
        write_cache(ck, df)
    return filter_by_date(df, start_date, end_date)


# ══════════════════════════════════════════
#  3.  LOF 基金
# ══════════════════════════════════════════

def fetch_lof_history(
    symbol: str,
    start_date: str = "20000101",
    end_date: Optional[str] = None,
    period: Literal["daily", "weekly", "monthly"] = "daily",
    adjust: Literal["", "qfq", "hfq"] = "qfq",
) -> pd.DataFrame:
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")
    ck = cache_key("lof", symbol, period, adjust)
    cached = read_cache(ck)
    if cached is not None:
        return filter_by_date(cached, start_date, end_date)

    err = None
    for _ in range(3):
        try:
            df = ak.fund_lof_hist_em(symbol=symbol, period=period,
                                     start_date=_validate_date(_FULL_START),
                                     end_date=_validate_date(_FULL_END), adjust=adjust)
            if not df.empty:
                df["日期"] = pd.to_datetime(df["日期"])
                df.set_index("日期", inplace=True)
                write_cache(ck, df)
                return filter_by_date(df, start_date, end_date)
        except _CONNECT_ERRORS as e:
            err = e
            time.sleep(2)
            continue

    try:
        df = ak.fund_etf_hist_em(symbol=symbol, period=period,
                                 start_date=_validate_date(start_date),
                                 end_date=_validate_date(end_date), adjust=adjust)
        if not df.empty:
            df["日期"] = pd.to_datetime(df["日期"])
            df.set_index("日期", inplace=True)
            write_cache(ck, df)
            return df
    except Exception as e:
        err = e

    try:
        nav_df = fetch_open_fund_nav(symbol=symbol, adjust=adjust)
        if not nav_df.empty:
            nav_df = nav_df.rename(columns={"单位净值": "收盘"})
            write_cache(ck, nav_df)
            return filter_by_date(nav_df, start_date, end_date)
    except Exception as nav_err:
        err = nav_err

    raise ConnectionError(
        f"无法获取LOF {symbol} 的数据。可能是不支持的 LOF 代码或 QDII 基金。"
        f"已尝试 kline、ETF 接口和基金净值三种路径均失败。({err})"
    )


# ══════════════════════════════════════════
#  4.  开放式基金
# ══════════════════════════════════════════

def _forward_adjust_nav(nav_df: pd.DataFrame, adjust: str) -> pd.DataFrame:
    if adjust != "qfq" or nav_df.empty or "单位净值" not in nav_df.columns:
        return nav_df
    df = nav_df.copy()
    s = df["单位净值"]
    ratio = s / s.shift(1)
    events = ratio[(ratio < 0.8) | (ratio > 1.25)]
    if events.empty:
        return nav_df
    cumulative = 1.0
    for idx in reversed(sorted(events.index)):
        cumulative *= events[idx]
        s.loc[s.index < idx] *= cumulative
    df["单位净值"] = s
    return df


def fetch_open_fund_nav(
    symbol: str,
    indicator: str = "单位净值走势",
    adjust: str = "",
) -> pd.DataFrame:
    ck = cache_key("open_fund", symbol, indicator, adjust) if adjust else cache_key("open_fund", symbol, indicator)
    cached = read_cache(ck)
    if cached is not None:
        return cached

    df = ak.fund_open_fund_info_em(symbol=symbol, indicator=indicator, period="成立来")
    if not df.empty:
        date_col = [c for c in df.columns if "日期" in c][0]
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
        if adjust:
            df = _forward_adjust_nav(df, adjust)
        write_cache(ck, df)
    return df


# ══════════════════════════════════════════
#  5.  指数
# ══════════════════════════════════════════

def fetch_index_history(
    symbol: str,
    start_date: str = "20000101",
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")
    ck = cache_key("index", symbol)
    cached = read_cache(ck)
    if cached is not None:
        return filter_by_date(cached, start_date, end_date)

    df = ak.stock_zh_index_daily_tx(symbol=symbol)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df = df.sort_index()
        write_cache(ck, df)
    return filter_by_date(df, start_date, end_date)


# ══════════════════════════════════════════
#  6.  境外资产 (yfinance)
# ══════════════════════════════════════════

_US_EXCHANGE_PREFIXES = ["105", "106", "107"]


def _em_us_kline(secid: str, period: str, adjust: str,
                 start: str, end: str) -> pd.DataFrame:
    period_map = {"daily": "101", "weekly": "102", "monthly": "103"}
    adjust_map = {"qfq": "1", "hfq": "2", "": "0"}
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
        "klt": period_map.get(period, "101"),
        "fqt": adjust_map.get(adjust, "0"),
        "secid": secid,
        "beg": start,
        "end": end,
    }
    data = _req_json(url, params)
    return _parse_em_klines(data, secid)


def fetch_us_history(
    symbol: str,
    start_date: str = "20000101",
    end_date: Optional[str] = None,
    period: Literal["daily", "weekly", "monthly"] = "daily",
    adjust: Literal["", "qfq", "hfq"] = "qfq",
) -> pd.DataFrame:
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")

    ck = cache_key("us", symbol, period)
    cached = read_cache(ck)
    if cached is not None:
        return filter_by_date(cached, start_date, end_date)

    symbol = symbol.upper()
    errs = []
    for prefix in _US_EXCHANGE_PREFIXES:
        try:
            full = _em_us_kline(f"{prefix}.{symbol}", period, adjust,
                                start=start_date, end=end_date)
            if not full.empty:
                write_cache(ck, full)
                return filter_by_date(full, start_date, end_date)
        except Exception as e:
            errs.append(f"{prefix}: {e}")
            continue

    raise ConnectionError(
        f"无法获取 {symbol} 的数据。\n已尝试 {_US_EXCHANGE_PREFIXES} 均失败:\n"
        + "\n".join(errs)
        + "\n常见代码: QQQ, TQQQ, GLD, SPY, MSFT, AAPL"
    )


# ══════════════════════════════════════════
#  7.  统一入口
# ══════════════════════════════════════════

def fetch_history(
    asset_type: AssetType,
    symbol: str,
    start_date: str = "20000101",
    end_date: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")

    def _try_fetch(ft: str) -> Optional[pd.DataFrame]:
        try:
            if ft == "open_fund":
                df = fetch_open_fund_nav(
                    symbol=symbol, indicator=kwargs.get("indicator", "单位净值走势"),
                    adjust=kwargs.get("adjust", ""),
                )
                if not df.empty:
                    df = df.rename(columns={"单位净值": "收盘"})
                    return filter_by_date(df, start_date, end_date)
                return None
            f = {
                "stock": fetch_stock_history,
                "etf": fetch_etf_history,
                "lof": fetch_lof_history,
                "index": fetch_index_history,
                "us": fetch_us_history,
            }.get(ft)
            if f is None:
                return None
            return f(symbol=symbol, start_date=start_date, end_date=end_date, **kwargs)
        except Exception:
            return None

    result = _try_fetch(asset_type)
    if result is not None and not result.empty:
        return result

    _NO_FALLBACK = {"us"}
    if asset_type in _NO_FALLBACK:
        raise ConnectionError(
            f"无法获取 {symbol} 的数据。请确认代码在 Yahoo Finance 中存在。"
        )

    fallback_order = ["open_fund", "etf", "lof", "stock", "index"]
    for ft in fallback_order:
        if ft == asset_type:
            continue
        result = _try_fetch(ft)
        if result is not None and not result.empty:
            return result

    raise ConnectionError(
        f"无法获取 {symbol} 的数据。已尝试多种数据源均失败。\n"
        f"建议: 确认代码正确, 或在侧边栏尝试切换资产类型。"
    )


def get_price_series(df: pd.DataFrame) -> Optional[pd.Series]:
    for col in ["收盘", "收盘价", "单位净值", "close", "close_price"]:
        if col in df.columns:
            return df[col].sort_index()
    return None


# ══════════════════════════════════════════
#  7.  辅助函数
# ══════════════════════════════════════════

def _validate_date(date_str: str) -> str:
    if len(date_str) != 8:
        raise ValueError(f"日期格式错误: {date_str}")
    return date_str


def get_etf_list() -> pd.DataFrame:
    df = ak.fund_etf_fund_daily_em()
    return df[["基金代码", "基金简称"]] if not df.empty else df


def get_open_fund_list() -> pd.DataFrame:
    df = ak.fund_open_fund_daily_em()
    return df[["基金代码", "基金简称"]] if not df.empty else df


def ensure_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    df_out = df.copy()
    price_col = None
    for c in ["收盘", "收盘价", "收盘价(元)", "close"]:
        if c in df_out.columns:
            price_col = c
            break
    if price_col is None:
        return df_out
    for target in ["开盘", "最高", "最低"]:
        if target not in df_out.columns:
            df_out[target] = df_out[price_col]
    return df_out


def add_premium_rate(
    df: pd.DataFrame,
    symbol: str,
    asset_type: str = "etf",
) -> pd.DataFrame:
    if asset_type not in ("etf", "lof"):
        return df
    df_out = df.copy()
    close_col = None
    for c in ["收盘", "收盘价", "close"]:
        if c in df_out.columns:
            close_col = c
            break
    if close_col is None:
        df_out["溢价率"] = 0.0
        return df_out
    try:
        nav_df = fetch_etf_nav_history(symbol)
        if not nav_df.empty and "单位净值" in nav_df.columns:
            nav = nav_df["单位净值"]
            df_out["单位净值"] = nav.reindex(df_out.index, method="ffill", tolerance=5)
            df_out["溢价率"] = np.where(
                df_out["单位净值"].notna() & (df_out["单位净值"] != 0),
                (df_out[close_col] - df_out["单位净值"]) / df_out["单位净值"] * 100,
                0.0,
            )
        else:
            df_out["溢价率"] = 0.0
    except Exception:
        df_out["溢价率"] = 0.0
    return df_out


def fetch_etf_realtime_premium(symbol: str) -> float:
    try:
        df = ak.fund_etf_spot_em()
        row = df[df["代码"] == symbol]
        if not row.empty:
            return float(row.iloc[0].get("基金折价率", 0))
    except Exception:
        pass
    return 0.0


# ══════════════════════════════════════════
#  8.  情感数据获取
# ══════════════════════════════════════════

SENTIMENT_SOURCES = {}


def _get_sentiment_sources():
    """延迟加载情感数据源."""
    global SENTIMENT_SOURCES
    if not SENTIMENT_SOURCES:
        from data.sentiment.guba import GubaSource
        from data.sentiment.news import NewsSource
        SENTIMENT_SOURCES = {
            "guba": GubaSource(),
            "news": NewsSource(),
        }
    return SENTIMENT_SOURCES


def fetch_sentiment_data(
    symbol: str,
    asset_type: str = "stock",
    start_date: str = "20000101",
    end_date: Optional[str] = None,
    use_llm: bool = False,
    llm_client=None,
) -> pd.DataFrame:
    """获取情感数据, 返回日频 DataFrame, 索引为 DatetimeIndex.

    合并所有可用情感数据源 (当前: 东方财富股吧).
    列: sentiment_score, post_volume, bull_bear_ratio, disagreement, heat_index
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")

    sources = _get_sentiment_sources()
    all_daily = []

    for src_name, src in sources.items():
        try:
            df = src.fetch(symbol, start_date, end_date,
                           use_llm=use_llm, llm_client=llm_client)
            if df is not None and not df.empty:
                all_daily.append(df)
        except Exception:
            continue

    if not all_daily:
        return pd.DataFrame()

    merged = all_daily[0]
    for df in all_daily[1:]:
        for col in df.columns:
            if col not in merged.columns:
                merged[col] = df[col]
            else:
                merged[col] = merged[col].fillna(df[col])

    return merged


clear_expired()
