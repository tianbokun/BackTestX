"""
数据获取模块 - 基于 AKShare 封装
支持: A股个股, ETF基金, LOF基金, 开放式基金, 指数

特点:
  - 自动重试 + 备用数据源
  - 本地文件缓存 (按 symbol 缓存全量数据, 30 天过期)
  - 多次查询同一 symbol 不同日期区间不再重新请求
"""

import os
import json
import hashlib
import time
from typing import Optional, Literal
from datetime import datetime, date
from pathlib import Path

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import akshare as ak

AssetType = Literal["stock", "etf", "lof", "open_fund", "index"]

# ── 缓存 ──
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
CACHE_EXPIRE_SECS = 30 * 24 * 3600  # 30 天

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


# ══════════════════════════════════════════
#  缓存工具
# ══════════════════════════════════════════

def _cache_key(*parts: str) -> str:
    return hashlib.md5("_".join(parts).encode()).hexdigest()


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.parquet"


def _read_cache(key: str) -> Optional[pd.DataFrame]:
    path = _cache_path(key)
    if not path.exists():
        return None
    age = datetime.now().timestamp() - path.stat().st_mtime
    if age > CACHE_EXPIRE_SECS:
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def _write_cache(key: str, df: pd.DataFrame):
    if df.empty:
        return
    try:
        df.to_parquet(_cache_path(key))
    except Exception:
        pass


def _filter_by_date(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """从已缓存的 DataFrame 中截取日期区间, 返回副本"""
    if df.empty:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        return df
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    return df[(df.index >= start) & (df.index <= end)].copy()


def _clear_expired():
    now = datetime.now().timestamp()
    for f in CACHE_DIR.glob("*.parquet"):
        if now - f.stat().st_mtime > CACHE_EXPIRE_SECS * 2:
            f.unlink(missing_ok=True)


# ══════════════════════════════════════════
#  HTTP 请求 (tenacity 重试)
# ══════════════════════════════════════════

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


# ══════════════════════════════════════════
#  内部: 直接调用 East Money API
# ══════════════════════════════════════════

def _parse_em_klines(data_json: dict, symbol: str) -> pd.DataFrame:
    """解析东方财富 kline 返回为 DataFrame"""
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
    """通用东方财富 kline API"""
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

    ck = _cache_key("stock", symbol, period, adjust)
    cached = _read_cache(ck)
    if cached is not None:
        return _filter_by_date(cached, start_date, end_date)

    # 尝试 East Money API
    mc = 1 if symbol.startswith("6") else 0
    err = None
    for _ in range(3):
        try:
            full = _em_kline(symbol, mc, period, adjust)
            if not full.empty:
                _write_cache(ck, full)
                return _filter_by_date(full, start_date, end_date)
        except _CONNECT_ERRORS as e:
            err = e
            time.sleep(2)
            continue

    # 降级: AKShare
    try:
        df = ak.stock_zh_a_hist(symbol=symbol, period=period,
                                start_date=_validate_date(start_date),
                                end_date=_validate_date(end_date), adjust=adjust)
        if not df.empty:
            df["日期"] = pd.to_datetime(df["日期"])
            df.set_index("日期", inplace=True)
            _write_cache(ck, df)
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

    ck = _cache_key("etf", symbol, period, adjust)
    cached = _read_cache(ck)
    if cached is not None:
        return _filter_by_date(cached, start_date, end_date)

    err = None
    for _ in range(3):
        try:
            full = _em_kline(symbol, 0, period, adjust)
            if not full.empty:
                _write_cache(ck, full)
                return _filter_by_date(full, start_date, end_date)
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
            _write_cache(ck, df)
            return df
    except Exception as e:
        err = e

    raise ConnectionError(
        f"无法获取ETF {symbol} 的数据。可能是代码不存在、类型选择有误或网络波动。"
        f"({err})"
    )


def fetch_etf_nav_history(
    symbol: str,
    start_date: str = "20000101",
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")
    ck = _cache_key("etf_nav", symbol)
    cached = _read_cache(ck)
    if cached is not None:
        return _filter_by_date(cached, start_date, end_date)

    df = ak.fund_etf_fund_info_em(
        fund=symbol,
        start_date=_validate_date(_FULL_START),
        end_date=_validate_date(_FULL_END),
    )
    if not df.empty:
        df["净值日期"] = pd.to_datetime(df["净值日期"])
        df.set_index("净值日期", inplace=True)
        _write_cache(ck, df)
    return _filter_by_date(df, start_date, end_date)


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
    ck = _cache_key("lof", symbol, period, adjust)
    cached = _read_cache(ck)
    if cached is not None:
        return _filter_by_date(cached, start_date, end_date)

    err = None
    for _ in range(3):
        try:
            df = ak.fund_lof_hist_em(symbol=symbol, period=period,
                                     start_date=_validate_date(_FULL_START),
                                     end_date=_validate_date(_FULL_END), adjust=adjust)
            if not df.empty:
                df["日期"] = pd.to_datetime(df["日期"])
                df.set_index("日期", inplace=True)
                _write_cache(ck, df)
                return _filter_by_date(df, start_date, end_date)
        except _CONNECT_ERRORS as e:
            err = e
            time.sleep(2)
            continue

    raise ConnectionError(
        f"无法获取LOF {symbol} 的数据。({err})"
    )


# ══════════════════════════════════════════
#  4.  开放式基金
# ══════════════════════════════════════════

def fetch_open_fund_nav(
    symbol: str,
    indicator: str = "单位净值走势",
) -> pd.DataFrame:
    ck = _cache_key("open_fund", symbol, indicator)
    cached = _read_cache(ck)
    if cached is not None:
        return cached

    df = ak.fund_open_fund_info_em(symbol=symbol, indicator=indicator, period="成立来")
    if not df.empty:
        date_col = [c for c in df.columns if "日期" in c][0]
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
        _write_cache(ck, df)
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
    ck = _cache_key("index", symbol)
    cached = _read_cache(ck)
    if cached is not None:
        return _filter_by_date(cached, start_date, end_date)

    df = ak.stock_zh_index_daily_tx(symbol=symbol)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df = df.sort_index()
        _write_cache(ck, df)
    return _filter_by_date(df, start_date, end_date)


# ══════════════════════════════════════════
#  6.  统一入口
# ══════════════════════════════════════════

ASSET_TYPE_CONFIG = {
    "stock": {
        "label": "A股个股",
        "price_label": "收盘",
        "search_hint": "输入股票代码, 如 000001, 600519",
    },
    "etf": {
        "label": "ETF基金",
        "price_label": "收盘",
        "search_hint": "输入ETF代码, 如 510300, 513100",
    },
    "lof": {
        "label": "LOF基金",
        "price_label": "收盘",
        "search_hint": "输入LOF代码, 如 160719",
    },
    "open_fund": {
        "label": "开放式基金",
        "price_label": None,
        "search_hint": "输入基金代码, 如 110011, 000001",
    },
    "index": {
        "label": "指数",
        "price_label": "close",
        "search_hint": "输入指数代码, 如 sh000001, sh000300",
    },
}


def fetch_history(
    asset_type: AssetType,
    symbol: str,
    start_date: str = "20000101",
    end_date: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")

    fetchers = {
        "stock": fetch_stock_history,
        "etf": fetch_etf_history,
        "lof": fetch_lof_history,
        "open_fund": lambda symbol, start_date, end_date, **kw: fetch_open_fund_nav(
            symbol=symbol, indicator=kw.get("indicator", "单位净值走势")
        ),
        "index": fetch_index_history,
    }

    fetcher = fetchers.get(asset_type)
    if fetcher is None:
        raise ValueError(f"不支持的资产类型: {asset_type}")

    return fetcher(symbol=symbol, start_date=start_date, end_date=end_date, **kwargs)


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


_clear_expired()
