"""
数据获取模块 - 基于 AKShare 封装
支持: A股个股, ETF基金, LOF基金, 开放式基金, 指数

特点:
  - 自动重试 (连接断开时指数退避, 最多 5 次)
  - 本地文件缓存 (每日刷新, 减少重复请求)
  - 备用数据源 (主源失败时自动切换)
"""

import os
import json
import hashlib
from typing import Optional, Literal
from datetime import datetime, date
from pathlib import Path

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import akshare as ak

# ── 资产类型 ──
AssetType = Literal["stock", "etf", "lof", "open_fund", "index"]

# ── 缓存配置 ──
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
CACHE_EXPIRE_SECS = 12 * 3600  # 12 小时

# ── HTTP 会话 (模拟浏览器, 避免被反爬) ──
_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Referer": "https://quote.eastmoney.com/",
    "Connection": "keep-alive",
})

_CONNECT_ERRORS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)


# ── 工具函数 ──

def _validate_date(date_str: str) -> str:
    if len(date_str) != 8:
        raise ValueError(f"日期格式错误, 应为 YYYYMMDD, 实际: {date_str}")
    return date_str


def _cache_key(*parts: str) -> str:
    raw = "_".join(parts)
    return hashlib.md5(raw.encode()).hexdigest()


def _cache_path(cache_key: str) -> Path:
    return CACHE_DIR / f"{cache_key}.csv"


def _read_cache(cache_key: str) -> Optional[pd.DataFrame]:
    path = _cache_path(cache_key)
    if not path.exists():
        return None
    age = datetime.now().timestamp() - path.stat().st_mtime
    if age > CACHE_EXPIRE_SECS:
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df
    except Exception:
        return None


def _write_cache(cache_key: str, df: pd.DataFrame):
    if df.empty:
        return
    try:
        df.to_csv(_cache_path(cache_key))
    except Exception:
        pass


def _clear_expired_cache():
    now = datetime.now().timestamp()
    for f in CACHE_DIR.glob("*.csv"):
        if now - f.stat().st_mtime > CACHE_EXPIRE_SECS * 2:
            f.unlink(missing_ok=True)


def _build_retry_decorator():
    return retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(_CONNECT_ERRORS),
        reraise=True,
    )


@_build_retry_decorator()
def _request_json(url: str, params: dict) -> dict:
    """带重试和超时的 JSON API 请求"""
    r = _SESSION.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


# ══════════════════════════════════════════
#  内部: 直接调用 East Money API (绕过 AKShare)
#  原因: AKShare 没有 retry/custom headers, 连接不稳时会直接抛异常
# ══════════════════════════════════════════

def _em_stock_hist(
    symbol: str,
    period: str = "daily",
    start_date: str = "20000101",
    end_date: str = "20500101",
    adjust: str = "qfq",
) -> pd.DataFrame:
    """
    直接调用东方财富 API 获取 A 股历史行情
    比 AKShare 的 stock_zh_a_hist 更稳定 (自定义 headers + retry)
    """
    market_code = 1 if symbol.startswith("6") else 0
    adjust_map = {"qfq": "1", "hfq": "2", "": "0"}
    period_map = {"daily": "101", "weekly": "102", "monthly": "103"}

    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
        "klt": period_map.get(period, "101"),
        "fqt": adjust_map.get(adjust, "0"),
        "secid": f"{market_code}.{symbol}",
        "beg": start_date,
        "end": end_date,
    }

    data_json = _request_json(url, params)
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


def _em_etf_hist(
    symbol: str,
    period: str = "daily",
    start_date: str = "20000101",
    end_date: str = "20500101",
    adjust: str = "qfq",
) -> pd.DataFrame:
    """东方财富 ETF 历史行情"""
    params = {
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
    }
    period_map = {"daily": "101", "weekly": "102", "monthly": "103"}
    adjust_map = {"qfq": "1", "hfq": "2", "": "0"}
    params["klt"] = period_map.get(period, "101")
    params["fqt"] = adjust_map.get(adjust, "0")
    params["secid"] = f"0.{symbol}"
    params["beg"] = start_date
    params["end"] = end_date

    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    data_json = _request_json(url, params)
    if not (data_json.get("data") and data_json["data"].get("klines")):
        return pd.DataFrame()

    rows = [item.split(",") for item in data_json["data"]["klines"]]
    cols = ["日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅", "涨跌幅", "涨跌额", "换手率"]
    df = pd.DataFrame(rows, columns=cols[:len(rows[0])])
    for col in df.columns:
        if col == "日期":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["日期"] = pd.to_datetime(df["日期"])
    df.set_index("日期", inplace=True)
    return df


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
    ck = _cache_key("stock", symbol, period, adjust, start_date, end_date)
    cached = _read_cache(ck)
    if cached is not None and not cached.empty:
        return cached

    exc = None
    for attempt in range(3):
        try:
            df = _em_stock_hist(symbol, period, start_date, end_date, adjust)
            if not df.empty:
                _write_cache(ck, df)
                return df
        except _CONNECT_ERRORS as e:
            exc = e
            import time
            time.sleep(2 ** attempt)
            continue

    try:
        df = ak.stock_zh_a_hist(symbol=symbol, period=period,
                                start_date=_validate_date(start_date),
                                end_date=_validate_date(end_date), adjust=adjust)
        if not df.empty:
            df["日期"] = pd.to_datetime(df["日期"])
            df.set_index("日期", inplace=True)
            _write_cache(ck, df)
            return df
    except Exception as e2:
        exc = e2

    raise ConnectionError(
        f"无法获取股票 {symbol} 的历史数据。"
        f"请检查网络连接或稍后重试。"
        f"({exc})"
    ) from exc


# ══════════════════════════════════════════
#  2.  ETF 基金 (场内交易价格)
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
    ck = _cache_key("etf", symbol, period, adjust, start_date, end_date)
    cached = _read_cache(ck)
    if cached is not None and not cached.empty:
        return cached

    exc = None
    for attempt in range(3):
        try:
            df = _em_etf_hist(symbol, period, start_date, end_date, adjust)
            if not df.empty:
                _write_cache(ck, df)
                return df
        except _CONNECT_ERRORS as e:
            exc = e
            import time
            time.sleep(2 ** attempt)
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
    except Exception as e2:
        exc = e2

    raise ConnectionError(
        f"无法获取ETF {symbol} 的历史数据。"
        f"请检查网络连接或稍后重试。"
        f"({exc})"
    ) from exc


def fetch_etf_nav_history(
    symbol: str,
    start_date: str = "20000101",
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")
    ck = _cache_key("etf_nav", symbol, start_date, end_date)
    cached = _read_cache(ck)
    if cached is not None:
        return cached
    df = ak.fund_etf_fund_info_em(
        fund=symbol,
        start_date=_validate_date(start_date),
        end_date=_validate_date(end_date),
    )
    if not df.empty:
        df["净值日期"] = pd.to_datetime(df["净值日期"])
        df.set_index("净值日期", inplace=True)
        _write_cache(ck, df)
    return df


# ══════════════════════════════════════════
#  3.  LOF 基金 (场内交易价格)
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
    ck = _cache_key("lof", symbol, period, adjust, start_date, end_date)
    cached = _read_cache(ck)
    if cached is not None and not cached.empty:
        return cached

    exc = None
    for attempt in range(3):
        try:
            df = ak.fund_lof_hist_em(symbol=symbol, period=period,
                                     start_date=_validate_date(start_date),
                                     end_date=_validate_date(end_date), adjust=adjust)
            if not df.empty:
                df["日期"] = pd.to_datetime(df["日期"])
                df.set_index("日期", inplace=True)
                _write_cache(ck, df)
                return df
        except _CONNECT_ERRORS as e:
            exc = e
            import time
            time.sleep(2 ** attempt)
            continue

    raise ConnectionError(
        f"无法获取LOF {symbol} 的历史数据。"
        f"请检查网络连接或稍后重试。"
        f"({exc})"
    ) from exc


# ══════════════════════════════════════════
#  4.  开放式基金 (场外基金净值)
# ══════════════════════════════════════════

def fetch_open_fund_nav(
    symbol: str,
    indicator: str = "单位净值走势",
) -> pd.DataFrame:
    ck = _cache_key("open_fund", symbol, indicator)
    cached = _read_cache(ck)
    if cached is not None and not cached.empty:
        return cached

    df = ak.fund_open_fund_info_em(symbol=symbol, indicator=indicator, period="成立来")
    if df.empty:
        return df
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
    ck = _cache_key("index", symbol, start_date, end_date)
    cached = _read_cache(ck)
    if cached is not None and not cached.empty:
        return cached

    df = ak.stock_zh_index_daily_tx(symbol=symbol)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df = df.sort_index()
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    df = df[(df.index >= start_ts) & (df.index <= end_ts)]
    _write_cache(ck, df)
    return df


# ══════════════════════════════════════════
#  6.  统一入口
# ══════════════════════════════════════════

ASSET_TYPE_CONFIG = {
    "stock": {
        "label": "A股个股",
        "price_label": "收盘",
        "nav_label": None,
        "search_hint": "输入股票代码, 如 000001, 600519",
        "suffix": "",
    },
    "etf": {
        "label": "ETF基金",
        "price_label": "收盘",
        "nav_label": "单位净值",
        "search_hint": "输入ETF代码, 如 510300, 513100",
        "suffix": "",
    },
    "lof": {
        "label": "LOF基金",
        "price_label": "收盘",
        "nav_label": "单位净值",
        "search_hint": "输入LOF代码, 如 160719",
        "suffix": "",
    },
    "open_fund": {
        "label": "开放式基金",
        "price_label": None,
        "nav_label": "单位净值",
        "search_hint": "输入基金代码, 如 110011, 000001",
        "suffix": "",
    },
    "index": {
        "label": "指数",
        "price_label": "close",
        "nav_label": None,
        "search_hint": "输入指数代码, 如 sh000001, sh000300",
        "suffix": "",
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
            prices = df[col]
            prices = prices.sort_index()
            return prices
    return None


# ══════════════════════════════════════════
#  7.  基金列表查询
# ══════════════════════════════════════════

def get_etf_list() -> pd.DataFrame:
    df = ak.fund_etf_fund_daily_em()
    if df.empty:
        return df
    return df[["基金代码", "基金简称"]]


def get_open_fund_list() -> pd.DataFrame:
    df = ak.fund_open_fund_daily_em()
    if df.empty:
        return df
    return df[["基金代码", "基金简称"]]


# 启动时清理过期缓存
_clear_expired_cache()
