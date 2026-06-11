"""纳斯达克100市盈率(PE TTM)数据获取

多源策略:
  1. yfinance (通过代理) - 获取当前 trailingPE, 追加到历史缓存
  2. 价格反推 - 利用 QQQ 历史价格 × 当前 EPS 估算历史 PE 序列
  3. 用户手动输入作为 final fallback

新获取的数据会追加到本地 parquet 缓存中, 避免覆盖历史。
"""

import os
import time
from datetime import datetime
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

import pandas as pd
import streamlit as st

from data.cache import cache_key, read_cache, write_cache

PE_CACHE_PREFIX = "ndx_pe_ttm"
YFINANCE_RETRIES = 2
YFINANCE_BACKOFF = 3
YFINANCE_TIMEOUT = 10


def _proxy_for_yfinance():
    """Inject proxy env vars so yfinance uses the Clash proxy."""
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    if proxy:
        os.environ["HTTP_PROXY"] = proxy
        os.environ["HTTPS_PROXY"] = proxy


def _try_yfinance_once() -> Optional[pd.DataFrame]:
    import yfinance as yf

    _proxy_for_yfinance()
    ticker = yf.Ticker("QQQ")
    info = ticker.info
    trailing_pe = info.get("trailingPE")
    if trailing_pe is None:
        return None

    today = datetime.now().strftime("%Y-%m-%d")
    df = pd.DataFrame({"date": [pd.Timestamp(today)], "pe": [float(trailing_pe)]})
    df.set_index("date", inplace=True)
    return df


def _try_yfinance_with_retry() -> Optional[pd.DataFrame]:
    for attempt in range(1, YFINANCE_RETRIES + 1):
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(_try_yfinance_once)
                return fut.result(timeout=YFINANCE_TIMEOUT)
        except FuturesTimeout:
            if attempt < YFINANCE_RETRIES:
                time.sleep(YFINANCE_BACKOFF)
            else:
                st.warning(f"yfinance 超时 (已重试{YFINANCE_RETRIES}次)")
        except Exception as e:
            if attempt < YFINANCE_RETRIES:
                time.sleep(YFINANCE_BACKOFF)
            else:
                st.warning(f"yfinance 获取失败: {e}")
    return None


def _seed_from_price_history(today_pe: float) -> Optional[pd.DataFrame]:
    """用 QQQ 历史价格 × 当前 EPS 估算历史 PE, 作为种子数据.

    原理: PE ≈ Price / EPS_TTM, EPS_TTM 变化相对缓慢,
    用当前 EPS 回推历史 PE 可得到一个可用的近似序列.
    """
    import yfinance as yf

    _proxy_for_yfinance()
    try:
        qqq = yf.download("QQQ", period="2y", auto_adjust=True, progress=False)
    except Exception:
        return None
    if qqq is None or qqq.empty:
        return None

    # yfinance Ticker.download() returns MultiIndex columns: ("Close","QQQ"), etc.
    if isinstance(qqq.columns, pd.MultiIndex):
        qqq.columns = qqq.columns.droplevel(1)
    close_col = "Close" if "Close" in qqq.columns else "Adj Close"
    if close_col not in qqq.columns:
        return None

    prices = qqq[close_col].dropna()
    if len(prices) < 2:
        return None

    current_price = float(prices.iloc[-1])
    eps = current_price / today_pe
    if eps <= 0:
        return None

    pe_series = prices / eps
    records = []
    for dt, val in pe_series.items():
        dt_str = pd.Timestamp(dt).strftime("%Y-%m-%d")
        records.append({"date": dt_str, "pe": round(float(val), 2)})

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df = df[~df.index.duplicated(keep="last")]
    return df


def _merge_and_cache(ck: str, new_df: pd.DataFrame):
    existing = read_cache(ck)
    if existing is not None and not existing.empty:
        combined = pd.concat([existing, new_df])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        write_cache(ck, combined)
    else:
        write_cache(ck, new_df)


def fetch_ndx_pe(force_refresh: bool = False) -> Optional[pd.DataFrame]:
    """获取纳斯达克100 PE(TTM) 数据, 返回 date|pe 的 DataFrame (date 为 index).

    策略:
      1. 非强制刷新时优先读缓存
      2. yfinance 获取今日 PE → 追加到缓存
      3. 首次获取时自动用历史价格反推种子数据
      4. 以上均失败则返回 None
    """
    ck = cache_key(PE_CACHE_PREFIX)

    if not force_refresh:
        cached = read_cache(ck)
        if cached is not None and not cached.empty:
            return cached

    today_df = _try_yfinance_with_retry()
    if today_df is not None and not today_df.empty:
        today_pe = float(today_df["pe"].iloc[-1])

        existing = read_cache(ck)
        if existing is None or existing.empty:
            seed = _seed_from_price_history(today_pe)
            if seed is not None and not seed.empty:
                _merge_and_cache(ck, seed)

        _merge_and_cache(ck, today_df)
        return read_cache(ck)

    return None


def get_latest_pe(force_refresh: bool = False) -> Optional[float]:
    df = fetch_ndx_pe(force_refresh=force_refresh)
    if df is None or df.empty:
        return None
    return float(df["pe"].iloc[-1])


def get_pe_percentile(
    pe_value: Optional[float] = None, force_refresh: bool = False
) -> Optional[dict]:
    df = fetch_ndx_pe(force_refresh=force_refresh)
    if df is None or df.empty:
        return None

    if pe_value is None:
        pe_value = float(df["pe"].iloc[-1])

    series = df["pe"].dropna()
    if len(series) < 2:
        return {
            "pe": pe_value,
            "pct": None,
            "min": float(series.min()) if len(series) > 0 else pe_value,
            "max": float(series.max()) if len(series) > 0 else pe_value,
            "count": len(series),
        }

    pct = (series < pe_value).sum() / len(series) * 100
    return {
        "pe": pe_value,
        "pct": round(pct, 1),
        "min": float(series.min()),
        "max": float(series.max()),
        "count": len(series),
    }


def manual_set_pe(pe_value: float) -> pd.DataFrame:
    ck = cache_key(PE_CACHE_PREFIX)
    today = datetime.now().strftime("%Y-%m-%d")
    df = pd.DataFrame({"date": [pd.Timestamp(today)], "pe": [pe_value]})
    df.set_index("date", inplace=True)

    existing = read_cache(ck)
    if existing is not None and not existing.empty:
        combined = pd.concat([existing, df])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        write_cache(ck, combined)
    else:
        write_cache(ck, df)
    return df
