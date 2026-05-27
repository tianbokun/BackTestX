"""本地 Parquet 缓存

按 symbol 缓存全量数据, 30 天过期。
多次查询同一 symbol 不同日期区间不再重新请求。
"""

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

CACHE_DIR = Path(__file__).parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
CACHE_EXPIRE_SECS = 30 * 24 * 3600  # 30 天


def cache_key(*parts: str) -> str:
    return hashlib.md5("_".join(parts).encode()).hexdigest()


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.parquet"


def read_cache(key: str) -> Optional[pd.DataFrame]:
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


def write_cache(key: str, df: pd.DataFrame):
    if df.empty:
        return
    try:
        df.to_parquet(_cache_path(key))
    except Exception:
        pass


def filter_by_date(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """从已缓存的 DataFrame 中截取日期区间, 返回副本"""
    if df.empty:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        return df
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    return df[(df.index >= start) & (df.index <= end)].copy()


def clear_expired():
    now = datetime.now().timestamp()
    for f in CACHE_DIR.glob("*.parquet"):
        if now - f.stat().st_mtime > CACHE_EXPIRE_SECS * 2:
            f.unlink(missing_ok=True)
