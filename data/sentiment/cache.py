"""情感数据 Parquet 缓存.

与 data/cache.py 同模式, 但 TTL 按数据源特性配置:
  - 股吧原始帖子: 7 天
  - 股吧日频聚合: 1 天
  - 新闻日频:     1 天
  - 词典分析结果:  30 天
"""

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "cache" / "sentiment"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TTL_MAP = {
    "guba_raw":       7 * 86400,
    "guba_aggregated": 1 * 86400,
    "news_raw":        7 * 86400,
    "news_aggregated": 1 * 86400,
    "lexicon":        30 * 86400,
    "llm":             7 * 86400,
}


def cache_key(source: str, symbol: str, *parts: str) -> str:
    raw = "_".join([source, symbol] + list(parts))
    return hashlib.md5(raw.encode()).hexdigest()


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.parquet"


def read_cache(key: str, source: str = "guba_aggregated") -> Optional[pd.DataFrame]:
    path = _cache_path(key)
    if not path.exists():
        return None
    ttl = TTL_MAP.get(source, 86400)
    age = datetime.now().timestamp() - path.stat().st_mtime
    if age > ttl:
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def write_cache(key: str, df: pd.DataFrame):
    if df is None or df.empty:
        return
    try:
        path = _cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
    except Exception:
        pass
