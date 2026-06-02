"""情绪数据历史持久化.

存储路径: data/history/sentiment/
  stock_daily/{symbol}_{source}.parquet   — 日频聚合 (按 date 去重)
  stock_raw/{symbol}_{source}.parquet     — raw posts (按 date+text 去重)
  sector_snapshot.parquet                 — 板块快照 (按 snapshot_date+board_name 去重)
"""

from datetime import date, datetime
from pathlib import Path

import pandas as pd

HISTORY_DIR = Path(__file__).resolve().parent.parent.parent / "history" / "sentiment"
STOCK_DAILY_DIR = HISTORY_DIR / "stock_daily"
STOCK_RAW_DIR = HISTORY_DIR / "stock_raw"
SECTOR_FILE = HISTORY_DIR / "sector_snapshot.parquet"


def _ensure_dirs():
    STOCK_DAILY_DIR.mkdir(parents=True, exist_ok=True)
    STOCK_RAW_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def _write_parquet(path: Path, df: pd.DataFrame):
    df.to_parquet(path, compression="zstd", index=False)


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


# ── 个股日频聚合 ────────────────────────────────────────

def _daily_path(symbol: str, source: str) -> Path:
    return STOCK_DAILY_DIR / f"{symbol}_{source}.parquet"


def append_stock_daily(symbol: str, source: str, df: pd.DataFrame):
    """追加日频聚合数据 (按 date 去重)."""
    if df is None or df.empty:
        return
    _ensure_dirs()
    path = _daily_path(symbol, source)
    old = _read_parquet(path)
    combined = pd.concat([old, df], ignore_index=True)
    if "date" in combined.columns:
        combined = combined.drop_duplicates(subset=["date"], keep="last")
    combined = combined.sort_values("date").reset_index(drop=True)
    _write_parquet(path, combined)


def load_stock_daily(
    symbol: str, source: str,
    start: str = None, end: str = None,
) -> pd.DataFrame:
    """读取个股历史日频聚合."""
    df = _read_parquet(_daily_path(symbol, source))
    if df.empty:
        return df
    df = df.sort_values("date")
    if start:
        df = df[df["date"] >= start]
    if end:
        df = df[df["date"] <= end]
    return df.reset_index(drop=True)


# ── 个股 raw posts ──────────────────────────────────────

def _raw_path(symbol: str, source: str) -> Path:
    return STOCK_RAW_DIR / f"{symbol}_{source}.parquet"


def append_stock_raw(symbol: str, source: str, df: pd.DataFrame):
    """追加 raw posts (按 date+text 去重)."""
    if df is None or df.empty:
        return
    _ensure_dirs()
    path = _raw_path(symbol, source)
    old = _read_parquet(path)
    combined = pd.concat([old, df], ignore_index=True)
    dedup_cols = [c for c in ["date", "text"] if c in combined.columns]
    if dedup_cols:
        combined = combined.drop_duplicates(subset=dedup_cols, keep="last")
    combined = combined.sort_values("date").reset_index(drop=True)
    _write_parquet(path, combined)


def load_stock_raw(
    symbol: str, source: str,
    start: str = None, end: str = None,
    limit: int = 500,
) -> pd.DataFrame:
    """读取个股历史 raw posts."""
    df = _read_parquet(_raw_path(symbol, source))
    if df.empty:
        return df
    df = df.sort_values("date", ascending=False)
    if start:
        df = df[df["date"] >= start]
    if end:
        df = df[df["date"] <= end]
    return df.head(limit).reset_index(drop=True)


# ── 板块快照 ────────────────────────────────────────────

def append_sector_snapshot(df: pd.DataFrame):
    """追加板块快照 (按 snapshot_date+board_name 去重)."""
    if df is None or df.empty:
        return
    _ensure_dirs()
    today = date.today().isoformat()
    df = df.copy()
    df["snapshot_date"] = today

    old = _read_parquet(SECTOR_FILE)
    if not old.empty:
        common = [c for c in old.columns if c in df.columns]
        if "snapshot_date" not in common:
            common = list(old.columns)
        df = df[common] if not set(common).issuperset(df.columns) else df
    combined = pd.concat([old, df], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["snapshot_date", "board_name"], keep="last",
    )
    combined = combined.sort_values(["snapshot_date", "rank"]).reset_index(drop=True)
    _write_parquet(SECTOR_FILE, combined)


def load_sector_history(snapshot_date: str = None) -> pd.DataFrame:
    """读取板块快照.

    Args:
        snapshot_date: 如 "2025-06-02", None 返回全部.
    """
    df = _read_parquet(SECTOR_FILE)
    if df.empty:
        return df
    if snapshot_date:
        df = df[df["snapshot_date"] == snapshot_date]
    return df.reset_index(drop=True)


def list_sector_snapshot_dates() -> list[str]:
    """返回已有板块快照的日期列表 (降序)."""
    df = _read_parquet(SECTOR_FILE)
    if df.empty:
        return []
    dates = sorted(df["snapshot_date"].unique(), reverse=True)
    return list(dates)
