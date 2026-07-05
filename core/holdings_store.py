"""持仓分析持久化存储

每次上传的持仓文件 + 解析结果按时间戳保存到 .cache/holdings/。
保留全部历史记录，无自动清理。
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

HOLDINGS_DIR = Path(__file__).parent.parent / ".cache" / "holdings"
META_SUFFIX = "_meta.json"
DATA_SUFFIX = "_data.parquet"


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _meta_path(ts: str) -> Path:
    return HOLDINGS_DIR / f"{ts}{META_SUFFIX}"


def _data_path(ts: str) -> Path:
    return HOLDINGS_DIR / f"{ts}{DATA_SUFFIX}"


def save_record(df: pd.DataFrame, filename: str) -> str:
    """保存一次持仓分析结果，返回记录的时间戳标识。"""
    HOLDINGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = _ts()

    total = float(df["资产情况"].sum())
    grp = df.groupby("分类")["资产情况"].sum()
    meta = {
        "timestamp": ts,
        "filename": filename,
        "total_asset": total,
        "categories": {k: float(v) for k, v in grp.items()},
    }
    _meta_path(ts).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    df.to_parquet(_data_path(ts))
    return ts


def load_record(ts: str) -> Optional[pd.DataFrame]:
    """加载指定时间戳的持仓数据。"""
    p = _data_path(ts)
    if not p.exists():
        return None
    return pd.read_parquet(p)


def load_meta(ts: str) -> Optional[dict]:
    """加载指定时间戳的元数据。"""
    p = _meta_path(ts)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def list_records() -> list[dict]:
    """返回所有历史记录元数据（按时间倒序）。"""
    if not HOLDINGS_DIR.exists():
        return []
    metas = []
    for f in sorted(HOLDINGS_DIR.glob(f"*{META_SUFFIX}"), reverse=True):
        try:
            metas.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            continue
    return metas
