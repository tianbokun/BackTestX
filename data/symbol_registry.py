import json
from pathlib import Path
from typing import Optional
from datetime import date

REGISTRY_PATH = Path(__file__).parent / "symbol_registry.json"

_DEFAULT_SYMBOLS = [
    {"symbol": "515790", "name": "光伏ETF", "asset_type": "etf", "tags": ["主题ETF", "新能源"]},
    {"symbol": "159790", "name": "碳中和ETF", "asset_type": "etf", "tags": ["主题ETF", "新能源"]},
    {"symbol": "512660", "name": "军工ETF", "asset_type": "etf", "tags": ["主题ETF"]},
    {"symbol": "588000", "name": "科创50ETF", "asset_type": "etf", "tags": ["宽基ETF"]},
    {"symbol": "515070", "name": "AI智能ETF", "asset_type": "etf", "tags": ["主题ETF", "科技"]},
    {"symbol": "515030", "name": "新能源车ETF", "asset_type": "etf", "tags": ["主题ETF", "新能源"]},
    {"symbol": "512480", "name": "半导体ETF", "asset_type": "etf", "tags": ["主题ETF", "科技"]},
    {"symbol": "159995", "name": "芯片ETF", "asset_type": "etf", "tags": ["主题ETF", "科技"]},
    {"symbol": "515050", "name": "5GETF", "asset_type": "etf", "tags": ["主题ETF", "科技"]},
    {"symbol": "159819", "name": "人工智能ETF", "asset_type": "etf", "tags": ["主题ETF", "科技"]},
    {"symbol": "515900", "name": "央企创新ETF", "asset_type": "etf", "tags": ["主题ETF"]},
    {"symbol": "512010", "name": "医药ETF", "asset_type": "etf", "tags": ["主题ETF"]},
    {"symbol": "512580", "name": "环保ETF", "asset_type": "etf", "tags": ["主题ETF", "新能源"]},
    {"symbol": "510880", "name": "红利ETF", "asset_type": "etf", "tags": ["宽基ETF"]},
    {"symbol": "159915", "name": "创业板ETF", "asset_type": "etf", "tags": ["宽基ETF"]},
]


def _load() -> dict:
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"version": 1, "symbols": []}


def _save(registry: dict):
    registry["version"] = 1
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)


def migrate():
    registry = _load()
    if registry["symbols"]:
        return 0
    for ds in _DEFAULT_SYMBOLS:
        registry["symbols"].append({
            "symbol": ds["symbol"],
            "name": ds["name"],
            "asset_type": ds["asset_type"],
            "start_date": "20200101",
            "tags": list(ds["tags"]),
            "notes": "",
        })
    _save(registry)
    return len(_DEFAULT_SYMBOLS)


class SymbolRegistry:

    @staticmethod
    def list(asset_type: Optional[str] = None, tag: Optional[str] = None) -> list[dict]:
        registry = _load()
        symbols = registry["symbols"]
        if asset_type:
            symbols = [s for s in symbols if s["asset_type"] == asset_type]
        if tag:
            symbols = [s for s in symbols if tag in s.get("tags", [])]
        return list(symbols)

    @staticmethod
    def get(symbol: str) -> Optional[dict]:
        registry = _load()
        for s in registry["symbols"]:
            if s["symbol"] == symbol:
                return dict(s)
        return None

    @staticmethod
    def add(symbol: str, name: str, asset_type: str,
            start_date: str = "20200101", tags: Optional[list] = None,
            notes: str = "") -> bool:
        registry = _load()
        if any(s["symbol"] == symbol for s in registry["symbols"]):
            return False
        registry["symbols"].append({
            "symbol": symbol,
            "name": name,
            "asset_type": asset_type,
            "start_date": start_date,
            "tags": tags or [],
            "notes": notes,
        })
        _save(registry)
        return True

    @staticmethod
    def update(symbol: str, **kwargs) -> bool:
        registry = _load()
        for s in registry["symbols"]:
            if s["symbol"] == symbol:
                allowed = {"name", "asset_type", "start_date", "tags", "notes"}
                for k, v in kwargs.items():
                    if k in allowed:
                        s[k] = v
                _save(registry)
                return True
        return False

    @staticmethod
    def remove(symbol: str) -> bool:
        registry = _load()
        before = len(registry["symbols"])
        registry["symbols"] = [s for s in registry["symbols"] if s["symbol"] != symbol]
        if len(registry["symbols"]) < before:
            _save(registry)
            return True
        return False

    @staticmethod
    def fetch_data(symbol: str, adjust: str = "qfq") -> Optional["pd.DataFrame"]:
        import pandas as pd
        from data.cache import cache_key, read_cache
        from data_fetcher import fetch_history

        entry = SymbolRegistry.get(symbol)
        if entry is None:
            return None

        asset_type = entry["asset_type"]
        start_date = entry.get("start_date", "20200101")
        today_str = date.today().strftime("%Y%m%d")

        ck = cache_key(asset_type, symbol, "daily", adjust)
        cached = read_cache(ck)
        if cached is not None and not cached.empty:
            idx = cached.index
            cached_start = str(idx.min().date()).replace("-", "")
            cached_end = str(idx.max().date()).replace("-", "")
            if cached_start <= start_date and cached_end >= today_str:
                return cached.loc[start_date:today_str] if start_date in idx else cached

        df = fetch_history(
            asset_type=asset_type, symbol=symbol,
            start_date=start_date, end_date=today_str, adjust=adjust,
        )
        return df

    @staticmethod
    def get_cache_info(symbol: str) -> dict:
        from data.cache import cache_key, read_cache
        entry = SymbolRegistry.get(symbol)
        if entry is None:
            return {"status": "not_found"}
        ck = cache_key(entry["asset_type"], symbol, "daily", entry.get("adjust", "qfq"))
        cached = read_cache(ck)
        if cached is None or cached.empty:
            return {"status": "no_cache"}
        idx = cached.index
        return {
            "status": "cached",
            "start": str(idx.min().date()),
            "end": str(idx.max().date()),
            "rows": len(cached),
        }
