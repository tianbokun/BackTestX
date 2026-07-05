import json
from pathlib import Path
from typing import Optional, Iterable
from datetime import date

REGISTRY_PATH = Path(__file__).parent / "symbol_registry.json"

CACHE_KEY_PATTERNS: dict[str, list[tuple[str, ...]]] = {
    "stock": [
        ("stock", "{sym}", period, adj)
        for period in ("daily", "weekly", "monthly")
        for adj in ("", "qfq", "hfq")
    ] + [
        ("open_fund", "{sym}", "单位净值走势"),
        ("open_fund", "{sym}", "单位净值走势", "qfq"),
    ],
    "etf": [
        ("etf", "{sym}", period, adj)
        for period in ("daily", "weekly", "monthly")
        for adj in ("", "qfq", "hfq")
    ] + [
        ("open_fund", "{sym}", "单位净值走势"),
        ("open_fund", "{sym}", "单位净值走势", "qfq"),
    ],
    "lof": [
        ("lof", "{sym}", period, adj)
        for period in ("daily", "weekly", "monthly")
        for adj in ("", "qfq", "hfq")
    ] + [
        ("open_fund", "{sym}", "单位净值走势"),
        ("open_fund", "{sym}", "单位净值走势", "qfq"),
    ],
    "open_fund": [
        ("open_fund", "{sym}", ind, adj)
        for ind in ("单位净值走势", "累计净值走势")
        for adj in ("", "qfq")
    ] + [
        ("open_fund", "{sym}", "daily"),
        ("open_fund", "{sym}", "daily", "qfq"),
    ],
    "index": [
        ("index", "{sym}"),
    ],
    "us": [
        ("us", "{sym}", period)
        for period in ("1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
    ],
}


def _clear_symbol_cache(symbol: str, asset_type: str):
    from data.cache import cache_key, CACHE_DIR

    patterns = CACHE_KEY_PATTERNS.get(asset_type, [])
    seen: set[str] = set()
    for parts_tmpl in patterns:
        parts = tuple(p.replace("{sym}", symbol) for p in parts_tmpl)
        key = cache_key(*parts)
        if key in seen:
            continue
        seen.add(key)
        fpath = CACHE_DIR / f"{key}.parquet"
        if fpath.exists():
            fpath.unlink()


def _load() -> dict:
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"version": 1, "symbols": []}


def _save(registry: dict):
    registry["version"] = 1
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)


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
        entry = SymbolRegistry.get(symbol)
        before = len(registry["symbols"])
        registry["symbols"] = [s for s in registry["symbols"] if s["symbol"] != symbol]
        if len(registry["symbols"]) < before:
            _save(registry)
            if entry is not None:
                _clear_symbol_cache(symbol, entry["asset_type"])
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
    def autofetch_meta(symbol: str, asset_type: str) -> dict:
        import pandas as pd
        from data.fetcher import fetch_history, get_etf_list, get_open_fund_list

        if not symbol.strip():
            raise ValueError("请输入代码")
        df = fetch_history(asset_type=asset_type, symbol=symbol.strip(), start_date="19900101")
        if df is None or df.empty:
            raise ValueError(f"无法获取 {symbol} 的数据，请检查代码和资产类型")
        start_date = str(df.index.min().date()).replace("-", "")

        name = None
        if asset_type in ("etf", "lof"):
            try:
                lst = get_etf_list()
                if lst is not None and not lst.empty:
                    row = lst[lst["基金代码"] == symbol.strip()]
                    if not row.empty:
                        name = str(row.iloc[0]["基金简称"])
            except Exception:
                pass
        elif asset_type == "open_fund":
            try:
                lst = get_open_fund_list()
                if lst is not None and not lst.empty:
                    row = lst[lst["基金代码"] == symbol.strip()]
                    if not row.empty:
                        name = str(row.iloc[0]["基金简称"])
            except Exception:
                pass
        elif asset_type == "stock":
            try:
                import akshare as ak
                info = ak.stock_individual_info_em(symbol=symbol.strip())
                if not info.empty:
                    row = info[info["item"] == "股票名称"]
                    if not row.empty:
                        name = str(row.iloc[0]["value"])
            except Exception:
                pass
        elif asset_type == "us":
            try:
                import yfinance as yf
                ticker = yf.Ticker(symbol.strip())
                info = ticker.info
                name = info.get("shortName") or info.get("longName") or symbol.strip()
            except Exception:
                pass
        if not name:
            name = symbol.strip()

        return {"name": name, "start_date": start_date}

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
