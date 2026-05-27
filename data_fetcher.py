"""
[已迁移] 数据获取模块

此文件保留为向后兼容的 shim, 实际实现在 data/fetcher.py。
新代码请导入 data.fetcher 或 data.cache / data.asset_config。
"""

from data.fetcher import (
    AssetType,
    fetch_history,
    fetch_stock_history,
    fetch_etf_history,
    fetch_etf_nav_history,
    fetch_lof_history,
    fetch_open_fund_nav,
    fetch_index_history,
    get_price_series,
    get_etf_list,
    get_open_fund_list,
    ensure_ohlc,
    add_premium_rate,
    fetch_etf_realtime_premium,
)

from data.asset_config import ASSET_TYPE_CONFIG
from data.cache import (
    CACHE_DIR,
    CACHE_EXPIRE_SECS,
    cache_key,
    read_cache,
    write_cache,
    filter_by_date,
    clear_expired,
)
