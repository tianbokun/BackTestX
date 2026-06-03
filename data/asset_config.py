"""资产类型配置

定义受支持的资产类型及其显示标签、价格列名等元数据。
"""

ASSET_TYPE_CONFIG = {
    "stock": {
        "label": "A股个股",
        "label_en": "A-Share Stock",
        "price_label": "收盘",
        "search_hint": "输入股票代码, 如 000001, 600519",
        "search_hint_en": "Enter stock code, e.g. 000001, 600519",
    },
    "etf": {
        "label": "ETF基金",
        "label_en": "ETF Fund",
        "price_label": "收盘",
        "search_hint": "输入ETF代码, 如 510300, 513100",
        "search_hint_en": "Enter ETF code, e.g. 510300, 513100",
    },
    "lof": {
        "label": "LOF基金",
        "label_en": "LOF Fund",
        "price_label": "收盘",
        "search_hint": "输入LOF代码, 如 160719",
        "search_hint_en": "Enter LOF code, e.g. 160719",
    },
    "open_fund": {
        "label": "开放式基金",
        "label_en": "Open-End Fund",
        "price_label": None,
        "search_hint": "输入基金代码, 如 110011, 000001",
        "search_hint_en": "Enter fund code, e.g. 110011, 000001",
    },
    "index": {
        "label": "指数",
        "label_en": "Index",
        "price_label": "close",
        "search_hint": "输入指数代码, 如 sh000001, sh000300",
        "search_hint_en": "Enter index code, e.g. sh000001, sh000300",
    },
    "us": {
        "label": "境外资产 (美股/ETF/商品)",
        "label_en": "Overseas Assets (US Stocks/ETF/Commodities)",
        "price_label": "Close",
        "search_hint": "输入 Yahoo Finance 代码, 如 QQQ, TQQQ, GLD, GC=F",
        "search_hint_en": "Enter Yahoo Finance code, e.g. QQQ, TQQQ, GLD, GC=F",
    },
}
