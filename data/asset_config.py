"""资产类型配置

定义受支持的资产类型及其显示标签、价格列名等元数据。
"""

ASSET_TYPE_CONFIG = {
    "stock": {
        "label": "A股个股",
        "price_label": "收盘",
        "search_hint": "输入股票代码, 如 000001, 600519",
    },
    "etf": {
        "label": "ETF基金",
        "price_label": "收盘",
        "search_hint": "输入ETF代码, 如 510300, 513100",
    },
    "lof": {
        "label": "LOF基金",
        "price_label": "收盘",
        "search_hint": "输入LOF代码, 如 160719",
    },
    "open_fund": {
        "label": "开放式基金",
        "price_label": None,
        "search_hint": "输入基金代码, 如 110011, 000001",
    },
    "index": {
        "label": "指数",
        "price_label": "close",
        "search_hint": "输入指数代码, 如 sh000001, sh000300",
    },
}
