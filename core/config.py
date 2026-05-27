"""共享配置: 费用参数 / 频率映射 / 日均乘数"""

# ── 费用默认值 ──
DEFAULT_COMMISSION_RATE = 0.00025   # 佣金率 (0.025%)
DEFAULT_MIN_COMMISSION = 5.0        # 最低佣金 (元)
DEFAULT_STAMP_DUTY = 0.001          # 印花税率 (0.1%)

# ── 日均乘数: amount 理解为"日均投入", 各频率按交易日天数放大 ──
DAILY_MULTIPLIER = {
    "daily": 1,
    "weekly": 5,
    "biweekly": 10,
    "monthly": 22,
    "quarterly": 66,
    "yearly": 252,
}

# ── 频率标签映射 ──
freq_map = {
    "daily": "每日",
    "weekly": "每周",
    "biweekly": "每两周",
    "monthly": "每月",
    "quarterly": "每季度",
    "yearly": "每年",
}
