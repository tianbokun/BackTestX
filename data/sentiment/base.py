from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class SentimentSource(ABC):
    """情感数据源基类.

    所有子类输出统一的日频 DataFrame, 包含以下列:
      - sentiment_score: 日平均情感得分 (-1~1, >0 偏多)
      - post_volume: 当日帖子/新闻数量
      - bull_bear_ratio: 看多/看空比值 (0~∞, >1 偏多)
      - disagreement: 分歧度 (0~1, 越高分歧越大)
      - heat_index: 热度指数 (阅读/评论加权)

    子类可扩展额外的列, 但须在 EXTRA_COLUMNS 中声明.
    """

    BASE_COLUMNS = [
        "sentiment_score",
        "post_volume",
        "bull_bear_ratio",
        "disagreement",
        "heat_index",
    ]

    EXTRA_COLUMNS: list[str] = []

    @property
    def all_columns(self) -> list[str]:
        return self.BASE_COLUMNS + self.EXTRA_COLUMNS

    @abstractmethod
    def fetch(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        use_llm: bool = False,
        llm_client=None,
    ) -> pd.DataFrame:
        """获取情感数据, 返回日频 DataFrame, 索引为 DatetimeIndex."""

    @abstractmethod
    def name(self) -> str:
        """数据源名称 (如 'guba', 'news')."""
