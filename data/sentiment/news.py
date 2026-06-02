"""财经新闻情感数据源.

从东方财富获取个股新闻, 可选 DeepSeek API 进行情感分析和基本面认知提取.
"""

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from data.sentiment.base import SentimentSource
from data.sentiment.cache import cache_key, read_cache, write_cache
from data.sentiment.lexicon import score_batch as lexicon_scores


class NewsSource(SentimentSource):
    """财经新闻情感数据源."""

    EXTRA_COLUMNS = [
        "news_sentiment",
        "news_volume",
        "news_heat",
        "analyst_sentiment",
        "report_count",
    ]

    def name(self) -> str:
        return "news"

    def _fetch_news(self, symbol: str) -> pd.DataFrame:
        """使用 AKShare 获取个股新闻."""
        try:
            import akshare as ak
            df = ak.stock_news_em(symbol=symbol)
            if df is None or df.empty:
                return pd.DataFrame()
            date_col = None
            for col in df.columns:
                if "时间" in col or "日期" in col or "date" in col.lower():
                    date_col = col
                    break
            if date_col is None:
                return pd.DataFrame()
            df["date"] = pd.to_datetime(df[date_col]).dt.date
            text_col = None
            for col in df.columns:
                if "标题" in col or "title" in col.lower():
                    text_col = col
                    break
            if text_col is None:
                text_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
            df["text"] = df[text_col].astype(str)
            read_col = None
            for col in df.columns:
                if "阅读" in col or "点击" in col or "read" in col.lower():
                    read_col = col
                    break
            df["read_count"] = pd.to_numeric(df[read_col], errors="coerce").fillna(0) if read_col else 0
            return df[["date", "text", "read_count"]]
        except Exception:
            return pd.DataFrame()

    def _lexicon_analyze(self, df_news: pd.DataFrame) -> pd.DataFrame:
        """用词典分析新闻情感."""
        if df_news.empty:
            return pd.DataFrame()
        scores = lexicon_scores(df_news["text"].tolist())
        df_news["score"] = scores
        grouped = df_news.groupby("date")
        def _agg(g):
            n = len(g)
            scores = g["score"].values
            sentiment_mean = float(np.mean(scores))
            sentiment_std = float(np.std(scores)) if n > 1 else 0.0
            bull = float(np.sum(scores > 0.05))
            bear = float(np.sum(scores < -0.05))
            bull_bear = bull / max(bear, 1)
            disagreement = sentiment_std / 2.0 if n > 1 else 0.0
            heat = int(g["read_count"].sum())
            return pd.Series({
                "news_sentiment": round(sentiment_mean, 4),
                "news_volume": n,
                "bull_bear_ratio": round(bull_bear, 4),
                "disagreement": round(min(disagreement, 1.0), 4),
                "news_heat": heat,
                "sentiment_score": round(sentiment_mean, 4),
                "post_volume": n,
                "heat_index": heat,
            })
        daily = grouped.apply(_agg)
        daily.index = pd.to_datetime(daily.index)
        return daily.sort_index()

    def fetch(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        use_llm: bool = False,
        llm_client=None,
    ) -> pd.DataFrame:
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        ck = cache_key("news", symbol, start_date, end_date)
        cached = read_cache(ck, "news_aggregated")
        if cached is not None:
            return cached

        raw_ck = cache_key("news_raw", symbol)
        raw_cached = read_cache(raw_ck, "news_raw")
        if raw_cached is not None:
            df_news = raw_cached
        else:
            df_news = self._fetch_news(symbol)
            if not df_news.empty:
                write_cache(raw_ck, df_news)

        if df_news.empty:
            return pd.DataFrame(columns=self.all_columns)

        daily = self._lexicon_analyze(df_news)
        daily = daily[(daily.index >= start) & (daily.index <= end)]

        if use_llm and llm_client is not None:
            daily = self._llm_upgrade(daily, df_news, start, end, llm_client)

        write_cache(ck, daily)
        return daily

    def _llm_upgrade(self, daily: pd.DataFrame, df_raw: pd.DataFrame,
                     start: pd.Timestamp, end: pd.Timestamp,
                     llm_client) -> pd.DataFrame:
        df_filtered = df_raw[
            (pd.to_datetime(df_raw["date"]) >= start) &
            (pd.to_datetime(df_raw["date"]) <= end)
        ]
        if df_filtered.empty:
            return daily

        top = df_filtered.nlargest(min(30, len(df_filtered)), "read_count")
        articles = [
            {"title": t, "content": t}
            for t in top["text"].tolist()
        ]

        try:
            results = llm_client.extract_fundamentals(articles)
        except Exception:
            return daily

        date_scores = {}
        for i, (_, row) in enumerate(top.iterrows()):
            d = str(row["date"])
            if d not in date_scores:
                date_scores[d] = []
            if i < len(results):
                date_scores[d].append(results[i].get("sentiment_score", 0))

        for date_str, scores in date_scores.items():
            if date_str in daily.index:
                llm_avg = float(np.mean(scores))
                daily.loc[date_str, "news_sentiment"] = round(llm_avg, 4)
                daily.loc[date_str, "sentiment_score"] = round(
                    llm_avg * 0.7 + daily.loc[date_str, "sentiment_score"] * 0.3, 4
                )
                daily.loc[date_str, "analyst_sentiment"] = round(llm_avg, 4)

        return daily
