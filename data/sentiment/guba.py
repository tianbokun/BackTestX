"""东方财富股吧情感数据源.

从东方财富股吧获取个股讨论帖子, 使用情感词典 (可选的 LLM) 分析情感,
聚合成日频情感指标.

数据源优先级:
  1. 东方财富股吧 JSON API (直接 HTTP)
  2. AKShare 微博舆情报告 (fallback, 全市场情绪)
  3. AKShare 东方财富个股新闻 (fallback, 仅新闻)
"""

import re
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests

from data.sentiment.base import SentimentSource
from data.sentiment.cache import cache_key, read_cache, write_cache
from data.sentiment.lexicon import analyze_batch

# ── 东方财富股吧 API ──
GUBA_API = "https://guba.eastmoney.com/interface/GetList"

_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Referer": "https://guba.eastmoney.com/",
})

# A 股 market code: 1 for 上海 (6xx), 0 for 深圳 (0/3xx)
def _market_code(symbol: str) -> int:
    return 1 if symbol.startswith("6") else 0


def _fetch_guba_posts(symbol: str, page: int = 1, page_size: int = 50) -> list[dict]:
    """调用东方财富股吧 API 获取帖子列表."""
    params = {
        "type": "1",
        "code": symbol,
        "page": str(page),
        "pageSize": str(page_size),
        "sort": "1",
    }
    try:
        r = _SESSION.get(GUBA_API, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []

    raw_list = data.get("data", {}).get("list", []) if isinstance(data, dict) else []
    posts = []
    for item in raw_list:
        title = item.get("post_title", "") or item.get("title", "")
        content = item.get("post_content", "") or item.get("content", "")
        post_date = item.get("post_date", "") or item.get("date", "")
        text = f"{title} {content}".strip()
        if not text:
            continue
        try:
            dt = pd.to_datetime(post_date)
        except Exception:
            continue
        posts.append({
            "text": text,
            "date": dt,
            "read_count": int(item.get("read_count", 0) or item.get("click_count", 0)),
            "comment_count": int(item.get("comment_count", 0) or item.get("reply_count", 0)),
            "user_id": item.get("user_id", "") or item.get("uid", ""),
        })
    return posts


def _fetch_guba_pages(symbol: str, max_pages: int = 5) -> list[dict]:
    """翻页获取帖子."""
    all_posts = []
    for p in range(1, max_pages + 1):
        posts = _fetch_guba_posts(symbol, page=p)
        if not posts:
            break
        all_posts.extend(posts)
    return all_posts


def _aggregate_daily(posts: list[dict]) -> pd.DataFrame:
    """将帖子聚合成日频情感指标."""
    if not posts:
        return pd.DataFrame()

    df = pd.DataFrame(posts)

    # 词典情感分析
    scores = analyze_batch(df["text"].tolist())
    df["score"] = [s["score"] for s in scores]
    df["confidence"] = [s["confidence"] for s in scores]

    df["date"] = pd.to_datetime(df["date"]).dt.date

    grouped = df.groupby("date")

    def _agg(g):
        n = len(g)
        scores = g["score"].values
        sentiment_mean = float(np.mean(scores))
        sentiment_std = float(np.std(scores)) if n > 1 else 0.0

        bull = float(np.sum(scores > 0.05))
        bear = float(np.sum(scores < -0.05))
        bull_bear = bull / max(bear, 1)

        disagreement = sentiment_std / 2.0 if n > 1 else 0.0
        read_sum = int(g["read_count"].sum())
        comment_sum = int(g["comment_count"].sum())
        heat = read_sum + comment_sum * 3

        return pd.Series({
            "sentiment_score": round(sentiment_mean, 4),
            "post_volume": n,
            "bull_bear_ratio": round(bull_bear, 4),
            "disagreement": round(min(disagreement, 1.0), 4),
            "heat_index": heat,
            "avg_confidence": round(float(np.mean(g["confidence"])), 4),
        })

    daily = grouped.apply(_agg)
    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()
    return daily


class GubaSource(SentimentSource):
    """东方财富股吧情感数据源."""

    EXTRA_COLUMNS = ["avg_confidence"]

    def name(self) -> str:
        return "guba"

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

        ck = cache_key("guba", symbol, start_date, end_date)
        cached = read_cache(ck, "guba_aggregated")
        if cached is not None:
            return cached

        raw_ck = cache_key("guba_raw", symbol)
        raw_cached = read_cache(raw_ck, "guba_raw")

        if raw_cached is not None:
            posts = raw_cached.to_dict("records")
        else:
            posts = _fetch_guba_pages(symbol, max_pages=5)
            if posts:
                raw_df = pd.DataFrame(posts)
                write_cache(raw_ck, raw_df)

        if not posts:
            return pd.DataFrame(columns=self.all_columns)

        daily = _aggregate_daily(posts)
        daily = daily[(daily.index >= start) & (daily.index <= end)]

        # LLM 升级: 对高热度帖子重新打分 (在 _aggregate_daily 基础上覆盖)
        if use_llm and llm_client is not None:
            daily = self._llm_upgrade(daily, posts, start, end, llm_client)

        write_cache(ck, daily)
        return daily

    def _llm_upgrade(self, daily: pd.DataFrame, posts: list[dict],
                     start: pd.Timestamp, end: pd.Timestamp,
                     llm_client) -> pd.DataFrame:
        """用 LLM 对高热度帖子重新打分, 更新 daily."""
        df_posts = pd.DataFrame(posts)
        df_posts["date"] = pd.to_datetime(df_posts["date"]).dt.date
        df_posts = df_posts[(pd.to_datetime(df_posts["date"]) >= start) &
                            (pd.to_datetime(df_posts["date"]) <= end)]
        if df_posts.empty:
            return daily

        top_posts = df_posts.nlargest(min(50, len(df_posts)), "read_count")
        texts = top_posts["text"].tolist()

        try:
            llm_results = llm_client.analyze_sentiment(texts, symbol="")
        except Exception:
            return daily

        score_map: dict[str, list[float]] = {}
        for i, row in top_posts.iterrows():
            d = str(row["date"])
            if d not in score_map:
                score_map[d] = []
            if i < len(llm_results):
                score_map[d].append(llm_results[i].get("score", 0))

        for date_str, scores in score_map.items():
            if date_str in daily.index:
                llm_avg = float(np.mean(scores))
                lex_score = daily.loc[date_str, "sentiment_score"]
                daily.loc[date_str, "sentiment_score"] = round((llm_avg * 0.6 + lex_score * 0.4), 4)

        return daily
