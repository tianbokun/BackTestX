"""板块情绪数据获取与综合评分.

合并多个东方财富数据源, 计算每个板块的综合情绪得分.
"""

import random
import time
from typing import Optional

import numpy as np
import pandas as pd

_SESSION = None


def _get_session():
    global _SESSION
    if _SESSION is None:
        import requests
        _SESSION = requests.Session()
        _SESSION.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
            "Referer": "https://data.eastmoney.com/",
        })
    return _SESSION


def _ak_call(fn, retries=3, delay=2, *args, **kwargs):
    """带重试和延迟的 AKShare 调用，使用共享 Session 减少被识别为爬虫的概率.

    重试策略: 指数退避 + 随机 jitter, 捕获 ConnectionError 时先换 Session.
    """
    from requests.exceptions import ConnectionError as ReqConnectionError

    for attempt in range(retries + 1):
        try:
            return fn(*args, **kwargs)
        except ReqConnectionError as e:
            # 服务端主动断开 → 重置 session, 下次重新建立连接
            global _SESSION
            _SESSION = None
            if attempt < retries:
                wait = delay * (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait)
                continue
            raise
        except Exception as e:
            if attempt < retries:
                wait = delay * (attempt + 1) + random.uniform(0, 0.5)
                time.sleep(wait)
                continue
            raise


def fetch_concept_boards() -> pd.DataFrame:
    """获取概念板块列表 (486个), 含涨跌家数."""
    import akshare as ak
    df = _ak_call(ak.stock_board_concept_name_em)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns={
        "板块名称": "board_name",
        "涨跌幅": "change_pct",
        "上涨家数": "advance",
        "下跌家数": "decline",
        "换手率": "turnover",
        "总市值": "market_value",
        "领涨股票": "lead_stock",
        "领涨股票-涨跌幅": "lead_stock_pct",
        "最新价": "price",
    })
    df["board_type"] = "concept"
    return df


def fetch_industry_boards() -> pd.DataFrame:
    """获取行业板块列表, 含涨跌家数."""
    import akshare as ak
    df = _ak_call(ak.stock_board_industry_name_em)
    if df is None or df.empty:
        return pd.DataFrame()
    rename_map = {}
    for c in df.columns:
        if "板块名称" in c:
            rename_map[c] = "board_name"
        elif "涨跌幅" in c:
            rename_map[c] = "change_pct"
        elif "上涨家数" in c:
            rename_map[c] = "advance"
        elif "下跌家数" in c:
            rename_map[c] = "decline"
        elif "换手率" in c:
            rename_map[c] = "turnover"
        elif "总市值" in c:
            rename_map[c] = "market_value"
        elif "领涨股票" in c and "涨跌幅" not in c:
            rename_map[c] = "lead_stock"
        elif "领涨股票" in c and "涨跌幅" in c:
            rename_map[c] = "lead_stock_pct"
        elif "最新价" in c or "最新" in c:
            rename_map[c] = "price"
    df = df.rename(columns=rename_map)
    df["board_type"] = "industry"
    return df


def fetch_board_anomalies() -> pd.DataFrame:
    """获取所有板块异动数据 (含主力资金).

    Returns:
        DataFrame 含 board_name, change_pct, main_force_net_inflow,
        anomaly_count, anomaly_direction
    """
    import akshare as ak
    df = _ak_call(ak.stock_board_change_em)
    if df is None or df.empty:
        return pd.DataFrame()

    def _parse_direction(row):
        direction = row.get("板块异动最频繁个股及所属类型-买卖方向", "")
        if "买入" in str(direction):
            return 1
        elif "卖出" in str(direction):
            return -1
        return 0

    out = pd.DataFrame()
    out["board_name"] = df["板块名称"]
    out["change_pct"] = pd.to_numeric(df["涨跌幅"], errors="coerce")
    out["main_force_net_inflow"] = pd.to_numeric(df["主力净流入"], errors="coerce")
    out["anomaly_count"] = pd.to_numeric(df["板块异动总次数"], errors="coerce").fillna(0).astype(int)
    out["anomaly_direction"] = df.apply(_parse_direction, axis=1)
    out["anomaly_lead_stock"] = df["板块异动最频繁个股及所属类型-股票名称"]
    return out


def _normalize(s: pd.Series, cap: float = 3.0) -> pd.Series:
    """鲁棒归一化到 [-1, 1], 用分位数截断 outlier."""
    clipped = s.clip(lower=s.quantile(0.05), upper=s.quantile(0.95))
    r = clipped.max() - clipped.min()
    if r < 1e-10:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (clipped - clipped.min()) / r * 2 - 1


def compute_sector_sentiment(
    board_type: str = "concept",
    w_breadth: float = 0.30,
    w_money: float = 0.25,
    w_anomaly: float = 0.20,
    w_change: float = 0.15,
    w_heat: float = 0.10,
) -> pd.DataFrame:
    """计算板块综合情绪得分.

    Args:
        board_type: "concept" | "industry" | "all"
        w_*: 各因子权重

    Returns:
        DataFrame 按 sentiment_score 降序排列, 列:
        board_name, board_type, sentiment_score, change_pct,
        breadth_ratio, main_force_net_inflow, anomaly_count,
        anomaly_direction, adv_dec_ratio, lead_stock

        如果全部 API 调用失败则返回空 DataFrame.
    """
    # 1. 获取板块列表
    boards = []
    if board_type in ("concept", "all"):
        b = fetch_concept_boards()
        if not b.empty:
            boards.append(b)
    if board_type in ("industry", "all"):
        b = fetch_industry_boards()
        if not b.empty:
            boards.append(b)

    if not boards:
        return pd.DataFrame()

    df = pd.concat(boards, ignore_index=True)
    if df.empty:
        return df

    # 2. 获取异动数据 (含主力资金) — 可选, 失败不影响基础结果
    try:
        anomalies = fetch_board_anomalies()
        if not anomalies.empty:
            anomaly_cols = [c for c in anomalies.columns if c not in ("board_name", "change_pct")]
            df = df.merge(anomalies[["board_name"] + anomaly_cols], on="board_name", how="left")
    except Exception:
        pass

    # 3. 计算因子
    total = df["advance"] + df["decline"]
    df["breadth_ratio"] = np.where(total > 0, (df["advance"] - df["decline"]) / total, 0.0)

    df["change_factor"] = _normalize(df["change_pct"].fillna(0))
    df["money_factor"] = _normalize(df["main_force_net_inflow"].fillna(0))

    anomaly_norm = _normalize(df["anomaly_count"].fillna(0).astype(float))
    df["anomaly_factor"] = anomaly_norm * df["anomaly_direction"].fillna(0).map({1: 1, -1: 1, 0: 0})
    df["anomaly_factor"] = df["anomaly_factor"].fillna(0)

    df["heat_factor"] = _normalize(df["turnover"].fillna(0))

    # 4. 综合情绪分
    df["sentiment_score"] = (
        w_breadth * df["breadth_ratio"]
        + w_money * df["money_factor"]
        + w_anomaly * df["anomaly_factor"]
        + w_change * df["change_factor"]
        + w_heat * df["heat_factor"]
    )
    df["sentiment_score"] = df["sentiment_score"].clip(-1, 1).round(4)

    # 5. 补齐辅助列
    df["adv_dec_ratio"] = np.where(
        df["decline"] > 0,
        (df["advance"] / df["decline"]).round(2),
        df["advance"].astype(float),
    )
    for col in ["main_force_net_inflow", "anomaly_count", "lead_stock"]:
        if col not in df.columns:
            df[col] = 0 if col != "lead_stock" else ""

    cols = [
        "board_name", "board_type", "sentiment_score",
        "change_pct", "breadth_ratio", "adv_dec_ratio",
        "main_force_net_inflow", "anomaly_count",
        "anomaly_direction", "lead_stock", "turnover",
    ]
    result = df[[c for c in cols if c in df.columns]].copy()
    result = result.sort_values("sentiment_score", ascending=False).reset_index(drop=True)
    result["rank"] = range(1, len(result) + 1)
    return result
