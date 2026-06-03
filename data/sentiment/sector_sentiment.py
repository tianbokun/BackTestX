"""板块情绪数据获取与综合评分.

合并多个东方财富数据源, 计算每个板块的综合情绪得分.
"""

import hashlib
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
)

_SESSION = None

# ── 板块数据缓存 ──────────────────────────────────────────
_SECTOR_CACHE_DIR = Path(__file__).parent.parent.parent / "cache" / "sector"
_SECTOR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
SECTOR_CACHE_TTL = 3600  # 秒, 1 小时


def _cache_key(name: str) -> str:
    return hashlib.md5(f"sector_{name}".encode()).hexdigest()


def _read_cache(name: str) -> Optional[pd.DataFrame]:
    path = _SECTOR_CACHE_DIR / f"{_cache_key(name)}.parquet"
    if not path.exists():
        return None
    age = datetime.now().timestamp() - path.stat().st_mtime
    if age > SECTOR_CACHE_TTL:
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def _write_cache(name: str, df: pd.DataFrame):
    if df.empty:
        return
    try:
        path = _SECTOR_CACHE_DIR / f"{_cache_key(name)}.parquet"
        df.to_parquet(path)
    except Exception:
        pass


def _clear_expired_sector_cache():
    """清理过期缓存 (TTL 两倍)."""
    now = datetime.now().timestamp()
    for f in _SECTOR_CACHE_DIR.glob("*.parquet"):
        if now - f.stat().st_mtime > SECTOR_CACHE_TTL * 2:
            f.unlink(missing_ok=True)


# ── Session ──────────────────────────────────────────────

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


def _reset_session():
    """重置全局 Session, 断开旧连接."""
    global _SESSION
    _SESSION = None


def _ak_call(fn, *args, **kwargs):
    """带指数退避重试的 AKShare 调用.

    最多尝试 30 分钟, 使用 tenacity 自动重试, 捕获 ConnectionError /
    ConnectionResetError / OSError, 每次重试前重置 Session 并提示用户.
    """
    from requests.exceptions import ConnectionError as ReqConnectionError

    def _on_retry(retry_state):
        """重试前重置 Session 并提示用户."""
        _reset_session()
        exc = retry_state.outcome.exception()
        attempt = retry_state.attempt_number
        wait = int(retry_state.next_action.sleep) if retry_state.next_action else 2
        elapsed = int(retry_state.seconds_since_start)
        remaining = max(0, 1800 - elapsed)
        msg = (
            f"⚠️ 连接失败 (第{attempt}次重试, 等待{wait}s, "
            f"剩余{remaining // 60}分{remaining % 60}秒): {type(exc).__name__}"
        )
        print(f"[sector_sentiment] {msg}", flush=True)
        try:
            import streamlit as st
            st.toast(msg, icon="🔄")
        except Exception:
            pass

    @retry(
        stop=stop_after_delay(1800) | stop_after_attempt(50),
        wait=wait_exponential(multiplier=1.5, min=2, max=300),
        retry=retry_if_exception_type(
            (ReqConnectionError, ConnectionResetError, OSError)
        ),
        before_sleep=_on_retry,
        reraise=True,
    )
    def _call():
        return fn(*args, **kwargs)

    return _call()


def fetch_concept_boards(use_cache: bool = True) -> pd.DataFrame:
    """获取概念板块列表 (486个), 含涨跌家数.

    Args:
        use_cache: 是否使用本地缓存 (1 小时 TTL).
    """
    if use_cache:
        cached = _read_cache("concept_boards")
        if cached is not None:
            return cached
    import akshare as ak
    df = _ak_call(ak.stock_board_concept_name_em)
    if df is None or df.empty:
        return pd.DataFrame()
    _write_cache("concept_boards", df)
    df = df.rename(columns={
        "板块名称": "board_name",
        "板块代码": "board_code",
        "涨跌幅": "change_pct",
        "上涨家数": "advance",
        "下跌家数": "decline",
        "换手率": "turnover",
        "总市值": "market_value",
        "领涨股票": "lead_stock",
        "领涨股票-涨跌幅": "lead_stock_pct",
        "最新价": "price",
    })
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    df["board_type"] = "concept"
    return df


def fetch_industry_boards(use_cache: bool = True) -> pd.DataFrame:
    """获取行业板块列表, 含涨跌家数.

    Args:
        use_cache: 是否使用本地缓存 (1 小时 TTL).

    列名精确匹配已知的 AKShare 映射, 避免 `in` 匹配导致误将多个不同列
    映射到同一英文名 (如 "涨跌幅" 和 "行业涨跌幅" 都被映射为 change_pct).
    不认识的列自动丢弃.
    """
    if use_cache:
        cached = _read_cache("industry_boards")
        if cached is not None:
            return cached
    import akshare as ak
    df = _ak_call(ak.stock_board_industry_name_em)
    if df is None or df.empty:
        return pd.DataFrame()
    _write_cache("industry_boards", df)
    if df is None or df.empty:
        return pd.DataFrame()

    KNOWN = {
        "板块名称": "board_name",
        "涨跌幅": "change_pct",
        "上涨家数": "advance",
        "下跌家数": "decline",
        "换手率": "turnover",
        "总市值": "market_value",
        "领涨股票": "lead_stock",
        "领涨股票-涨跌幅": "lead_stock_pct",
        "最新价": "price",
    }
    rename_map = {k: v for k, v in KNOWN.items() if k in df.columns}
    df = df.rename(columns=rename_map)
    wanted = list(rename_map.values())
    df = df[[c for c in wanted if c in df.columns]]
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
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


def fetch_board_constituents(board_code: str, top_n: int = 5) -> pd.DataFrame:
    """获取板块成分股行情.

    Args:
        board_code: 板块代码, 如 "BK0890"
        top_n: 返回前 N 支股票 (按涨跌幅绝对值排序)

    Returns:
        DataFrame 含 symbol, name, change_pct, price, turnover
    """
    import akshare as ak
    df = _ak_call(ak.stock_board_concept_cons_em, symbol=board_code)
    if df is None or df.empty:
        return pd.DataFrame()

    col_map = {}
    for c in df.columns:
        if "代码" in c:
            col_map[c] = "symbol"
        elif "名称" in c:
            col_map[c] = "name"
        elif c == "涨跌幅":
            col_map[c] = "change_pct"
        elif "最新价" in c or "最新" in c:
            col_map[c] = "price"
        elif "换手率" in c:
            col_map[c] = "turnover"
    df = df.rename(columns=col_map)
    out = df[[c for c in ["symbol", "name", "change_pct", "price", "turnover"] if c in df.columns]].copy()
    if "symbol" not in out.columns:
        return pd.DataFrame()
    out = out.dropna(subset=["symbol"])
    if out.empty:
        return pd.DataFrame()
    if "change_pct" in out.columns:
        out["change_pct"] = pd.to_numeric(out["change_pct"], errors="coerce")
        out["abs_change"] = out["change_pct"].abs()
        out = out.sort_values("abs_change", ascending=False).head(top_n)
        out = out.drop(columns=["abs_change"])
    else:
        out = out.head(top_n)
    return out.reset_index(drop=True)


def _normalize(s: pd.Series, cap: float = 3.0) -> pd.Series:
    """鲁棒归一化到 [-1, 1], 用分位数截断 outlier."""
    import warnings
    if isinstance(s, pd.DataFrame):
        dup = s.columns.tolist()
        warnings.warn(f"_normalize 收到 DataFrame 而非 Series，列重复: {dup}，取第一列降级")
        s = s.iloc[:, 0]
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

    # 2b. 去重 — 防止列名重复导致 df["col"] 返回 DataFrame
    import warnings as _warnings
    dup_cols = df.columns[df.columns.duplicated()].tolist()
    if dup_cols:
        _warnings.warn(f"compute_sector_sentiment 发现重复列: {dup_cols}，自动去重")
        df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # 2c. 补齐异常数据的缺失列 (异动接口失败时 protect 后续计算)
    for col in ["main_force_net_inflow", "anomaly_count", "anomaly_direction", "anomaly_lead_stock"]:
        if col not in df.columns:
            df[col] = 0 if col != "anomaly_lead_stock" else ""

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
    for col in ["lead_stock"]:
        if col not in df.columns:
            df[col] = ""

    cols = [
        "board_name", "board_code", "board_type", "sentiment_score",
        "change_pct", "breadth_ratio", "adv_dec_ratio",
        "main_force_net_inflow", "anomaly_count",
        "anomaly_direction", "lead_stock", "turnover",
    ]
    result = df[[c for c in cols if c in df.columns]].copy()
    result = result.sort_values("sentiment_score", ascending=False).reset_index(drop=True)
    result["rank"] = range(1, len(result) + 1)
    return result


def clear_sector_cache():
    """清空板块缓存, 下次 fetch 时重新请求."""
    for f in _SECTOR_CACHE_DIR.glob("*.parquet"):
        f.unlink(missing_ok=True)


# 启动时清理过期缓存
_clear_expired_sector_cache()
