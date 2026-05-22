"""
网格搜索模块
支持超参数遍历 + Walk-Forward 交叉验证 + 结果持久化
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Callable

import pandas as pd
import numpy as np

from .strategies import run_dropbuy_backtest, DropBuyResult


RESULTS_DIR = Path(__file__).parent / "strategy_results"
RESULTS_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════
#  数据结构
# ══════════════════════════════════════════════════════════════

@dataclass
class TrialResult:
    """单次网格搜索的完整记录"""
    fold_id: int
    X: float           # 跌幅阈值 (%)
    Y: float           # 每期买入金额 (元)
    train_invested: float = 0.0
    train_return_pct: float = 0.0
    train_annualized: float = 0.0
    train_num_trades: int = 0
    val_invested: float = 0.0
    val_return_pct: float = 0.0
    val_annualized: float = 0.0
    val_num_trades: int = 0
    # 总投入 (train + val)
    combined_invested: float = 0.0
    combined_return_pct: float = 0.0
    combined_annualized: float = 0.0


@dataclass
class FoldResult:
    """单折的结果"""
    fold_id: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    best_params: Dict[str, Any] = field(default_factory=dict)
    best_val_return: float = -999.0
    trials: List[TrialResult] = field(default_factory=list)


@dataclass
class GridSearchResult:
    """完整的网格搜索结果"""
    symbol: str
    total_start: str
    total_end: str
    param_grid: Dict[str, List[float]]
    n_folds: int
    folds: List[FoldResult] = field(default_factory=list)
    best_params_overall: Dict[str, Any] = field(default_factory=dict)
    avg_val_return: float = 0.0
    timestamp: str = ""
    trials_df: pd.DataFrame = field(default_factory=pd.DataFrame)


# ══════════════════════════════════════════════════════════════
#  核心引擎
# ══════════════════════════════════════════════════════════════

def _make_walk_forward_splits(
    price_series: pd.Series,
    total_start: str,
    total_end: str,
    n_folds: int,
) -> List[Tuple[str, str, str, str]]:
    """
    生成 Walk-Forward 数据划分

    将总区间切为 n_folds 段, 每段作为验证集;
    训练集从总起点到该段之前。

    返回 [(train_start, train_end, val_start, val_end), ...]
    """
    prices = price_series.sort_index().dropna()
    start_ts = pd.Timestamp(total_start)
    end_ts = pd.Timestamp(total_end)
    prices = prices[(prices.index >= start_ts) & (prices.index <= end_ts)]

    if len(prices) < n_folds * 20:
        raise ValueError(f"数据点不足 ({len(prices)}), 无法划分 {n_folds} 折")

    # 按时间等分 (确保每段至少有一些数据)
    total_days = (prices.index[-1] - prices.index[0]).days
    fold_days = total_days // n_folds

    splits = []
    for i in range(n_folds):
        val_end_idx = (i + 1) * fold_days
        if i == n_folds - 1:
            val_end_idx = total_days

        val_start_ts = prices.index[0] + pd.Timedelta(days=i * fold_days)
        val_end_ts = prices.index[0] + pd.Timedelta(days=val_end_idx)

        # 训练集: 从最初到验证集开始 (不包含验证集)
        train_start_ts = prices.index[0]
        train_end_ts = val_start_ts - pd.Timedelta(days=1)

        # 确保验证集至少有交易日
        val_prices = prices[(prices.index >= val_start_ts) & (prices.index <= val_end_ts)]
        if len(val_prices) < 5:
            continue

        splits.append((
            train_start_ts.strftime("%Y-%m-%d"),
            train_end_ts.strftime("%Y-%m-%d"),
            val_start_ts.strftime("%Y-%m-%d"),
            val_end_ts.strftime("%Y-%m-%d"),
        ))

    return splits


def _run_trial(
    price_series: pd.Series,
    X: float,
    Y: float,
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
    fold_id: int,
    max_total: float = 0,
) -> TrialResult:
    """
    运行一组 X, Y 在训练集和验证集上的表现

    总投资上限 max_total 是累计的: 先在训练阶段消耗, 剩余额度留给验证阶段,
    模拟投资者总体预算固定的真实场景。
    """

    train_res = run_dropbuy_backtest(
        price_series, X=X, Y=Y,
        start_date=train_start, end_date=train_end,
        max_total=max_total,
    )

    remaining = max(0, max_total - train_res.total_invested) if max_total > 0 else 0
    val_res = run_dropbuy_backtest(
        price_series, X=X, Y=Y,
        start_date=val_start, end_date=val_end,
        max_total=remaining,
    )

    comb_invested = train_res.total_invested + val_res.total_invested
    comb_ret = (
        (train_res.total_return_pct * train_res.total_invested +
         val_res.total_return_pct * val_res.total_invested) /
        max(comb_invested, 1e-9)
    )

    return TrialResult(
        fold_id=fold_id,
        X=X, Y=Y,
        train_invested=train_res.total_invested,
        train_return_pct=train_res.total_return_pct,
        train_annualized=train_res.annualized_return_pct,
        train_num_trades=train_res.num_investments,
        val_invested=val_res.total_invested,
        val_return_pct=val_res.total_return_pct,
        val_annualized=val_res.annualized_return_pct,
        val_num_trades=val_res.num_investments,
        combined_invested=comb_invested,
        combined_return_pct=round(comb_ret, 2),
        combined_annualized=(
            (train_res.annualized_return_pct + val_res.annualized_return_pct) / 2
            if train_res.annualized_return_pct * val_res.annualized_return_pct != 0
            else max(train_res.annualized_return_pct, val_res.annualized_return_pct)
        ),
    )


def run_grid_search(
    price_series: pd.Series,
    symbol: str,
    total_start: str,
    total_end: str,
    X_range: List[float],
    Y_range: List[float],
    n_folds: int = 4,
    max_total: float = 0,
) -> GridSearchResult:
    """
    执行网格搜索 + Walk-Forward 交叉验证

    Parameters
    ----------
    price_series : pd.Series
        价格序列
    symbol : str
        代码 (仅用于标识)
    total_start, total_end : str
        回测总区间
    X_range : list of float
        X (跌幅阈值) 候选值
    Y_range : list of float
        Y (买入金额) 候选值
    n_folds : int
        折数
    max_total : float
        总投资上限 (训练+验证共享此额度)

    Returns
    -------
    GridSearchResult
    """
    timestr = time.strftime("%Y%m%d_%H%M%S")
    splits = _make_walk_forward_splits(price_series, total_start, total_end, n_folds)
    n_folds_actual = len(splits)

    if n_folds_actual == 0:
        raise ValueError("无法生成任何有效的数据划分")

    all_trials: List[TrialResult] = []
    folds: List[FoldResult] = []

    for fold_id, (tr_s, tr_e, vl_s, vl_e) in enumerate(splits):
        fold_trials: List[TrialResult] = []

        for X in X_range:
            for Y in Y_range:
                trial = _run_trial(
                    price_series, X, Y,
                    tr_s, tr_e, vl_s, vl_e,
                    fold_id, max_total,
                )
                fold_trials.append(trial)
                all_trials.append(trial)

        # 该折最优: 按验证集年化收益率排序
        best = max(fold_trials, key=lambda t: t.val_annualized)
        folds.append(FoldResult(
            fold_id=fold_id,
            train_start=tr_s, train_end=tr_e,
            val_start=vl_s, val_end=vl_e,
            best_params={"X": best.X, "Y": best.Y},
            best_val_return=best.val_annualized,
            trials=fold_trials,
        ))

    # 全局最优: 各折最优参数的验证集年化平均值
    param_scores: Dict[Tuple[float, float], List[float]] = {}
    for trial in all_trials:
        key = (trial.X, trial.Y)
        if key not in param_scores:
            param_scores[key] = []
        param_scores[key].append(trial.val_annualized)

    best_params_overall = {}
    best_avg_score = -999.0
    for (X, Y), scores in param_scores.items():
        avg = np.mean(scores)
        if avg > best_avg_score:
            best_avg_score = avg
            best_params_overall = {"X": X, "Y": Y}

    # 组装结果 DataFrame
    rows = []
    for trial in all_trials:
        d = asdict(trial)
        d.pop("fold_id")
        row = {"fold": trial.fold_id, **d}
        rows.append(row)

    trials_df = pd.DataFrame(rows)

    return GridSearchResult(
        symbol=symbol,
        total_start=total_start,
        total_end=total_end,
        param_grid={"X": X_range, "Y": Y_range},
        n_folds=n_folds_actual,
        folds=folds,
        best_params_overall=best_params_overall,
        avg_val_return=round(best_avg_score, 2),
        timestamp=timestr,
        trials_df=trials_df,
    )


# ══════════════════════════════════════════════════════════════
#  持久化
# ══════════════════════════════════════════════════════════════

def save_result(result: GridSearchResult) -> Path:
    """
    将网格搜索结果保存到磁盘

    保存内容:
    1. trials_df  → strategy_results/{symbol}_{timestamp}_trials.parquet
    2. 摘要 JSON  → strategy_results/{symbol}_{timestamp}_summary.json
    3. 每折明细   → strategy_results/{symbol}_{timestamp}_fold_{id}.json

    Returns
    -------
    保存目录路径
    """
    ts = result.timestamp
    sym = result.symbol
    base = RESULTS_DIR / f"{sym}_{ts}"
    base.mkdir(parents=True, exist_ok=True)

    # 1. 所有试验
    if not result.trials_df.empty:
        p = base / "trials.parquet"
        result.trials_df.to_parquet(p)

    # 2. 摘要
    summary = {
        "symbol": result.symbol,
        "total_start": result.total_start,
        "total_end": result.total_end,
        "n_folds": result.n_folds,
        "param_grid": result.param_grid,
        "best_params_overall": result.best_params_overall,
        "avg_val_return": result.avg_val_return,
        "timestamp": result.timestamp,
        "folds": [],
    }
    for fold in result.folds:
        fdict = {
            "fold_id": fold.fold_id,
            "train_start": fold.train_start,
            "train_end": fold.train_end,
            "val_start": fold.val_start,
            "val_end": fold.val_end,
            "best_params": fold.best_params,
            "best_val_return": fold.best_val_return,
        }
        summary["folds"].append(fdict)

        # 每折单独 JSON
        fold_json = base / f"fold_{fold.fold_id}.json"
        fold_json.write_text(json.dumps(fdict, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_path = base / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return base


def list_saved_results(symbol: Optional[str] = None) -> List[Dict]:
    """列出所有已保存的搜索结果"""
    results = []
    for d in sorted(RESULTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        summary_file = d / "summary.json"
        if not summary_file.exists():
            continue
        try:
            data = json.loads(summary_file.read_text(encoding="utf-8"))
            if symbol and data.get("symbol") != symbol:
                continue
            data["dir"] = str(d)
            results.append(data)
        except (json.JSONDecodeError, KeyError):
            continue
    return results


def load_result(dir_path: str) -> GridSearchResult:
    """从目录加载网格搜索结果"""
    base = Path(dir_path)
    summary = json.loads((base / "summary.json").read_text(encoding="utf-8"))

    trials_df = pd.DataFrame()
    trials_file = base / "trials.parquet"
    if trials_file.exists():
        trials_df = pd.read_parquet(trials_file)

    folds = []
    for fd in summary.get("folds", []):
        fold_path = base / f"fold_{fd['fold_id']}.json"
        fold_data = json.loads(fold_path.read_text(encoding="utf-8")) if fold_path.exists() else fd
        fold_trials = []
        if not trials_df.empty:
            ft = trials_df[trials_df["fold"] == fd["fold_id"]]
            for _, row in ft.iterrows():
                fold_trials.append(TrialResult(
                    fold_id=int(row["fold"]),
                    X=row["X"], Y=row["Y"],
                    train_invested=row["train_invested"],
                    train_return_pct=row["train_return_pct"],
                    train_annualized=row["train_annualized"],
                    train_num_trades=int(row["train_num_trades"]),
                    val_invested=row["val_invested"],
                    val_return_pct=row["val_return_pct"],
                    val_annualized=row["val_annualized"],
                    val_num_trades=int(row["val_num_trades"]),
                    combined_invested=row["combined_invested"],
                    combined_return_pct=row["combined_return_pct"],
                    combined_annualized=row["combined_annualized"],
                ))
        folds.append(FoldResult(
            fold_id=fd["fold_id"],
            train_start=fd["train_start"],
            train_end=fd["train_end"],
            val_start=fd["val_start"],
            val_end=fd["val_end"],
            best_params=fd.get("best_params", {}),
            best_val_return=fd.get("best_val_return", 0),
            trials=fold_trials,
        ))

    return GridSearchResult(
        symbol=summary["symbol"],
        total_start=summary["total_start"],
        total_end=summary["total_end"],
        param_grid=summary["param_grid"],
        n_folds=summary["n_folds"],
        folds=folds,
        best_params_overall=summary["best_params_overall"],
        avg_val_return=summary["avg_val_return"],
        timestamp=summary["timestamp"],
        trials_df=trials_df,
    )
