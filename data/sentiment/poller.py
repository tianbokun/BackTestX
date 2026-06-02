"""情感数据 7x24 独立轮询脚本.

可作为 cron job 或 systemd service 运行.
每 N 分钟拉取所有注册代码的最新股吧 + 新闻数据, 更新 Parquet 缓存.

用法:
  python -m data.sentiment.poller                 # 一次轮询
  python -m data.sentiment.poller --daemon        # 持续轮询 (默认每30分钟)
  python -m data.sentiment.poller --interval 60   # 每60分钟轮询
  python -m data.sentiment.poller --symbol 159941 --symbol 588870  # 指定代码
"""

import argparse
import time
from datetime import datetime
from typing import Optional

import pandas as pd

from data.symbol_registry import SymbolRegistry
from data.fetcher import fetch_sentiment_data, _get_sentiment_sources


def poll_once(symbols: Optional[list[str]] = None, verbose: bool = True):
    """执行一次轮询: 拉取所有/指定代码的情感数据."""
    sources = _get_sentiment_sources()
    if verbose:
        print(f"[{datetime.now():%H:%M:%S}] 情感数据轮询开始 源: {list(sources.keys())}")

    if symbols is None:
        all_symbols = SymbolRegistry.list()
        symbols = [s["symbol"] for s in all_symbols]

    for sym in symbols:
        for src_name in sources:
            try:
                df = fetch_sentiment_data(sym)
                if df is not None and not df.empty:
                    if verbose:
                        print(f"  ✓ {sym} [{src_name}] {len(df)} 天数据 "
                              f"情感: {df['sentiment_score'].mean():.3f} "
                              f"帖量: {int(df['post_volume'].sum())}")
                else:
                    if verbose:
                        print(f"  - {sym} [{src_name}] 无数据")
            except Exception as e:
                if verbose:
                    print(f"  ✗ {sym} [{src_name}] {e}")

    if verbose:
        print(f"[{datetime.now():%H:%M:%S}] 轮询完成")


def poll_daemon(interval_minutes: int = 30, symbols: Optional[list[str]] = None):
    """持续轮询守护进程."""
    print(f"情感数据轮询守护进程启动 (间隔={interval_minutes}分钟)")
    print("按 Ctrl+C 停止")
    while True:
        try:
            poll_once(symbols, verbose=True)
        except Exception as e:
            print(f"轮询异常: {e}")
        time.sleep(interval_minutes * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="情感数据轮询")
    parser.add_argument("--daemon", action="store_true", help="持续轮询模式")
    parser.add_argument("--interval", type=int, default=30, help="轮询间隔(分钟)")
    parser.add_argument("--symbol", action="append", dest="symbols", help="指定代码(可多次)")
    args = parser.parse_args()

    if args.daemon:
        poll_daemon(args.interval, args.symbols)
    else:
        poll_once(args.symbols)
