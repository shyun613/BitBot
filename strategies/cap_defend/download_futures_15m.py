#!/usr/bin/env python3
"""A2: BTC/ETH/SOL 15m kline downloader.

기존 1h CSV 옆에 15m CSV를 추가한다. funding은 8h 단위라 별도 다운로드 불필요.
증분 업데이트 default; --full로 전체 재다운로드.
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from download_futures_data import download_klines, DATA_DIR

A2_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
INTERVAL = "15m"
START = "2020-09-01"


def main():
    incremental = "--full" not in sys.argv
    for sym in A2_SYMBOLS:
        out = os.path.join(DATA_DIR, f"{sym}_{INTERVAL}.csv")
        start_ts_override = None
        if incremental and os.path.exists(out):
            existing = pd.read_csv(out, parse_dates=["Date"], index_col="Date")
            if len(existing) > 0:
                start_ts_override = int(existing.index[-1].timestamp() * 1000) + 1
                print(f"{sym}: incremental from {existing.index[-1]}")
            else:
                print(f"{sym}: empty file, full redownload")
        else:
            print(f"{sym}: full download from {START}")

        df = download_klines(sym, interval=INTERVAL, start=START, start_ts_override=start_ts_override)
        if df is None or len(df) == 0:
            print(f"{sym}: no new data")
            continue
        if incremental and os.path.exists(out) and start_ts_override:
            existing = pd.read_csv(out, parse_dates=["Date"], index_col="Date")
            df = pd.concat([existing, df]).sort_index()
            df = df[~df.index.duplicated(keep="last")]
        df.to_csv(out)
        print(f"{sym}: saved {len(df)} bars -> {out}")


if __name__ == "__main__":
    main()
