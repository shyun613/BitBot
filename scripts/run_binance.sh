#!/bin/bash
# run_binance.sh — 바이낸스 USDT-M 선물 V21 auto_trade 래퍼.
# cron (V21): 5 9,13,17,21,1,5 * * * /home/ubuntu/workspace/BitBot/scripts/run_binance.sh --dry-run
set -e

REPO=/home/ubuntu/workspace/BitBot
LOG=$REPO/trade/run_binance.log
LOCK=/tmp/run_binance_wrapper.lock

exec 200>"$LOCK"
if ! flock -n 200; then
    echo "[$(date '+%F %T')] run_binance: wrapper lock busy, skip" >> "$LOG"
    exit 0
fi

cd "$REPO/trade"
"$REPO/.venv/bin/python" auto_trade_binance.py "$@" >> "$LOG" 2>&1
