#!/bin/bash
# run_coin.sh — 업비트 V21 현물 executor 래퍼.
# cron (V21): 5 9 * * * /home/ubuntu/workspace/BitBot/scripts/run_coin.sh --dry-run
set -e

REPO=/home/ubuntu/workspace/BitBot
LOG=$REPO/trade/run_coin.log
LOCK=/tmp/run_coin_wrapper.lock

exec 200>"$LOCK"
if ! flock -n 200; then
    echo "[$(date '+%F %T')] run_coin: wrapper lock busy, skip" >> "$LOG"
    exit 0
fi

cd "$REPO/trade"
"$REPO/.venv/bin/python" executor_coin.py "$@" >> "$LOG" 2>&1
