#!/usr/bin/env python3
"""A2 backtest CLI."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from a2_backtest import run_a2

START = '2020-10-01'
END = '2026-04-15'

ABSOLUTE_GATE = {'Sharpe': 1.5, 'Cal': 1.5, 'MDD': -0.30}


def main():
    m = run_a2(start=START, end=END, with_15m=True)

    print()
    print('=' * 72)
    print(f'A2 결과 ({START} ~ {END})')
    print('=' * 72)
    print(f'  Sharpe : {m.get("Sharpe", 0):.2f}')
    print(f'  CAGR   : {m.get("CAGR", 0):+.1%}')
    print(f'  MDD    : {m.get("MDD", 0):+.1%}')
    print(f'  Calmar : {m.get("Cal", 0):.2f}')
    print(f'  MDDm   : {m.get("MDD_m_avg", 0):+.1%} (avg of 30-day samples)')
    print(f'  Cal_m  : {m.get("Cal_m", 0):.2f}')
    print(f'  Liq    : {m.get("Liq", 0)}')
    print(f'  Stops  : {m.get("Stops", 0)}')
    print(f'  TP     : {m.get("TP", 0)} (partial 발동)')
    print(f'  Rebal  : {m.get("Rebal", 0)}')
    print()

    # Absolute gate
    fails = []
    if m.get('Sharpe', 0) < ABSOLUTE_GATE['Sharpe']:
        fails.append(f"Sharpe {m.get('Sharpe', 0):.2f} < {ABSOLUTE_GATE['Sharpe']}")
    if m.get('Cal', 0) < ABSOLUTE_GATE['Cal']:
        fails.append(f"Cal {m.get('Cal', 0):.2f} < {ABSOLUTE_GATE['Cal']}")
    if m.get('MDD', 0) < ABSOLUTE_GATE['MDD']:
        fails.append(f"MDD {m.get('MDD', 0):+.1%} < {ABSOLUTE_GATE['MDD']:+.1%}")

    if fails:
        print(f'  GATE FAIL: {", ".join(fails)}')
    else:
        print('  GATE PASS')


if __name__ == '__main__':
    main()
