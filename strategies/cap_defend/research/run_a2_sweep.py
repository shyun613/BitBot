#!/usr/bin/env python3
"""A2 sweep — DOE 기반 단계적 그리드.

Phase A: stop_kind × stop_pct × tp_partial × tp_trail = 7×3×3×3 = 189
Phase B: A 최적 lock 후 indicator × cap × st_w = 27
Phase C: 각 axis top-3 cross = ~30

본 스크립트는 Phase A만 실행. B/C는 결과 보고 추가.
"""
from __future__ import annotations

import os
import sys
import time
import csv
import itertools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from a2_backtest import run_a2
from a2_strategy import DEFAULT_COIN_CAPS

START = '2020-10-01'
END = '2026-04-15'

STOP_KINDS = [
    'prev_close_pct',
    'highest_close_since_entry_pct',
    'highest_high_since_entry_pct',
    'rolling_high_close_pct',
    'rolling_high_high_pct',
    'atr_highest_high_since_entry',
    'atr_rolling_high_high',
]
STOP_PCTS = [0.08]
TP_PARTIALS = [0.0, 0.05, 0.10]
TP_TRAILS = [0.025]

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'a2_sweep_results.csv')


def main():
    rows = []
    grid = list(itertools.product(STOP_KINDS, STOP_PCTS, TP_PARTIALS, TP_TRAILS))
    print(f'Phase A: {len(grid)} configs')
    t0 = time.time()
    fieldnames = [
        'stop_kind', 'stop_pct', 'tp_partial_pct', 'tp_trail_pct',
        'Sharpe', 'CAGR', 'MDD', 'Cal', 'Cal_m', 'MDD_m_avg',
        'Liq', 'Stops', 'TP', 'Rebal', 'elapsed',
    ]
    # Open file once, write incrementally so we can monitor progress
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, (sk, sp, tpp, tpt) in enumerate(grid, 1):
            t1 = time.time()
            ek = {
                'stop_kind': sk,
                'stop_pct': sp,
                'stop_lookback_bars': 168 if sk.startswith('rolling') else 0,
                'stop_atr_lookback_bars': 14 if sk.startswith('atr') else 0,
                'stop_atr_mult': 2.0 if sk.startswith('atr') else 0.0,
                'tp_partial_pct': tpp,
                'tp_trail_pct': tpt,
            }
            try:
                m = run_a2(start=START, end=END,
                          engine_kwargs=ek, with_15m=True, verbose=False)
            except Exception as e:
                print(f'  [{i}/{len(grid)}] {sk}/{sp}/{tpp}/{tpt}: ERROR {e}')
                continue
            row = {
                'stop_kind': sk, 'stop_pct': sp,
                'tp_partial_pct': tpp, 'tp_trail_pct': tpt,
                'Sharpe': round(m.get('Sharpe', 0), 3),
                'CAGR': round(m.get('CAGR', 0), 4),
                'MDD': round(m.get('MDD', 0), 4),
                'Cal': round(m.get('Cal', 0), 3),
                'Cal_m': round(m.get('Cal_m', 0), 3),
                'MDD_m_avg': round(m.get('MDD_m_avg', 0), 4),
                'Liq': m.get('Liq', 0),
                'Stops': m.get('Stops', 0),
                'TP': m.get('TP', 0),
                'Rebal': m.get('Rebal', 0),
                'elapsed': round(time.time() - t1, 1),
            }
            w.writerow(row)
            f.flush()
            rows.append(row)
            elapsed = time.time() - t0
            eta = (len(grid) - i) * elapsed / max(i, 1)
            print(f'  [{i}/{len(grid)}] {sk[:30]}/{sp}/{tpp}/{tpt} '
                  f'Cal={row["Cal"]} Sh={row["Sharpe"]} MDD={row["MDD"]:+.1%} '
                  f'({row["elapsed"]}s; eta {eta/60:.0f}m)')

    print(f'\nDone. Results -> {OUT} ({len(rows)} rows). Total {(time.time()-t0)/60:.1f}min')

    # Print top 10 by Calmar
    rows.sort(key=lambda r: -r['Cal'])
    print('\nTop 10 by Calmar:')
    for r in rows[:10]:
        print(f"  Cal={r['Cal']:5.2f} Sh={r['Sharpe']:5.2f} CAGR={r['CAGR']:+.1%} "
              f"MDD={r['MDD']:+.1%} TP={r['TP']:4d} Stops={r['Stops']:4d} | "
              f"{r['stop_kind']}/{r['stop_pct']}/tp={r['tp_partial_pct']}/tr={r['tp_trail_pct']}")


if __name__ == '__main__':
    main()
