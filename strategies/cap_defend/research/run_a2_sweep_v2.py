#!/usr/bin/env python3
"""A2 sweep v2 — swing-only, stop+TP grid."""
from __future__ import annotations
import os, sys, time, csv, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a2_backtest import run_a2

START = '2020-10-01'
END = '2026-04-15'

STOP_KINDS = [
    'none',
    'prev_close_pct',
    'highest_close_since_entry_pct',
    'highest_high_since_entry_pct',
    'rolling_high_close_pct',
    'rolling_high_high_pct',
    'atr_highest_high_since_entry',
    'atr_rolling_high_high',
]
STOP_PCTS = [0.05, 0.08, 0.12]
TP_PARTIALS = [0.0, 0.05, 0.10]
TP_TRAILS = [0.025]

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'a2_sweep_v2_results.csv')


def main():
    rows = []
    grid = []
    for sk in STOP_KINDS:
        if sk == 'none':
            for tpp in TP_PARTIALS:
                grid.append((sk, 0.0, tpp, 0.025))
        else:
            for sp, tpp in itertools.product(STOP_PCTS, TP_PARTIALS):
                grid.append((sk, sp, tpp, 0.025))
    print('Stage 1 sweep:', len(grid), 'configs (swing-only, no overlay)')
    t0 = time.time()
    fieldnames = ['stop_kind','stop_pct','tp_partial_pct','tp_trail_pct',
                  'Sharpe','CAGR','MDD','Cal','Cal_m',
                  'Liq','Stops','TP','Rebal','elapsed']
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, (sk, sp, tpp, tpt) in enumerate(grid, 1):
            t1 = time.time()
            ek = {
                'stop_kind': sk, 'stop_pct': sp,
                'stop_lookback_bars': 168 if sk.startswith('rolling') else 0,
                'stop_atr_lookback_bars': 14 if sk.startswith('atr') else 0,
                'stop_atr_mult': 2.0 if sk.startswith('atr') else 0.0,
                'tp_partial_pct': tpp,
                'tp_trail_pct': tpt,
            }
            try:
                m = run_a2(start=START, end=END,
                          combo_weights={'swing': 1.0, 'overlay': 0.0},
                          engine_kwargs=ek, with_15m=False, verbose=False)
            except Exception as e:
                print('  ERROR', sk, sp, tpp, e)
                continue
            row = {
                'stop_kind': sk, 'stop_pct': sp,
                'tp_partial_pct': tpp, 'tp_trail_pct': tpt,
                'Sharpe': round(m.get('Sharpe', 0), 3),
                'CAGR': round(m.get('CAGR', 0), 4),
                'MDD': round(m.get('MDD', 0), 4),
                'Cal': round(m.get('Cal', 0), 3),
                'Cal_m': round(m.get('Cal_m', 0), 3),
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
            print('  [%d/%d] %s/%s/tp=%s Cal=%.2f Sh=%.2f MDD=%+.1f%% (%.1fs; eta %.0fm)' % (
                i, len(grid), sk[:30], sp, tpp,
                row['Cal'], row['Sharpe'], row['MDD']*100,
                row['elapsed'], eta/60))
    print('\nDone. Results -> %s (%d rows). Total %.1fmin' % (OUT, len(rows), (time.time()-t0)/60))
    rows.sort(key=lambda r: -r['Cal'])
    print('\nTop 10 by Calmar:')
    for r in rows[:10]:
        print('  Cal=%6.2f Sh=%5.2f CAGR=%+.1f%% MDD=%+.1f%% TP=%4d Stops=%4d Rebal=%5d | %s/%s/tp=%s' % (
            r['Cal'], r['Sharpe'], r['CAGR']*100, r['MDD']*100,
            r['TP'], r['Stops'], r['Rebal'],
            r['stop_kind'], r['stop_pct'], r['tp_partial_pct']))


if __name__ == '__main__':
    main()
