"""A2 backtest entry — data load + trace build + engine run.

Usage:
    from a2_backtest import run_a2
    metrics = run_a2(start='2020-10-01', end='2026-04-15', params={...})
"""
from __future__ import annotations

import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from a1_backtest import load_data
from a2_engine import HedgeAccountEngine, combine_targets_hedge
from a2_strategy import build_a2_traces, DEFAULT_COIN_CAPS, DEFAULT_COMBO_WEIGHTS


def _slice(d, start, end):
    out = {}
    for k, df in d.items():
        out[k] = df.loc[(df.index >= start) & (df.index <= end)]
    return out


def run_a2(start='2020-10-01', end='2026-04-15',
           params=None, coin_caps=None, combo_weights=None,
           engine_kwargs=None, with_15m=True, verbose=True):
    """Run A2 backtest. Returns metrics dict."""
    t0 = time.time()
    if verbose:
        print(f'A2 backtest ({start} ~ {end})')

    # Load
    if verbose:
        print('  Loading 1h, 4h...')
    bars_1h, funding_1h = load_data('1h')
    bars_4h, _ = load_data('4h')
    bars_1h = _slice(bars_1h, start, end)
    bars_4h = _slice(bars_4h, start, end)

    bars_15m = {}
    if with_15m:
        if verbose:
            print('  Loading 15m...')
        bars_15m_full, _ = load_data('15m')
        if bars_15m_full:
            bars_15m = _slice(bars_15m_full, start, end)
            if verbose:
                print(f'    15m loaded: {sorted(bars_15m.keys())}')
        elif verbose:
            print('    15m unavailable — overlay degrades to 1h-only')

    if verbose:
        for c, df in bars_1h.items():
            print(f'    {c} 1h: {len(df)} bars [{df.index[0]} ~ {df.index[-1]}]')

    # Build traces
    if verbose:
        print('  Building traces...')
    coin_caps = coin_caps or DEFAULT_COIN_CAPS
    traces = build_a2_traces(bars_1h, bars_4h, bars_15m, coin_caps=coin_caps, params=params)
    if verbose:
        for k, t in traces.items():
            print(f'    {k}: {len(t)} entries')

    # Combine on 1h grid
    if verbose:
        print('  Combining...')
    weights = combo_weights or DEFAULT_COMBO_WEIGHTS
    btc_idx = bars_1h['BTC'].index
    combined = combine_targets_hedge(traces, weights, btc_idx)
    if verbose:
        print(f'    combined: {len(combined)} 1h slots')

    # Engine
    if verbose:
        print('  Running engine...')
    eng_kwargs = {
        'leverage': 3.0,
        'tx_cost': 0.0004,
        'maint_rate': 0.004,
        'initial_capital': 10000.0,
        'stop_kind': 'highest_close_since_entry_pct',
        'stop_pct': 0.08,
        'tp_partial_pct': 0.05,
        'tp_partial_frac': 0.5,
        'tp_trail_pct': 0.025,
        'max_gross': 3.0,
        'cash_floor': 0.05,
        'coin_caps': coin_caps,
        'fill_mode': 'open',
    }
    if engine_kwargs:
        eng_kwargs.update(engine_kwargs)
    engine = HedgeAccountEngine(bars_1h, funding_1h, **eng_kwargs)
    metrics = engine.run(combined)

    if verbose:
        print(f'  Elapsed: {time.time()-t0:.1f}s')
    return metrics
