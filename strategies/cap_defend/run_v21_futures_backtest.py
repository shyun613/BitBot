#!/usr/bin/env python3
"""V21 L3 선물 앙상블 백테스트 (ENS_fut_L3_k3_12652d57, 2026-04-17 확정).

실거래 SSOT: trade/auto_trade_binance.py STRATEGIES.
- 3멤버 EW (1/3씩): 4h_S240_SN120, 4h_S240_SN30, 4h_S120_SN120
- 고정 3x 레버리지, 스탑 없음, 캐시 게이트 없음.
- 앙상블 분산만으로 방어.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest_futures_full import load_data, run
from futures_ensemble_engine import SingleAccountEngine, combine_targets


START = '2020-10-01'
END = '2026-03-28'

# V21 STRATEGIES — auto_trade_binance.py:63-100 SSOT 미러
V21_STRATEGIES = {
    '4h_S240_SN120': dict(
        interval='4h', sma_bars=240, mom_short_bars=20, mom_long_bars=720,
        health_mode='mom2vol', vol_mode='daily', vol_threshold=0.05,
        snap_interval_bars=120, canary_hyst=0.015, n_snapshots=3,
    ),
    '4h_S240_SN30': dict(
        interval='4h', sma_bars=240, mom_short_bars=20, mom_long_bars=480,
        health_mode='mom2vol', vol_mode='daily', vol_threshold=0.05,
        snap_interval_bars=30, canary_hyst=0.015, n_snapshots=3,
    ),
    '4h_S120_SN120': dict(
        interval='4h', sma_bars=120, mom_short_bars=20, mom_long_bars=720,
        health_mode='mom2vol', vol_mode='daily', vol_threshold=0.05,
        snap_interval_bars=120, canary_hyst=0.015, n_snapshots=3,
    ),
}
V21_COMBO = {k: 1/3 for k in V21_STRATEGIES}


def generate_trace(data, cfg):
    run_cfg = dict(cfg)
    interval = run_cfg.pop('interval')
    bars, funding = data[interval]
    trace = []
    run(
        bars, funding,
        interval=interval,
        leverage=1.0,  # 트레이스는 비중만, 실제 레버리지는 엔진에서
        start_date=START, end_date=END,
        _trace=trace,
        **run_cfg,
    )
    return trace


def main():
    t0 = time.time()
    print(f'V21 L3 백테스트 ({START} ~ {END})')
    print(f'  전략: {list(V21_STRATEGIES)}')
    print(f'  가중: EW 1/3 each, 고정 3x, 스탑/가드 없음')
    print()
    print('Loading data...')
    intervals = sorted({cfg['interval'] for cfg in V21_STRATEGIES.values()} | {'1h'})
    data = {iv: load_data(iv) for iv in intervals}
    bars_1h, funding_1h = data['1h']
    all_dates = bars_1h['BTC'].index[(bars_1h['BTC'].index >= START) & (bars_1h['BTC'].index <= END)]

    print('Generating traces...')
    traces = {}
    for name, cfg in V21_STRATEGIES.items():
        t1 = time.time()
        traces[name] = generate_trace(data, cfg)
        print(f'  {name}: {len(traces[name])} rebal points ({time.time()-t1:.1f}s)')

    combined = combine_targets({k: traces[k] for k in V21_COMBO}, V21_COMBO, all_dates)

    print('Running V21 single-account engine (fixed 3x, no stop, no gate)...')
    engine = SingleAccountEngine(
        bars_1h, funding_1h,
        leverage=3.0,
        leverage_mode='fixed',
        per_coin_leverage_mode='none',
        stop_kind='none',
        stop_pct=0.0,
        stop_gate='always',
        stop_gate_cash_threshold=0.0,
        fill_mode='open',
        tx_cost=0.0004,
        maint_rate=0.004,
        initial_capital=10000.0,
    )
    m = engine.run(combined)

    print()
    print('=' * 72)
    print(f'V21 L3 결과 ({START} ~ {END})')
    print('=' * 72)
    print(f'  Sharpe : {m["Sharpe"]:.2f}')
    print(f'  CAGR   : {m["CAGR"]:+.1%}')
    print(f'  MDD    : {m["MDD"]:+.1%}')
    print(f'  Calmar : {m["Cal"]:.2f}')
    print(f'  MDDm   : {m.get("MDD_m_avg", 0):+.1%} (avg of 30-day samples)')
    print(f'  Cal_m  : {m.get("Cal_m", 0):.2f}')
    print(f'  Liq    : {m["Liq"]}')
    print(f'  Stops  : {m.get("Stops", 0)}')
    print(f'  Rebal  : {m.get("Rebal", 0)}')
    print()
    print(f'Elapsed: {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
