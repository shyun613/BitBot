"""A2 strategy — swing(4h) + short-term overlay(1h+15m) traces.

Traces are list of {'date': ts, 'target': {coin: {'long': w, 'short': w}}}.
HedgeAccountEngine.combine_targets_hedge가 1h grid로 합산.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from a2_indicators import sma, donchian, tsmom, rsi, bollinger


DEFAULT_COIN_CAPS = {
    'BTC': {'long': 1.5, 'short': 1.0},
    'ETH': {'long': 1.0, 'short': 0.7},
    'SOL': {'long': 0.5, 'short': 0.0},
}

DEFAULT_PARAMS = {
    'swing_sma_n': 200,
    'swing_donchian_n': 55,
    'swing_tsmom_n': 84,
    'swing_rsi_n': 14,
    'swing_rsi_overbought': 75.0,
    'swing_long_min_votes': 2,
    'overlay_sma_1h': 48,
    'overlay_bb_n': 20,
    'overlay_bb_k': 2.0,
    'overlay_weight_cap': 0.3,
}


def _all_index_union(d):
    """Union sorted index across dict-of-DataFrame."""
    if not d:
        return pd.DatetimeIndex([])
    out = set()
    for df in d.values():
        out |= set(df.index)
    return pd.DatetimeIndex(sorted(out))


def compute_swing_traces(bars_4h, coin_caps=None, params=None):
    """4h close 시점에 long/short 시그널 계산.
    - long: SMA200 / Donchian55 / TSMOM84 중 min_votes 이상 ON, RSI14 ≤ overbought.
    - short: AND-gate (Close<SMA, Close<Donch_low, TSMOM<0).
    """
    coin_caps = coin_caps or DEFAULT_COIN_CAPS
    p = {**DEFAULT_PARAMS, **(params or {})}

    sigs = {}
    for coin, df in bars_4h.items():
        close = df['Close']
        high = df['High']
        low = df['Low']
        sma_s = sma(close, p['swing_sma_n'])
        donch_up, donch_lo = donchian(high, low, p['swing_donchian_n'])
        tsmom_s = tsmom(close, p['swing_tsmom_n'])
        rsi_s = rsi(close, p['swing_rsi_n'])

        c_l1 = close > sma_s
        c_l2 = close > donch_up
        c_l3 = tsmom_s > 0
        votes = c_l1.astype(int).fillna(0) + c_l2.astype(int).fillna(0) + c_l3.astype(int).fillna(0)
        long_sig = (votes >= p['swing_long_min_votes']) & (rsi_s <= p['swing_rsi_overbought'])

        c_s1 = close < sma_s
        c_s2 = close < donch_lo
        c_s3 = tsmom_s < 0
        short_sig = c_s1 & c_s2 & c_s3

        sigs[coin] = (long_sig.fillna(False), short_sig.fillna(False))

    all_dates = _all_index_union(bars_4h)
    trace = []
    for date in all_dates:
        target = {}
        for coin, (long_s, short_s) in sigs.items():
            if date not in long_s.index:
                continue
            wL = coin_caps.get(coin, {}).get('long', 1.0) if bool(long_s.loc[date]) else 0.0
            wS = coin_caps.get(coin, {}).get('short', 1.0) if bool(short_s.loc[date]) else 0.0
            if wL > 0 or wS > 0:
                target[coin] = {'long': float(wL), 'short': float(wS)}
        trace.append({'date': date + pd.Timedelta('4h'), 'target': target})
    return trace


def compute_overlay_traces(bars_1h, bars_15m=None, coin_caps=None, params=None):
    """1h 그리드에 score = clip(sma_1h_dir + bb15_breakout, -1, +1) → long/short cap 매핑."""
    coin_caps = coin_caps or DEFAULT_COIN_CAPS
    p = {**DEFAULT_PARAMS, **(params or {})}
    bars_15m = bars_15m or {}

    sigs = {}
    for coin, df in bars_1h.items():
        close = df['Close']
        sma_s = sma(close, p['overlay_sma_1h'])
        sma_dir = pd.Series(0, index=close.index, dtype='int64')
        sma_dir[close > sma_s] = 1
        sma_dir[close < sma_s] = -1

        bb_dir = pd.Series(0, index=close.index, dtype='int64')
        if coin in bars_15m and len(bars_15m[coin]) > 0:
            df15 = bars_15m[coin]
            _, up, lo = bollinger(df15['Close'], p['overlay_bb_n'], p['overlay_bb_k'])
            b15 = pd.Series(0, index=df15.index, dtype='int64')
            b15[df15['Close'] > up] = 1
            b15[df15['Close'] < lo] = -1
            b1h = b15.resample('1h').last().fillna(0)
            b1h = b1h.reindex(close.index, method='ffill').fillna(0).astype('int64')
            bb_dir = b1h

        score = (sma_dir + bb_dir).clip(-1, 1).astype(float)
        sigs[coin] = score

    all_dates = _all_index_union(bars_1h)
    overlay_w = float(p['overlay_weight_cap'])
    trace = []
    for date in all_dates:
        target = {}
        for coin, score_s in sigs.items():
            if date not in score_s.index:
                continue
            sc = float(score_s.loc[date])
            cap_l = coin_caps.get(coin, {}).get('long', 1.0)
            cap_s = coin_caps.get(coin, {}).get('short', 1.0)
            wL = cap_l * overlay_w * max(0.0, sc)
            wS = cap_s * overlay_w * max(0.0, -sc)
            if wL > 0 or wS > 0:
                target[coin] = {'long': float(wL), 'short': float(wS)}
        trace.append({'date': date + pd.Timedelta('1h'), 'target': target})
    return trace


def build_a2_traces(bars_1h, bars_4h, bars_15m=None, coin_caps=None, params=None):
    """Returns dict ready for combine_targets_hedge."""
    swing = compute_swing_traces(bars_4h, coin_caps=coin_caps, params=params)
    overlay = compute_overlay_traces(bars_1h, bars_15m=bars_15m, coin_caps=coin_caps, params=params)
    return {'swing': swing, 'overlay': overlay}


DEFAULT_COMBO_WEIGHTS = {'swing': 1.0, 'overlay': 0.0}
