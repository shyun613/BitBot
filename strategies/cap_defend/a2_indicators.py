"""A2 indicators — vectorized pandas Series implementations.

기존 V21 helper는 scalar(특정 index 값) 반환이지만, A2 sweep/엔진은 Series 단위 처리가 필요.
모든 함수는 pandas Series in -> Series out, look-ahead bias 없음.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------- Trend ----------

def sma(close, n):
    return close.rolling(n).mean()


def ema(close, span):
    return close.ewm(span=span, adjust=False).mean()


def tsmom(close, n):
    """Time-series momentum: close[t]/close[t-n] - 1."""
    return close / close.shift(n) - 1.0


def donchian(high, low, n):
    """t 시점 upper/lower (직전 n봉 기준, t 봉은 미포함 → look-ahead 없음)."""
    upper = high.shift(1).rolling(n).max()
    lower = low.shift(1).rolling(n).min()
    return upper, lower


# ---------- Mean reversion ----------

def rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.fillna(50.0)


def bollinger(close, n=20, k=2.0):
    mid = close.rolling(n).mean()
    std = close.rolling(n).std(ddof=0)
    upper = mid + k * std
    lower = mid - k * std
    return mid, upper, lower


def macd(close, fast=12, slow=26, signal_n=9):
    macd_line = ema(close, fast) - ema(close, slow)
    sig = macd_line.ewm(span=signal_n, adjust=False).mean()
    hist = macd_line - sig
    return macd_line, sig, hist


# ---------- Volatility / Trend strength ----------

def atr(high, low, close, n=14):
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False).mean()


def adx(high, low, close, n=14):
    up = high.diff()
    dn = -low.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=high.index)
    tr_smooth = atr(high, low, close, n)
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / n, adjust=False).mean() / tr_smooth.replace(0, np.nan)
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / n, adjust=False).mean() / tr_smooth.replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_out = dx.ewm(alpha=1.0 / n, adjust=False).mean()
    return adx_out.fillna(0.0), plus_di.fillna(0.0), minus_di.fillna(0.0)


# ---------- Divergence (look-ahead-safe, pivot 확정 후 시점에 신호) ----------

def _confirmed_pivot_high(s, n):
    """t 시점에 t-n 봉이 trailing 2n+1 윈도우의 max인지 판정."""
    return (s.shift(n) == s.rolling(2 * n + 1).max()) & s.shift(n).notna()


def _confirmed_pivot_low(s, n):
    return (s.shift(n) == s.rolling(2 * n + 1).min()) & s.shift(n).notna()


def _generic_divergence(price, indicator, pivot_n, lookback):
    """
    Bullish: 가장 최근 두 confirmed pivot LOW에서 price LL & indicator HL.
    Bearish: 가장 최근 두 confirmed pivot HIGH에서 price HH & indicator LH.
    """
    bull = pd.Series(False, index=price.index)
    bear = pd.Series(False, index=price.index)

    p_lows = _confirmed_pivot_low(price, pivot_n)
    p_highs = _confirmed_pivot_high(price, pivot_n)

    low_idx = np.flatnonzero(p_lows.to_numpy())
    high_idx = np.flatnonzero(p_highs.to_numpy())

    for i in range(1, len(low_idx)):
        cur, prev = low_idx[i], low_idx[i - 1]
        if cur - prev > lookback:
            continue
        cur_pivot = cur - pivot_n
        prev_pivot = prev - pivot_n
        if cur_pivot < 0 or prev_pivot < 0:
            continue
        if price.iat[cur_pivot] < price.iat[prev_pivot] and indicator.iat[cur_pivot] > indicator.iat[prev_pivot]:
            bull.iat[cur] = True

    for i in range(1, len(high_idx)):
        cur, prev = high_idx[i], high_idx[i - 1]
        if cur - prev > lookback:
            continue
        cur_pivot = cur - pivot_n
        prev_pivot = prev - pivot_n
        if cur_pivot < 0 or prev_pivot < 0:
            continue
        if price.iat[cur_pivot] > price.iat[prev_pivot] and indicator.iat[cur_pivot] < indicator.iat[prev_pivot]:
            bear.iat[cur] = True

    return bull, bear


def rsi_divergence(close, rsi_s, pivot_n=5, lookback=30):
    return _generic_divergence(close, rsi_s, pivot_n, lookback)


def macd_hist_divergence(close, hist, pivot_n=5, lookback=30):
    return _generic_divergence(close, hist, pivot_n, lookback)


__all__ = [
    "sma", "ema", "tsmom", "donchian",
    "rsi", "bollinger", "macd",
    "atr", "adx",
    "rsi_divergence", "macd_hist_divergence",
]
