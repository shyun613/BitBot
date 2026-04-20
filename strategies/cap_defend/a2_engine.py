"""A2 HedgeAccountEngine — long+short 분리, 7-mode stop, partial TP + trailing,
익스포저 정규화 (gross+CASH 산식). V21 엔진 무수정.

target 형태: {coin: {'long': w_L >= 0, 'short': w_S >= 0}, 'CASH': w_C}
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest_futures_full import SLIPPAGE_MAP


class HedgeAccountEngine:
    def __init__(self, bars_1h, funding,
                 leverage=3.0, tx_cost=0.0004, maint_rate=0.004,
                 initial_capital=10000.0,
                 # Stop
                 stop_kind='none', stop_pct=0.0, stop_lookback_bars=0,
                 stop_atr_lookback_bars=14, stop_atr_mult=2.0,
                 stop_liq_buffer_pct=0.05,
                 reentry_cooldown_bars=0,
                 # Partial TP + trailing
                 tp_partial_pct=0.0, tp_partial_frac=0.5, tp_trail_pct=0.0,
                 # Allocation
                 max_gross=3.0, cash_floor=0.05,
                 coin_caps=None,  # {coin: {'long': X, 'short': Y}}
                 # Execution
                 fill_mode='open'):
        self.bars = bars_1h
        self.funding = funding
        self.leverage = leverage
        self.tx_cost = tx_cost
        self.maint_rate = maint_rate
        self.initial_capital = initial_capital
        self.stop_kind = stop_kind
        self.stop_pct = stop_pct
        self.stop_lookback_bars = stop_lookback_bars
        self.stop_atr_lookback_bars = stop_atr_lookback_bars
        self.stop_atr_mult = stop_atr_mult
        self.stop_liq_buffer_pct = stop_liq_buffer_pct
        self.reentry_cooldown_bars = reentry_cooldown_bars
        self.tp_partial_pct = tp_partial_pct
        self.tp_partial_frac = tp_partial_frac
        self.tp_trail_pct = tp_trail_pct
        self.max_gross = max_gross
        self.cash_floor = cash_floor
        self.coin_caps = coin_caps or {}
        self.fill_mode = fill_mode

    # ---------- helpers ----------
    def _get_price(self, coin, date):
        df = self.bars.get(coin)
        if df is None: return 0
        ci = df.index.get_indexer([date], method='ffill')[0]
        return float(df['Close'].iloc[ci]) if ci >= 0 else 0

    def _get_fill_price(self, coin, date):
        df = self.bars.get(coin)
        if df is None: return 0
        if self.fill_mode == 'close':
            return self._get_price(coin, date)
        try:
            ci = df.index.get_loc(date)
        except KeyError:
            return 0
        return float(df['Open'].iloc[ci])

    def _get_bar_index(self, coin, date):
        df = self.bars.get(coin)
        if df is None: return -1
        return df.index.get_indexer([date], method='ffill')[0]

    def _calc_atr(self, coin, ci, lookback):
        df = self.bars.get(coin)
        if df is None or lookback <= 0 or ci < lookback + 1:
            return None
        high = df['High'].iloc[ci - lookback:ci].values
        low = df['Low'].iloc[ci - lookback:ci].values
        prev_close = df['Close'].iloc[ci - lookback - 1:ci - 1].values
        if len(high) != lookback or len(prev_close) != lookback:
            return None
        tr = np.maximum.reduce([
            high - low,
            np.abs(high - prev_close),
            np.abs(low - prev_close),
        ])
        atr = float(np.mean(tr))
        return atr if atr > 0 else None

    # ---------- liq prices ----------
    def _get_liq_price_long(self, coin):
        qty = self.long_holdings.get(coin, 0)
        entry = self.long_entry.get(coin, 0)
        margin = self.long_margin.get(coin, 0)
        if qty <= 0 or entry <= 0: return None
        denom = qty * (1.0 - self.maint_rate)
        if denom <= 0: return None
        liq = (qty * entry - margin) / denom
        return liq if liq > 0 else None

    def _get_liq_price_short(self, coin):
        qty = self.short_holdings.get(coin, 0)
        entry = self.short_entry.get(coin, 0)
        margin = self.short_margin.get(coin, 0)
        if qty <= 0 or entry <= 0: return None
        denom = qty * (1.0 + self.maint_rate)
        if denom <= 0: return None
        liq = (qty * entry + margin) / denom
        return liq if liq > 0 else None

    def _floor_above_liq_long(self, coin, stop_price):
        if stop_price is None or self.stop_liq_buffer_pct <= 0:
            return stop_price
        liq = self._get_liq_price_long(coin)
        if liq is None or liq <= 0: return stop_price
        return max(stop_price, liq * (1.0 + self.stop_liq_buffer_pct))

    def _cap_below_liq_short(self, coin, stop_price):
        """Short은 가격 상승이 위험 → cap을 liq보다 낮게 유지."""
        if stop_price is None or self.stop_liq_buffer_pct <= 0:
            return stop_price
        liq = self._get_liq_price_short(coin)
        if liq is None or liq <= 0: return stop_price
        return min(stop_price, liq * (1.0 - self.stop_liq_buffer_pct))

    # ---------- stop prices (7-mode) ----------
    def _get_stop_price_long(self, coin, date, target=None):
        df = self.bars.get(coin)
        if df is None or self.stop_kind == 'none': return None
        ci = self._get_bar_index(coin, date)
        if ci <= 0: return None

        if self.stop_kind == 'prev_close_pct':
            if self.stop_pct <= 0: return None
            ref = float(df['Close'].iloc[ci - 1])
        elif self.stop_kind == 'highest_close_since_entry_pct':
            if self.stop_pct <= 0: return None
            start_ci = self.long_entry_bar.get(coin, -1)
            if start_ci < 0: return None
            ref = float(np.max(df['Close'].iloc[start_ci:ci]))
        elif self.stop_kind == 'highest_high_since_entry_pct':
            if self.stop_pct <= 0: return None
            start_ci = self.long_entry_bar.get(coin, -1)
            if start_ci < 0: return None
            ref = float(np.max(df['High'].iloc[start_ci:ci]))
        elif self.stop_kind == 'rolling_high_close_pct':
            if self.stop_pct <= 0 or self.stop_lookback_bars <= 0 or ci < self.stop_lookback_bars: return None
            ref = float(np.max(df['Close'].iloc[ci - self.stop_lookback_bars:ci]))
        elif self.stop_kind == 'rolling_high_high_pct':
            if self.stop_pct <= 0 or self.stop_lookback_bars <= 0 or ci < self.stop_lookback_bars: return None
            ref = float(np.max(df['High'].iloc[ci - self.stop_lookback_bars:ci]))
        elif self.stop_kind == 'atr_highest_high_since_entry':
            start_ci = self.long_entry_bar.get(coin, -1)
            atr = self._calc_atr(coin, ci, self.stop_atr_lookback_bars)
            if start_ci < 0 or atr is None or self.stop_atr_mult <= 0: return None
            ref = float(np.max(df['High'].iloc[start_ci:ci]))
            return self._floor_above_liq_long(coin, max(ref - self.stop_atr_mult * atr, 0.0))
        elif self.stop_kind == 'atr_rolling_high_high':
            atr = self._calc_atr(coin, ci, self.stop_atr_lookback_bars)
            if self.stop_lookback_bars <= 0 or ci < self.stop_lookback_bars or atr is None or self.stop_atr_mult <= 0:
                return None
            ref = float(np.max(df['High'].iloc[ci - self.stop_lookback_bars:ci]))
            return self._floor_above_liq_long(coin, max(ref - self.stop_atr_mult * atr, 0.0))
        else:
            return None

        if ref <= 0: return None
        return self._floor_above_liq_long(coin, ref * (1.0 - self.stop_pct))

    def _get_stop_price_short(self, coin, date, target=None):
        """Short stop: ref는 *_low_since_entry / rolling_low / atr 기반, ref*(1+stop_pct)."""
        df = self.bars.get(coin)
        if df is None or self.stop_kind == 'none': return None
        ci = self._get_bar_index(coin, date)
        if ci <= 0: return None

        if self.stop_kind == 'prev_close_pct':
            if self.stop_pct <= 0: return None
            ref = float(df['Close'].iloc[ci - 1])
        elif self.stop_kind == 'highest_close_since_entry_pct':
            # short에서는 lowest_close_since_entry 의미로 사용
            if self.stop_pct <= 0: return None
            start_ci = self.short_entry_bar.get(coin, -1)
            if start_ci < 0: return None
            ref = float(np.min(df['Close'].iloc[start_ci:ci]))
        elif self.stop_kind == 'highest_high_since_entry_pct':
            if self.stop_pct <= 0: return None
            start_ci = self.short_entry_bar.get(coin, -1)
            if start_ci < 0: return None
            ref = float(np.min(df['Low'].iloc[start_ci:ci]))
        elif self.stop_kind == 'rolling_high_close_pct':
            if self.stop_pct <= 0 or self.stop_lookback_bars <= 0 or ci < self.stop_lookback_bars: return None
            ref = float(np.min(df['Close'].iloc[ci - self.stop_lookback_bars:ci]))
        elif self.stop_kind == 'rolling_high_high_pct':
            if self.stop_pct <= 0 or self.stop_lookback_bars <= 0 or ci < self.stop_lookback_bars: return None
            ref = float(np.min(df['Low'].iloc[ci - self.stop_lookback_bars:ci]))
        elif self.stop_kind == 'atr_highest_high_since_entry':
            start_ci = self.short_entry_bar.get(coin, -1)
            atr = self._calc_atr(coin, ci, self.stop_atr_lookback_bars)
            if start_ci < 0 or atr is None or self.stop_atr_mult <= 0: return None
            ref = float(np.min(df['Low'].iloc[start_ci:ci]))
            sp = ref + self.stop_atr_mult * atr
            return self._cap_below_liq_short(coin, sp)
        elif self.stop_kind == 'atr_rolling_high_high':
            atr = self._calc_atr(coin, ci, self.stop_atr_lookback_bars)
            if self.stop_lookback_bars <= 0 or ci < self.stop_lookback_bars or atr is None or self.stop_atr_mult <= 0:
                return None
            ref = float(np.min(df['Low'].iloc[ci - self.stop_lookback_bars:ci]))
            return self._cap_below_liq_short(coin, ref + self.stop_atr_mult * atr)
        else:
            return None

        if ref <= 0: return None
        return self._cap_below_liq_short(coin, ref * (1.0 + self.stop_pct))

    # ---------- exits ----------
    def _execute_long_exit(self, coin, date, exit_price):
        ci = self._get_bar_index(coin, date)
        if ci < 0: return False
        df = self.bars[coin]
        low = float(df['Low'].iloc[ci])
        if low <= 0 or low > exit_price:
            return False
        slip = SLIPPAGE_MAP.get(coin, 0.0005)
        cur_open = float(df['Open'].iloc[ci])
        ep = min(cur_open, exit_price) * (1 - slip)
        qty = self.long_holdings[coin]
        pnl = qty * (ep - self.long_entry[coin])
        tx = qty * ep * self.tx_cost
        self.capital += self.long_margin[coin] + pnl - tx
        self._clear_long(coin)
        if self.reentry_cooldown_bars > 0:
            self.reentry_long[coin] = self.reentry_cooldown_bars
        return True

    def _execute_short_exit(self, coin, date, exit_price):
        ci = self._get_bar_index(coin, date)
        if ci < 0: return False
        df = self.bars[coin]
        high = float(df['High'].iloc[ci])
        if high <= 0 or high < exit_price:
            return False
        slip = SLIPPAGE_MAP.get(coin, 0.0005)
        cur_open = float(df['Open'].iloc[ci])
        ep = max(cur_open, exit_price) * (1 + slip)
        qty = self.short_holdings[coin]
        pnl = qty * (self.short_entry[coin] - ep)
        tx = qty * ep * self.tx_cost
        self.capital += self.short_margin[coin] + pnl - tx
        self._clear_short(coin)
        if self.reentry_cooldown_bars > 0:
            self.reentry_short[coin] = self.reentry_cooldown_bars
        return True

    def _execute_partial_long(self, coin, date, tp_price):
        if self.tp_partial_frac <= 0 or self.tp_partial_frac >= 1: return False
        ci = self._get_bar_index(coin, date)
        if ci < 0: return False
        df = self.bars[coin]
        slip = SLIPPAGE_MAP.get(coin, 0.0005)
        cur_open = float(df['Open'].iloc[ci])
        ep = max(cur_open, tp_price) * (1 - slip)  # TP fills at >= tp_price
        qty = self.long_holdings[coin]
        sell_qty = qty * self.tp_partial_frac
        sell_margin = self.long_margin[coin] * self.tp_partial_frac
        pnl = sell_qty * (ep - self.long_entry[coin])
        tx = sell_qty * ep * self.tx_cost
        self.capital += sell_margin + pnl - tx
        self.long_holdings[coin] -= sell_qty
        self.long_margin[coin] -= sell_margin
        self.long_partial_done[coin] = True
        return True

    def _execute_partial_short(self, coin, date, tp_price):
        if self.tp_partial_frac <= 0 or self.tp_partial_frac >= 1: return False
        ci = self._get_bar_index(coin, date)
        if ci < 0: return False
        df = self.bars[coin]
        slip = SLIPPAGE_MAP.get(coin, 0.0005)
        cur_open = float(df['Open'].iloc[ci])
        ep = min(cur_open, tp_price) * (1 + slip)
        qty = self.short_holdings[coin]
        buy_qty = qty * self.tp_partial_frac
        buy_margin = self.short_margin[coin] * self.tp_partial_frac
        pnl = buy_qty * (self.short_entry[coin] - ep)
        tx = buy_qty * ep * self.tx_cost
        self.capital += buy_margin + pnl - tx
        self.short_holdings[coin] -= buy_qty
        self.short_margin[coin] -= buy_margin
        self.short_partial_done[coin] = True
        return True

    def _clear_long(self, coin):
        for d in (self.long_holdings, self.long_entry, self.long_margin,
                  self.long_entry_bar, self.long_peak, self.long_partial_done):
            d.pop(coin, None)

    def _clear_short(self, coin):
        for d in (self.short_holdings, self.short_entry, self.short_margin,
                  self.short_entry_bar, self.short_trough, self.short_partial_done):
            d.pop(coin, None)

    # ---------- exposure normalization ----------
    def _normalize_target(self, target):
        """Plan §1: gross 산정 → MAX_GROSS×(1-CASH_FLOOR) 한도 → 비례 downscale."""
        if not target: return {}
        norm = {}
        for coin, sides in target.items():
            if coin == 'CASH': continue
            if not isinstance(sides, dict): continue
            wL = max(0.0, float(sides.get('long', 0.0)))
            wS = max(0.0, float(sides.get('short', 0.0)))
            cap = self.coin_caps.get(coin, {})
            wL = min(wL, cap.get('long', wL))
            wS = min(wS, cap.get('short', wS))
            if wL > 0 or wS > 0:
                norm[coin] = {'long': wL, 'short': wS}
        gross_long = sum(s['long'] for s in norm.values())
        gross_short = sum(s['short'] for s in norm.values())
        gross_total = gross_long + gross_short
        usable = self.max_gross * (1.0 - self.cash_floor)
        if gross_total > usable and gross_total > 0:
            scale = usable / gross_total
            for coin in norm:
                norm[coin]['long'] *= scale
                norm[coin]['short'] *= scale
            gross_total = usable
        cash = max(0.0, 1.0 - gross_total / max(self.max_gross, 1e-9))
        norm['CASH'] = cash
        return norm

    # ---------- rebalance ----------
    def _execute_rebalance(self, target, date):
        target = self._normalize_target(target)
        if not target: return

        # Compute PV
        pv = self.capital
        for coin in self.long_holdings:
            cur = self._get_price(coin, date)
            if cur > 0:
                pv += self.long_margin[coin] + self.long_holdings[coin] * (cur - self.long_entry[coin])
        for coin in self.short_holdings:
            cur = self._get_price(coin, date)
            if cur > 0:
                pv += self.short_margin[coin] + self.short_holdings[coin] * (self.short_entry[coin] - cur)
        if pv <= 0: return

        # Compute target qty per coin per side
        # Convert weight (fraction of pv) to margin: each unit weight = pv * w * 0.95 margin × leverage notional
        target_qty_long = {}
        target_qty_short = {}
        target_margin_long = {}
        target_margin_short = {}
        for coin, sides in target.items():
            if coin == 'CASH': continue
            cur = self._get_fill_price(coin, date)
            if cur <= 0: continue
            wL = sides.get('long', 0.0)
            wS = sides.get('short', 0.0)
            if wL > 0:
                tmgn = pv * wL / self.leverage  # weight = notional fraction; margin = notional/leverage
                tnot = pv * wL
                target_qty_long[coin] = tnot / cur
                target_margin_long[coin] = tmgn
            if wS > 0:
                tmgn = pv * wS / self.leverage
                tnot = pv * wS
                target_qty_short[coin] = tnot / cur
                target_margin_short[coin] = tmgn

        # Reduce-first: close removed sides, partial reduce existing
        self._reduce_side(self.long_holdings, self.long_entry, self.long_margin,
                         self.long_entry_bar, target_qty_long, target_margin_long,
                         date, side='long')
        self._reduce_side(self.short_holdings, self.short_entry, self.short_margin,
                         self.short_entry_bar, target_qty_short, target_margin_short,
                         date, side='short')

        # Open / increase positions
        self._open_side(self.long_holdings, self.long_entry, self.long_margin,
                       self.long_entry_bar, self.long_peak, self.long_partial_done,
                       self.reentry_long,
                       target_qty_long, target_margin_long, date, side='long')
        self._open_side(self.short_holdings, self.short_entry, self.short_margin,
                       self.short_entry_bar, self.short_trough, self.short_partial_done,
                       self.reentry_short,
                       target_qty_short, target_margin_short, date, side='short')

    def _reduce_side(self, holdings, entry, margin, entry_bar,
                    tqty, tmgn, date, side):
        for coin in list(holdings.keys()):
            cur = self._get_fill_price(coin, date)
            if cur <= 0: continue
            slip = SLIPPAGE_MAP.get(coin, 0.0005)
            if coin not in tqty:
                # Full close
                qty = holdings[coin]
                if side == 'long':
                    ep = cur * (1 - slip)
                    pnl = qty * (ep - entry[coin])
                else:
                    ep = cur * (1 + slip)
                    pnl = qty * (entry[coin] - ep)
                tx = qty * ep * self.tx_cost
                self.capital += margin[coin] + pnl - tx
                if side == 'long':
                    self._clear_long(coin)
                else:
                    self._clear_short(coin)
            else:
                delta = tqty[coin] - holdings[coin]
                if delta < -holdings[coin] * 0.05:
                    sell_qty = -delta
                    sell_frac = sell_qty / holdings[coin]
                    sell_margin = margin[coin] * sell_frac
                    if side == 'long':
                        ep = cur * (1 - slip)
                        pnl = sell_qty * (ep - entry[coin])
                    else:
                        ep = cur * (1 + slip)
                        pnl = sell_qty * (entry[coin] - ep)
                    tx = sell_qty * ep * self.tx_cost
                    self.capital += sell_margin + pnl - tx
                    holdings[coin] -= sell_qty
                    margin[coin] -= sell_margin

    def _open_side(self, holdings, entry, margin, entry_bar, peak_or_trough,
                  partial_done, reentry, tqty, tmgn, date, side):
        for coin, q in tqty.items():
            cur = self._get_fill_price(coin, date)
            if cur <= 0: continue
            if coin in reentry and coin not in holdings:
                continue
            slip = SLIPPAGE_MAP.get(coin, 0.0005)
            if coin not in holdings:
                if side == 'long':
                    ep = cur * (1 + slip)
                else:
                    ep = cur * (1 - slip)
                m = tmgn[coin]
                notional = m * self.leverage
                qty = notional / ep
                tx = notional * self.tx_cost
                if self.capital >= m + tx:
                    self.capital -= (m + tx)
                    holdings[coin] = qty
                    entry[coin] = ep
                    margin[coin] = m
                    entry_bar[coin] = self._get_bar_index(coin, date)
                    peak_or_trough[coin] = ep
                    partial_done[coin] = False
            else:
                delta = q - holdings[coin]
                if delta > holdings[coin] * 0.05:
                    if side == 'long':
                        ep = cur * (1 + slip)
                    else:
                        ep = cur * (1 - slip)
                    add_notional = delta * ep
                    add_margin = add_notional / self.leverage
                    tx = add_notional * self.tx_cost
                    if self.capital >= add_margin + tx:
                        self.capital -= (add_margin + tx)
                        total = holdings[coin] + delta
                        entry[coin] = (entry[coin] * holdings[coin] + ep * delta) / total
                        holdings[coin] = total
                        margin[coin] += add_margin

    # ---------- main loop ----------
    def run(self, target_series):
        self.capital = self.initial_capital
        self.long_holdings = {}; self.short_holdings = {}
        self.long_entry = {}; self.short_entry = {}
        self.long_margin = {}; self.short_margin = {}
        self.long_entry_bar = {}; self.short_entry_bar = {}
        self.long_peak = {}; self.short_trough = {}
        self.long_partial_done = {}; self.short_partial_done = {}
        self.reentry_long = {}; self.reentry_short = {}

        liq_count = stop_count = tp_count = rebal_count = 0
        pv_list = []

        norm_funding = {}
        for coin, fr in self.funding.items():
            norm_funding[coin] = fr.copy()
            norm_funding[coin].index = norm_funding[coin].index.floor('h')

        prev_target = {}

        for date, target in target_series:
            # Decrement reentry cooldowns
            for coin in list(self.reentry_long.keys()):
                self.reentry_long[coin] -= 1
                if self.reentry_long[coin] <= 0:
                    del self.reentry_long[coin]
            for coin in list(self.reentry_short.keys()):
                self.reentry_short[coin] -= 1
                if self.reentry_short[coin] <= 0:
                    del self.reentry_short[coin]

            # Process LONG: peak update → partial TP → trailing/stop → liq
            for coin in list(self.long_holdings.keys()):
                df = self.bars.get(coin)
                if df is None: continue
                ci = df.index.get_indexer([date], method='ffill')[0]
                if ci < 0: continue
                low = float(df['Low'].iloc[ci])
                high = float(df['High'].iloc[ci])
                if low <= 0: continue

                if ci >= 1:
                    prev_close = float(df['Close'].iloc[ci - 1])
                    if prev_close > self.long_peak.get(coin, 0):
                        self.long_peak[coin] = prev_close

                # Partial TP check (before stop)
                if not self.long_partial_done.get(coin, False) and self.tp_partial_pct > 0:
                    tp_price = self.long_entry[coin] * (1 + self.tp_partial_pct)
                    if high >= tp_price:
                        if self._execute_partial_long(coin, date, tp_price):
                            tp_count += 1

                # If partial done → trailing stop only
                if self.long_partial_done.get(coin, False) and self.tp_trail_pct > 0:
                    trail = self.long_peak.get(coin, self.long_entry[coin]) * (1.0 - self.tp_trail_pct)
                    trail = self._floor_above_liq_long(coin, trail)
                    if low <= trail:
                        if self._execute_long_exit(coin, date, trail):
                            stop_count += 1
                        continue

                # Otherwise normal stop + liq
                stop_price = self._get_stop_price_long(coin, date, target) if self.stop_kind != 'none' else None
                liq_price = self._get_liq_price_long(coin)
                hit_stop = stop_price is not None and low <= stop_price
                hit_liq = liq_price is not None and low <= liq_price
                if hit_stop and (not hit_liq or stop_price > liq_price):
                    if self._execute_long_exit(coin, date, stop_price):
                        stop_count += 1
                    continue
                if hit_liq:
                    qty = self.long_holdings[coin]
                    pnl = qty * (low - self.long_entry[coin])
                    eq = self.long_margin[coin] + pnl
                    returned = max(eq - max(eq, 0) * 0.015, 0)
                    self.capital += returned
                    self._clear_long(coin)
                    liq_count += 1

            # Process SHORT: trough update → partial TP → trailing/stop → liq
            for coin in list(self.short_holdings.keys()):
                df = self.bars.get(coin)
                if df is None: continue
                ci = df.index.get_indexer([date], method='ffill')[0]
                if ci < 0: continue
                low = float(df['Low'].iloc[ci])
                high = float(df['High'].iloc[ci])
                if high <= 0: continue

                if ci >= 1:
                    prev_close = float(df['Close'].iloc[ci - 1])
                    if prev_close < self.short_trough.get(coin, float('inf')):
                        self.short_trough[coin] = prev_close

                if not self.short_partial_done.get(coin, False) and self.tp_partial_pct > 0:
                    tp_price = self.short_entry[coin] * (1.0 - self.tp_partial_pct)
                    if low <= tp_price:
                        if self._execute_partial_short(coin, date, tp_price):
                            tp_count += 1

                if self.short_partial_done.get(coin, False) and self.tp_trail_pct > 0:
                    trail = self.short_trough.get(coin, self.short_entry[coin]) * (1.0 + self.tp_trail_pct)
                    trail = self._cap_below_liq_short(coin, trail)
                    if high >= trail:
                        if self._execute_short_exit(coin, date, trail):
                            stop_count += 1
                        continue

                stop_price = self._get_stop_price_short(coin, date, target) if self.stop_kind != 'none' else None
                liq_price = self._get_liq_price_short(coin)
                hit_stop = stop_price is not None and high >= stop_price
                hit_liq = liq_price is not None and high >= liq_price
                if hit_stop and (not hit_liq or stop_price < liq_price):
                    if self._execute_short_exit(coin, date, stop_price):
                        stop_count += 1
                    continue
                if hit_liq:
                    qty = self.short_holdings[coin]
                    pnl = qty * (self.short_entry[coin] - high)
                    eq = self.short_margin[coin] + pnl
                    returned = max(eq - max(eq, 0) * 0.015, 0)
                    self.capital += returned
                    self._clear_short(coin)
                    liq_count += 1

            # Funding
            all_held = set(list(self.long_holdings.keys()) + list(self.short_holdings.keys()))
            for coin in all_held:
                fr_series = norm_funding.get(coin)
                if fr_series is None: continue
                if date in fr_series.index:
                    fr = float(fr_series.loc[date])
                    if fr != 0 and not np.isnan(fr):
                        cur = self._get_price(coin, date)
                        if cur > 0:
                            if coin in self.long_holdings:
                                self.capital -= self.long_holdings[coin] * cur * fr
                            if coin in self.short_holdings:
                                self.capital += self.short_holdings[coin] * cur * fr
            self.capital = max(self.capital, 0)

            # Rebalance
            need_rebal = (target != prev_target)
            if need_rebal and target:
                self._execute_rebalance(target, date)
                rebal_count += 1

            # Compute PV
            pv = self.capital
            for coin in self.long_holdings:
                cur = self._get_price(coin, date)
                if cur > 0:
                    pv += self.long_margin[coin] + self.long_holdings[coin] * (cur - self.long_entry[coin])
            for coin in self.short_holdings:
                cur = self._get_price(coin, date)
                if cur > 0:
                    pv += self.short_margin[coin] + self.short_holdings[coin] * (self.short_entry[coin] - cur)
            pv_list.append({'Date': date, 'Value': max(pv, 0)})
            prev_target = target

        if not pv_list:
            return {}
        pvdf = pd.DataFrame(pv_list).set_index('Date')
        eq = pvdf['Value']
        eq_daily = eq.resample('D').last().dropna()
        yrs = (eq_daily.index[-1] - eq_daily.index[0]).days / 365.25
        if eq_daily.iloc[-1] <= 0 or yrs <= 0:
            return {'Sharpe': 0, 'CAGR': -1, 'MDD': -1, 'Cal': 0,
                    'Liq': liq_count, 'Stops': stop_count, 'TP': tp_count,
                    'Rebal': rebal_count, '_equity': eq}
        cagr = (eq_daily.iloc[-1] / eq_daily.iloc[0]) ** (1 / yrs) - 1
        dr = eq_daily.pct_change().dropna()
        sh = dr.mean() / dr.std() * np.sqrt(365) if dr.std() > 0 else 0
        mdd = (eq / eq.cummax() - 1).min()
        cal = cagr / abs(mdd) if mdd != 0 else 0
        mdd_m_list = []
        for offset in range(0, 30, 3):
            sampled = eq_daily.iloc[offset::30]
            if len(sampled) >= 2:
                mdd_m_list.append(float((sampled / sampled.cummax() - 1).min()))
        if mdd_m_list:
            mdd_m_avg = float(np.mean(mdd_m_list))
        else:
            mdd_m_avg = mdd
        cal_m = cagr / abs(mdd_m_avg) if mdd_m_avg != 0 else 0
        return {'Sharpe': sh, 'CAGR': cagr, 'MDD': mdd, 'Cal': cal,
                'MDD_m_avg': mdd_m_avg, 'Cal_m': cal_m,
                'Liq': liq_count, 'Stops': stop_count, 'TP': tp_count,
                'Rebal': rebal_count, '_equity': eq}


def combine_targets_hedge(traces, weights, all_dates_1h):
    """nested dict trace를 1h 기준 합산.

    각 trace entry: {'date': ts, 'target': {coin: {'long': w, 'short': w}, ...}}
    """
    sorted_traces = {}
    for key, trace in traces.items():
        entries = [(entry['date'], entry['target']) for entry in trace]
        entries.sort(key=lambda x: x[0])
        sorted_traces[key] = entries

    trace_idx = {key: 0 for key in traces}
    combined_series = []
    latest_targets = {key: {} for key in traces}

    for date in all_dates_1h:
        for key in traces:
            entries = sorted_traces[key]
            idx = trace_idx[key]
            while idx < len(entries) and entries[idx][0] <= date:
                latest_targets[key] = entries[idx][1]
                idx += 1
            trace_idx[key] = idx

        merged = {}
        for key, w in weights.items():
            for coin, sides in latest_targets[key].items():
                if coin == 'CASH' or not isinstance(sides, dict):
                    continue
                if coin not in merged:
                    merged[coin] = {'long': 0.0, 'short': 0.0}
                merged[coin]['long'] += sides.get('long', 0.0) * w
                merged[coin]['short'] += sides.get('short', 0.0) * w
        combined_series.append((date, merged))

    return combined_series
