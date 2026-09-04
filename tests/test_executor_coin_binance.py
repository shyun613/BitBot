#!/usr/bin/env python3
"""executor_coin_binance.py 단위 테스트 (fake Binance client).

pytest 가 없는 환경이라 plain assert + __main__ 러너로 동작한다.
pytest 가 설치되면 `pytest tests/test_executor_coin_binance.py` 로도 그대로 수집된다.

실행:
  cd ~/workspace/BitBot && .venv/bin/python tests/test_executor_coin_binance.py

커버 범위:
  ① exchangeInfo 필터 경계 (stepSize 절사 / tickSize 올림 / minQty / minNotional /
     MARKET_LOT_SIZE / maxQty 캡)
  ② free vs locked 분리 (평가액=total, 주문가능=free)
  ③ 가격 누락 → BalanceIncomplete → run_once exit 2 (+ ignore 목록 예외)
  ④ 불명 응답 후 clientOrderId 재조회로 중복 매도 방지 (C1)
  ⑤ dry-run 에서 주문 API 미호출 · state 미저장
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from decimal import Decimal

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRADE_DIR = os.path.join(REPO_DIR, 'trade')
sys.path.insert(0, TRADE_DIR)

from binance.exceptions import BinanceAPIException  # noqa: E402

import executor_coin_binance as ecb  # noqa: E402
from common.health_guard import UnknownExecutionError  # noqa: E402

# 실 로그 파일 오염 방지 — 모듈 전역 log() 를 캡처로 교체
LOG_LINES = []
ecb.log = lambda msg, *a, **kw: LOG_LINES.append(str(msg))

# 운영 디렉터리 오염 방지 — state/WAL 경로를 테스트 전용 임시 디렉터리로 격리
_TMP_DIR = tempfile.mkdtemp(prefix='ecb_test_')
ecb.CACHE_DIR = _TMP_DIR
atexit.register(lambda: shutil.rmtree(_TMP_DIR, ignore_errors=True))

# 테스트 속도용 지연 제거
ecb.ORDER_LOOKUP_RETRY_DELAY = 0
ecb.FILL_CHECK_DELAY_SEC = 0
ecb.ORDER_WAIT_SEC = 0
ecb.SELL_RETRY_DELAYS = [0, 0, 0, 0]
ecb.AMBIGUOUS_LOOKUP_DELAYS = [0, 0, 0]
ecb.FILL_POLL_DELAYS = [0, 0, 0]


# ═══ Fake Binance client ═══
class FakeAPIException(BinanceAPIException):
    """BinanceAPIException 생성자 의존 없이 code 만 세팅."""

    def __init__(self, code, msg='fake'):
        Exception.__init__(self, f'APIError(code={code}): {msg}')
        self.code = code
        self.message = msg
        self.status_code = 400
        self.response = None
        self.request = None


class FakeTransportError(Exception):
    """네트워크 타임아웃 등 (retryable 로 판정되어야 함)."""


def _sym(symbol, base, status='TRADING', step='0.001', min_qty='0.01', max_qty='1000',
         mkt_step='0', mkt_min_qty='0', mkt_max_qty='0', tick='0.01',
         min_price='0.01', max_price='100000', min_notional='5', max_notional='0',
         legacy_min_notional=None, pct_bid_up=None, pct_bid_down=None,
         extra_filters=None):
    filters = [
        {'filterType': 'LOT_SIZE', 'stepSize': step, 'minQty': min_qty, 'maxQty': max_qty},
        {'filterType': 'PRICE_FILTER', 'tickSize': tick, 'minPrice': min_price, 'maxPrice': max_price},
        # 개수 제한 필터 — 파싱은 하되 수량 산정엔 무관 (미지 필터로 잡히면 안 됨)
        {'filterType': 'MAX_NUM_ORDERS', 'maxNumOrders': 200},
    ]
    if legacy_min_notional is not None:
        filters.append({'filterType': 'MIN_NOTIONAL', 'minNotional': legacy_min_notional,
                        'applyToMarket': True, 'avgPriceMins': 5})
    else:
        filters.append({'filterType': 'NOTIONAL', 'minNotional': min_notional,
                        'maxNotional': max_notional,
                        'applyMinToMarket': True, 'applyMaxToMarket': False})
    if mkt_step != '0' or mkt_max_qty != '0' or mkt_min_qty != '0':
        filters.append({'filterType': 'MARKET_LOT_SIZE', 'stepSize': mkt_step,
                        'minQty': mkt_min_qty, 'maxQty': mkt_max_qty})
    if pct_bid_up is not None:
        filters.append({'filterType': 'PERCENT_PRICE_BY_SIDE',
                        'bidMultiplierUp': pct_bid_up, 'bidMultiplierDown': pct_bid_down or '0',
                        'askMultiplierUp': pct_bid_up, 'askMultiplierDown': pct_bid_down or '0',
                        'avgPriceMins': 5})
    if extra_filters:
        filters.extend(extra_filters)
    return {'symbol': symbol, 'baseAsset': base, 'quoteAsset': 'USDT',
            'status': status, 'filters': filters}


DEFAULT_SYMBOLS = [
    _sym('BTCUSDT', 'BTC', step='0.00001', min_qty='0.00001', max_qty='9000',
         mkt_max_qty='115', tick='0.01', min_notional='5'),
    # 필터 경계 시험용 — LOT step 0.001 / MARKET step 0.01 / minQty 0.01 / minNotional 12
    _sym('TSTUSDT', 'TST', step='0.001', min_qty='0.01', max_qty='1000',
         mkt_step='0.01', mkt_max_qty='500', tick='0.01', min_notional='12',
         max_notional='50000'),
    _sym('LUNCUSDT', 'LUNC', step='1', min_qty='1', max_qty='90000000', tick='0.00000001',
         min_price='0.00000001', min_notional='5'),
    _sym('HALTUSDT', 'HALT', status='HALT', step='0.001', min_qty='0.001', min_notional='5'),
]

DEFAULT_PRICES = {'BTCUSDT': 100000.0, 'TSTUSDT': 100.0, 'LUNCUSDT': 0.00005,
                  'HALTUSDT': 1.0, 'NOMKTUSDT': 1.0}


class FakeClient:
    def __init__(self, symbols=None, balances=None, prices=None, strict_no_orders=False):
        self._symbols = symbols if symbols is not None else DEFAULT_SYMBOLS
        self._balances = dict(balances or {})          # asset -> (free, locked)
        self._prices = dict(prices if prices is not None else DEFAULT_PRICES)
        self.strict_no_orders = strict_no_orders
        self.submissions = []      # 전송 시도한 주문 params (거부 포함)
        self.accepted = {}         # clientOrderId -> order (거래소가 실제 접수)
        self.cancel_calls = []
        self.open_orders = []
        self.open_orders_error = None
        self.open_orders_payload = None   # list 가 아닌 malformed 응답 주입
        # 주문 상태 머신
        #  'ok'            정상 접수·체결
        #  'lost_response' 접수·체결됐지만 응답 유실 (타임아웃)
        #  'not_placed'    접수 안 됨 (네트워크 오류) — 모호
        #  'delayed'       응답 유실 + 조회에도 지연 노출 (visible_after 번째 조회부터 보임)
        #  'reject'        거래소 명시적 사전 거절 (reject_code)
        #  'partial'       부분 체결 (partial_ratio)
        self.sell_behavior = 'ok'
        self.recover_after_first = True
        self.reject_code = -2010
        self.visible_after = 2     # delayed 시 몇 번째 조회부터 보이는지
        self.partial_ratio = 0.4
        self.lookup_fail = False
        self.hide_from_lookup = set()   # 접수됐지만 조회에선 -2013 (NOT_FOUND) 로 보이는 coid
        self.free_qty_error_assets = set()
        self.lookup_calls = 0
        self._delay_counter = {}
        self._seq = 0

    # ─ 읽기 ─
    def get_exchange_info(self):
        return {'symbols': self._symbols}

    def get_account(self, **kw):
        assert kw.get('recvWindow'), 'get_account 에 recvWindow 필요'
        return {'balances': [{'asset': a, 'free': str(f), 'locked': str(l)}
                             for a, (f, l) in self._balances.items()]}

    def get_all_tickers(self):
        return [{'symbol': s, 'price': str(p)} for s, p in self._prices.items()]

    def get_asset_balance(self, asset, **kw):
        assert kw.get('recvWindow'), 'get_asset_balance 에 recvWindow 필요'
        if asset in self.free_qty_error_assets:
            raise FakeTransportError(f'balance query failed for {asset}')
        f, l = self._balances.get(asset, (0, 0))
        return {'asset': asset, 'free': str(f), 'locked': str(l)}

    def get_symbol_ticker(self, symbol):
        if symbol not in self._prices:
            raise FakeTransportError(f'no price for {symbol}')
        return {'symbol': symbol, 'price': str(self._prices[symbol])}

    def get_open_orders(self, **kw):
        assert kw.get('recvWindow'), 'get_open_orders 에 recvWindow 필요'
        if self.open_orders_error:
            raise FakeTransportError(self.open_orders_error)
        if self.open_orders_payload is not None:
            return self.open_orders_payload
        return list(self.open_orders)

    def cancel_order(self, **kw):
        self.cancel_calls.append(kw)
        coid = kw.get('origClientOrderId')
        if coid and coid in self.accepted:
            self.accepted[coid]['status'] = 'CANCELED'
        self.open_orders = [o for o in self.open_orders
                            if o.get('clientOrderId') != coid and o.get('orderId') != kw.get('orderId')]
        return {'status': 'CANCELED'}

    def get_order(self, symbol=None, orderId=None, origClientOrderId=None, **kw):
        self.lookup_calls += 1
        if self.lookup_fail:
            raise FakeTransportError('lookup down')
        if origClientOrderId is not None:
            if origClientOrderId in self.hide_from_lookup:
                raise FakeAPIException(-2013, 'Order does not exist.')
            o = self.accepted.get(origClientOrderId)
            if o is None:
                raise FakeAPIException(-2013, 'Order does not exist.')
            # delayed visibility — 지정 횟수 전까지는 없는 것처럼 응답
            n = self._delay_counter.get(origClientOrderId)
            if n is not None:
                self._delay_counter[origClientOrderId] = n + 1
                if n + 1 < self.visible_after:
                    raise FakeAPIException(-2013, 'Order does not exist.')
            return o
        for o in self.accepted.values():
            if o['orderId'] == orderId:
                return o
        raise FakeAPIException(-2013, 'Order does not exist.')

    # ─ 주문 ─
    def _settle(self, symbol, side, qty):
        """체결 반영 — 실제 거래소처럼 free 잔량을 감소시킨다."""
        base = None
        for s in self._symbols:
            if s['symbol'] == symbol:
                base = s['baseAsset']
                break
        if base is None:
            return
        price = float(self._prices.get(symbol, 0.0))
        bf, bl = self._balances.get(base, (0.0, 0.0))
        uf, ul = self._balances.get('USDT', (0.0, 0.0))
        if side == 'SELL':
            self._balances[base] = (max(0.0, bf - qty), bl)
            self._balances['USDT'] = (uf + qty * price, ul)
        else:
            self._balances[base] = (bf + qty, bl)
            self._balances['USDT'] = (max(0.0, uf - qty * price), ul)

    def _mk_order(self, params, status='FILLED', executed=None, side='SELL',
                  settle_qty=None):
        self._seq += 1
        qty = params['quantity']
        if settle_qty is not None:
            self._settle(params['symbol'], side, float(settle_qty))
        elif status == 'FILLED':
            self._settle(params['symbol'], side, float(qty))
        return {'symbol': params['symbol'], 'orderId': self._seq,
                'clientOrderId': params.get('newClientOrderId'),
                'origQty': qty, 'executedQty': qty if executed is None else executed,
                'status': status, 'price': params.get('price', '0')}

    def order_market_sell(self, **params):
        if self.strict_no_orders:
            raise AssertionError('dry-run 인데 order_market_sell 호출됨')
        self.submissions.append(dict(params, side='SELL'))
        coid = params.get('newClientOrderId')
        behavior = self.sell_behavior
        if behavior != 'ok' and self.recover_after_first:
            self.sell_behavior = 'ok'
        if behavior == 'reject':
            raise FakeAPIException(self.reject_code, 'rejected before matching')
        if behavior == 'lost_response':
            self.accepted[coid] = self._mk_order(params, side='SELL')  # 거래소는 접수·체결
            raise FakeTransportError('Read timed out')                  # 응답만 유실
        if behavior == 'delayed':
            self.accepted[coid] = self._mk_order(params, side='SELL')
            self._delay_counter[coid] = 0                               # 조회에도 지연 노출
            raise FakeTransportError('Read timed out')
        if behavior == 'not_placed':
            raise FakeTransportError('Connection aborted')              # 접수 안 됨(모호)
        if behavior == 'partial':
            qty = float(params['quantity'])
            filled = qty * self.partial_ratio
            o = self._mk_order(params, status='EXPIRED', executed=str(filled),
                               side='SELL', settle_qty=filled)
            self.accepted[coid] = o
            return o
        self.accepted[coid] = self._mk_order(params, side='SELL')
        return self.accepted[coid]

    def order_limit_buy(self, **params):
        if self.strict_no_orders:
            raise AssertionError('dry-run 인데 order_limit_buy 호출됨')
        self.submissions.append(dict(params, side='BUY'))
        coid = params.get('newClientOrderId')
        self.accepted[coid] = self._mk_order(params, status='FILLED', side='BUY')
        return self.accepted[coid]


def _api(client=None, dry_run=False, ignore_assets=()):
    client = client or FakeClient()
    return ecb.BinanceSpotAPI(dry_run=dry_run, ignore_assets=ignore_assets,
                              client=client, run_id='t000000000000')


# ═══ ① 필터 경계 ═══
def test_filter_step_and_tick():
    api = _api()
    # LOT_SIZE stepSize 0.001 절사
    qs, qd, reason = api.prepare_qty('TST', 1.23456, 100.0, market=False)
    assert reason == '', reason
    assert qs == '1.234', qs
    # MARKET_LOT_SIZE stepSize 0.01 이 시장가에 추가 적용
    qs, qd, reason = api.prepare_qty('TST', 1.23456, 100.0, market=True)
    assert reason == '' and qs == '1.23', (qs, reason)
    # tickSize 0.01 올림
    ps, pd, reason = api.prepare_price('TST', 100.001)
    assert reason == '' and ps == '100.01', (ps, reason)
    # 정확히 tick 배수면 그대로
    ps, _, _ = api.prepare_price('TST', 100.00)
    assert ps == '100.00', ps


def test_filter_min_qty_and_notional():
    api = _api()
    # minQty 0.01 미만
    _, _, reason = api.prepare_qty('TST', 0.005, 100.0, market=False)
    assert 'minQty' in reason, reason
    # 심볼 minNotional 12 > 전역 10 → 12 적용
    assert abs(api.min_notional('TST') - 12.0) < 1e-9
    _, _, reason = api.prepare_qty('TST', 0.05, 100.0, market=False)  # notional 5
    assert '최소주문' in reason, reason
    # 심볼 minNotional 5 < 전역 10 → 전역 10 적용
    assert abs(api.min_notional('BTC') - 10.0) < 1e-9
    _, _, reason = api.prepare_qty('BTC', 0.00005, 100000.0, market=False)  # notional 5
    assert '최소주문' in reason, reason
    # 통과 케이스
    qs, _, reason = api.prepare_qty('BTC', 0.001, 100000.0, market=False)   # notional 100
    assert reason == '' and qs == '0.00100', (qs, reason)


def test_filter_max_qty_cap():
    api = _api()
    # MARKET_LOT_SIZE maxQty 500 캡 (LOT maxQty 1000 보다 작음)
    qs, qd, reason = api.prepare_qty('TST', 600, 100.0, market=True)
    assert reason == '' and float(qs) == 500.0, (qs, reason)
    # 지정가는 LOT_SIZE maxQty 1000 적용 (maxNotional 50000 에 걸리지 않는 가격으로)
    qs, _, reason = api.prepare_qty('TST', 1500, 10.0, market=False)
    assert reason == '' and float(qs) == 1000.0, (qs, reason)


def test_filter_max_notional_cap_limit_order():
    api = _api()
    # maxNotional 50000, 지정가(applyMaxToMarket=False 라 지정가에만 적용)
    qs, _, reason = api.prepare_qty('TST', 900, 100.0, market=False)  # notional 90000
    assert reason == '' and float(qs) == 500.0, (qs, reason)  # 50000/100


def test_filter_no_market():
    api = _api()
    _, _, reason = api.prepare_qty('NOPE', 1.0, 1.0, market=False)
    assert 'USDT 마켓 없음' in reason, reason
    assert api.symbol_of('NOPE') is None
    assert api.symbol_status('HALT') == 'HALT'


def test_decimal_no_float_roundtrip():
    """LUNC 처럼 tick 이 1e-8 인 심볼도 지수표기 없이 문자열화."""
    api = _api()
    ps, _, reason = api.prepare_price('LUNC', 0.000045678)
    assert reason == '' and 'e' not in ps.lower(), ps
    assert ps == '0.00004568', ps
    # step 1 절사 (notional 이 최소주문 10 을 넘는 가격으로)
    qs, _, reason = api.prepare_qty('LUNC', 81160.57, 0.0005, market=True)
    assert reason == '' and qs == '81160', (qs, reason)
    # 같은 수량이라도 notional 이 10 미만이면 거부
    _, _, reason = api.prepare_qty('LUNC', 81160.57, 0.00005, market=True)
    assert '최소주문' in reason, reason


# ═══ ② free vs locked ═══
def test_free_vs_locked():
    client = FakeClient(balances={'BTC': (1.0, 0.5), 'USDT': (100.0, 50.0)})
    api = _api(client)
    bal = api.get_balance()
    # 평가액은 total(free+locked)
    assert abs(bal['BTC'] - 1.5 * 100000.0) < 1e-6, bal
    assert abs(bal['USDT'] - 150.0) < 1e-9, bal
    # 주문 가능 수량은 free
    assert abs(api.get_free_qty('BTC') - 1.0) < 1e-12
    assert abs(api.get_total_qty('BTC') - 1.5) < 1e-12
    assert abs(api.get_free_qty('USDT') - 100.0) < 1e-12


def test_sell_uses_free_only():
    """locked 수량은 매도 대상에서 제외된다 (free 1 만 주문, locked 9 는 건드리지 않음)."""
    client = FakeClient(balances={'TST': (1.0, 9.0), 'USDT': (0.0, 0.0)})
    api = _api(client)
    ok, filled = api.sell_market_robust('TST', 10.0)   # 10 요청, free 는 1
    assert len(client.submissions) == 1, client.submissions
    assert float(client.submissions[0]['quantity']) == 1.0, client.submissions
    assert abs(filled - 1.0) < 1e-9, filled
    # free 소진 후 locked 를 건드리지 않고 종료 (원본 '매도 잔량 없음' 규약)
    assert client._balances['TST'][1] == 9.0, client._balances


# ═══ ③ 잔고 불완전 → BalanceIncomplete / exit 2 ═══
def test_balance_incomplete_on_missing_price():
    # 시세 맵은 살아있지만 BTCUSDT 가격만 비정상(0) → 해당 자산만 평가 불가
    client = FakeClient(balances={'BTC': (1.0, 0.0), 'USDT': (10.0, 0.0)},
                        prices={'BTCUSDT': 0.0, 'TSTUSDT': 100.0})
    api = _api(client)
    try:
        api.get_balance()
    except ecb.BalanceIncomplete as e:
        assert 'BTC' in str(e) and '가격없음' in str(e), str(e)
    else:
        raise AssertionError('BalanceIncomplete 가 발생해야 함')
    # 시세 맵 자체가 비면 전체 중단
    client2 = FakeClient(balances={'BTC': (1.0, 0.0), 'USDT': (10.0, 0.0)}, prices={})
    try:
        _api(client2).get_balance()
    except ecb.BalanceIncomplete as e:
        assert '가격 일괄조회' in str(e), str(e)
    else:
        raise AssertionError('BalanceIncomplete 가 발생해야 함')


def test_balance_incomplete_on_missing_symbol():
    client = FakeClient(balances={'GHOST': (5.0, 0.0), 'USDT': (10.0, 0.0)})
    api = _api(client)
    try:
        api.get_balance()
    except ecb.BalanceIncomplete as e:
        assert 'GHOST' in str(e) and '마켓없음' in str(e), str(e)
    else:
        raise AssertionError('BalanceIncomplete 가 발생해야 함')


def test_balance_ignore_list():
    client = FakeClient(balances={'GHOST': (5.0, 0.0), 'USDT': (10.0, 0.0)})
    api = _api(client, ignore_assets=('ghost',))
    bal = api.get_balance()
    assert 'GHOST' not in bal and bal['USDT'] == 10.0, bal


def test_run_once_exit2_on_incomplete_balance():
    """주문 전에 exit 2 로 중단되는지 (C2)."""
    client = FakeClient(balances={'GHOST': (5.0, 0.0), 'USDT': (10.0, 0.0)},
                        strict_no_orders=True)
    api = _api(client, dry_run=True)
    orig_status = ecb.cle.fetch_upbit_market_status
    orig_state = ecb.STATE_FILE
    ecb.cle.fetch_upbit_market_status = lambda session: {}
    ecb.STATE_FILE = '__test_missing_state__.json'
    try:
        rc = ecb.run_once(dry_run=True, api=api)
    finally:
        ecb.cle.fetch_upbit_market_status = orig_status
        ecb.STATE_FILE = orig_state
    assert rc == 2, rc
    assert client.submissions == [], client.submissions
    assert not os.path.exists(os.path.join(ecb.CACHE_DIR, '__test_missing_state__.json'))


def test_cancel_all_blocks_symbol_on_leftover():
    client = FakeClient(balances={'USDT': (100.0, 0.0)})
    api = _api(client)
    # 취소해도 남아있는 미체결 (취소 실패 시나리오)
    client.open_orders = [{'symbol': 'TSTUSDT', 'orderId': 77, 'clientOrderId': 'x'}]
    client.cancel_order = lambda **kw: (_ for _ in ()).throw(FakeTransportError('cancel failed'))
    blocked = api.cancel_all()
    assert blocked == {'TST'}, blocked


def test_cancel_all_unknown_returns_none():
    client = FakeClient(balances={'USDT': (100.0, 0.0)})
    api = _api(client)
    client.open_orders_error = 'open orders down'
    assert api.cancel_all() is None


# ═══ ④ 멱등성 — 불명 응답 후 재조회 ═══
def test_lost_response_does_not_double_sell():
    """응답만 유실되고 실제로는 체결된 경우 재주문하지 않는다 (C1)."""
    client = FakeClient(balances={'TST': (5.0, 0.0), 'USDT': (0.0, 0.0)})
    client.sell_behavior = 'lost_response'
    api = _api(client)
    ok, filled = api.sell_market_robust('TST', 5.0)
    assert len(client.submissions) == 1, f'중복 주문 발생: {client.submissions}'
    assert len(client.accepted) == 1, client.accepted
    assert ok is True and abs(filled - 5.0) < 1e-9, (ok, filled)


def test_ambiguous_network_error_never_reorders():
    """네트워크 오류(모호)는 -2013 조회 결과와 무관하게 재주문 금지 (라운드2 정책).

    이전 정책은 -2013 을 '미접수 확정'으로 보고 재주문했으나, 주문이 아직 전파
    중일 수 있으므로 이제는 UnknownExecutionError → health lock 이 정답이다.
    """
    client = FakeClient(balances={'TST': (5.0, 0.0), 'USDT': (0.0, 0.0)})
    client.sell_behavior = 'not_placed'   # 접수 안 됐지만 거래소가 명시하지 않음
    client.recover_after_first = False
    api = _api(client)
    try:
        api.sell_market_robust('TST', 5.0)
    except UnknownExecutionError as e:
        assert 'TST' in str(e), str(e)
    else:
        raise AssertionError('모호한 실패는 UnknownExecutionError 여야 함')
    assert len(client.submissions) == 1, f'모호 상태에서 재주문됨: {client.submissions}'


def test_explicit_reject_retries_with_new_client_order_id():
    """거래소가 명시적으로 사전 거절(-2010)한 경우에만 새 coid 로 재시도한다."""
    client = FakeClient(balances={'TST': (5.0, 0.0), 'USDT': (0.0, 0.0)})
    client.sell_behavior = 'reject'
    client.reject_code = -1021           # transient reject (recvWindow) → 재시도 대상
    api = _api(client)
    ok, filled = api.sell_market_robust('TST', 5.0)
    assert len(client.submissions) == 2, client.submissions
    assert len(client.accepted) == 1, client.accepted
    coids = [x['newClientOrderId'] for x in client.submissions]
    assert coids[0] != coids[1], coids
    assert ok is True and abs(filled - 5.0) < 1e-9, (ok, filled)


def test_deterministic_reject_stops_immediately():
    """결정적 거절(-1013 필터 위반)은 재시도하지 않고 즉시 중단."""
    client = FakeClient(balances={'TST': (5.0, 0.0), 'USDT': (0.0, 0.0)})
    client.sell_behavior = 'reject'
    client.reject_code = -1013
    client.recover_after_first = False
    api = _api(client)
    ok, filled = api.sell_market_robust('TST', 5.0)
    assert len(client.submissions) == 1, client.submissions
    assert ok is False and filled == 0.0, (ok, filled)


def test_delayed_visibility_does_not_double_sell():
    """응답 유실 + 조회 지연 노출 — 지연 재조회로 접수를 확인하고 재주문하지 않는다."""
    client = FakeClient(balances={'TST': (5.0, 0.0), 'USDT': (0.0, 0.0)})
    client.sell_behavior = 'delayed'
    client.visible_after = 2      # 두 번째 조회부터 보임
    api = _api(client)
    ok, filled = api.sell_market_robust('TST', 5.0)
    assert len(client.submissions) == 1, f'중복 주문 발생: {client.submissions}'
    assert ok is True and abs(filled - 5.0) < 1e-9, (ok, filled)


def test_accepted_but_lookup_not_found_is_unknown():
    """접수 응답을 받았는데 이후 조회가 계속 NOT_FOUND → 재매도 금지, UnknownExecutionError."""
    client = FakeClient(balances={'TST': (5.0, 0.0), 'USDT': (0.0, 0.0)})
    api = _api(client)
    orig = client.order_market_sell

    def _sell(**params):
        res = orig(**params)
        client.hide_from_lookup.add(params['newClientOrderId'])
        res = dict(res)
        res.pop('status', None)          # POST 응답도 최종상태 fallback 불가
        res.pop('executedQty', None)
        return res
    client.order_market_sell = _sell
    try:
        api.sell_market_robust('TST', 5.0)
    except UnknownExecutionError as e:
        assert '체결량 확정 불가' in str(e), str(e)
    else:
        raise AssertionError('NOT_FOUND fallback 은 UnknownExecutionError 여야 함')
    assert len(client.submissions) == 1, client.submissions


def test_market_partial_fill_reported():
    """시장가 부분체결은 체결분만 인정하고 미완으로 보고한다."""
    client = FakeClient(balances={'TST': (5.0, 0.0), 'USDT': (0.0, 0.0)})
    client.sell_behavior = 'partial'
    client.partial_ratio = 0.4
    client.recover_after_first = False
    api = _api(client)
    ok, filled = api.sell_market_robust('TST', 5.0)
    assert ok is False, '부분체결은 미완으로 보고돼야 함'
    assert 0 < filled < 5.0, filled
    assert len(client.submissions) <= ecb.SELL_MAX_ATTEMPTS, client.submissions
    # 체결분만 인정 — 첫 주문은 요청 5, 체결 2
    assert abs(float(client.accepted[client.submissions[0]['newClientOrderId']]['executedQty'])
               - 2.0) < 1e-6


def test_free_qty_lookup_failure_is_fail_closed():
    """free 잔량 조회 실패는 0 이 아니라 '불명' → 주문하지 않는다."""
    client = FakeClient(balances={'TST': (5.0, 0.0), 'USDT': (0.0, 0.0)})
    client.free_qty_error_assets = {'TST'}
    api = _api(client)
    assert api.get_free_qty('TST') is None
    ok, filled = api.sell_market_robust('TST', 5.0)
    assert ok is False and filled == 0.0, (ok, filled)
    assert client.submissions == [], client.submissions


def test_liquidation_not_complete_when_free_unknown():
    """청산 완료는 '조회 성공 AND free==0/dust' 일 때만."""
    client = FakeClient(balances={'HALT': (100.0, 0.0), 'USDT': (0.0, 0.0)})
    client.free_qty_error_assets = {'HALT'}
    api = _api(client)
    state = {}
    liq, failed = ecb.liquidate_coins(['HALT'], 'test', api, state)
    assert liq == [] and failed == ['HALT'], (liq, failed)
    assert client.submissions == [], client.submissions


def test_cancel_all_malformed_recheck_returns_none():
    """재조회 응답이 list 가 아니거나 필수 필드가 없으면 불명 → None."""
    client = FakeClient(balances={'USDT': (100.0, 0.0)})
    api = _api(client)
    client.open_orders_payload = {'not': 'a list'}
    assert api.cancel_all() is None
    client.open_orders_payload = [{'orderId': 1}]        # symbol 누락
    assert api.cancel_all() is None
    client.open_orders_payload = ['garbage']
    assert api.cancel_all() is None


def test_lcm_step_non_integer_multiple():
    """LOT_SIZE 0.001 / MARKET_LOT_SIZE 0.0025 → 공통 배수 0.005 (max() 는 오답)."""
    assert ecb._lcm_step(Decimal('0.001'), Decimal('0.0025')) == Decimal('0.005')
    assert ecb._lcm_step(Decimal('0.001'), Decimal('0.01')) == Decimal('0.01')
    assert ecb._lcm_step(Decimal('0'), Decimal('0.01')) == Decimal('0.01')
    assert ecb._lcm_step(Decimal('0.01'), Decimal('0')) == Decimal('0.01')

    syms = [_sym('ODDUSDT', 'ODD', step='0.001', min_qty='0.001', max_qty='1000',
                 mkt_step='0.0025', mkt_max_qty='1000', tick='0.01', min_notional='5')]
    api = _api(FakeClient(symbols=syms, balances={'USDT': (100.0, 0.0)},
                          prices={'ODDUSDT': 100.0}))
    qs, qd, reason = api.prepare_qty('ODD', 1.23456, 100.0, market=True)
    assert reason == '', reason
    assert qd == Decimal('1.230'), qd            # 0.005 배수
    assert qd % Decimal('0.001') == 0 and qd % Decimal('0.0025') == 0, qd
    # 지정가는 LOT_SIZE 만 → 0.001 배수
    qs, qd, reason = api.prepare_qty('ODD', 1.23456, 100.0, market=False)
    assert reason == '' and qd == Decimal('1.234'), (qs, qd, reason)


def test_executable_min_notional():
    """minNotional / minQty×price / step×price 중 최대값."""
    syms = [_sym('BIGUSDT', 'BIG', step='0.5', min_qty='2', max_qty='1000',
                 tick='0.01', min_notional='5')]
    api = _api(FakeClient(symbols=syms, balances={'USDT': (100.0, 0.0)},
                          prices={'BIGUSDT': 30.0}))
    # minQty 2 × 30 = 60 > 전역 10, step 0.5 × 30 = 15
    assert abs(api.executable_min_notional('BIG', 30.0) - 60.0) < 1e-9
    # 가격 정보가 없으면 정적 하한
    assert abs(api.executable_min_notional('BIG', 0.0) - 10.0) < 1e-9


def test_needs_rebalance_ignores_unexecutable_dust():
    """실행 불가 dust 는 리밸 완료 판정에서 제외돼 영구 True 루프가 생기지 않는다."""
    syms = [_sym('BIGUSDT', 'BIG', step='0.5', min_qty='2', max_qty='1000',
                 tick='0.01', min_notional='5')]
    api = _api(FakeClient(symbols=syms, balances={'USDT': (100.0, 0.0)},
                          prices={'BIGUSDT': 30.0}))
    balance = {'USDT': 70.0, 'BIG': 30.0}     # 잔여 30 < 실행최소 60
    target = {'Cash': 1.0}
    min_fn = lambda c: api.executable_min_notional(c, api.last_price(c) or 30.0)
    assert ecb.coin_needs_rebalance(target, balance, 100.0, min_fn=min_fn) is False
    # 정적 minNotional(10)만 쓰면 True 로 남아 매일 헛도는 회귀
    assert ecb.coin_needs_rebalance(target, balance, 100.0) is True


def test_percent_price_is_advisory_only():
    """PERCENT_PRICE 는 advisory — 경계 의심이어도 주문은 시도하고 로그만 남긴다 (m)."""
    syms = [_sym('PPUSDT', 'PP', step='0.001', min_qty='0.001', tick='0.01',
                 min_notional='5', pct_bid_up='1.05', pct_bid_down='0.95')]
    api = _api(FakeClient(symbols=syms, balances={'USDT': (1000.0, 0.0)},
                          prices={'PPUSDT': 100.0}))
    n0 = len(LOG_LINES)
    ps, _, reason = api.prepare_price('PP', 200.0, ref_price=100.0, side='BUY')
    assert reason == '' and ps == '200.00', (ps, reason)   # 차단하지 않음
    assert any('advisory' in l for l in LOG_LINES[n0:]), LOG_LINES[n0:]
    # 검사 자체는 여전히 위반을 탐지한다
    assert 'PERCENT_PRICE' in api.check_percent_price('PP', Decimal('200'), 100.0, 'BUY')
    ps, _, reason = api.prepare_price('PP', 100.3, ref_price=100.0, side='BUY')
    assert reason == '' and ps == '100.30', (ps, reason)


def test_legacy_min_notional_filter():
    syms = [_sym('LEGUSDT', 'LEG', step='0.001', min_qty='0.001', tick='0.01',
                 legacy_min_notional='15')]
    api = _api(FakeClient(symbols=syms, balances={'USDT': (1000.0, 0.0)},
                          prices={'LEGUSDT': 100.0}))
    assert abs(api.min_notional('LEG') - 15.0) < 1e-9
    _, _, reason = api.prepare_qty('LEG', 0.12, 100.0, market=False)   # notional 12
    assert '최소주문' in reason, reason


# ═══ WAL ═══
class _WalDir:
    """CACHE_DIR 을 임시 디렉터리로 바꾸는 컨텍스트."""

    def __enter__(self):
        self.d = tempfile.mkdtemp(prefix='wal_')
        self.orig = ecb.CACHE_DIR
        ecb.CACHE_DIR = self.d
        return self.d

    def __exit__(self, *a):
        ecb.CACHE_DIR = self.orig
        shutil.rmtree(self.d, ignore_errors=True)


def _wal_records(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def _intent(coid, symbol='TSTUSDT', side='SELL', qty='5.00', otype='MARKET', ts=None):
    return {'event': 'intent', 'ts': time.time() if ts is None else ts, 'coid': coid,
            'symbol': symbol, 'side': side, 'qty': qty, 'type': otype,
            'recv_window_ms': ecb.RECV_WINDOW_MS}


def test_wal_intent_and_resolved_records():
    """live 주문은 intent(POST 직전 ts + recvWindow) → resolved(post_ts) 를 남긴다."""
    client = FakeClient(balances={'TST': (5.0, 0.0), 'USDT': (0.0, 0.0)})
    api = _api(client)
    with _WalDir() as d:
        api.sell_market_robust('TST', 5.0)
        wal = os.path.join(d, ecb.ORDER_WAL_FILE)
        assert os.path.exists(wal)
        recs = _wal_records(wal)
        assert recs[0]['event'] == 'intent' and recs[0]['symbol'] == 'TSTUSDT'
        assert recs[0]['recv_window_ms'] == ecb.RECV_WINDOW_MS, recs[0]
        assert recs[0]['ts'] > 0
        res = [r for r in recs if r['event'] == 'resolved']
        assert res and res[0]['coid'] == recs[0]['coid'], recs
        assert res[0]['post_ts'] is not None, res[0]
        # 전부 해소 → compaction 으로 비워지되 파일은 유지 (append-only, 삭제 금지)
        ok, pending = api.reconcile_wal()
        assert ok is True and pending == [], (ok, pending)
        assert os.path.exists(wal) and os.path.getsize(wal) == 0


def test_wal_unresolved_is_never_auto_resolved_by_age():
    """아무리 오래된 intent 라도 시간만으로 NOT_PLACED 확정하지 않는다 (라운드3 C)."""
    client = FakeClient(balances={'TST': (5.0, 0.0), 'USDT': (0.0, 0.0)})
    api = _api(client)
    with _WalDir() as d:
        wal = os.path.join(d, ecb.ORDER_WAL_FILE)
        with open(wal, 'w') as f:
            f.write(json.dumps(_intent('bsOLD', ts=time.time() - 86400)) + '\n')  # 하루 전
        ok, pending = api.reconcile_wal()
        assert ok is False, '시간 경과로 자동 해소되면 안 됨'
        assert pending and 'bsOLD' in pending[0], pending
        assert os.path.getsize(wal) > 0, '미해결 intent 는 WAL 에 남아야 함'


def test_wal_late_visibility_resolves():
    """늦게 조회 가능해진 주문은 권위 조회로 해소된다."""
    client = FakeClient(balances={'TST': (5.0, 0.0), 'USDT': (0.0, 0.0)})
    api = _api(client)
    with _WalDir() as d:
        wal = os.path.join(d, ecb.ORDER_WAL_FILE)
        coid = 'bsLATE'
        client.accepted[coid] = {'symbol': 'TSTUSDT', 'orderId': 99, 'clientOrderId': coid,
                                 'origQty': '5.000', 'executedQty': '5.000', 'status': 'FILLED'}
        with open(wal, 'w') as f:
            f.write(json.dumps(_intent(coid)) + '\n')
        ok, pending = api.reconcile_wal()
        assert ok is True and pending == [], (ok, pending)
        assert any(r['event'] == 'resolved' and r['coid'] == coid
                   for r in _wal_records(wal)), _wal_records(wal)


def test_wal_corrupt_line_fails_reconcile():
    """손상/스키마 불일치 행은 무시하지 않고 reconcile 실패 (fail-closed)."""
    client = FakeClient(balances={'USDT': (10.0, 0.0)})
    api = _api(client)
    _i = _intent('x')
    for bad in ('{not json',
                json.dumps(['list', 'not', 'dict']),
                json.dumps({'event': 'intent'}),                       # coid 없음
                json.dumps({'event': 'intent', 'coid': 'x'}),          # ts/symbol 없음
                json.dumps({'event': 'weird', 'coid': 'x', 'ts': 1.0}),   # 미지 event
                # 라운드4: 이벤트별 필수 필드
                json.dumps({k: v for k, v in _i.items() if k != 'side'}),
                json.dumps({k: v for k, v in _i.items() if k != 'qty'}),
                json.dumps({k: v for k, v in _i.items() if k != 'type'}),
                json.dumps({k: v for k, v in _i.items() if k != 'ts'}),
                json.dumps({k: v for k, v in _i.items() if k != 'recv_window_ms'}),
                json.dumps(dict(_i, side='HOLD')),
                json.dumps(dict(_i, type='STOP')),
                json.dumps(dict(_i, qty='0')),
                json.dumps(dict(_i, qty='nan')),
                # resolved 는 status 필수 + 선행 intent 필수
                json.dumps({'event': 'resolved', 'coid': 'x', 'ts': 1.0}),
                json.dumps({'event': 'resolved', 'coid': 'x', 'ts': 1.0,
                            'status': 'FILLED'})):
        with _WalDir() as d:
            wal = os.path.join(d, ecb.ORDER_WAL_FILE)
            with open(wal, 'w') as f:
                f.write(bad + '\n')
            ok, pending = api.reconcile_wal()
            assert ok is False and pending, (bad, ok, pending)


def test_wal_manual_mark_resolved():
    """운영자 수동 해소는 intent 가 있을 때만 동작하고 기록을 남긴다."""
    client = FakeClient(balances={'USDT': (10.0, 0.0)})
    api = _api(client)
    with _WalDir() as d:
        wal = os.path.join(d, ecb.ORDER_WAL_FILE)
        with open(wal, 'w') as f:
            f.write(json.dumps(_intent('bsMAN')) + '\n')
        assert api.mark_wal_resolved('bsNOPE') is False       # 없는 coid
        assert api.mark_wal_resolved('bsMAN') is True
        recs = _wal_records(wal)
        assert any(r['event'] == 'resolved' and r['status'] == 'MANUAL_RESOLVED'
                   for r in recs), recs
        client.lookup_fail = True
        ok, pending = api.reconcile_wal()
        assert ok is True and pending == [], (ok, pending)   # 수동 해소분은 조회 없이 통과


def test_wal_append_only_across_restart():
    """재시작(새 API 인스턴스)해도 기존 WAL 을 덮어쓰지 않고 append 한다."""
    with _WalDir() as d:
        wal = os.path.join(d, ecb.ORDER_WAL_FILE)
        c1 = FakeClient(balances={'TST': (5.0, 0.0), 'USDT': (0.0, 0.0)})
        _api(c1).sell_market_robust('TST', 5.0)
        n1 = len(_wal_records(wal))
        c2 = FakeClient(balances={'TST': (5.0, 0.0), 'USDT': (0.0, 0.0)})
        _api(c2).sell_market_robust('TST', 5.0)
        n2 = len(_wal_records(wal))
        assert n2 > n1, (n1, n2)


def test_wal_not_written_in_dry_run():
    client = FakeClient(balances={'LUNC': (1000000.0, 0.0), 'USDT': (1000.0, 0.0)},
                        strict_no_orders=True)
    api = _api(client, dry_run=True)
    with _WalDir() as d:
        ecb.execute_delta({'BTC': 0.5, 'Cash': 0.5}, api, [], True,
                          balance={'USDT': 1000.0, 'LUNC': 50.0})
        assert not os.path.exists(os.path.join(d, ecb.ORDER_WAL_FILE))


# ═══ 라운드3 신규 위험 경로 ═══
def test_client_order_id_unique_across_restart():
    """같은 초에 재시작해도 coid 가 충돌하지 않는다 (uuid4 난수)."""
    ids = set()
    for _ in range(20):
        api = _api(FakeClient(balances={'USDT': (1.0, 0.0)}), )
        for _ in range(3):
            ids.add(api._next_client_order_id('S', 'BTC'))
    assert len(ids) == 60, len(ids)
    # 긴 base asset 이어도 난수/seq 는 절단되지 않는다 (coin 꼬리만 잘림)
    api = _api(FakeClient(balances={'USDT': (1.0, 0.0)}))
    long_coin = 'VERYLONGBASEASSETNAME1234567890'
    cid = api._next_client_order_id('S', long_coin)
    assert len(cid) <= 36, cid
    head = f'{ecb.CLIENT_ORDER_PREFIX}'
    assert cid.startswith(head), cid
    rand = cid[len(head):len(head) + ecb.COID_RANDOM_LEN]
    assert len(rand) == ecb.COID_RANDOM_LEN, cid
    assert 'S1-' in cid, cid          # side + sequence 보존


def test_malformed_get_order_is_unknown():
    """get_order 응답 검증 실패(심볼/ID 불일치, 미지 status, executedQty>origQty)는 미확정."""
    base = {'symbol': 'TSTUSDT', 'clientOrderId': 'c1', 'origQty': '1', 'executedQty': '1',
            'status': 'FILLED'}
    assert ecb._validate_order_resp(base, 'TSTUSDT', 'c1') == ''
    assert 'symbol' in ecb._validate_order_resp(dict(base, symbol='XUSDT'), 'TSTUSDT', 'c1')
    assert 'clientOrderId' in ecb._validate_order_resp(base, 'TSTUSDT', 'other')
    assert 'status' in ecb._validate_order_resp(dict(base, status='WEIRD'), 'TSTUSDT', 'c1')
    assert 'executedQty' in ecb._validate_order_resp(dict(base, executedQty='2'), 'TSTUSDT', 'c1')
    assert 'executedQty' in ecb._validate_order_resp(dict(base, executedQty='-1'), 'TSTUSDT', 'c1')
    assert 'origQty' in ecb._validate_order_resp(dict(base, origQty=None), 'TSTUSDT', 'c1')
    assert 'dict' in ecb._validate_order_resp('nope', 'TSTUSDT', 'c1')

    # 실제 경로: 조회가 malformed 이고 POST 응답도 최종상태가 아니면 → UnknownExecutionError
    client = FakeClient(balances={'TST': (5.0, 0.0), 'USDT': (0.0, 0.0)})
    api = _api(client)
    orig_lookup = client.get_order
    orig_sell = client.order_market_sell

    def _bad(**kw):
        o = dict(orig_lookup(**kw))
        o['executedQty'] = '999'       # origQty 초과 → 검증 실패
        return o

    def _sell(**params):
        res = dict(orig_sell(**params))
        res.pop('status', None)        # POST 응답 fallback 불가
        res.pop('executedQty', None)
        return res
    client.get_order = _bad
    client.order_market_sell = _sell
    try:
        api.sell_market_robust('TST', 5.0)
    except UnknownExecutionError:
        pass
    else:
        raise AssertionError('malformed 응답은 UnknownExecutionError 여야 함')
    assert len(client.submissions) == 1, client.submissions


def test_liquidation_fails_when_locked_remains():
    """free 만 팔리고 locked 가 남으면 청산 완료로 인정하지 않는다 (M)."""
    syms = [_sym('HALTUSDT', 'HALT', status='HALT', step='0.001', min_qty='0.001',
                 min_notional='5')]
    client = FakeClient(symbols=syms, balances={'HALT': (10.0, 50.0), 'USDT': (0.0, 0.0)},
                        prices={'HALTUSDT': 1.0})
    api = _api(client)
    state = {}
    liq, failed = ecb.liquidate_coins(['HALT'], 'test', api, state)
    assert liq == [] and failed == ['HALT'], (liq, failed, client._balances)
    assert 'HALT' in state['permanent_block'], state


def test_liquidation_fails_when_symbol_blocked():
    """cancel_all 이 blocked 로 표시한 심볼은 청산 완료로 인정하지 않는다."""
    syms = [_sym('HALTUSDT', 'HALT', status='HALT', step='0.001', min_qty='0.001',
                 min_notional='5')]
    client = FakeClient(symbols=syms, balances={'HALT': (10.0, 0.0), 'USDT': (0.0, 0.0)},
                        prices={'HALTUSDT': 1.0})
    api = _api(client)
    liq, failed = ecb.liquidate_coins(['HALT'], 'test', api, {}, blocked_coins={'HALT'})
    assert liq == [] and failed == ['HALT'], (liq, failed)


def test_buy_unknown_final_state_raises():
    """취소 후 최종상태 미확인 매수는 (True, note) 가 아니라 UnknownExecutionError (축소 철회)."""
    client = FakeClient(balances={'USDT': (1000.0, 0.0)})
    api = _api(client)
    orig = client.order_limit_buy

    def _buy(**params):
        res = orig(**params)
        client.hide_from_lookup.add(params['newClientOrderId'])
        return res
    client.order_limit_buy = _buy
    try:
        api.buy_limit('TST', 500.0)
    except UnknownExecutionError as e:
        assert 'TSTUSDT' in str(e) or 'TST' in str(e), str(e)
    else:
        raise AssertionError('매수 미확정은 UnknownExecutionError 여야 함')


def test_grid_min_notional_boundary():
    """리뷰 예시: step=0.03, price=100, minNotional=10 → 실제 최소는 $12."""
    syms = [_sym('GRDUSDT', 'GRD', step='0.03', min_qty='0.03', max_qty='1000',
                 tick='0.01', min_notional='10')]
    api = _api(FakeClient(symbols=syms, balances={'USDT': (100.0, 0.0)},
                          prices={'GRDUSDT': 100.0}))
    assert abs(api.executable_min_notional('GRD', 100.0) - 12.0) < 1e-9, \
        api.executable_min_notional('GRD', 100.0)
    # 경계값 정확히 $12 는 주문 가능해야 한다 (>= 통일)
    qs, qd, reason = api.prepare_qty('GRD', 12.0 / 100.0, 100.0, market=True)
    assert reason == '' and qd == Decimal('0.12'), (qs, qd, reason)
    # $11.99 는 격자상 주문 불가
    _, _, reason = api.prepare_qty('GRD', 11.99 / 100.0, 100.0, market=True)
    assert reason, reason
    # 매수 판정은 예상 지정가(+0.3%) 기준
    assert api.buy_min_notional('GRD', 100.0) > 12.0


def test_grid_boundary_no_permanent_loop():
    """경계값에서 후보 생성(>=)과 완료 판정이 일치해 영구 루프가 생기지 않는다."""
    syms = [_sym('GRDUSDT', 'GRD', step='0.03', min_qty='0.03', max_qty='1000',
                 tick='0.01', min_notional='10')]
    api = _api(FakeClient(symbols=syms, balances={'USDT': (88.0, 0.0), 'GRD': (0.12, 0.0)},
                          prices={'GRDUSDT': 100.0}))
    api.get_balance()                      # last_price 채우기
    balance = {'USDT': 88.0, 'GRD': 12.0}
    target = {'Cash': 1.0}                 # GRD 전량 매도 목표
    min_fn = lambda c: api.executable_min_notional(c, api.last_price(c))
    # 잔여 12 == 실행최소 12 → 주문 가능하므로 리밸 필요로 판정돼야 한다
    assert ecb.coin_needs_rebalance(target, balance, 100.0, min_fn=min_fn) is True
    # 11.99 면 주문 불가 → 리밸 불필요 (영구 True 루프 없음)
    assert ecb.coin_needs_rebalance(target, {'USDT': 88.01, 'GRD': 11.99}, 100.0,
                                    min_fn=min_fn) is False


def test_balance_requires_free_and_locked():
    """free/locked 필수 — 부재/None/빈문자열/bool 이면 BalanceIncomplete."""
    for bad in ({'asset': 'BTC', 'free': '1.0'},                    # locked 없음
                {'asset': 'BTC', 'free': None, 'locked': '0'},
                {'asset': 'BTC', 'free': '', 'locked': '0'},
                {'asset': 'BTC', 'free': True, 'locked': '0'},
                {'asset': 'BTC', 'free': 'nan', 'locked': '0'},
                {'asset': 'BTC', 'free': '-1', 'locked': '0'}):
        client = FakeClient(balances={'USDT': (10.0, 0.0)})
        client.get_account = lambda b=bad: {'balances': [
            {'asset': 'USDT', 'free': '10', 'locked': '0'}, b]}
        try:
            _api(client).get_balance()
        except ecb.BalanceIncomplete:
            pass
        else:
            raise AssertionError(f'BalanceIncomplete 필요: {bad}')


def test_get_free_qty_none_on_missing_field():
    client = FakeClient(balances={'TST': (1.0, 0.0), 'USDT': (0.0, 0.0)})
    api = _api(client)
    client.get_asset_balance = lambda asset: {'asset': asset, 'free': '1.0'}   # locked 없음
    assert api.get_free_qty('TST') is None
    client.get_asset_balance = lambda asset: {'asset': asset, 'free': '', 'locked': '0'}
    assert api.get_free_qty('TST') is None


def test_state_loader_fail_closed():
    """state JSON 손상/타입 불일치는 빈 state 로 삼키지 않는다."""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'state.json')
        st, err = ecb.load_state_strict(p)
        assert st == {} and err == '', (st, err)          # 부재만 초기 state 허용
        open(p, 'w').write('{broken')
        st, err = ecb.load_state_strict(p)
        assert st is None and '손상' in err, (st, err)
        open(p, 'w').write('[1,2,3]')
        st, err = ecb.load_state_strict(p)
        assert st is None and '타입' in err, (st, err)
        open(p, 'w').write('{"a": 1}')
        st, err = ecb.load_state_strict(p)
        assert st == {'a': 1} and err == '', (st, err)


def test_run_once_exit2_on_corrupt_state():
    client = FakeClient(balances={'USDT': (10.0, 0.0)}, strict_no_orders=True)
    api = _api(client, dry_run=True)
    orig_state = ecb.STATE_FILE
    ecb.STATE_FILE = '__corrupt_state__.json'
    try:
        with open(os.path.join(ecb.CACHE_DIR, ecb.STATE_FILE), 'w') as f:
            f.write('{oops')
        rc = ecb.run_once(dry_run=True, api=api)
        assert rc == 2, rc
        assert client.submissions == [], client.submissions
    finally:
        try:
            os.remove(os.path.join(ecb.CACHE_DIR, ecb.STATE_FILE))
        except OSError:
            pass
        ecb.STATE_FILE = orig_state


def test_redaction_applies_to_child_loggers():
    """LogRecordFactory 방식이라 자식 로거 레코드도 마스킹된다."""
    rec = logging.getLogger('urllib3.connectionpool').makeRecord(
        'urllib3.connectionpool', logging.WARNING, __file__, 1,
        'failed https://api.telegram.org/bot123456:AAH-secretTokenValue/sendMessage', (), None)
    assert 'AAH-secretTokenValue' not in rec.getMessage(), rec.getMessage()
    assert 'REDACTED' in rec.getMessage(), rec.getMessage()


# ═══ 라운드4 신규 ═══
def test_liquidation_free_zero_but_locked_remains():
    """free==0 이어도 locked 가 실행 최소금액 이상이면 청산 실패 (라운드4 C)."""
    syms = [_sym('HALTUSDT', 'HALT', status='HALT', step='0.001', min_qty='0.001',
                 min_notional='5')]
    client = FakeClient(symbols=syms, balances={'HALT': (0.0, 50.0), 'USDT': (0.0, 0.0)},
                        prices={'HALTUSDT': 1.0})
    api = _api(client)
    state = {}
    liq, failed = ecb.liquidate_coins(['HALT'], 'test', api, state)
    assert liq == [] and failed == ['HALT'], (liq, failed)
    assert 'HALT' in state['permanent_block'], state
    assert client.submissions == [], '주문 없이 실패로 판정돼야 함'


def test_liquidation_free_zero_but_blocked():
    """free==0 이어도 blocked 심볼이면 청산 완료로 인정하지 않는다 (라운드4 C)."""
    syms = [_sym('HALTUSDT', 'HALT', status='HALT', step='0.001', min_qty='0.001',
                 min_notional='5')]
    client = FakeClient(symbols=syms, balances={'HALT': (0.0, 0.0), 'USDT': (0.0, 0.0)},
                        prices={'HALTUSDT': 1.0})
    api = _api(client)
    state = {}
    liq, failed = ecb.liquidate_coins(['HALT'], 'test', api, state, blocked_coins={'HALT'})
    assert liq == [] and failed == ['HALT'], (liq, failed)
    assert 'HALT' in state['permanent_block'], state
    # blocked 가 아니면 free=0/locked=0 은 정상 완료(이미 청산됨)로 통과
    liq2, failed2 = ecb.liquidate_coins(['HALT'], 'test', api, {}, blocked_coins=set())
    assert liq2 == [] and failed2 == [], (liq2, failed2)


def test_post_fallback_must_pass_validation():
    """조회 실패 시 쓰는 POST 응답 fallback 도 _validate_order_resp 를 통과해야 한다."""
    for mutate, why in (
            (lambda o: dict(o, symbol='WRONGUSDT'), 'symbol 불일치'),
            (lambda o: dict(o, executedQty='999'), 'executedQty > origQty'),
            (lambda o: dict(o, clientOrderId='other'), 'coid 불일치'),
            (lambda o: dict(o, status='WEIRD'), '미지 status')):
        client = FakeClient(balances={'TST': (5.0, 0.0), 'USDT': (0.0, 0.0)})
        api = _api(client)
        orig_sell = client.order_market_sell

        def _sell(_m=mutate, _o=orig_sell, **params):
            return _m(dict(_o(**params)))
        client.order_market_sell = _sell
        client.lookup_fail = True          # 조회는 계속 실패 → fallback 경로
        try:
            api.sell_market_robust('TST', 5.0)
        except UnknownExecutionError:
            pass
        else:
            raise AssertionError(f'malformed fallback 은 UnknownExecutionError 여야 함: {why}')
        assert len(client.submissions) == 1, (why, client.submissions)


def test_post_fallback_accepted_when_valid():
    """검증을 통과한 최종 상태 POST 응답은 조회 실패 시에도 신뢰한다."""
    client = FakeClient(balances={'TST': (5.0, 0.0), 'USDT': (0.0, 0.0)})
    api = _api(client)
    client.lookup_fail = True
    ok, filled = api.sell_market_robust('TST', 5.0)
    assert ok is True and abs(filled - 5.0) < 1e-9, (ok, filled)
    assert len(client.submissions) == 1, client.submissions


def test_wal_resolved_requires_preceding_intent():
    """선행 intent 없는 resolved 한 줄로는 해소되지 않는다 (reconcile 실패)."""
    client = FakeClient(balances={'USDT': (10.0, 0.0)})
    api = _api(client)
    with _WalDir() as d:
        wal = os.path.join(d, ecb.ORDER_WAL_FILE)
        with open(wal, 'w') as f:
            f.write(json.dumps(_intent('bsA')) + '\n')
            f.write(json.dumps({'event': 'resolved', 'coid': 'bsGHOST', 'ts': time.time(),
                                'status': 'FILLED'}) + '\n')
        ok, pending = api.reconcile_wal()
        assert ok is False and '선행 intent 없음' in pending[0], pending


def test_wal_resolved_write_failure_reported():
    """resolved 기록 실패는 수동 해소에서 실패로 보고된다 (라운드4 m)."""
    client = FakeClient(balances={'USDT': (10.0, 0.0)})
    api = _api(client)
    with _WalDir() as d:
        wal = os.path.join(d, ecb.ORDER_WAL_FILE)
        with open(wal, 'w') as f:
            f.write(json.dumps(_intent('bsMAN')) + '\n')
        orig = api._wal_append
        api._wal_append = lambda rec, critical: False if not critical else orig(rec, critical)
        assert api.mark_wal_resolved('bsMAN') is False
        api._wal_append = orig
        assert api.mark_wal_resolved('bsMAN') is True


def test_needs_rebalance_direction_aware_min():
    """완료 판정도 방향별 최소금액을 쓴다 (매도=market, 매수=limit)."""
    calls = []

    def sell_min(_c):
        calls.append(('sell', _c))
        return 100.0

    def buy_min(_c):
        calls.append(('buy', _c))
        return 100.0

    # 매수 방향(목표 > 보유) → buy_min 이 쓰여야 함
    ecb.coin_needs_rebalance({'BTC': 0.5, 'Cash': 0.5}, {'USDT': 100.0, 'BTC': 0.0}, 100.0,
                             min_fn=sell_min, buy_min_fn=buy_min)
    assert ('buy', 'BTC') in calls and ('sell', 'BTC') not in calls, calls
    calls.clear()
    # 매도 방향(보유 > 목표) → sell_min
    ecb.coin_needs_rebalance({'Cash': 1.0}, {'USDT': 50.0, 'BTC': 50.0}, 100.0,
                             min_fn=sell_min, buy_min_fn=buy_min)
    assert ('sell', 'BTC') in calls and ('buy', 'BTC') not in calls, calls


def test_redaction_covers_exception_traceback():
    """exc_info 로 뒤늦게 렌더링되는 traceback 텍스트도 마스킹된다 (라운드4 m)."""
    import io
    stream = io.StringIO()
    h = logging.StreamHandler(stream)
    h.setFormatter(logging.Formatter('%(message)s'))
    lg = logging.getLogger('urllib3.connectionpool.test')
    lg.handlers = [h]
    lg.propagate = False
    lg.setLevel(logging.ERROR)
    ecb._wrap_handler(h, ecb._RedactingFilter())
    try:
        raise RuntimeError('POST https://api.telegram.org/bot777:AAH-secretTok3n/sendMessage '
                           'signature=deadbeefcafe failed')
    except RuntimeError:
        lg.exception('call failed')
    out = stream.getvalue()
    assert 'AAH-secretTok3n' not in out, out
    assert 'deadbeefcafe' not in out, out
    assert 'REDACTED' in out, out


# ═══ 라운드5 마무리 ═══
def test_wal_rejects_duplicate_and_bad_status():
    """intent/resolved 중복, 허용되지 않은 resolved status 는 reconcile 실패."""
    client = FakeClient(balances={'USDT': (10.0, 0.0)})
    api = _api(client)
    i = _intent('bsDUP')
    cases = [
        # intent coid 중복
        ([i, dict(i, ts=i['ts'] + 1)], 'intent coid 중복'),
        # resolved coid 중복
        ([i, {'event': 'resolved', 'coid': 'bsDUP', 'ts': i['ts'], 'status': 'FILLED'},
          {'event': 'resolved', 'coid': 'bsDUP', 'ts': i['ts'], 'status': 'FILLED'}],
         'resolved coid 중복'),
        # 허용되지 않은 status
        ([i, {'event': 'resolved', 'coid': 'bsDUP', 'ts': i['ts'], 'status': 'WHATEVER'}],
         'resolved status 허용값 아님'),
    ]
    for rows, why in cases:
        with _WalDir() as d:
            wal = os.path.join(d, ecb.ORDER_WAL_FILE)
            with open(wal, 'w') as f:
                for r in rows:
                    f.write(json.dumps(r) + '\n')
            ok, pending = api.reconcile_wal()
            assert ok is False and why in pending[0], (why, ok, pending)
    # 정상 status 는 통과
    for st in ('FILLED', 'NOT_PLACED', 'MANUAL_RESOLVED', 'CANCELED'):
        with _WalDir() as d:
            wal = os.path.join(d, ecb.ORDER_WAL_FILE)
            with open(wal, 'w') as f:
                f.write(json.dumps(i) + '\n')
                f.write(json.dumps({'event': 'resolved', 'coid': 'bsDUP',
                                    'ts': i['ts'], 'status': st}) + '\n')
            ok, pending = api.reconcile_wal()
            assert ok is True and pending == [], (st, ok, pending)


def test_non_trading_detection_includes_boundary_holdings():
    """정확히 최소주문 금액인 보유분도 청산 대상에 포함된다 (라운드5 m)."""
    syms = [_sym('HALTUSDT', 'HALT', status='HALT', step='0.001', min_qty='0.001',
                 min_notional='5'),
            _sym('BTCUSDT', 'BTC', step='0.00001', min_qty='0.00001', min_notional='5')]
    client = FakeClient(symbols=syms,
                        balances={'HALT': (10.0, 0.0), 'USDT': (0.0, 0.0)},
                        prices={'HALTUSDT': 1.0, 'BTCUSDT': 100000.0})
    api = _api(client)
    balance = api.get_balance()
    assert abs(balance['HALT'] - 10.0) < 1e-9, balance     # 정확히 $10 (= MIN_ORDER_USDT)
    held = [k for k, v in balance.items() if k != ecb.CASH_ASSET and v > 0]
    assert 'HALT' in held, held
    assert ecb.detect_non_trading(held, api) == ['HALT']
    # dust 보유는 청산 루틴에 들어가도 liquidation_state 가 완료로 판정 (실패 아님)
    client._balances['HALT'] = (0.000001, 0.0)
    done, why = ecb.liquidation_state(api, 'HALT', set())
    assert done is True, (done, why)


def _exception_output(mod, use_formatter: bool) -> str:
    import io
    stream = io.StringIO()
    h = logging.StreamHandler(stream)
    if use_formatter:
        h.setFormatter(logging.Formatter('%(message)s'))
    lg = logging.getLogger(f'urllib3.nofmt.{mod.__name__}.{use_formatter}')
    lg.handlers = [h]
    lg.propagate = False
    lg.setLevel(logging.ERROR)
    try:
        raise RuntimeError('https://api.telegram.org/bot4242:AAH-noFormatterTok3nX/sendMessage '
                           'signature=deadbeefcafe1234 boom')
    except RuntimeError:
        lg.exception('boom')
    return stream.getvalue()


def test_redaction_without_formatter_on_handler():
    """setFormatter 를 호출하지 않은 핸들러의 logger.exception 도 마스킹된다 (라운드5 m)."""
    out = _exception_output(ecb, use_formatter=False)
    assert 'AAH-noFormatterTok3nX' not in out, out
    assert 'deadbeefcafe1234' not in out, out
    assert 'REDACTED' in out, out


def test_redaction_clears_original_exc_text():
    """wrapped formatter 가 record.exc_text 를 마스킹된 값으로 교체한다."""
    import io
    rec_holder = {}

    class _Capture(logging.Handler):
        def emit(self, record):
            self.format(record)
            rec_holder['rec'] = record

    h = _Capture()
    lg = logging.getLogger('urllib3.exctext.test')
    lg.handlers = [h]
    lg.propagate = False
    lg.setLevel(logging.ERROR)
    try:
        raise RuntimeError('bot9999:AAH-excTextTok3nValue leaked')
    except RuntimeError:
        lg.exception('x')
    rec = rec_holder['rec']
    assert rec.exc_text and 'AAH-excTextTok3nValue' not in rec.exc_text, rec.exc_text


# ═══ compare_daily_report.py 파서/판정 (라운드3 M — 기존 테스트 0건) ═══
sys.path.insert(0, os.path.join(REPO_DIR, 'scripts'))
import compare_daily_report as cdr  # noqa: E402

_DAY = '2026-08-31'


def _mk_log(path, lines):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def _rec(t, msg, rid='--------', day=_DAY):
    return f'[{day} {t}] [{rid}] {msg}'


def _start(t, dry=True, rid='--------', label='바낸현물'):
    return _rec(t, f'═══ {label} Executor 시작 (dry_run={dry}, now=x) ═══', rid)


_TGT = '  combined target: BTC:33.3%, ETH:33.3%, SOL:33.3% (cash=0.0%)'
_MEMBER = '  D_SMA42 target: BTC:33.3%, ETH:33.3%, SOL:33.3% (cash=0.0%)'


def _run_compare(up_lines, bn_lines, day=_DAY, up_rot=None, bn_rot=None):
    d = tempfile.mkdtemp(prefix='cmp_')
    up = os.path.join(d, 'executor_coin.log')
    bn = os.path.join(d, 'executor_coin_binance.log')
    _mk_log(up, up_lines)
    _mk_log(bn, bn_lines)
    if up_rot:
        _mk_log(up + '.' + day, up_rot)
    if bn_rot:
        _mk_log(bn + '.' + day, bn_rot)
    old = cdr.SIDES
    cdr.SIDES = [('업비트', up), ('바낸현물', bn)]
    try:
        sides = []
        for n, p_ in cdr.SIDES:
            sr = cdr.SideResult(n, p_)
            sr.parse(day)
            sides.append(sr)
        body, warn = cdr.build_report(day, sides)
    finally:
        cdr.SIDES = old
        shutil.rmtree(d, ignore_errors=True)
    return body, warn, sides


def test_compare_rotated_and_active_merged():
    """실행이 회전 경계에서 갈려도 active+회전 파일을 시간순 병합한다."""
    rot = [_start('00:05:00', label='코인'), _rec('00:05:05', '  [engine] D_SMA42 카나리 ON 🟢')]
    act = [_rec('00:05:10', _MEMBER), _rec('00:05:11', _TGT),
           _rec('00:05:12', '  ✅ 거래 완료 (silent).')]
    bn = [_start('00:05:02'), _rec('00:05:10', _MEMBER), _rec('00:05:11', _TGT),
          _rec('00:05:12', '  ✅ 거래 완료 (silent).')]
    body, warn, sides = _run_compare(act, bn, up_rot=rot)
    up = sides[0]
    assert up.ran is True, up.problems
    assert up.run_time == '00:05:00', up.run_time      # 회전 파일의 시작 로그를 잡아야 함
    assert up.combined and set(up.coins()) == {'BTC', 'ETH', 'SOL'}, up.combined
    assert '✅ 코인 집합 일치' in body, body


def test_compare_multiline_message_merged():
    """타임스탬프 없는 후속 줄은 직전 레코드에 병합된다."""
    up = [_start('00:05:00', label='코인'), _rec('00:05:10', _MEMBER), _rec('00:05:11', _TGT),
          _rec('00:05:12', '  사전 알림 (silent): 목표 (앙상블):'),
          '  BTC: 33.33%', '  ⚠ 매도 미완 SOL: filled 0.1/1.0',
          _rec('00:05:13', '  ✅ 거래 완료 (silent).')]
    bn = [_start('00:05:02'), _rec('00:05:10', _MEMBER), _rec('00:05:11', _TGT),
          _rec('00:05:12', '  ✅ 거래 완료 (silent).')]
    body, warn, sides = _run_compare(up, bn)
    # 병합된 줄 안의 실패 마커도 high-watermark 로 잡혀야 한다
    assert sides[0].result_kind == 'warn', (sides[0].result_kind, sides[0].issues)
    assert warn is True


def test_compare_one_side_missing():
    up = [_start('00:05:00', label='코인'), _rec('00:05:11', _TGT),
          _rec('00:05:12', '  ✅ 거래 완료 (silent).')]
    body, warn, sides = _run_compare(up, ['(빈 로그)'])
    assert warn is True and '미실행' in body, body
    assert '한쪽 미실행' in body, body


def test_compare_cron_window_rule():
    """사이클 ID 가 없으면 양쪽 모두 cron 창(09:05~09:12 KST) 안일 때만 ✅."""
    # 00:05 UTC == 09:05 KST → 창 안
    up = [_start('00:05:00', label='코인'), _rec('00:05:11', _TGT),
          _rec('00:05:12', '  ✅ 거래 완료 (silent).')]
    bn = [_start('00:05:30'), _rec('00:05:41', _TGT),
          _rec('00:05:42', '  ✅ 거래 완료 (silent).')]
    body, warn, _ = _run_compare(up, bn)
    assert '✅ 실행 쌍 정합 (양쪽 cron 창 내' in body, body

    # 창 밖(수동 실행) → ✅ 금지
    bn2 = [_start('05:52:00'), _rec('05:52:11', _TGT),
           _rec('05:52:12', '  ✅ 거래 완료 (silent).')]
    body2, warn2, _ = _run_compare(up, bn2)
    assert '실행 쌍 확인 불가' in body2 and warn2 is True, body2
    assert 'cron 창 밖' in body2, body2


def test_compare_cycle_id_pairing():
    """양쪽 모두 사이클 ID 가 있으면 ID 일치를 요구한다."""
    up = [_start('00:05:00', rid='0905abcd', label='코인'), _rec('00:05:11', _TGT, '0905abcd'),
          _rec('00:05:12', '  ✅ 거래 완료 (silent).', '0905abcd')]
    bn = [_start('00:05:30', rid='0905abcd'), _rec('00:05:41', _TGT, '0905abcd'),
          _rec('00:05:42', '  ✅ 거래 완료 (silent).', '0905abcd')]
    body, warn, sides = _run_compare(up, bn)
    assert sides[0].cycle_id == '0905abcd' and sides[1].cycle_id == '0905abcd'
    assert '사이클 ID 0905abcd 일치' in body, body

    bn2 = [_start('00:05:30', rid='0905ffff'), _rec('00:05:41', _TGT, '0905ffff'),
           _rec('00:05:42', '  ✅ 거래 완료 (silent).', '0905ffff')]
    body2, warn2, _ = _run_compare(up, bn2)
    assert '사이클 ID 0905abcd vs 0905ffff' in body2 and warn2 is True, body2


def test_compare_kst_utc_boundary():
    """KST 날짜 기준 + UTC 로그 시각 매핑 (09:05 KST = 00:05 UTC 같은 날짜)."""
    assert cdr._in_cron_window('00:05:00') is True     # 09:05 KST
    assert cdr._in_cron_window('00:12:00') is True     # 09:12 KST
    assert cdr._in_cron_window('00:13:00') is False
    assert cdr._in_cron_window('15:05:00') is False    # 00:05 KST (다음날 새벽)
    assert cdr._in_cron_window('bad') is False
    # 서비스 날짜는 KST 기준
    import datetime as _dt
    try:
        from zoneinfo import ZoneInfo
        expect = _dt.datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d')
        assert cdr._service_today() == expect, cdr._service_today()
    except ImportError:
        pass


def test_compare_wal_markers_are_errors():
    """WAL 미해결/미체결 확인 불가는 error 로 잡히고 '거래 완료'가 덮지 못한다."""
    up = [_start('00:05:00', label='코인'), _rec('00:05:11', _TGT),
          _rec('00:05:12', '  ✅ 거래 완료 (silent).')]
    for marker, label in (('  🚨 WAL 미해결: bsX (TSTUSDT SELL 1.0) — 조회 실패',
                           '미해결 주문 WAL (수동 확인 필요)'),
                          ('❌ 미체결 상태 확인 불가 → fail-closed', '미체결 상태 확인 불가'),
                          ('❌ state 손상 → 실행 중단', 'state 파일 손상')):
        bn = [_start('00:05:02'), _rec('00:05:10', _MEMBER), _rec('00:05:11', _TGT),
              _rec('00:05:12', marker),
              _rec('00:05:13', '  ✅ 거래 완료 (silent).')]
        body, warn, sides = _run_compare(up, bn)
        assert sides[1].result_kind == 'error', (marker, sides[1].result_kind)
        assert label in sides[1].result_label or label in sides[1].issues, sides[1].issues
        assert warn is True


def test_compare_redaction_covers_child_loggers():
    rec = logging.getLogger('urllib3.connectionpool').makeRecord(
        'urllib3.connectionpool', logging.WARNING, __file__, 1,
        'POST https://api.telegram.org/bot999:AAH-tok3nValue/sendMessage failed', (), None)
    assert 'AAH-tok3nValue' not in rec.getMessage(), rec.getMessage()


def test_compare_redaction_covers_exception_traceback():
    """compare 프로세스도 exc_info traceback 을 마스킹한다."""
    import io
    stream = io.StringIO()
    h = logging.StreamHandler(stream)
    h.setFormatter(logging.Formatter('%(message)s'))
    lg = logging.getLogger('requests.test_cmp')
    lg.handlers = [h]
    lg.propagate = False
    lg.setLevel(logging.ERROR)
    cdr._wrap_handler(h, cdr._RedactingFilter())
    try:
        raise RuntimeError('https://api.telegram.org/bot42:AAH-cmpTok3n/sendMessage boom')
    except RuntimeError:
        lg.exception('send failed')
    out = stream.getvalue()
    assert 'AAH-cmpTok3n' not in out and 'REDACTED' in out, out


# ═══ 러너 ═══
def _all_tests():
    g = globals()
    return [(n, g[n]) for n in sorted(g) if n.startswith('test_') and callable(g[n])]


def main():
    failures = []
    for name, fn in _all_tests():
        try:
            fn()
            print(f'  PASS  {name}')
        except Exception as e:
            import traceback
            failures.append((name, e))
            print(f'  FAIL  {name}: {e}')
            traceback.print_exc()
    total = len(_all_tests())
    print(f'\n{total - len(failures)}/{total} passed')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
