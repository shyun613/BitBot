#!/usr/bin/env python3
"""Cap Defend 코인 현물 Executor — Binance USDT 마켓 이식판.

executor_coin.py (Upbit KRW) 의 거래소 레이어만 Binance 현물로 교체한 판본.
전략 흐름(엔진 호출/V24 파라미터/스냅샷/드리프트/카나리/refill/freshness)과
유니버스 필터(Upbit 상장·유의종목 필터 포함)는 원본과 100% 동일하게 유지한다.
→ 두 executor 의 target 을 1:1 비교하기 위한 검증용.

구조:
  - 신호: coin_live_engine.compute_live_targets
          (V24: Binance spot kline → 1D 단일 멤버 D_SMA42, snap=217×7, drift=0.10)
  - 체결: Binance Spot USDT (python-binance)
  - 상태: binance_spot_state.json (schema_version=V24)
  - alloc_transit cap: trade_state.json (공유 control-plane, executor_stock/auto_trade_binance 와 동일)

실행 순서:
  1. flock /tmp/coin_executor_binance.lock
  2. BinanceSpotAPI (exchangeInfo 필터 1회 로드) + 미체결 취소 (남은 미체결 심볼은 리밸 제외)
  3. 잔고 스냅샷 1회 — 평가 불가 자산이 있으면 BalanceIncomplete → 주문 전 exit 2
  4. 거래정지(status != TRADING) 보유 감지 → 즉시 시장가 청산 (freshness 무관, 체결 검증 후에만 완료 처리)
  5. compute_live_targets() 호출 (엔진 내부에서 freshness / 카나리 / 갭 / 앙상블 처리)
  6. all_fresh=False 면 리밸런싱 스킵 (청산만 수행)
  7. Cash buffer 2% 적용
  8. Notional cap 비활성 (필요 시 상수로 재활성)
  9. Delta 매매 (매도 → 매수), dust <10 USDT → 전량 매도
 10. 상태 저장 + 텔레그램 사전/사후 알림

주문 멱등성 (C1):
  모든 주문에 uuid4 기반 고유 newClientOrderId 를 부여하고, live 모드에서는 POST 직전 WAL
  (binance_spot_orders.wal, append-only jsonl + file/dir fsync)에 intent 를 남긴 뒤
  최종 확인 후 resolved 를 append 한다. run_once 시작 시 미해결 엔트리를 coid 권위 조회로만
  해소하며(시간 경과로 자동 해소하지 않는다), 확인 불가면 exit 2 + 알림.
  운영자가 거래소 주문내역을 직접 확인한 뒤에만 --wal-mark-resolved <coid> 로 수동 해소한다.

  재시도 정책 (거래소가 '접수 안 함'을 명시한 경우에만 재주문):
    - 명시적 사전 거절(DEFINITE_REJECT_CODES / -1100~-1132 파라미터 오류)  → 미접수 확정. 재시도 가능.
    - timeout / 5xx / 연결 오류 / -1000·-1001·-1006·-1007 / 응답 형식 이상 → **미확정**.
      2s·5s 지연 후 coid 재조회, 그래도 -2013(없음)이면 전파 지연 가능성이 남으므로
      재주문하지 않고 UnknownExecutionError → health lock.
    - 접수된 주문의 체결량도 최종(FINAL) 상태 조회로만 확정. 조회 실패·NOT_FOUND·비최종
      상태는 잔량 재매도 금지 → UnknownExecutionError.

dry-run 규약 (억제 범위 명시):
  억제하는 것 = ① 모든 주문/취소 API 호출 ② state 파일 저장 ③ 텔레그램 발송(로그로만 출력).
  억제하지 않는 것 = 로그 파일 기록, flock 파일 생성/truncate, 엔진의 universe/exchangeInfo
  디스크 캐시 갱신, 읽기 전용 계좌·시세 조회. (원본 executor_coin.py 와 동일한 규약)

  --state-ref PATH (엔진 상태 참조):
    dry-run 은 state 를 저장하지 않아 매 실행 7개 스냅샷을 새로 초기화한다 → 업비트 LIVE 와
    같은 날 다른 target 이 나올 수 있다. 그래서 업비트 실행기의 V24 state 파일을 **읽기 전용**
    출처로 참조해 같은 스냅샷에서 출발한다 (참조 파일은 어떤 경로에서도 절대 쓰지 않는다).
    dry-run: 매 실행 참조(members/last_target_snapshot/schema_version/last_member_targets 덮어쓰기),
             자체 state 저장은 종전대로 생략. drift 평가 보유비중은 참조 목표를 보유 중이라
             가정한다 (dust 계좌의 실잔고로 평가하면 매일 drift 발화 → refill v2 가 업비트와
             다른 코인으로 교체됨). 참조가 이미 오늘 봉을 처리했으면(같은 봉) '새 봉 없음'
             스킵을 하지 않고 참조의 최종 target(refill v2 반영)으로 비교를 진행한다.
    live:    자체 state 에 members 가 없을 때만 1회 시드, 이후에는 자체 state 만 사용
             (시드 때 참조의 cash buffer 정책도 한 번 같이 들여온다). 시드 당일 참조가 이미
             오늘 봉을 처리했어도 참조의 최종 target(refill 반영)으로 진행하고 그 값을 저장한다
             — 이때 참조 members 를 복원하고(엔진이 stale 기준으로 돌린 refill 되돌리기)
             drift 는 실잔고 vs 참조 최종 target 으로 다시 계산한다. 시드 실행이 Freshness
             미달로 끝나면 state 를 저장하지 않는다 (stale 이 굳지 않게, 다음 실행에서 재시드).
    참조의 cash buffer(spot_cash_buffer/cash_buffer/buffer_pct)가 있으면 그대로 빌려 쓴다 —
    target 스케일과 drift 가정 보유비중을 업비트와 같은 기준으로 맞추기 위해서다.
    참조 키는 엄격하게 본다: members 는 엔진 MEMBERS 와 정확히 같아야 하고, 가중치 키는
    대문자 티커(또는 현금 키)만 허용한다 (메타키·소문자 키는 거부).
    참조 실패(파일 부재/JSON 손상/스키마 불일치/가중치 이상 등)는 fail-closed — fresh 초기화로
    대체하지 않고 exit 2 로 중단한다. 참조 경로는 .json 파일만 허용하며, 이 실행기가 쓰는
    파일(자체 state/로그/락/캐시)과 같은 파일(심볼릭·하드링크 포함)이면 거부한다.
    한계: ① 새 봉 경로의 refill v2 는 '참조 목표 보유' 가정 drift 로 평가하므로, 업비트가
          실잔고 drift 로 그날 refill 을 발화하면 shadow 는 refill 이전 target 을 낼 수 있다
          (그날 리포트에 불일치로 드러나고 다음 날 정합). ② live 시드 당일 업비트가 이미 오늘
          봉을 처리했으면 참조의 최종 target 을 쓰고 저장하므로 stale 은 남지 않는다. 다만
          시드 직후 첫 리밸런싱은 실잔고 drift 로 한 번에 목표까지 채우려 하므로 체결 여력을
          확인할 것.
          ③ universe_cache 는 참조하지 않고 dry-run 에선 자체 캐시도 버린다 (TTL 20h 라
          양쪽이 같은 분 안에 같은 소스로 재구성하고, 같은 봉 경로는 참조 최종 target 으로
          덮으므로 결과에 영향 없음).

exit 코드 계약 (원본 executor_coin.py parity — 의도적으로 동일하게 유지):
  0 = 정상 종료 / 정상 스킵(새 봉 없음·target 불변·목표 근접) / health lock 활성 / flock 충돌
  1 = freshness 미달 스킵
  2 = 에러 (엔진 호출 실패, 잔고 스냅샷 불완전, 미체결 취소 확인 실패, 치명 예외)
  3 = 청산 실패 fail-closed, UNKNOWN_EXECUTION (health lock 동반)
  ※ 원본과 동일. monitoring 이 0/1/2 만 가정하지 않도록 3 도 실패로 취급할 것.

Usage:
  python3 executor_coin_binance.py
  python3 executor_coin_binance.py --dry-run
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import logging
import math
import os, json, math
import re
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pyupbit
import requests
from pyupbit import request_api as pyupbit_request_api

from binance.client import Client
from binance.exceptions import BinanceAPIException

from common.io import save_json
from common.notify import send_telegram as _send_tg
from common.health_guard import HealthGuard, UnknownExecutionError

try:
    from common.logging_utils import setup_file_logger, make_log_fn
except ImportError:
    setup_file_logger = None  # type: ignore
    make_log_fn = None  # type: ignore

import coin_live_engine as cle

try:
    from config import (
        BINANCE_API_KEY, BINANCE_API_SECRET,
        TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
    )
except ImportError:
    BINANCE_API_KEY = os.environ.get('BINANCE_API_KEY', '')
    BINANCE_API_SECRET = os.environ.get('BINANCE_API_SECRET', '')
    TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')


# ═══ 상수 ═══
STATE_FILE = 'binance_spot_state.json'
# alloc_transit 은 주식/선물과 공유하는 control-plane 파일에서 읽는다 (executor_stock.py:62 동일 규약)
ALLOC_TRANSIT_STATE_FILE = 'trade_state.json'
ALLOC_SPOT_RATIO = 0.25  # V24 (옵션 D, 2026-05-23)
LOCK_FILE = '/tmp/coin_executor_binance.lock'
LOG_FILE = 'executor_coin_binance.log'
CACHE_DIR = os.path.dirname(os.path.abspath(__file__))

EXCHANGE_LABEL = '바낸현물'          # 텔레그램/로그 prefix
QUOTE_ASSET = 'USDT'                # 마켓 quote
CASH_ASSET = 'USDT'                 # 잔고 dict 의 현금 키 (원본의 'KRW' 자리)
CLIENT_ORDER_PREFIX = 'bs'          # newClientOrderId prefix
COID_RANDOM_LEN = 10                # uuid4 기반 난수 길이 (전역 유일성)

CASH_BUFFER_DEFAULT = 0.02          # 총자산의 2%는 USDT 유지
NOTIONAL_CAP_FRACTION = 1.00        # 1.0이면 비활성
MIN_ORDER_USDT = 10.0               # 최소주문 (심볼별 minNotional 과 max 로 결합)
MIN_ORDER_USDT_D = Decimal('10')
DUST_USDT = 10.0                    # 이보다 작은 잔여는 전량 매도
LIMIT_PRICE_SLIP = 0.003            # 매수 지정가 +0.3%
ORDER_WAIT_SEC = 5                  # 매수 후 미체결 취소 대기
LIQUIDATION_MAX_RETRIES = 3
# V24 매도 robust retry — 코인 spot 매도 무조건 성공 보장 (2026-05-08)
SELL_RETRY_DELAYS = [1, 3, 5, 10]
SELL_MAX_ATTEMPTS = 5
SELL_TIMEOUT_SEC = 60
EXCHANGE_INFO_MAX_RETRIES = 3
ORDER_LOOKUP_MAX_RETRIES = 2        # clientOrderId 단건 조회 내부 재시도
ORDER_LOOKUP_RETRY_DELAY = 1
FILL_CHECK_DELAY_SEC = 2            # 주문 접수 후 체결 조회까지 대기
AMBIGUOUS_LOOKUP_DELAYS = [0, 2, 5]  # 모호한 POST 후 재조회 지연 (전파 지연 대비)
FILL_POLL_DELAYS = [0, 2, 5]         # 체결 확정 폴링 지연
ORDER_WAL_FILE = 'binance_spot_orders.wal'
RECV_WINDOW_MS = 5000               # 주문/조회 recvWindow 명시

ORDER_NOT_EXIST_CODE = -2013        # "Order does not exist"
OPEN_STATUSES = {'NEW', 'PARTIALLY_FILLED', 'PENDING_NEW', 'PENDING_CANCEL'}
FINAL_STATUSES = {'FILLED', 'CANCELED', 'EXPIRED', 'REJECTED', 'EXPIRED_IN_MATCH'}
KNOWN_STATUSES = OPEN_STATUSES | FINAL_STATUSES

# 거래소가 요청 검증 단계에서 거절 → 주문이 접수되지 않았음이 보장됨 (재주문 안전)
DEFINITE_REJECT_CODES = {
    -1003,  # TOO_MANY_REQUESTS (레이트리밋에서 거절)
    -1010,  # ERROR_MSG_RECEIVED
    -1013,  # INVALID_MESSAGE / filter failure
    -1015,  # TOO_MANY_ORDERS
    -1021,  # 타임스탬프 recvWindow 밖 → 처리 전 거절
    -1022,  # INVALID_SIGNATURE
    -2010,  # NEW_ORDER_REJECTED (잔고 부족 등)
    -2015,  # 잘못된 API-key / 권한 / IP
    -2016,  # NO_TRADING_WINDOW
}
PARAM_ERROR_MIN, PARAM_ERROR_MAX = -1132, -1100   # 파라미터 검증 실패 계열
# 사전 거절 중 잠시 후 재시도할 가치가 있는 것 (그 외 거절은 결정적이라 즉시 중단)
TRANSIENT_REJECT_CODES = {-1003, -1015, -1021, -2016}


# 주문 수량/가격 산정에 영향이 없어 무시해도 되는 필터 (개수 제한 등)
KNOWN_IGNORABLE_FILTERS = {
    'ICEBERG_PARTS', 'TRAILING_DELTA', 'MAX_NUM_ORDERS', 'MAX_NUM_ALGO_ORDERS',
    'MAX_NUM_ICEBERG_ORDERS', 'MAX_POSITION', 'MAX_NUM_ORDER_LISTS',
    'MAX_NUM_ORDER_AMENDS', 'EXCHANGE_MAX_NUM_ORDERS', 'EXCHANGE_MAX_NUM_ALGO_ORDERS',
}


def _validate_order_resp(o, symbol: str, coid: str) -> str:
    """get_order/주문 응답 엄격 검증. 이상이면 사유 문자열, 정상이면 ''.

    symbol/clientOrderId 일치, status 가 알려진 값, origQty/executedQty 가 유한하고
    0 <= executedQty <= origQty 인지까지 확인한다 (malformed 응답이 체결량으로 둔갑 방지).
    """
    if not isinstance(o, dict):
        return f'dict 아님({type(o).__name__})'
    if o.get('symbol') != symbol:
        return f'symbol 불일치({o.get("symbol")!r} != {symbol!r})'
    if o.get('clientOrderId') != coid:
        return f'clientOrderId 불일치({o.get("clientOrderId")!r} != {coid!r})'
    st = o.get('status')
    if not isinstance(st, str) or st not in KNOWN_STATUSES:
        return f'미지 status({st!r})'
    for k in ('origQty', 'executedQty'):
        v = o.get(k)
        if v is None or isinstance(v, bool):
            return f'{k} 누락'
        try:
            d = Decimal(str(v))
        except Exception:
            return f'{k} 파싱 실패({v!r})'
        if not d.is_finite() or d < 0:
            return f'{k} 비정상({v!r})'
    if Decimal(str(o['executedQty'])) > Decimal(str(o['origQty'])):
        return f'executedQty({o["executedQty"]}) > origQty({o["origQty"]})'
    return ''


def _wal_num(rec: Dict, key: str, positive: bool = True) -> str:
    v = rec.get(key)
    if v is None or isinstance(v, bool):
        return f'{key} 누락'
    try:
        d = Decimal(str(v))
    except Exception:
        return f'{key} 파싱 실패({v!r})'
    if not d.is_finite() or (positive and d <= 0):
        return f'{key} 비정상({v!r})'
    return ''


RESOLVED_STATUSES = KNOWN_STATUSES | {'NOT_PLACED', 'MANUAL_RESOLVED'}


def _validate_wal_record(rec: Dict, seen_intents: Dict[str, Dict],
                         seen_resolved: set) -> str:
    """WAL 한 행의 이벤트별 필수 필드 검증. 이상이면 사유, 정상이면 ''.

    coid 는 writer 경로에서 유일하므로 intent/resolved 중복은 의미적 손상으로 보고 거부한다
    (동일 coid 재사용 시 후속 intent 가 기존 resolved 로 해소되는 것을 차단 — 라운드5 m).
    """
    ev = rec.get('event')
    coid = rec.get('coid')
    if not isinstance(coid, str) or not coid:
        return 'coid 누락'
    if not isinstance(rec.get('ts'), (int, float)) or isinstance(rec.get('ts'), bool):
        return 'ts 누락'
    bad = _wal_num(rec, 'ts')
    if bad:
        return bad
    if ev == 'intent':
        if coid in seen_intents:
            return f'intent coid 중복({coid})'
        for k in ('symbol', 'side', 'type'):
            v = rec.get(k)
            if not isinstance(v, str) or not v:
                return f'intent {k} 누락'
        if rec['side'] not in ('BUY', 'SELL'):
            return f'intent side 이상({rec["side"]!r})'
        if rec['type'] not in ('MARKET', 'LIMIT'):
            return f'intent type 이상({rec["type"]!r})'
        bad = _wal_num(rec, 'qty')
        if bad:
            return f'intent {bad}'
        bad = _wal_num(rec, 'recv_window_ms')
        if bad:
            return f'intent {bad}'
        return ''
    if ev == 'resolved':
        st = rec.get('status')
        if not isinstance(st, str) or not st:
            return 'resolved status 누락'
        if st not in RESOLVED_STATUSES:
            return f'resolved status 허용값 아님({st!r})'
        if coid not in seen_intents:
            return f'resolved 에 선행 intent 없음({coid})'
        if coid in seen_resolved:
            return f'resolved coid 중복({coid})'
        return ''
    return f'미지 event={ev!r}'


def _is_definite_reject(code) -> bool:
    """이 에러코드면 주문이 접수되지 않았음이 확정인가."""
    if not isinstance(code, int):
        return False
    return code in DEFINITE_REJECT_CODES or PARAM_ERROR_MIN <= code <= PARAM_ERROR_MAX


class BalanceIncomplete(Exception):
    """보유 자산 중 평가 불가(심볼/가격 없음)가 있어 PV 를 신뢰할 수 없음.

    이 상태로 delta 매매를 진행하면 PV 가 과소평가되어 보유분을 대량 매도할 수 있으므로
    주문 전에 반드시 중단한다 (exit 2).
    """


def _strict_num(value, label: str) -> float:
    """잔고 수량 엄격 파싱 — 변환 실패/NaN/inf/음수는 조용히 넘기지 않고 중단시킨다 (M)."""
    if value is None or isinstance(value, bool):
        raise BalanceIncomplete(f'{label} 값 이상: {value!r}')
    if isinstance(value, str) and not value.strip():
        raise BalanceIncomplete(f'{label} 빈 문자열')
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise BalanceIncomplete(f'{label} 숫자 변환 실패: {value!r}')
    if not math.isfinite(f):
        raise BalanceIncomplete(f'{label} 비유한값(NaN/inf): {value!r}')
    return f


def _patch_pyupbit_remaining_req_parser():
    """pyupbit Remaining-Req 파싱 실패를 무해화.

    (본 판본은 pyupbit 를 유니버스 필터용 public OHLCV 조회에만 사용하지만,
     원본과의 diff 최소화를 위해 패치는 그대로 둔다.)
    """
    orig_parse = pyupbit_request_api._parse

    def _safe_parse(remaining_req: str):
        try:
            return orig_parse(remaining_req)
        except Exception:
            return {"group": "unknown", "min": 0, "sec": 0}

    pyupbit_request_api._parse = _safe_parse


_patch_pyupbit_remaining_req_parser()


CAP_RATIO_FLOOR = 0.10  # cap_ratio < floor → 거래 중단 fallback (1.0 처리)
ALLOC_TRANSIT_STALE_HOURS = 26
CAP_DEFEND_MIN_EXCESS = 0.01  # cap_ratio < 0.99 면 cap_defend 매도 발동 (any_new_bar 우회)


def _validate_cap_ratio(val, sleeve_name: str, log_fn=None):
    """cap_ratio 검증. 0 < cr ≤ 1, finite. invalid → 1.0 fallback + ERROR 로그."""
    _l = log_fn if log_fn else (lambda m: None)
    try:
        cr = float(val)
    except Exception:
        _l(f'  🚨 alloc_transit cap_ratio[{sleeve_name}] parse 실패 ({val!r}) → fallback 1.0')
        return 1.0
    if not math.isfinite(cr) or cr <= 0:
        _l(f'  🚨 alloc_transit cap_ratio[{sleeve_name}]={cr} invalid → fallback 1.0')
        return 1.0
    if cr < CAP_RATIO_FLOOR:
        _l(f'  🚨 alloc_transit cap_ratio[{sleeve_name}]={cr:.4f} < floor {CAP_RATIO_FLOOR} → SKIP (fallback 1.0)')
        return 1.0
    if cr > 1.0:
        return 1.0
    return cr


def _read_alloc_transit_cap_ratio_spot():
    """공유 trade_state.json 의 alloc_transit active 면 spot cap_ratio (≤1.0) 반환. 아니면 None.

    schema 손상 / parse 실패 / cap_ratio invalid → 1.0 fallback + ERROR 로그.
    """
    _p = os.path.join(CACHE_DIR, ALLOC_TRANSIT_STATE_FILE)
    try:
        if not os.path.exists(_p):
            return None
        with open(_p, 'r') as f:
            obj = json.load(f)
        at = obj.get('alloc_transit')
        if not at or not at.get('active'):
            return None
        cr_raw = (at.get('cap_ratio') or {}).get('spot')
        if cr_raw is None:
            try:
                log('  🚨 alloc_transit active 하나 cap_ratio[spot] missing → fallback 1.0')
            except Exception:
                pass
            return 1.0
        cr = _validate_cap_ratio(cr_raw, 'spot', log_fn=(lambda m: log(m)))
        # stale guard
        try:
            import time as _t
            age_h = (_t.time() - os.path.getmtime(_p)) / 3600
            if age_h > ALLOC_TRANSIT_STALE_HOURS:
                log(f'  🚨 alloc_transit state stale (age {age_h:.1f}h) → cap 무시')
                return None
            log(f'  alloc_transit cap_ratio[spot]={cr:.4f} (mtime age {age_h:.1f}h)')
        except Exception:
            pass
        return cr
    except json.JSONDecodeError as ex:
        try:
            log(f'  🚨 alloc_transit JSON parse 실패: {ex} → fallback (cap 없음)')
        except Exception:
            pass
        return None
    except Exception:
        return None


# ═══ 비밀정보 마스킹 (m: 외부 예외 원문이 토큰/서명을 담을 수 있음) ═══
_TG_TOKEN_RE = re.compile(r'bot\d+:[A-Za-z0-9_\-]{10,}')
_SIG_RE = re.compile(r'((?:signature|apiKey|api_key|token)=)[A-Za-z0-9%_\-./+]+', re.I)
_TG_URL_RE = re.compile(r'(api\.telegram\.org/)[^\s\'"]+')
_HDR_RE = re.compile(r"('?X-MBX-APIKEY'?\s*:\s*'?)[A-Za-z0-9]+", re.I)


def _redact(value) -> str:
    """로그로 나가는 문자열에서 토큰/API 키/서명을 마스킹."""
    s = str(value)
    s = _TG_TOKEN_RE.sub('bot<REDACTED>', s)
    s = _TG_URL_RE.sub(r'\1<REDACTED>', s)
    s = _SIG_RE.sub(r'\1<REDACTED>', s)
    s = _HDR_RE.sub(r'\1<REDACTED>', s)
    for _secret in (BINANCE_API_KEY, BINANCE_API_SECRET, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID):
        if _secret and len(str(_secret)) >= 8:
            s = s.replace(str(_secret), '<REDACTED>')
    return s


class _RedactingFilter(logging.Filter):
    """핸들러에 붙여 쓰는 마스킹 필터 (자식 로거 레코드까지 확실히 적용)."""

    def filter(self, record: logging.LogRecord) -> bool:
        _redact_record(record)
        return True


def _redact_record(record: logging.LogRecord) -> logging.LogRecord:
    try:
        msg = record.getMessage()
    except Exception:
        return record
    red = _redact(msg)
    if red != msg:
        record.msg = red
        record.args = ()
    return record


class _RedactingFormatter(logging.Formatter):
    """핸들러 formatter 를 감싸 exception/stack 텍스트까지 마스킹.

    LogRecordFactory 는 record.msg 만 다루므로 exc_info 가 뒤늦게 렌더링하는
    traceback 원문(토큰/서명 포함 가능)을 놓친다.
    """

    def __init__(self, inner: logging.Formatter):
        super().__init__()
        self._inner = inner

    def format(self, record):
        out = self._inner.format(record)
        # 원본 exc_text 가 남아 뒤따르는 unwrapped handler 로 새지 않게 교체
        if getattr(record, 'exc_text', None):
            record.exc_text = _redact(record.exc_text)
        return _redact(out)

    def formatException(self, ei):
        return _redact(self._inner.formatException(ei))

    def formatStack(self, stack_info):
        return _redact(self._inner.formatStack(stack_info))


def _wrap_handler(h: logging.Handler, filt: logging.Filter):
    h.addFilter(filt)
    fmt = h.formatter or logging.Formatter()
    if not isinstance(fmt, _RedactingFormatter):
        logging.Handler.setFormatter(h, _RedactingFormatter(fmt))


def _patch_set_formatter():
    """이후 추가되는 formatter 도 자동으로 감싸지게 한다."""
    if getattr(logging.Handler.setFormatter, '_redacting', False):
        return
    _orig = logging.Handler.setFormatter

    def _patched(self, fmt):
        if fmt is not None and not isinstance(fmt, _RedactingFormatter):
            fmt = _RedactingFormatter(fmt)
        return _orig(self, fmt)

    _patched._redacting = True
    logging.Handler.setFormatter = _patched


def _patch_handler_format():
    """formatter 를 아예 설정하지 않은 핸들러(기본 formatter 경로)까지 커버.

    setFormatter 를 호출하지 않고 추가된 핸들러도 traceback 이 마스킹되도록
    Handler.format 자체를 감싼다.
    """
    if getattr(logging.Handler.format, '_redacting', False):
        return
    _orig = logging.Handler.format

    def _patched(self, record):
        out = _orig(self, record)
        if getattr(record, 'exc_text', None):
            record.exc_text = _redact(record.exc_text)
        return _redact(out)

    _patched._redacting = True
    logging.Handler.format = _patched


def _install_redaction():
    """logger.addFilter 는 자식 로거 레코드에 적용되지 않는다 (Python logging 규약).

    LogRecordFactory 로 프로세스의 모든 레코드 메시지를 마스킹하고, 핸들러 formatter 를
    감싸 exc_info traceback 텍스트까지 마스킹한다.
    """
    _prev_factory = logging.getLogRecordFactory()

    def _factory(*args, **kwargs):
        return _redact_record(_prev_factory(*args, **kwargs))

    logging.setLogRecordFactory(_factory)
    _patch_set_formatter()
    _patch_handler_format()
    _filt = _RedactingFilter()
    for _lg in [logging.getLogger()] + [logging.getLogger(n) for n in
                                        ('common.notify', 'binance', 'urllib3', 'requests',
                                         'executor_coin_binance', LOG_FILE)]:
        for _h in list(getattr(_lg, 'handlers', []) or []):
            _wrap_handler(_h, _filt)


# ═══ 로거 ═══
LOG_PATH = os.path.join(CACHE_DIR, LOG_FILE)
if setup_file_logger and make_log_fn:
    _logger = setup_file_logger(LOG_FILE, LOG_PATH)
    # 래퍼(run_trade_oracle.sh)가 export 한 사이클 ID — 로그 [rid] 칸에 8자로 찍힌다.
    # (업비트판 executor_coin.py 는 수정 금지 대상이라 rid 를 못 찍으므로 비교기는
    #  양쪽 ID 가 모두 있을 때만 ID 일치를 요구하고, 아니면 cron 창 규칙으로 판정한다.)
    _run_id_ref = [os.environ.get('BITBOT_CYCLE_ID', '')]
    log = make_log_fn(_logger, _run_id_ref)
else:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')
    _logger = logging.getLogger('executor_coin_binance')
    def log(msg: str, level: str = 'info'):  # type: ignore
        getattr(_logger, level, _logger.info)(msg)

_install_redaction()


# ═══ 텔레그램 버퍼 ═══
_tg_events: List[str] = []

def _tg(msg: str):
    _tg_events.append(msg)

def _flush_telegram(dry_run: bool = False):
    if not _tg_events:
        return
    # dry_run 시 텔레그램 silent (로그에만 출력)
    if dry_run:
        log('  [DRY] telegram silent — events: ' + ' | '.join(_tg_events))
        _tg_events.clear()
        return
    payload = f'[{EXCHANGE_LABEL}]\n' + '\n'.join(_tg_events)
    try:
        _send_tg(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, payload)
    except Exception as e:
        log(f'텔레그램 전송 실패: {_redact(e)}')
    _tg_events.clear()


# ═══ 호가/수량 단위 (Binance exchangeInfo 필터 — Decimal 원문 보존) ═══
def _dec(v, default: str = '0') -> Decimal:
    """exchangeInfo 원문 문자열 → Decimal (float 왕복 없음)."""
    try:
        return Decimal(str(v))
    except Exception:
        return Decimal(default)


def _step_exp(step: Decimal) -> int:
    """step 의 유효 소수 자릿수."""
    if step is None or step <= 0:
        return 8
    try:
        return max(0, -step.normalize().as_tuple().exponent)
    except Exception:
        return 8


def _floor_to_step(value: Decimal, step: Decimal) -> Decimal:
    """stepSize 단위 절사."""
    if step is None or step <= 0:
        return value
    return (value / step).to_integral_value(rounding=ROUND_DOWN) * step


def _ceil_to_step(value: Decimal, step: Decimal) -> Decimal:
    """tickSize 단위 올림."""
    if step is None or step <= 0:
        return value
    return (value / step).to_integral_value(rounding=ROUND_UP) * step


def _fmt_step(value: Decimal, step: Decimal) -> str:
    """API 전송용 고정 소수 문자열 (지수표기 방지)."""
    exp = _step_exp(step)
    q = value.quantize(Decimal(1).scaleb(-exp), rounding=ROUND_DOWN)
    return format(q, 'f')


def _lcm_step(a: Decimal, b: Decimal) -> Decimal:
    """두 stepSize 의 최소공배수 (양쪽 필터를 동시에 만족하는 최소 단위).

    LOT_SIZE 와 MARKET_LOT_SIZE 는 각각 독립적으로 적용되므로 max() 가 아니라
    공통 배수여야 한다 (예: 0.001 과 0.0025 → 0.005).
    """
    if a is None or a <= 0:
        return b if b and b > 0 else Decimal('0')
    if b is None or b <= 0:
        return a
    exp = max(_step_exp(a), _step_exp(b))
    scale = Decimal(1).scaleb(exp)
    ia = int((a * scale).to_integral_value(rounding=ROUND_DOWN))
    ib = int((b * scale).to_integral_value(rounding=ROUND_DOWN))
    if ia <= 0 or ib <= 0:
        return max(a, b)
    g = math.gcd(ia, ib)
    return (Decimal(ia // g) * Decimal(ib)) / scale


def _is_step_multiple(value: Decimal, step: Decimal) -> bool:
    if step is None or step <= 0:
        return True
    try:
        return (value % step) == 0
    except Exception:
        return False


def _sanitize_client_order_id(raw: str) -> str:
    """Binance newClientOrderId 규격: ^[\\.A-Za-z0-9_:/-]{1,36}$"""
    cleaned = re.sub(r'[^A-Za-z0-9_:./-]', '', raw)
    return cleaned[:36] or 'bs0'


# ═══ Binance Spot API 래퍼 ═══
class BinanceSpotAPI:
    def __init__(self, dry_run: bool = False, ignore_assets: Sequence[str] = (),
                 client=None, run_id: Optional[str] = None):
        self.client = client if client is not None else Client(BINANCE_API_KEY, BINANCE_API_SECRET)
        self.dry_run = dry_run
        self.ignore_assets = {str(a).upper() for a in (ignore_assets or ())}
        self.run_id = run_id or datetime.now(timezone.utc).strftime('%y%m%d%H%M%S')
        self._order_seq = 0
        self._last_post_ts: Optional[float] = None
        self._last_price_map: Dict[str, float] = {}
        self._filters: Dict[str, Dict] = {}      # symbol -> 필터 (Decimal)
        self._base_to_symbol: Dict[str, str] = {}
        self._load_exchange_filters()

    # ─── exchangeInfo 필터 (시작 시 1회 로드·캐시) ───
    def _load_exchange_filters(self):
        info = None
        last_err = None
        for attempt in range(1, EXCHANGE_INFO_MAX_RETRIES + 1):
            try:
                info = self.client.get_exchange_info()
                break
            except Exception as e:
                last_err = e
                log(f'  exchangeInfo 조회 attempt {attempt} 오류: {_redact(e)}')
                if attempt >= EXCHANGE_INFO_MAX_RETRIES:
                    raise RuntimeError(f'Binance exchangeInfo 로드 실패: {last_err}')
                time.sleep(2 * attempt)
        for s in ((info or {}).get('symbols') or []):
            if s.get('quoteAsset') != QUOTE_ASSET:
                continue
            sym = s.get('symbol', '')
            base = s.get('baseAsset', '')
            if not sym or not base:
                continue
            f = {
                'symbol': sym, 'base': base, 'status': s.get('status', ''),
                'step': Decimal('0'), 'min_qty': Decimal('0'), 'max_qty': Decimal('0'),
                'mkt_step': Decimal('0'), 'mkt_min_qty': Decimal('0'), 'mkt_max_qty': Decimal('0'),
                'tick': Decimal('0'), 'min_price': Decimal('0'), 'max_price': Decimal('0'),
                'min_notional': Decimal('0'), 'max_notional': Decimal('0'),
                'apply_min_to_market': True, 'apply_max_to_market': False,
                # PERCENT_PRICE / PERCENT_PRICE_BY_SIDE (지정가 허용 범위)
                'bid_mult_up': Decimal('0'), 'bid_mult_down': Decimal('0'),
                'ask_mult_up': Decimal('0'), 'ask_mult_down': Decimal('0'),
                'unknown_filters': [],
            }
            for ft in (s.get('filters') or []):
                t = ft.get('filterType')
                if t == 'LOT_SIZE':
                    f['step'] = _dec(ft.get('stepSize'))
                    f['min_qty'] = _dec(ft.get('minQty'))
                    f['max_qty'] = _dec(ft.get('maxQty'))
                elif t == 'MARKET_LOT_SIZE':
                    f['mkt_step'] = _dec(ft.get('stepSize'))
                    f['mkt_min_qty'] = _dec(ft.get('minQty'))
                    f['mkt_max_qty'] = _dec(ft.get('maxQty'))
                elif t == 'PRICE_FILTER':
                    f['tick'] = _dec(ft.get('tickSize'))
                    f['min_price'] = _dec(ft.get('minPrice'))
                    f['max_price'] = _dec(ft.get('maxPrice'))
                elif t == 'NOTIONAL':
                    f['min_notional'] = max(f['min_notional'], _dec(ft.get('minNotional')))
                    f['max_notional'] = max(f['max_notional'], _dec(ft.get('maxNotional')))
                    if 'applyMinToMarket' in ft:
                        f['apply_min_to_market'] = bool(ft.get('applyMinToMarket'))
                    if 'applyMaxToMarket' in ft:
                        f['apply_max_to_market'] = bool(ft.get('applyMaxToMarket'))
                elif t == 'MIN_NOTIONAL':
                    # legacy 필터 — applyToMarket 이 시장가 적용 여부
                    f['min_notional'] = max(f['min_notional'], _dec(ft.get('minNotional')))
                    if 'applyToMarket' in ft:
                        f['apply_min_to_market'] = bool(ft.get('applyToMarket'))
                elif t == 'PERCENT_PRICE':
                    up, down = _dec(ft.get('multiplierUp')), _dec(ft.get('multiplierDown'))
                    f['bid_mult_up'] = f['ask_mult_up'] = up
                    f['bid_mult_down'] = f['ask_mult_down'] = down
                elif t == 'PERCENT_PRICE_BY_SIDE':
                    f['bid_mult_up'] = _dec(ft.get('bidMultiplierUp'))
                    f['bid_mult_down'] = _dec(ft.get('bidMultiplierDown'))
                    f['ask_mult_up'] = _dec(ft.get('askMultiplierUp'))
                    f['ask_mult_down'] = _dec(ft.get('askMultiplierDown'))
                elif t in KNOWN_IGNORABLE_FILTERS:
                    pass
                else:
                    f['unknown_filters'].append(t)
            self._filters[sym] = f
            self._base_to_symbol[base] = sym
        unknown = sorted({t for v in self._filters.values() for t in v['unknown_filters']})
        log(f'  exchangeInfo 로드: {QUOTE_ASSET} 마켓 {len(self._filters)}종목 '
            f'(TRADING {sum(1 for v in self._filters.values() if v["status"] == "TRADING")})'
            + (f' — 미지 필터 {unknown} (검증 미적용, 진행)' if unknown else ''))

    def symbol_of(self, coin: str) -> Optional[str]:
        return self._base_to_symbol.get(coin)

    def symbol_status(self, coin: str) -> Optional[str]:
        sym = self.symbol_of(coin)
        if not sym:
            return None
        return self._filters.get(sym, {}).get('status')

    def _f(self, coin: str) -> Dict:
        return self._filters.get(self.symbol_of(coin) or '', {})

    def min_notional(self, coin: str) -> float:
        """심볼 minNotional 과 전역 최소주문의 max (가격 무관 하한)."""
        return float(max(MIN_ORDER_USDT_D, self._f(coin).get('min_notional', Decimal('0'))))

    def _qty_limits(self, coin: str, market: bool) -> Tuple[Decimal, Decimal, Decimal]:
        """(step, min_qty, max_qty) — 시장가면 LOT_SIZE·MARKET_LOT_SIZE 양쪽을 동시에 만족."""
        f = self._f(coin)
        step = f.get('step', Decimal('0'))
        min_qty = f.get('min_qty', Decimal('0'))
        max_qty = f.get('max_qty', Decimal('0'))
        if market:
            # 두 필터는 독립 적용 → step 은 공통 배수(LCM), min 은 큰 값, max 는 작은 값
            step = _lcm_step(step, f.get('mkt_step', Decimal('0')))
            if f.get('mkt_min_qty', Decimal('0')) > 0:
                min_qty = max(min_qty, f['mkt_min_qty'])
            if f.get('mkt_max_qty', Decimal('0')) > 0:
                max_qty = min(max_qty, f['mkt_max_qty']) if max_qty > 0 else f['mkt_max_qty']
        return step, min_qty, max_qty

    def executable_min_notional(self, coin: str, price: float, market: bool = True) -> float:
        """격자(step) 위에서 실제로 체결 가능한 최소 주문금액.

            qty = ceil(max(minQty, minNotional/price) / step) * step
            min_notional_eff = qty * price

        예) step=0.03, price=100, minNotional=10 → qty=0.12 → $12 (단순 max 는 $10 오답).
        이보다 작은 잔여는 어떤 방법으로도 주문할 수 없는 '불가피 dust'다.
        후보 생성·dust·리밸 완료 판정에서 공통으로 쓴다 (M).
        """
        f = self._f(coin)
        base = max(MIN_ORDER_USDT_D, f.get('min_notional', Decimal('0')))
        if price is None or price <= 0 or not f:
            return float(base)
        p = _dec(price)
        step, min_qty, _ = self._qty_limits(coin, market)
        need_qty = max(min_qty, base / p)
        if step > 0:
            units = (need_qty / step).to_integral_value(rounding=ROUND_UP)
            need_qty = units * step
        return float(need_qty * p)

    def expected_limit_price(self, coin: str, ref_price: float) -> float:
        """매수 판정용 예상 지정가 (현재가×1.003 을 tick 올림)."""
        if ref_price is None or ref_price <= 0:
            return 0.0
        tick = self._f(coin).get('tick', Decimal('0'))
        return float(_ceil_to_step(_dec(ref_price) * _dec(1 + LIMIT_PRICE_SLIP), tick))

    def buy_min_notional(self, coin: str, ref_price: float) -> float:
        """매수 후보/스킵 판정용 — 예상 지정가 기준 격자 최소금액."""
        px = self.expected_limit_price(coin, ref_price)
        return self.executable_min_notional(coin, px or ref_price, market=False)

    # ─── 주문 수량/가격 정규화 (M2: 필터 전량 검증) ───
    def check_percent_price(self, coin: str, price: Decimal, ref_price: float,
                            side: str) -> str:
        """PERCENT_PRICE / PERCENT_PRICE_BY_SIDE 검증. 위반이면 사유 문자열.

        정확한 기준은 거래소의 avgPrice(avgPriceMins) 이나 조회 API 가 별도라
        현재가를 근사 기준으로 쓴다 (보수적 — 위반 의심이면 주문하지 않음).
        """
        f = self._f(coin)
        if not f or ref_price is None or ref_price <= 0:
            return ''
        ref = _dec(ref_price)
        up = f['bid_mult_up'] if side == 'BUY' else f['ask_mult_up']
        down = f['bid_mult_down'] if side == 'BUY' else f['ask_mult_down']
        if up > 0 and price > ref * up:
            return f'가격 {price} > PERCENT_PRICE 상한 {ref * up}'
        if down > 0 and price < ref * down:
            return f'가격 {price} < PERCENT_PRICE 하한 {ref * down}'
        return ''

    def prepare_price(self, coin: str, price: float, ref_price: float = 0.0,
                      side: str = 'BUY') -> Tuple[Optional[str], Optional[Decimal], str]:
        """tickSize 올림 + PRICE_FILTER + PERCENT_PRICE 검증."""
        f = self._f(coin)
        if not f:
            return None, None, f'{QUOTE_ASSET} 마켓 없음'
        tick = f.get('tick', Decimal('0'))
        p = _ceil_to_step(_dec(price), tick)
        max_p = f.get('max_price', Decimal('0'))
        min_p = f.get('min_price', Decimal('0'))
        if max_p > 0 and p > max_p:
            p = _floor_to_step(max_p, tick)
        if p <= 0:
            return None, None, '가격 0'
        if min_p > 0 and p < min_p:
            return None, None, f'가격 {p} < minPrice {min_p}'
        if not _is_step_multiple(p, tick):
            return None, None, f'가격 {p} 이 tickSize {tick} 배수가 아님'
        reason = self.check_percent_price(coin, p, ref_price or price, side)
        if reason:
            # advisory — 권위 판정은 거래소(-1013 명시 거절)에 맡기고 주문은 시도한다
            log(f'  ⚠ {coin} PERCENT_PRICE 경계 의심(advisory, 주문은 시도): {reason}')
        return _fmt_step(p, tick), p, ''

    def _validate_order(self, coin: str, q: Decimal, p: Decimal, market: bool,
                        step: Decimal, min_qty: Decimal, max_qty: Decimal) -> str:
        """캡 적용 후 최종 (qty, price, notional) 전체 재검증. 위반이면 사유."""
        f = self._f(coin)
        if q is None or q <= 0:
            return f'stepSize({step}) 절사 후 수량 0'
        if not _is_step_multiple(q, step):
            return f'수량 {q} 이 stepSize {step} 배수가 아님'
        if min_qty > 0 and q < min_qty:
            return f'수량 {q} < minQty {min_qty}'
        if max_qty > 0 and q > max_qty:
            return f'수량 {q} > maxQty {max_qty}'
        if p is None or p <= 0:
            return ''  # 가격 미상(시장가 강행) — notional 검증 불가
        notional = q * p
        apply_min = (not market) or f.get('apply_min_to_market', True)
        min_not = max(MIN_ORDER_USDT_D,
                      f.get('min_notional', Decimal('0')) if apply_min else Decimal('0'))
        if notional < min_not:
            return f'notional ${notional:.2f} < 최소주문 ${min_not:.2f}'
        max_not = f.get('max_notional', Decimal('0'))
        apply_max = (not market) or f.get('apply_max_to_market', False)
        if max_not > 0 and apply_max and notional > max_not:
            return f'notional ${notional:.2f} > maxNotional ${max_not:.2f}'
        return ''

    def prepare_qty(self, coin: str, qty: float, price: float,
                    market: bool) -> Tuple[Optional[str], Optional[Decimal], str]:
        """LOT_SIZE / MARKET_LOT_SIZE / NOTIONAL 전량 검증 후 전송 문자열 생성.

        Returns (qty_str, qty_dec, skip_reason). skip_reason 이 비면 성공.
        maxQty / maxNotional 초과분은 분할하지 않고 캡하며(잔여는 다음 사이클),
        캡 적용 후 전체 제약을 한 번 더 재검증한다.
        """
        f = self._f(coin)
        if not f:
            return None, None, f'{QUOTE_ASSET} 마켓 없음'
        step, min_qty, max_qty = self._qty_limits(coin, market)
        p = _dec(price)

        q = _floor_to_step(_dec(qty), step)
        if max_qty > 0 and q > max_qty:
            q = _floor_to_step(max_qty, step)
            log(f'  ⚠ {coin} 수량 maxQty({max_qty}) 캡 → {q} (잔여는 다음 사이클)')
        max_not = f.get('max_notional', Decimal('0'))
        apply_max = (not market) or f.get('apply_max_to_market', False)
        if p > 0 and max_not > 0 and apply_max and q * p > max_not:
            q = _floor_to_step(max_not / p, step)
            log(f'  ⚠ {coin} 수량 maxNotional(${max_not}) 캡 → {q} (잔여는 다음 사이클)')

        reason = self._validate_order(coin, q, p, market, step, min_qty, max_qty)
        if reason:
            return None, None, reason
        return _fmt_step(q, step), q, ''

    # ─── 조회 ───
    def _get_all_prices(self) -> Dict[str, float]:
        rows = self.client.get_all_tickers()
        out: Dict[str, float] = {}
        if isinstance(rows, list):
            for r in rows:
                try:
                    sym, px = r['symbol'], float(r['price'])
                except Exception:
                    continue
                if math.isfinite(px) and px > 0:
                    out[sym] = px
        return out

    def last_price(self, coin: str) -> float:
        """직전 get_balance 가 받아온 시세 (executable_min_notional 용)."""
        sym = self.symbol_of(coin)
        if not sym:
            return 0.0
        return float(self._last_price_map.get(sym, 0.0))

    def get_balance(self) -> Dict[str, float]:
        """{asset: USDT 평가액}. 현금은 'USDT' 키. 평가는 total(free+locked) 기준.

        평가 불가(마켓 없음/가격 없음) 자산이 하나라도 있으면 BalanceIncomplete 를 raise 한다.
        (PV 과소평가 → 대량 오매도 방지. 알려진 미평가 더스트는 state 의
         balance_ignore_assets 로 화이트리스트 가능.)
        """
        result: Dict[str, float] = {}
        try:
            acct = self.client.get_account(recvWindow=RECV_WINDOW_MS)
        except Exception as e:
            raise BalanceIncomplete(f'잔고 조회 실패: {_redact(e)}')
        balances = acct.get('balances') if isinstance(acct, dict) else None
        if not isinstance(balances, list):
            raise BalanceIncomplete(f'잔고 응답 이상: {type(acct).__name__}')

        # 행 파싱 엄격화 (M) — 조용한 제외 금지. 이상 행이 있으면 즉시 중단.
        coin_rows: List[Dict] = []
        for b in balances:
            if not isinstance(b, dict):
                raise BalanceIncomplete(f'잔고 행 형식 이상: {type(b).__name__}')
            currency = b.get('asset')
            if not isinstance(currency, str) or not currency.strip():
                raise BalanceIncomplete(f'잔고 행 asset 이상: {currency!r}')
            if 'free' not in b or 'locked' not in b:
                raise BalanceIncomplete(f'{currency} 잔고 필수 필드 누락 (free/locked)')
            free = _strict_num(b.get('free'), f'{currency}.free')
            locked = _strict_num(b.get('locked'), f'{currency}.locked')
            if free < 0 or locked < 0:
                raise BalanceIncomplete(f'{currency} 수량 음수 (free={free}, locked={locked})')
            qty = free + locked
            if currency == CASH_ASSET:
                result[CASH_ASSET] = qty
            elif qty > 0:
                coin_rows.append({'currency': currency, 'qty': qty})
        result.setdefault(CASH_ASSET, 0.0)

        # 시세 맵은 항상 갱신 (executable_min_notional 이 참조)
        try:
            price_map = self._get_all_prices()
        except Exception as e:
            raise BalanceIncomplete(f'가격 일괄조회 실패: {_redact(e)}')
        if not price_map:
            raise BalanceIncomplete('가격 일괄조회 결과 없음')
        self._last_price_map = price_map

        if coin_rows:
            unpriced: List[str] = []
            for row in coin_rows:
                asset = row['currency']
                symbol = self.symbol_of(asset)
                price_val = price_map.get(symbol, 0.0) if symbol else 0.0
                if price_val > 0:
                    result[asset] = row['qty'] * price_val
                elif asset.upper() in self.ignore_assets:
                    log(f'  평가 불가 {asset} (qty={row["qty"]:.8f}) — ignore 목록 → 제외')
                else:
                    unpriced.append(f'{asset}(qty={row["qty"]:.8f},'
                                    f'{"마켓없음" if not symbol else "가격없음"})')
            if unpriced:
                raise BalanceIncomplete('평가 불가 자산: ' + ', '.join(unpriced))
        return result

    def _asset_qty(self, coin: str, want: str) -> Optional[float]:
        """free / locked / total 조회. 불명이면 None (0 으로 강등 금지, M)."""
        try:
            b = self.client.get_asset_balance(asset=coin, recvWindow=RECV_WINDOW_MS)
        except Exception as e:
            log(f'  🚨 잔량 조회 실패 {coin}: {_redact(e)} → 불명 처리')
            return None
        if not isinstance(b, dict) or 'free' not in b or 'locked' not in b:
            log(f'  🚨 잔량 응답 필수 필드 누락 {coin} → 불명 처리')
            return None
        try:
            free = _strict_num(b.get('free'), f'{coin}.free')
            locked = _strict_num(b.get('locked'), f'{coin}.locked')
        except BalanceIncomplete as e:
            log(f'  🚨 잔량 파싱 실패 {coin}: {e} → 불명 처리')
            return None
        if free < 0 or locked < 0:
            log(f'  🚨 잔량 음수 {coin} → 불명 처리')
            return None
        return {'free': free, 'locked': locked, 'total': free + locked}[want]

    def get_total_qty(self, coin: str) -> Optional[float]:
        """코인 수량 (free + locked). 불명이면 None."""
        return self._asset_qty(coin, 'total')

    def get_locked_qty(self, coin: str) -> Optional[float]:
        return self._asset_qty(coin, 'locked')

    def get_free_qty(self, coin: str) -> Optional[float]:
        """주문 가능 수량 (free only) — 주문 산출용 (M1).

        조회 실패·필드 부재·None/빈문자열/bool 은 0 으로 강등하지 않고 **None**(불명).
        (0 과 '불명'을 섞으면 청산 미완을 완료로 오판할 수 있음 — M)
        """
        return self._asset_qty(coin, 'free')

    def get_current_price(self, coin: str) -> float:
        symbol = self.symbol_of(coin)
        if not symbol:
            return 0.0
        try:
            t = self.client.get_symbol_ticker(symbol=symbol)
            price = float(t.get('price', 0) or 0) if isinstance(t, dict) else 0.0
            return price if price > 0 else 0.0
        except Exception:
            return 0.0

    def cancel_all(self) -> Optional[set]:
        """USDT 마켓 미체결 전량 취소.

        Returns: 취소 후에도 미체결이 남은 coin(base) 집합.
                 미체결 조회 자체가 실패해 상태를 알 수 없으면 None (호출자가 fail-closed).
        """
        orders = self._fetch_open_orders('미체결 조회')
        if orders is None:
            return None
        targets = [o for o in orders if o.get('symbol') in self._filters]
        skipped = len(orders) - len(targets)
        for o in targets:
            try:
                self.client.cancel_order(symbol=o['symbol'], orderId=o['orderId'],
                                         recvWindow=RECV_WINDOW_MS)
            except Exception as e:
                log(f'  취소 오류 {o.get("symbol")}/{o.get("orderId")}: {_redact(e)}')
        if targets:
            log(f'  미체결 {len(targets)}건 취소 시도'
                + (f' (비 {QUOTE_ASSET} 마켓 {skipped}건 제외)' if skipped else ''))
        # 취소 결과 재확인 — 응답도 동일하게 엄격 검증 (M). 불명이면 None → 호출부 exit 2
        remain = self._fetch_open_orders('미체결 재확인')
        if remain is None:
            return None
        blocked = set()
        for o in remain:
            f = self._filters.get(o.get('symbol'))
            if f:
                blocked.add(f['base'])
        if blocked:
            log(f'  ⚠ 취소 실패 미체결 잔존 → 리밸런싱 제외: {sorted(blocked)}')
        return blocked

    def _fetch_open_orders(self, what: str) -> Optional[List[Dict]]:
        """open orders 조회 + 응답/필수 필드 엄격 검증. 불명이면 None."""
        try:
            rows = self.client.get_open_orders(recvWindow=RECV_WINDOW_MS)
        except Exception as e:
            log(f'  🚨 {what} 오류: {_redact(e)} → 미체결 상태 불명')
            return None
        if not isinstance(rows, list):
            log(f'  🚨 {what} 응답이 list 아님 ({type(rows).__name__}) → 미체결 상태 불명')
            return None
        out: List[Dict] = []
        for o in rows:
            if not isinstance(o, dict):
                log(f'  🚨 {what} 항목 형식 이상 ({type(o).__name__}) → 미체결 상태 불명')
                return None
            sym = o.get('symbol')
            if not isinstance(sym, str) or not sym or 'orderId' not in o:
                log(f'  🚨 {what} 항목 필수 필드 누락 (symbol/orderId) → 미체결 상태 불명')
                return None
            out.append(o)
        return out

    @staticmethod
    def _reject_is_transient(e: Optional[Exception]) -> bool:
        """'미접수 확정' 거절 중 잠시 후 재시도할 가치가 있는가."""
        code = getattr(e, 'code', None) if isinstance(e, BinanceAPIException) else None
        return isinstance(code, int) and code in TRANSIENT_REJECT_CODES

    # ─── 주문 WAL (append-only, 프로세스 재시작 후에도 reconcile 가능하게) ───
    @property
    def wal_path(self) -> str:
        return os.path.join(CACHE_DIR, ORDER_WAL_FILE)

    @staticmethod
    def _fsync_dir(path: str):
        """파일 fsync 만으로는 디렉터리 엔트리가 영속되지 않는다 (fsync(2))."""
        d = os.path.dirname(os.path.abspath(path)) or '.'
        fd = os.open(d, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _wal_append(self, rec: Dict, critical: bool) -> bool:
        """append-only 기록. dry-run 은 기록하지 않는다 (True 반환).

        intent 기록 실패 = 주문 중단(fail-closed). resolved 기록 실패는 False 를 반환하며,
        정책상 주문 자체는 이미 확정됐으므로 사이클을 중단하지 않고 강한 경고만 남긴다
        (다음 실행의 reconcile 이 해당 coid 를 다시 확인한다).
        """
        if self.dry_run:
            return True
        existed = os.path.exists(self.wal_path)
        try:
            with open(self.wal_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                f.flush()
                os.fsync(f.fileno())
            if not existed:
                self._fsync_dir(self.wal_path)   # 새 파일명 자체를 영속화
            return True
        except Exception as e:
            msg = f'주문 WAL 기록 실패 ({rec.get("event")}): {_redact(e)}'
            log(f'  🚨 {msg}')
            if critical:
                raise RuntimeError(f'{msg} — 주문 중단 (fail-closed)')
            return False

    def _wal_intent(self, coid: str, symbol: str, side: str, qty: str, otype: str):
        """POST 직전 시각(ts)과 recvWindow 를 함께 남긴다 (정보용)."""
        self._wal_append({'event': 'intent', 'ts': time.time(), 'coid': coid,
                          'symbol': symbol, 'side': side, 'qty': qty, 'type': otype,
                          'recv_window_ms': RECV_WINDOW_MS}, critical=True)

    def _wal_resolved(self, coid: str, status: str, executed=None, post_ts=None,
                      note: str = '') -> bool:
        """해소 기록. 실패 시 False (호출자가 경고/실패 처리)."""
        ok = self._wal_append({'event': 'resolved', 'ts': time.time(), 'coid': coid,
                               'status': status, 'executedQty': executed,
                               'post_ts': post_ts, 'note': note}, critical=False)
        if not ok:
            log(f'  🚨 WAL 해소 기록 실패 {coid} (status={status}) — 다음 실행에서 재확인 필요')
        return ok

    def _read_wal(self) -> Tuple[Optional[Dict[str, Dict]], Optional[set], str]:
        """WAL 파싱. 손상/스키마 불일치 행은 무시하지 않고 실패로 보고한다.

        이벤트별 필수 필드·타입·유한성까지 검증하며, resolved 는 선행 intent 가 있어야 한다
        (valid-JSON malformed 한 줄이 intent 를 해소하지 못하게).
        """
        intents: Dict[str, Dict] = {}
        resolved: set = set()
        try:
            with open(self.wal_path, 'r', encoding='utf-8') as f:
                for lineno, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        return None, None, f'WAL {lineno}행 JSON 손상'
                    if not isinstance(rec, dict):
                        return None, None, f'WAL {lineno}행 타입 이상'
                    bad = _validate_wal_record(rec, intents, resolved)
                    if bad:
                        return None, None, f'WAL {lineno}행 {bad}'
                    if rec['event'] == 'intent':
                        intents[rec['coid']] = rec
                    else:
                        resolved.add(rec['coid'])
        except Exception as e:
            return None, None, f'WAL 읽기 실패: {_redact(e)}'
        return intents, resolved, ''

    def _compact_wal(self):
        """모든 엔트리가 resolved 인 경우에만 비우기 (tmp+fsync+rename+dir fsync)."""
        tmp = self.wal_path + '.tmp'
        try:
            with open(tmp, 'w', encoding='utf-8') as f:
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.wal_path)
            self._fsync_dir(self.wal_path)
            log('  WAL compaction 완료 (미해결 없음)')
        except Exception as e:
            log(f'  ⚠ WAL compaction 실패(무해): {_redact(e)}')
            try:
                os.remove(tmp)
            except Exception:
                pass

    def mark_wal_resolved(self, coid: str) -> bool:
        """운영자 수동 해소 — 거래소 주문내역을 직접 확인한 뒤에만 사용."""
        intents, resolved, err = (None, None, '')
        if os.path.exists(self.wal_path):
            intents, resolved, err = self._read_wal()
        if err:
            log(f'  🚨 수동 해소 실패 — {err}')
            return False
        if not intents or coid not in intents:
            log(f'  🚨 수동 해소 실패 — WAL 에 intent 없음: {coid}')
            return False
        if resolved and coid in resolved:
            log(f'  ℹ 이미 해소된 coid: {coid}')
            return True
        rec = intents[coid]
        if not self._wal_resolved(coid, 'MANUAL_RESOLVED',
                                  note='operator verified on exchange'):
            log(f'  🚨 수동 해소 실패 — WAL 기록 실패: {coid}')
            return False
        log(f'  ✍ 수동 해소 기록: {coid} ({rec.get("symbol")} {rec.get("side")} {rec.get("qty")})')
        return True

    def reconcile_wal(self) -> Tuple[bool, List[str]]:
        """미해결 WAL 엔트리를 coid 권위 조회로만 해소. Returns (모두 해소?, 미해결 설명들).

        시간 경과로 미접수를 확정하지 않는다 (라운드3 C). 조회로 최종 확인이 안 되면
        WAL 을 그대로 두고 호출자가 fail-closed 한다.
        """
        path = self.wal_path
        if not os.path.exists(path):
            return True, []
        intents, resolved, err = self._read_wal()
        if err:
            log(f'  🚨 {err} → reconcile 실패 (fail-closed)')
            return False, [err]

        pending = [r for c, r in intents.items() if c not in resolved]
        if not pending:
            self._compact_wal()
            return True, []

        log(f'  🔎 미해결 주문 WAL {len(pending)}건 → clientOrderId reconcile')
        unresolved: List[str] = []
        for rec in pending:
            coid, symbol = rec['coid'], rec['symbol']
            ok, order = self._lookup_client_order(symbol, coid)
            if ok and order is not None:
                st = str(order.get('status', ''))
                ex = order.get('executedQty')
                log(f'  ↩ WAL reconcile {coid} ({symbol}): status={st} executed={ex}')
                if not self._wal_resolved(coid, st, ex):
                    _tg(f'🚨 {EXCHANGE_LABEL} WAL 해소 기록 실패 {coid} '
                        f'(거래소 상태={st}) — 디스크 확인 필요')
                continue
            why = '조회 실패' if not ok else '거래소에 주문 없음(-2013) — 접수 여부 미확정'
            desc = (f'{coid} ({symbol} {rec.get("side")} {rec.get("qty")}, '
                    f'ts={rec.get("ts")}) — {why}')
            log(f'  🚨 WAL 미해결: {desc}')
            unresolved.append(desc)
        return (not unresolved), unresolved

    # ─── 주문 멱등성 (C1) ───
    def _next_client_order_id(self, side: str, coin: str) -> str:
        """전역 유일 주문 ID. uuid4 난수 + sequence 를 앞쪽에 두어
        36자 예산 초과 시 coin 꼬리만 잘리게 한다 (난수/seq 는 절대 절단되지 않음)."""
        self._order_seq += 1
        rand = uuid.uuid4().hex[:COID_RANDOM_LEN]
        head = f'{CLIENT_ORDER_PREFIX}{rand}{side}{self._order_seq}-'
        return _sanitize_client_order_id(head + coin)

    def _lookup_client_order(self, symbol: str, client_order_id: str) -> Tuple[bool, Optional[dict]]:
        """clientOrderId 로 주문 조회.

        Returns (조회성공?, order|None). order=None 은 거래소가 '그런 주문 없음(-2013)'이라
        응답했다는 뜻일 뿐, 모호한 POST 직후에는 '미접수 확정'이 아니다 (호출자가 판단).
        조회성공=False 는 결과 미확정.
        """
        last_err = None
        for attempt in range(1, ORDER_LOOKUP_MAX_RETRIES + 1):
            try:
                o = self.client.get_order(symbol=symbol, origClientOrderId=client_order_id,
                                          recvWindow=RECV_WINDOW_MS)
                bad = _validate_order_resp(o, symbol, client_order_id)
                if not bad:
                    return True, o
                last_err = f'응답 검증 실패: {bad}'
            except BinanceAPIException as e:
                if getattr(e, 'code', None) == ORDER_NOT_EXIST_CODE:
                    return True, None
                last_err = e
            except Exception as e:
                last_err = e
            if attempt < ORDER_LOOKUP_MAX_RETRIES:
                time.sleep(ORDER_LOOKUP_RETRY_DELAY)
        log(f'  🚨 주문 조회 실패 {symbol}/{client_order_id}: {_redact(last_err)}')
        return False, None

    def _submit_order(self, coin: str, symbol: str, client_order_id: str,
                      submit_fn, desc: str) -> Tuple[str, Optional[dict], Optional[Exception]]:
        """주문 전송 (+WAL intent) 및 결과 확정.

        Returns (outcome, order, error):
          'placed'   — 접수 확인됨 (order 는 주문/조회 응답)
          'rejected' — 거래소가 **명시적으로 사전 거절** → 미접수 확정 (재시도 안전)
        그 외(타임아웃/5xx/연결오류/응답 이상)는 지연 재조회로도 확정하지 못하면
        재주문하지 않고 UnknownExecutionError raise → health lock.
        """
        err: Optional[Exception] = None
        self._last_post_ts = time.time()   # 실제 POST 직전 시각 (정보용)
        try:
            res = submit_fn(newClientOrderId=client_order_id)
            if isinstance(res, dict) and ('orderId' in res or 'clientOrderId' in res):
                return 'placed', res, None
            log(f'  ⚠ {desc} 응답 형식 이상 → 미확정 처리: {_redact(type(res).__name__)}')
        except Exception as e:
            err = e
            code = getattr(e, 'code', None) if isinstance(e, BinanceAPIException) else None
            if _is_definite_reject(code):
                log(f'  {desc} 사전 거절 (code={code}) — 미접수 확정: {_redact(e)}')
                return 'rejected', None, e
            log(f'  ⚠ {desc} 모호한 실패: {_redact(e)} → clientOrderId 지연 재조회')

        # 모호 — 주문이 아직 전파 중일 수 있으므로 -2013 도 '미접수'로 보지 않는다.
        for delay in AMBIGUOUS_LOOKUP_DELAYS:
            if delay:
                time.sleep(delay)
            ok, order = self._lookup_client_order(symbol, client_order_id)
            if ok and order is not None:
                log(f'  ↩ {desc} 재조회로 접수 확인 (status={order.get("status")})')
                return 'placed', order, err
        raise UnknownExecutionError(
            f'{coin} {desc} 결과 미확정 — 재주문 금지 '
            f'(clientOrderId={client_order_id}, err={_redact(err)})')

    def _poll_order_fill(self, symbol: str, client_order_id: str,
                         response: Optional[dict] = None) -> Tuple[float, str]:
        """접수된 주문의 체결량을 **최종 상태**로만 확정한다.

        조회 실패 / NOT_FOUND / 비최종 상태가 계속되면 잔량 재매도 금지 →
        UnknownExecutionError. POST 응답이 이미 최종 상태면 그 값은 신뢰한다.
        """
        last = ''
        for delay in FILL_POLL_DELAYS:
            if delay:
                time.sleep(delay)
            ok, order = self._lookup_client_order(symbol, client_order_id)
            if ok and order is not None:
                status = str(order.get('status', ''))
                if status in FINAL_STATUSES:
                    try:
                        return float(order.get('executedQty', 0) or 0), status
                    except (TypeError, ValueError):
                        last = 'executedQty 파싱 실패'
                        break
                last = f'비최종 상태 {status}'
                continue
            last = 'NOT_FOUND' if ok else '조회 실패'
        if isinstance(response, dict):
            bad = _validate_order_resp(response, symbol, client_order_id)
            if bad:
                log(f'  🚨 POST 응답 fallback 검증 실패 ({bad}) → 미확정 처리')
            else:
                st = str(response.get('status', ''))
                if st in FINAL_STATUSES:
                    log(f'  ⚠ 체결 조회 불가({last}) → 검증된 최종 상태 주문 응답 사용 (status={st})')
                    return float(Decimal(str(response['executedQty']))), st
                log(f'  🚨 POST 응답 fallback 비최종 상태({st}) → 미확정 처리')
        raise UnknownExecutionError(
            f'{symbol} 체결량 확정 불가 ({last}) — 잔량 재주문 금지 '
            f'(clientOrderId={client_order_id})')

    # ─── 주문 ───
    def sell_market_robust(self, coin: str, target_qty: float) -> Tuple[bool, float]:
        """V24 robust 매도. 재시도 + fill 검증 + 잔량 재주문. Returns (success, total_filled).

        재시도 수량은 매번 현재 free 잔량 기준으로 재산출한다 (M3).
        """
        if target_qty <= 0:
            return True, 0.0
        if self.dry_run:
            log(f'  [DRY] robust sell {coin} qty={target_qty:.8f}')
            return True, target_qty
        symbol = self.symbol_of(coin)
        if not symbol:
            log(f'  ⚠ 매도 불가 {coin}: {QUOTE_ASSET} 마켓 없음')
            return False, 0.0
        total_filled = 0.0
        t0 = time.time()
        last_err = None
        for attempt in range(1, SELL_MAX_ATTEMPTS + 1):
            if time.time() - t0 > SELL_TIMEOUT_SEC:
                log(f'  ⚠ {coin} 매도 타임아웃 ({SELL_TIMEOUT_SEC}s) — filled={total_filled:.8f}/{target_qty:.8f}')
                break
            free_qty = self.get_free_qty(coin)
            if free_qty is None:
                log(f'  🚨 매도 중단 {coin}: free 잔량 불명 → fail-closed')
                return False, total_filled
            try_qty = min(max(0.0, target_qty - total_filled), free_qty)
            if try_qty <= 0:
                log(f'  매도 잔량 없음 ({coin}) 종료 (filled={total_filled:.8f})')
                return True, total_filled
            price = self.get_current_price(coin)
            qty_str, qty_dec, reason = self.prepare_qty(coin, try_qty, price, market=True)
            if reason:
                log(f'  매도 잔량 종료 ({coin}) — {reason}')
                return True, total_filled
            coid = self._next_client_order_id('S', coin)
            self._wal_intent(coid, symbol, 'SELL', qty_str, 'MARKET')
            outcome, order, err = self._submit_order(
                coin, symbol, coid,
                lambda **kw: self.client.order_market_sell(
                    symbol=symbol, quantity=qty_str, recvWindow=RECV_WINDOW_MS, **kw),
                f'매도 attempt {attempt} {coin}')
            if outcome == 'rejected':
                last_err = err
                self._wal_resolved(coid, 'NOT_PLACED')
                retryable = self._reject_is_transient(err)
                log(f'  매도 attempt {attempt} 미접수 확정 {coin} (retryable={retryable})')
                if not retryable or attempt >= SELL_MAX_ATTEMPTS:
                    break
                time.sleep(SELL_RETRY_DELAYS[min(attempt - 1, len(SELL_RETRY_DELAYS) - 1)])
                continue
            time.sleep(FILL_CHECK_DELAY_SEC)
            executed, status = self._poll_order_fill(symbol, coid, response=order)
            self._wal_resolved(coid, status, executed, post_ts=self._last_post_ts)
            total_filled += executed
            log(f'  매도 attempt {attempt} {coin}: try={float(qty_dec):.8f} executed={executed:.8f} '
                f'status={status} cumul_filled={total_filled:.8f}/{target_qty:.8f}')
            if total_filled >= target_qty * 0.999:
                return True, total_filled
            if executed <= 0 and status in ('CANCELED', 'EXPIRED', 'REJECTED'):
                log(f'  매도 {coin}: 미체결 종료 상태({status}) → 중단')
                break
            time.sleep(1)
        ok = total_filled >= target_qty * 0.999
        log(f'  ⚠ 매도 robust 종료 {coin}: ok={ok} filled={total_filled:.8f}/{target_qty:.8f} '
            f'last_err={_redact(last_err)}')
        return ok, total_filled

    def buy_limit(self, coin: str, usdt_amount: float) -> Tuple[bool, str, float]:
        """지정가 매수 (현재가×1.003, tickSize 올림) 후 ORDER_WAIT_SEC 대기 → 미체결 취소.

        Returns (ok, note, sent_notional). sent_notional 은 실제 전송한 주문 금액
        (필터 캡 반영) — 호출자는 이 값만 현금에서 차감한다.
        취소 후 최종 상태를 확인하지 못하면 매도와 동일하게 UnknownExecutionError.
        """
        min_notional = self.min_notional(coin)
        if usdt_amount < min_notional:
            return True, '', 0.0
        if self.dry_run:
            log(f'  [DRY] 지정가 매수 {coin} ${usdt_amount:,.2f}')
            return True, '', usdt_amount
        symbol = self.symbol_of(coin)
        if not symbol:
            log(f'  매수 실패 {coin}: {QUOTE_ASSET} 마켓 없음')
            return False, '', 0.0
        price = self.get_current_price(coin)
        if price <= 0:
            log(f'  매수 실패 {coin}: 현재가 조회 실패')
            return False, '', 0.0
        price_str, price_dec, preason = self.prepare_price(
            coin, price * (1 + LIMIT_PRICE_SLIP), ref_price=price, side='BUY')
        if preason:
            log(f'  매수 중단 {coin}: {preason}')
            return False, '', 0.0
        qty_str, qty_dec, qreason = self.prepare_qty(
            coin, usdt_amount / float(price_dec), float(price_dec), market=False)
        if qreason:
            log(f'  매수 스킵 {coin}: {qreason}')
            return True, '', 0.0
        sent_notional = float(qty_dec * price_dec)
        coid = self._next_client_order_id('B', coin)
        self._wal_intent(coid, symbol, 'BUY', qty_str, 'LIMIT')
        outcome, order, err = self._submit_order(
            coin, symbol, coid,
            lambda **kw: self.client.order_limit_buy(
                symbol=symbol, quantity=qty_str, price=price_str, timeInForce='GTC',
                recvWindow=RECV_WINDOW_MS, **kw),
            f'매수 {coin}')
        if outcome == 'rejected':
            self._wal_resolved(coid, 'NOT_PLACED')
            log(f'  매수 미접수 확정 {coin}')
            return False, '', 0.0
        log(f'  매수 {coin} 요청 ${usdt_amount:,.2f} → 전송 ${sent_notional:,.2f} '
            f'@ {price_str} qty={qty_str}')
        time.sleep(ORDER_WAIT_SEC)
        try:
            self.client.cancel_order(symbol=symbol, origClientOrderId=coid,
                                     recvWindow=RECV_WINDOW_MS)
        except Exception as e:
            log(f'  매수 취소 시도 {coin}: {_redact(e)} (체결/이미취소 가능 — 상태 재확인)')
        # 취소 후 최종상태 미확인 → 매도와 동일하게 fail-closed (health lock + exit 3)
        executed, status = self._poll_order_fill(symbol, coid, response=None)
        self._wal_resolved(coid, status, executed, post_ts=self._last_post_ts)
        log(f'  매수 결과 {coin}: status={status} executed={executed:.8f}/{qty_str}')
        if status in OPEN_STATUSES:
            note = f'{coin} 매수 미체결 잔존 (status={status}) — 다음 사이클 취소 대상'
            log(f'  ⚠ {note}')
            return True, note, sent_notional
        return True, '', sent_notional


# ═══ 거래정지 청산 ═══
def detect_non_trading(held_coins: List[str], api: BinanceSpotAPI) -> List[str]:
    """보유 코인 중 Binance USDT 마켓 status != TRADING 목록.

    원본(Upbit)의 detect_warning_suspended(유의/상폐) 대체.
    target 에서의 제외는 엔진(유니버스 필터)이 이미 처리하므로 여기서는 청산 대상만 판정."""
    out: List[str] = []
    for coin in held_coins:
        status = api.symbol_status(coin)
        if status is None:
            continue
        if status != 'TRADING':
            out.append(coin)
    return out


def liquidation_state(api: BinanceSpotAPI, coin: str,
                      blocked_coins: set) -> Tuple[Optional[bool], str]:
    """청산 완료 여부 판정. (True=완료, False=미완, None=불명) + 사유.

    fresh total(free+locked) 기준이며, locked 가 실행 최소금액 이상이거나 해당 심볼에
    미체결이 남아 있으면(blocked) 완료로 보지 않는다. free==0 경로도 반드시 여기를 지난다.
    """
    total_remain = api.get_total_qty(coin)
    locked_remain = api.get_locked_qty(coin)
    if total_remain is None or locked_remain is None:
        return None, '잔량 불명 (청산 완료 판정 불가)'
    price = api.get_current_price(coin)
    exec_min = api.executable_min_notional(coin, price, market=True)
    total_notional = total_remain * price if price > 0 else float('inf')
    locked_notional = locked_remain * price if price > 0 else float('inf')
    if coin in blocked_coins:
        return False, '미체결 잔존(blocked) — 청산 완료 인정 불가'
    if locked_remain > 0 and locked_notional >= exec_min:
        return False, (f'locked 잔량 {locked_remain:.8f} '
                       f'(≈${locked_notional:,.2f} ≥ 실행최소 ${exec_min:,.2f}) 잔존')
    if total_remain <= 0 or total_notional < exec_min:
        return True, (f'total 잔량 {total_remain:.8f} '
                      f'(≈${0 if price <= 0 else total_notional:,.2f} < 실행최소 ${exec_min:,.2f})')
    return False, f'total 잔량 {total_remain:.8f} (≈${total_notional:,.2f}) 잔존'


def liquidate_coins(coins: List[str], reason: str, api: BinanceSpotAPI,
                    state: Dict, blocked_coins: Optional[set] = None) -> Tuple[List[str], List[str]]:
    """시장가 전량 매도 (robust 매도 + 체결 검증, 3회 재시도). 실패 시 permanent_block 등록.

    완료 판정은 fresh total(free+locked) 기준이며, locked 잔량이 실행 dust 를 넘거나
    해당 심볼에 미체결이 남아 있으면(blocked) 실패로 본다 (라운드3 M).
    Returns: (liquidated, failed)."""
    blocked_coins = blocked_coins or set()
    permanent = state.setdefault('permanent_block', [])
    liquidated: List[str] = []
    failed: List[str] = []
    for coin in coins:
        qty = api.get_free_qty(coin)
        if qty is None:
            failed.append(coin)
            log(f'  🚨 {reason} 청산 불가 {coin}: free 잔량 불명 → fail-closed')
            _tg(f'🚨 {reason} 청산 불가 {coin} (잔량 조회 실패) → 수동 확인 필요')
            continue
        if qty <= 0:
            # free 가 0 이어도 locked/blocked 를 반드시 확인한다 (라운드4 C)
            done, why = liquidation_state(api, coin, blocked_coins)
            if done is True:
                continue
            if coin not in permanent:
                permanent.append(coin)
            failed.append(coin)
            log(f'  🚨 {reason} 청산 미완 {coin} (free=0, {why}) → permanent_block 등록')
            _tg(f'🚨 {reason} 청산 미완 {coin} — {why} → 수동 확인 필요')
            continue
        success = False
        last_err: Optional[str] = None
        for attempt in range(1, LIQUIDATION_MAX_RETRIES + 1):
            cur_free = api.get_free_qty(coin)
            if cur_free is None:
                last_err = 'free 잔량 불명'
                break
            try:
                api.sell_market_robust(coin, cur_free)
            except UnknownExecutionError:
                raise
            except Exception as e:
                last_err = _redact(e)
            if api.dry_run:
                success = True
                break
            # 완료 판정: fresh total(free+locked) + locked/blocked (공통 루틴)
            done, why = liquidation_state(api, coin, blocked_coins)
            if done is True:
                success = True
                log(f'  청산 검증 {coin}: {why} → 완료')
                break
            last_err = why
            time.sleep(2 * attempt)
        if success:
            liquidated.append(coin)
            log(f'  🧹 {reason} 청산: {coin} qty={qty:.8f}')
            _tg(f'{reason} 청산: {coin}')
        else:
            if coin not in permanent:
                permanent.append(coin)
            failed.append(coin)
            log(f'  🚨 {reason} 청산 실패 {coin} (err={last_err}) → permanent_block 등록')
            _tg(f'🚨 {reason} 청산 실패 {coin} → 수동 확인 필요 (permanent_block)')
    return liquidated, failed


# ═══ 현금 키 정규화 (실자금 계층 방어) ═══
# 엔진 내부 규약은 'CASH', executor 규약은 'Cash'. 상류가 규약을 어겨도 실자금 계층에서
# 현금이 코인 티커로 새지 않도록 여기서 다시 병합한다 (2026-08-20 헛주문 사고).
def _is_cash_key(k) -> bool:
    return str(k).lower() == 'cash'


def _norm_cash_map(target: Dict[str, float]) -> Dict[str, float]:
    """CASH/Cash/cash 를 'Cash' 하나로 병합. 메타키('_ts' 등)와 비수치 값은 제외."""
    out: Dict[str, float] = {}
    for k, v in (target or {}).items():
        if str(k).startswith('_'):
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        kk = 'Cash' if _is_cash_key(k) else k
        out[kk] = out.get(kk, 0.0) + fv
    return out


def targets_equal(a: Dict[str, float], b: Dict[str, float], tol: float = 0.005) -> bool:
    """두 target 이 경제적으로 같은가. 현금 키 표기·메타키 차이는 무시한다."""
    a, b = _norm_cash_map(a), _norm_cash_map(b)
    if not a or not b:
        return False
    for k in set(a) | set(b):
        if abs(a.get(k, 0.0) - b.get(k, 0.0)) > tol:
            return False
    return True


# ═══ Cash Buffer / Notional Cap ═══
def apply_cash_buffer(target: Dict[str, float], buffer_pct: float) -> Dict[str, float]:
    """최종 target × (1-buffer) 후 Cash += buffer."""
    target = _norm_cash_map(target)
    if buffer_pct <= 0:
        return dict(target)
    out: Dict[str, float] = {}
    for k, v in target.items():
        if k == 'Cash':
            continue
        out[k] = v * (1 - buffer_pct)
    out['Cash'] = target.get('Cash', 0.0) * (1 - buffer_pct) + buffer_pct
    return out


def apply_notional_cap(target: Dict[str, float], balance: Dict[str, float],
                       total_usdt: float, cap_fraction: float) -> Tuple[Dict[str, float], float]:
    """이번 실행 Σ|delta| ≤ cap_fraction으로 제한. 잔여는 다음 실행에서 자연 재계산(carryover 저장 없음)."""
    target = _norm_cash_map(target)
    if total_usdt <= 0 or cap_fraction <= 0 or cap_fraction >= 1:
        return dict(target), 0.0

    current_w: Dict[str, float] = {}
    for k, v in balance.items():
        if k == CASH_ASSET:
            continue
        current_w[k] = v / total_usdt
    current_w['Cash'] = balance.get(CASH_ASSET, 0.0) / total_usdt

    all_keys = (set(target.keys()) | set(current_w.keys())) - {CASH_ASSET}
    deltas = {k: target.get(k, 0.0) - current_w.get(k, 0.0) for k in all_keys}
    gross_delta = sum(abs(v) for v in deltas.values())

    if gross_delta <= cap_fraction + 1e-9:
        return dict(target), gross_delta

    shrink = cap_fraction / gross_delta
    scaled: Dict[str, float] = {}
    for k in all_keys:
        cw = current_w.get(k, 0.0)
        dw = deltas.get(k, 0.0)
        new_w = cw + dw * shrink
        if new_w > 1e-9 or k == 'Cash':
            scaled[k] = max(new_w, 0.0)
    s = sum(scaled.values())
    if s > 0:
        scaled = {k: v / s for k, v in scaled.items()}
    return scaled, cap_fraction


# ═══ Delta 매매 ═══
def execute_delta(target: Dict[str, float], api: BinanceSpotAPI,
                   permanent_block: List[str], dry_run: bool,
                   balance: Dict[str, float],
                   effective_pv_usdt: float = None,
                   blocked_coins: Optional[set] = None) -> set:
    """target vs 현재 잔고 비교 → 매도 먼저, 매수 나중.

    - balance: run_once 가 만든 단일 스냅샷 (C2 — 여기서 재조회하지 않는다)
    - dust (<10 USDT) 잔여 → 비율 매도 대신 전량 매도
    - permanent_block 코인은 신규 매수 금지
    - blocked_coins: 미체결 잔존 심볼 → 이번 사이클 주문 제외 (M1)
    - effective_pv_usdt: alloc_transit cap (옵션 D). None 이면 actual total 사용.

    Returns: 실행 불가 dust 로 스킵된 코인 집합 (rebalancing_needed 영구 True 방지용).
    """
    blocked_coins = blocked_coins or set()
    unexecutable: set = set()
    total = sum(balance.values())
    if total <= 0:
        log('  잔고 없음')
        return unexecutable

    # alloc_transit cap 적용
    pv_for_target = total if effective_pv_usdt is None else min(total, effective_pv_usdt)
    if effective_pv_usdt is not None and effective_pv_usdt < total:
        log(f'  🔴 execute_delta alloc_transit cap: ${total:,.2f} → ${pv_for_target:,.2f}')

    current_value: Dict[str, float] = {k: v for k, v in balance.items() if k != CASH_ASSET}

    # 주문 후보 하드 가드 — 어떤 표기의 현금 키도 티커가 되지 않는다.
    _odd_cash = [k for k in (target or {}) if _is_cash_key(k) and k != 'Cash']
    if _odd_cash:
        log(f'  ⚠ 비정규 현금 키 {_odd_cash} 감지 → Cash 로 병합 (주문 후보 제외)')
    target = _norm_cash_map(target)

    sells: List[Tuple[str, float, bool]] = []  # (coin, sell_usdt, sell_all)
    buys: List[Tuple[str, float]] = []

    all_tickers = set(current_value.keys()) | set(target.keys())
    for ticker in all_tickers:
        if _is_cash_key(ticker):
            continue
        if ticker in blocked_coins:
            log(f'  ⚠ 미체결 잔존 {ticker} → 이번 사이클 주문 제외')
            continue
        tgt_w = target.get(ticker, 0.0)
        cur_v = current_value.get(ticker, 0.0)
        tgt_v = tgt_w * pv_for_target
        delta_v = tgt_v - cur_v
        # 격자 최소 주문금액 — 매도는 시장가, 매수는 예상 지정가 기준 (비교는 전부 >=)
        px = api.last_price(ticker)
        sell_min = api.executable_min_notional(ticker, px, market=True)
        buy_min = api.buy_min_notional(ticker, px)

        if tgt_w <= 0 and cur_v > 0:
            if cur_v < sell_min:
                log(f'  ℹ 불가피 dust {ticker}: ${cur_v:,.2f} < 실행최소 ${sell_min:,.2f} → 주문 불가')
                unexecutable.add(ticker)
                continue
            sells.append((ticker, cur_v, True))
        elif -delta_v >= sell_min:
            remainder = cur_v - abs(delta_v)
            if remainder < max(DUST_USDT, sell_min):
                sells.append((ticker, cur_v, True))
            else:
                sells.append((ticker, abs(delta_v), False))
        elif delta_v >= buy_min:
            if ticker in permanent_block:
                log(f'  ⚠ permanent_block {ticker} 매수 스킵')
                continue
            buys.append((ticker, delta_v))
        elif delta_v > 0 and tgt_v > 0 and delta_v >= MIN_ORDER_USDT:
            # 목표는 있는데 격자 최소 단위에 못 미쳐 영원히 채울 수 없는 경우
            log(f'  ℹ 불가피 dust {ticker}: 매수 delta ${delta_v:,.2f} < 실행최소 ${buy_min:,.2f}')
            unexecutable.add(ticker)

    # 매도 — V24 robust retry + fill 검증 (sell_market_robust). 실패 시 즉시 텔레그램 alert.
    sell_failures: List[str] = []
    for coin, sell_usdt, sell_all in sells:
        if dry_run:
            qty_free = 0.0
        else:
            qty_free = api.get_free_qty(coin)
            if qty_free is None:
                log(f'  🚨 매도 스킵 {coin}: free 잔량 불명 → fail-closed')
                sell_failures.append(f'{coin}: free 잔량 조회 실패로 매도 미시도')
                continue
            if qty_free <= 0:
                log(f'  매도 스킵 {coin}: free 잔량 0 (locked 상태)')
                continue
        price = api.get_current_price(coin)

        if sell_all:
            if dry_run:
                qty_free = sell_usdt / price if price > 0 else 0.0
            sell_qty = qty_free
            est_usdt = sell_qty * price if price > 0 else sell_usdt
            if price <= 0:
                log(f'  ⚠ 전량매도 {coin}: 현재가 0 → qty 기반 시장가 강행')
        else:
            if price <= 0:
                log(f'  부분매도 스킵 {coin}: 현재가 0')
                continue
            if dry_run:
                qty_free = sell_usdt / price
            sell_qty = min(qty_free, sell_usdt / price)
            est_usdt = sell_qty * price

        if sell_qty <= 0:
            continue
        _min_notional = api.executable_min_notional(coin, price, market=True)
        if est_usdt < _min_notional:
            log(f'  매도 스킵 {coin}: est_usdt=${est_usdt:,.2f} < 실행최소 ${_min_notional:,.2f} (불가피 dust)')
            unexecutable.add(coin)
            continue
        log(f'  매도 시작 {coin} qty={sell_qty:.8f} ≈ ${est_usdt:,.2f} ({"전량" if sell_all else "부분"})')
        ok, filled = api.sell_market_robust(coin, sell_qty)
        if not ok:
            short_qty = max(0.0, sell_qty - filled)
            short_usdt = short_qty * price if price > 0 else 0
            sell_failures.append(f'{coin}: 요청 {sell_qty:.6f} / 체결 {filled:.6f} (잔량 ${short_usdt:,.2f})')
            log(f'  ⚠ 매도 미완 {coin}: filled {filled:.8f}/{sell_qty:.8f}')
    if sell_failures and not dry_run:
        alert = f'[{EXCHANGE_LABEL}] 🚨 V24 spot 매도 실패/부분체결 (다음 cron 재시도)\n' + '\n'.join(f'  - {m}' for m in sell_failures)
        try:
            _send_tg(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, alert)
        except Exception as e:
            log(f'  매도 실패 alert 전송 오류: {_redact(e)}')

    if buys:
        time.sleep(1)
        # 매도 직후 실제 주문 가능 현금만 사용 (free only, M1). PV/target 은 재계산하지 않는다.
        if dry_run:
            cash_free = balance.get(CASH_ASSET, 0.0)
        else:
            cash_free = api.get_free_qty(CASH_ASSET)
            if cash_free is None:
                log('  🚨 매수 중단: 주문 가능 USDT 불명 → fail-closed')
                _tg('🚨 매수 중단 — USDT free 잔량 조회 실패')
                return unexecutable
        cash_avail = cash_free * 0.995
        total_buy = sum(amt for _, amt in buys)
        scale = min(1.0, cash_avail / max(total_buy, 1.0))
        buy_notes: List[str] = []
        for coin, amt in buys:
            actual = amt * scale
            exec_min = api.buy_min_notional(coin, api.last_price(coin))
            if actual < exec_min:
                log(f'  매수 스킵 {coin}: ${actual:,.2f} < 실행최소 ${exec_min:,.2f} (불가피 dust)')
                unexecutable.add(coin)
                continue
            log(f'  매수 {coin} ${actual:,.2f}')
            ok, note, sent = api.buy_limit(coin, actual)
            if note:
                buy_notes.append(note)
            # 실제 전송한 금액만 차감 (필터 캡으로 줄어든 분은 다음 매수에 쓸 수 있게)
            cash_avail -= sent
        if buy_notes and not dry_run:
            alert = f'[{EXCHANGE_LABEL}] ⚠ 매수 확인 필요\n' + '\n'.join(f'  - {m}' for m in buy_notes)
            try:
                _send_tg(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, alert)
            except Exception as e:
                log(f'  매수 alert 전송 오류: {_redact(e)}')
    return unexecutable


# ═══ 사전 알림 ═══
import v24_report as v24r  # noqa: E402

USDT_PER_COIN_DUST = 5.0  # 5 USDT 미만 보유는 표시 생략


def _build_coin_holdings(balance: Dict[str, float], total_usdt: float,
                          api_get_price=None) -> list:
    """balance(USDT 환산) → 보유 list[{ticker, value_str, weight}]."""
    if total_usdt <= 0:
        return []
    out = []
    for k, v_usdt in sorted(balance.items(), key=lambda kv: -kv[1]):
        if v_usdt < USDT_PER_COIN_DUST:
            continue
        ticker = 'Cash' if k == CASH_ASSET else k
        out.append({'ticker': ticker, 'value_str': f'${v_usdt:,.2f}',
                    'weight': v_usdt / total_usdt})
    return out


def _build_coin_canary_lines(result) -> list:
    lines = []
    for mname, info in (result.canary_info or {}).items():
        if not info:
            continue
        on = info.get('on', False)
        ratio = info.get('ratio', 0.0)
        cur = info.get('cur', 0.0)
        sma = info.get('sma_val', 0.0)
        sma_p = info.get('sma_p', 0)
        flipped = result.canary_flipped.get(mname, False)
        flip_mark = ' *FLIP*' if flipped else ''
        lines.append(f"{mname}: {'ON 🟢' if on else 'OFF 🔴'} "
                     f"BTC ${cur:,.0f} vs SMA{sma_p} ${sma:,.0f} ratio={ratio:.4f}{flip_mark}")
    return lines


def build_coin_report(result, balance: Dict[str, float], total_usdt: float,
                      orders_text: str, status_extra: Optional[Dict[str, str]] = None) -> str:
    from datetime import timedelta
    KST = timezone(timedelta(hours=9))
    ts = datetime.now(KST).strftime('%Y-%m-%d %H:%M KST')
    target = {('Cash' if k == 'CASH' else k): v
              for k, v in result.combined_target.items() if not k.startswith('_')}
    canary_lines = _build_coin_canary_lines(result)
    holdings = _build_coin_holdings(balance, total_usdt)
    visible_positions = sum(1 for h in holdings if h['ticker'] != 'Cash')
    status = {
        'schema': 'V24',
        '평가액': f'${total_usdt:,.2f}',
        'ht': f'{result.drift_half_turnover:.4f}',
        'drift_threshold': f'{result.drift_threshold:.2f}',
        'drift_fire': '예 🔔' if result.drift_fire else '아니오',
        '리밸 대기': '아니오',
        '포지션 수': str(visible_positions),
    }
    if status_extra:
        status.update(status_extra)
    return v24r.build_report(
        asset_label='', emoji='🪙', name='Cap Defend Spot (Binance)',
        ts_str=ts, target=target,
        holdings=holdings,
        orders_text=orders_text, canary_lines=canary_lines, status=status)


def format_target_summary(combined: Dict[str, float],
                           member_targets: Dict[str, Dict[str, float]]) -> str:
    """레거시 헬퍼 — 일부 사전 알림 경로에서 호출. unified report 미사용."""
    lines = ['목표 (앙상블):']
    for k, v in sorted(combined.items(), key=lambda kv: -kv[1]):
        if k.startswith('_'):
            continue
        if v < 1e-4 and k != 'Cash':
            continue
        lines.append(f'  {k}: {v*100:.2f}%')
    for mname, mt in member_targets.items():
        tokens = [f'{k}={v*100:.1f}%' for k, v in sorted(mt.items(), key=lambda kv: -kv[1])
                  if v > 1e-4 and not k.startswith('_')]
        lines.append(f'  [{mname}] ' + ', '.join(tokens[:6]))
    return '\n'.join(lines)


def load_state_strict(state_path: str) -> Tuple[Optional[dict], str]:
    """state 로더 (fail-closed). 파일 부재만 초기 state 로 허용한다.

    common/io.load_json 은 JSON 손상도 {} 로 삼켜 permanent_block / rebalancing 상태를
    잃은 채 신규 주문이 가능해지므로, 이 이식판은 자체 strict 로더를 쓴다
    (원본 parity 대신 안전 우선 — 이 state 파일은 본 판본 전용이라 parity 부담 없음).
    """
    if not os.path.exists(state_path):
        return {}, ''
    try:
        with open(state_path, 'r', encoding='utf-8') as f:
            obj = json.load(f)
    except json.JSONDecodeError as e:
        return None, f'state JSON 손상: {e}'
    except Exception as e:
        return None, f'state 읽기 실패: {_redact(e)}'
    if not isinstance(obj, dict):
        return None, f'state 최상위 타입 이상: {type(obj).__name__}'
    return obj, ''


# 참조로 지정하면 안 되는 경로 — 이 실행기(또는 엔진)가 쓰기 대상으로 삼는 파일들
STATE_REF_DENY_FILES = ('universe_cg_cache.json', 'binance_exchinfo_cache.json',
                        'binance_exchange_info_cache.json', 'binance_universe_cache.json')


def _state_ref_path_error(ref_path: str, state_path: str) -> str:
    """참조 경로가 이 실행기의 쓰기 대상과 겹치면 사유, 아니면 ''.

    읽기 전용 참조라 해도 쓰기 파일을 가리키면 (실수든 오설정이든) 참조와 저장이 같은
    파일을 물어 상태가 서로를 덮어쓸 수 있으므로 아예 막는다. 심볼릭 링크는 realpath,
    하드링크는 samefile 로 본다.
    """
    base = os.path.basename(ref_path)
    if base.endswith('.tmp'):
        return '임시 파일 경로는 참조 불가'
    if not base.endswith('.json'):
        return '참조는 .json state 파일만 허용'
    rp = os.path.realpath(ref_path)
    deny = [state_path, state_path + '.tmp', LOG_PATH, LOCK_FILE]
    deny += [os.path.join(CACHE_DIR, n) for n in STATE_REF_DENY_FILES]
    try:   # HealthGuard 생성자는 읽기만 한다. 실패해도 preflight 자체는 계속.
        _hg = HealthGuard(name='coin_binance')
        deny += [_hg.health_file, _hg.health_file + '.tmp', _hg.lock_file, _hg.abort_log]
    except Exception:
        pass
    for d in deny:
        if rp == os.path.realpath(d):
            return f'쓰기 대상 경로와 충돌: {base}'
        try:
            if os.path.exists(d) and os.path.samefile(ref_path, d):
                return f'쓰기 대상 경로와 충돌: {base}'
        except OSError:
            pass
    return ''


STATE_REF_SUM_TOL = 0.01   # 참조 가중치 맵 합 허용 오차 (1.0 ± tol)
# 가중치 키로 허용하는 형태 — 대문자 티커. 그 외(메타키/소문자/빈 문자열)는 거부한다
# (현금 키 'Cash'/'cash' 는 _is_cash_key 로 따로 허용).
_STATE_REF_KEY_RE = re.compile(r'^[A-Z0-9]{1,20}$')
# 참조에서 빌려오는 cash buffer 정책 키 (업비트와 같은 buffer 로 맞춰야 target·drift 가 일치)
STATE_REF_BUFFER_KEYS = ('spot_cash_buffer', 'cash_buffer', 'buffer_pct')


def _state_ref_weight_error(w: Dict, label: str, allow_ts: bool = False) -> str:
    """참조 가중치 맵 검증.

    - 멤버 맵(snapshots[i] / last_combined)은 엔진이 '모든 키'에 산술을 하므로 '_ts' 같은
      메타키조차 허용하지 않는다 (allow_ts=False). 통과시키면 엔진 안에서 터진다 —
      즉 fail-closed 지점을 이미 지난 뒤라 막을 수 없다.
    - 상위 target 맵(last_target_snapshot / last_member_targets[m])만 '_ts' 를 선택 허용하되
      파싱 가능한 문자열이어야 한다.
    - 값은 bool 아닌 유한 실수 ≥ 0, 합은 1.0 ± STATE_REF_SUM_TOL (빈 맵·전부 0·합 2.0 거부).
      같은 봉 경로가 last_target_snapshot 을 정규화 없이 그대로 쓰므로 합까지 본다.
    """
    total = 0.0
    n_w = 0
    for k, v in w.items():
        if not isinstance(k, str):
            return f'참조 {label} 키 타입 이상: {k!r}'
        if k == '_ts':
            if not allow_ts:
                return f'참조 {label} 에 메타키 _ts 포함 (엔진이 가중치로 계산함)'
            if not isinstance(v, str) or cle.parse_utc_iso(v) is None:
                return f'참조 {label}._ts 이상: {v!r}'
            continue
        if not (_is_cash_key(k) or _STATE_REF_KEY_RE.match(k)):
            return f'참조 {label} 키 형식 이상: {k!r}'
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return f'참조 {label}[{k}] 값 타입 이상: {v!r}'
        try:
            fv = float(v)
            ok = math.isfinite(fv)
        except (OverflowError, TypeError, ValueError):
            return f'참조 {label}[{k}] 값 변환 실패: {v!r}'
        if not ok or fv < 0:
            return f'참조 {label}[{k}] 값 이상: {v!r}'
        total += fv
        n_w += 1
    if n_w == 0 or abs(total - 1.0) > STATE_REF_SUM_TOL:
        return f'참조 {label} 가중치 합 이상: {total:.6f} (기대 1.0 ± {STATE_REF_SUM_TOL})'
    return ''


def load_state_ref(path: str) -> Tuple[Optional[dict], str]:
    """다른 실행기(업비트 LIVE)의 V24 state 를 **읽기 전용** 참조로 로드한다.

    Returns: (엔진 상태 사본, '') 또는 (None, 실패사유). 실패는 첫 문제에서 즉시 반환하고
    호출자가 fail-closed 한다 (fresh 초기화로 대체하면 신호가 갈리므로).
    참조 파일은 열기만 하며 어떤 경우에도 쓰지 않는다. 반환값은 deep copy 라
    호출자가 mutate 해도 원본 dict/파일에 영향이 없다.
    """
    if not os.path.exists(path):
        return None, f'참조 파일 없음: {os.path.basename(path)}'
    try:
        with open(path, 'r', encoding='utf-8') as f:
            obj = json.load(f)
    except json.JSONDecodeError as e:
        return None, f'참조 state JSON 손상: {e}'
    except Exception as e:
        return None, f'참조 state 읽기 실패: {_redact(e)}'

    # 검증 전체를 감싼다 — 참조 파일은 이 프로세스가 만들지 않은 입력이라
    # 예상 못 한 형태로도 올 수 있다. 어떤 예외든 fresh 초기화가 아니라 fail-closed 로.
    try:
        if not isinstance(obj, dict):
            return None, f'참조 state 최상위 타입 이상: {type(obj).__name__}'

        sv = obj.get('schema_version')
        if sv != cle.SCHEMA_VERSION:
            return None, f'참조 schema_version 불일치: {sv!r} != {cle.SCHEMA_VERSION}'

        members = obj.get('members')
        if not isinstance(members, dict):
            return None, f'참조 members 타입 이상: {type(members).__name__}'
        unknown_m = sorted(set(members) - set(cle.MEMBERS))
        if unknown_m:
            return None, f'참조 members 에 미지 멤버: {unknown_m}'
        for mname, mcfg in cle.MEMBERS.items():
            ms = members.get(mname)
            if not isinstance(ms, dict):
                return None, f'참조 멤버 없음/타입 이상: {mname}'
            n_snap = mcfg['n_snapshots']
            snaps = ms.get('snapshots')
            if not isinstance(snaps, list) or len(snaps) != n_snap:
                got = len(snaps) if isinstance(snaps, list) else type(snaps).__name__
                return None, f'참조 {mname} snapshots 이상: {got} != {n_snap}'
            for si, snap_i in enumerate(snaps):
                if not isinstance(snap_i, dict):
                    return None, f'참조 {mname} snapshot 항목 타입 이상'
                werr = _state_ref_weight_error(snap_i, f'{mname} snapshots[{si}]')
                if werr:
                    return None, werr
            bc = ms.get('bar_counter')
            if isinstance(bc, bool) or not isinstance(bc, int) or bc < 0:
                return None, f'참조 {mname} bar_counter 이상: {bc!r}'
            lbt = ms.get('last_bar_ts')
            if not isinstance(lbt, str) or not lbt or cle.parse_utc_iso(lbt) is None:
                return None, f'참조 {mname} last_bar_ts 이상: {lbt!r}'
            if not isinstance(ms.get('canary_on'), bool):
                return None, f'참조 {mname} canary_on 타입 이상: {type(ms.get("canary_on")).__name__}'
            if not isinstance(ms.get('last_combined'), dict):
                return None, f'참조 {mname} last_combined 타입 이상: {type(ms.get("last_combined")).__name__}'
            werr = _state_ref_weight_error(ms['last_combined'], f'{mname} last_combined')
            if werr:
                return None, werr
            if 'snap_id' in ms:
                sid = ms['snap_id']
                if isinstance(sid, bool) or not isinstance(sid, int) or sid < 0:
                    return None, f'참조 {mname} snap_id 이상: {sid!r}'

        snap = obj.get('last_target_snapshot')
        if not isinstance(snap, dict):
            return None, f'참조 last_target_snapshot 타입 이상: {type(snap).__name__}'
        if not {k: v for k, v in snap.items() if k != '_ts'}:
            return None, '참조 last_target_snapshot 비어있음'
        werr = _state_ref_weight_error(snap, 'last_target_snapshot', allow_ts=True)
        if werr:
            return None, werr

        if 'last_member_targets' in obj:
            lmt = obj['last_member_targets']
            if not isinstance(lmt, dict):
                return None, f'참조 last_member_targets 타입 이상: {type(lmt).__name__}'
            unknown_mt = sorted(set(lmt) - set(cle.MEMBERS))
            if unknown_mt:
                return None, f'참조 last_member_targets 에 미지 멤버: {unknown_mt}'
            for mn, mt in lmt.items():
                if not isinstance(mt, dict):
                    return None, f'참조 last_member_targets[{mn}] 타입 이상: {type(mt).__name__}'
                werr = _state_ref_weight_error(mt, f'last_member_targets[{mn}]', allow_ts=True)
                if werr:
                    return None, werr

        for bk in STATE_REF_BUFFER_KEYS:
            if bk not in obj:
                continue
            bv = obj[bk]
            if isinstance(bv, bool) or not isinstance(bv, (int, float)):
                return None, f'참조 {bk} 타입 이상: {bv!r}'
            try:
                fbv = float(bv)
                ok_b = math.isfinite(fbv)
            except (OverflowError, TypeError, ValueError):
                return None, f'참조 {bk} 값 변환 실패: {bv!r}'
            if not ok_b or not (0.0 <= fbv < 0.5):
                return None, f'참조 {bk} 범위 이상: {bv!r} (0 ≤ v < 0.5)'

        out = {
            'members': copy.deepcopy(members),
            'last_target_snapshot': copy.deepcopy(snap),
            'schema_version': sv,
        }
        if 'last_member_targets' in obj:
            out['last_member_targets'] = copy.deepcopy(obj['last_member_targets'])
        for bk in STATE_REF_BUFFER_KEYS:
            if bk in obj:
                out[bk] = obj[bk]
        return out, ''
    except Exception as e:
        return None, f'참조 검증 중 예외: {_redact(e)}'


def _apply_state_ref(state: dict, ref: dict) -> None:
    """참조본의 엔진 상태 키만 현재 state 로 옮긴다 (참조 파일은 건드리지 않는다)."""
    for k in (('members', 'last_target_snapshot', 'schema_version', 'last_member_targets')
              + STATE_REF_BUFFER_KEYS):
        if k in ref:
            # 엔진이 state 를 mutate 해도 참조 사본이 따라 변하지 않도록 값마다 deep copy
            # (같은 봉 경로는 엔진 실행 뒤에 ref[...] 를 다시 읽는다)
            state[k] = copy.deepcopy(ref[k])


def _state_ref_summary(ref: dict) -> str:
    """참조 요약 (로그 1줄용) — members/bar_counter/last_bar/snapshots."""
    ms = ref.get('members') or {}
    names = sorted(ms)

    def _f(key):
        return ','.join(str((ms.get(n) or {}).get(key)) for n in names)

    snaps = ','.join(str(len((ms.get(n) or {}).get('snapshots') or [])) for n in names)
    return (f'schema={ref.get("schema_version")}, members={names}, '
            f'bar_counter={_f("bar_counter")}, last_bar={_f("last_bar_ts")}, snapshots={snaps}')


def _save_state_unless_dry(state_path: str, state: dict, dry_run: bool) -> None:
    """dry-run이면 state 저장을 건너뛰어 실거래 트리거가 오염되지 않게 한다."""
    if dry_run:
        log('  (dry-run) state 저장 생략')
        return
    save_json(state_path, state)




def coin_needs_rebalance(target: Dict[str, float], balance: Dict[str, float],
                          total: float, delta_pct_tol: float = 0.01,
                          min_fn=None, skip_coins: Optional[set] = None,
                          buy_min_fn=None) -> bool:
    """현재 잔고와 목표 사이 편차가 체결 가능한 크기로 남아있으면 True.

    선물 auto_trade_binance.needs_rebalance 와 동일 역할:
      - 체결액이 심볼 최소주문 미만이면 무시 (거래소 최소주문 미만은 의미 없음)
      - 그 외에는 현재 notional 대비 편차가 tol(1%) 넘으면 True
      - 목표에 있는데 보유 없으면 True
    min_fn: coin -> 매도(시장가) 실행 가능 최소주문. None 이면 전역 MIN_ORDER_USDT.
    buy_min_fn: coin -> 매수(지정가) 실행 가능 최소주문. None 이면 min_fn 과 동일.
                (후보 생성과 동일하게 delta 방향별 기준을 쓴다 — 라운드4 m)
    skip_coins: 실행 불가 dust 로 확인된 코인 (영구 True 루프 방지).
    """
    if total <= 0:
        return False
    _min = min_fn or (lambda _c: MIN_ORDER_USDT)
    _buy_min = buy_min_fn or _min
    skip_coins = skip_coins or set()
    target = _norm_cash_map(target)
    current_v = {k: v for k, v in balance.items() if k != CASH_ASSET}
    keys = set(current_v.keys()) | set(target.keys())
    for k in keys:
        if _is_cash_key(k) or k in skip_coins:
            continue
        tgt_v = target.get(k, 0.0) * total
        cur_v = current_v.get(k, 0.0)
        diff = tgt_v - cur_v
        min_k = _buy_min(k) if diff > 0 else _min(k)
        if abs(diff) < min_k:
            continue
        if cur_v <= 0:
            if tgt_v >= min_k:
                return True
            continue
        if abs(diff) / max(cur_v, 1.0) > delta_pct_tol:
            return True
    return False


def format_delta_preview(target: Dict[str, float], balance: Dict[str, float],
                          total: float) -> str:
    if total <= 0:
        return '잔고 없음'
    lines = ['예상 Delta:']
    current_v = {k: v for k, v in balance.items() if k != CASH_ASSET}
    all_keys = set(current_v.keys()) | set(target.keys())
    rows = []
    for k in all_keys:
        if k == 'Cash':
            continue
        tgt_v = target.get(k, 0.0) * total
        cur_v = current_v.get(k, 0.0)
        d = tgt_v - cur_v
        if abs(d) < MIN_ORDER_USDT:
            continue
        rows.append((k, d))
    rows.sort(key=lambda x: -abs(x[1]))
    for k, d in rows[:10]:
        sign = '+' if d > 0 else ''
        lines.append(f'  {k}: {sign}${d:,.2f}')
    return '\n'.join(lines) if len(lines) > 1 else '  (변화 없음)'


# ═══ run_once ═══
def run_once(dry_run: bool = False, api=None, state_ref: Optional[str] = None) -> int:
    """한 사이클 실행. 리턴: 0=정상, 1=freshness 스킵, 2=에러, 3=청산 실패 fail-closed.

    (원본 executor_coin.py 와 동일한 반환 계약 — 파일 헤더 exit 계약 참조)
    api: 테스트용 주입. None 이면 실제 BinanceSpotAPI 생성.
    state_ref: 다른 실행기(업비트 LIVE)의 V24 state 파일 경로. 읽기 전용 참조이며
        상대경로는 CACHE_DIR 기준 (STATE_FILE 과 동일 규약). dry-run 은 매 실행 참조,
        live 는 자체 state 에 members 가 없을 때 1회 시드. 참조 실패는 fail-closed (exit 2).
    """
    state_path = os.path.join(CACHE_DIR, STATE_FILE)
    state, state_err = load_state_strict(state_path)
    if state_err:
        log(f'❌ {state_err} → fail-closed (손상된 state 로 주문 금지)')
        _tg(f'❌ {EXCHANGE_LABEL} state 손상 → 실행 중단: {state_err}')
        _flush_telegram(dry_run)
        return 2
    if 'cash_buffer' in state and 'buffer_pct' not in state:
        state['buffer_pct'] = state['cash_buffer']
    elif 'buffer_pct' in state and 'cash_buffer' not in state:
        state['cash_buffer'] = state['buffer_pct']
    now = cle.utc_now()

    # 이전 사이클의 combined target 캡처 (engine 이 덮어쓰기 전)
    # legacy 스냅샷이 대문자 CASH 로 저장돼 있을 수 있으므로 로드 시점에 정규화 (허위 target_changed 방지)
    _prev_snap = state.get('last_target_snapshot') or {}
    prev_combined = _norm_cash_map(_prev_snap)

    log(f'═══ {EXCHANGE_LABEL} Executor 시작 (dry_run={dry_run}, now={cle.to_utc_iso(now)}) ═══')

    # state-ref — 업비트 LIVE 의 V24 state 를 엔진 상태 출처로 참조 (읽기 전용).
    # 시작 로그 뒤에 둔다 (일일 리포트는 마지막 시작 로그 이후 레코드만 읽는다).
    ref = None
    seeded_now = False       # 이번 실행에서 live 가 참조로 자체 state 를 시드했는가
    if state_ref is not None:
        ref_path = ''
        if state_ref.strip() == '':
            ref_err = '참조 경로 비어있음'
        else:
            ref_path = state_ref if os.path.isabs(state_ref) else os.path.join(CACHE_DIR, state_ref)
            ref_err = _state_ref_path_error(ref_path, state_path)
            if not ref_err:
                ref, ref_err = load_state_ref(ref_path)
        if ref_err:
            log(f'🚨 state-ref 참조 실패: {ref_err} → fail-closed (fresh 초기화로 대체하지 않음)')
            # 사유에 참조 파일 값이 섞일 수 있다 — 파일 로그 필터는 텔레그램을 안 덮는다
            _tg(f'🚨 {EXCHANGE_LABEL} state-ref 참조 실패 → 실행 중단: {_redact(ref_err)[:200]}')
            _flush_telegram(dry_run)
            return 2
        ref_name = os.path.basename(ref_path)
        if dry_run:
            _apply_state_ref(state, ref)
            # 예전 live 실행이 남긴 자체 universe 캐시를 쓰지 않는다 (참조와 다른 유니버스로
            # 갈릴 수 있다) — 업비트와 같은 소스로 매 실행 재구성한다
            state.pop('universe_cache', None)
            # 참조본으로 갈아끼운 뒤 prev target 재계산 — target_changed 를 참조본의
            # 직전 target 과 비교해야 업비트와 같은 기준이 된다
            prev_combined = _norm_cash_map(state['last_target_snapshot'])
            log(f'📎 state-ref: {ref_name} 참조 (dry-run, {_state_ref_summary(ref)})')
        elif not state.get('members'):
            _apply_state_ref(state, ref)
            seeded_now = True
            prev_combined = _norm_cash_map(state['last_target_snapshot'])
            log(f'📎 state-ref 시드: {ref_name} → 자체 state 초기화 '
                f'(live 최초 실행, 이후 자체 state 사용)')
        else:
            log('  ⚠ state-ref 무시: 자체 state 에 members 존재 (live 는 최초 1회만 시드)')

    session = requests.Session()
    if api is None:
        api = BinanceSpotAPI(dry_run=dry_run,
                             ignore_assets=state.get('balance_ignore_assets') or ())

    blocked_coins: set = set()
    if not dry_run:
        # 이전 실행이 남긴 미해결 주문 intent 먼저 정리 (프로세스 재시작 대비)
        wal_ok, wal_pending = api.reconcile_wal()
        if not wal_ok:
            log(f'❌ 미해결 주문 WAL 확인 불가 → fail-closed: {wal_pending}')
            _tg(f'❌ {EXCHANGE_LABEL} 미해결 주문 확인 불가 → 실행 중단. 수동 확인 필요:\n'
                + '\n'.join(f'  - {d}' for d in wal_pending))
            _save_state_unless_dry(state_path, state, dry_run)
            _flush_telegram(dry_run)
            return 2
        blocked = api.cancel_all()
        if blocked is None:
            log('❌ 미체결 상태 확인 불가 → fail-closed (리밸런싱 스킵)')
            _tg(f'❌ {EXCHANGE_LABEL} 미체결 조회 실패 → 실행 중단')
            _save_state_unless_dry(state_path, state, dry_run)
            _flush_telegram(dry_run)
            return 2
        blocked_coins = blocked

    # 유니버스 필터용 Upbit 상태 (원본과 동일 — 엔진에 그대로 전달)
    upbit_status = cle.fetch_upbit_market_status(session)

    # 잔고 스냅샷 — 이후 모든 PV/target/delta 계산의 단일 기준 (C2)
    try:
        balance = api.get_balance()
    except BalanceIncomplete as e:
        log(f'❌ 잔고 스냅샷 불완전: {e} → 주문 전 중단')
        _tg(f'❌ {EXCHANGE_LABEL} 잔고 평가 불가 → 실행 중단: {_redact(e)}')
        _save_state_unless_dry(state_path, state, dry_run)
        _flush_telegram(dry_run)
        return 2

    # 거래정지 감지 (freshness 무관, 매번 수행) — 위 스냅샷 기준
    # 실행 가능/ dust 판정은 liquidation_state 가 하므로 여기서는 양수 보유분을 전부 넘긴다
    # (경계값 $10 처럼 실제 매도 가능한 잔량이 누락되지 않도록 — 라운드5 m)
    held_coins = [k for k, v in balance.items() if k != CASH_ASSET and v > 0]
    non_trading = detect_non_trading(held_coins, api)
    to_liquidate = list(non_trading)
    if to_liquidate:
        log(f'  🚨 거래정지(non-TRADING) 보유: {to_liquidate}')
        _, failed_liq = liquidate_coins(to_liquidate, 'Binance 거래정지', api, state,
                                        blocked_coins=blocked_coins)
        try:
            balance = api.get_balance()
        except BalanceIncomplete as e:
            log(f'❌ 청산 후 잔고 스냅샷 불완전: {e} → 중단')
            _tg(f'❌ {EXCHANGE_LABEL} 청산 후 잔고 평가 불가 → 실행 중단')
            _save_state_unless_dry(state_path, state, dry_run)
            _flush_telegram(dry_run)
            return 2
        if failed_liq:
            log(f'❌ 청산 실패 {failed_liq} → fail-closed (리밸런싱 스킵)')
            _tg(f'❌ {EXCHANGE_LABEL} 청산 실패 {failed_liq} → 실행 중단')
            _save_state_unless_dry(state_path, state, dry_run)
            _flush_telegram(dry_run)
            return 3

    def _upbit_ohlcv(ticker: str):
        try:
            return pyupbit.get_ohlcv(ticker, interval='day', count=260)
        except Exception:
            return None

    # V24: cur_w 산출 (자본금 기준 비중) — drift 트리거 평가용
    # balance: {'USDT': float, 'BTC': float, ...} 모두 USDT 평가액 (단일 스냅샷)
    cur_balance_for_w = balance
    total_for_w = sum(cur_balance_for_w.values()) if cur_balance_for_w else 0.0

    # alloc_transit cap (옵션 D pure, 2026-05-23)
    _spot_cap_ratio = _read_alloc_transit_cap_ratio_spot()
    _pv_basis = total_for_w
    if _spot_cap_ratio is not None and _spot_cap_ratio < 1.0:
        _pv_basis = total_for_w * _spot_cap_ratio
        log(f'  🔴 alloc_transit cap_ratio={_spot_cap_ratio:.3f} → pv ${total_for_w:,.2f} → ${_pv_basis:,.2f}')

    cur_w_input: Dict[str, float] = {}
    if _pv_basis > 0:
        for k, v in cur_balance_for_w.items():
            key = 'Cash' if k == CASH_ASSET else k
            cur_w_input[key] = cur_w_input.get(key, 0.0) + (float(v) / _pv_basis)

    # state-ref dry-run 한정: drift 평가용 보유비중을 '참조 목표를 보유 중'으로 가정한다.
    # dry-run 계좌는 dust 뿐이라 실잔고로 평가하면 매일 drift 가 발화하고, refill v2 가
    # 업비트(목표 근접)라면 바꾸지 않았을 스냅샷 코인을 교체해 신호 비교가 오염된다.
    # 실잔고는 balance/total_usdt/execute_delta 쪽에서 그대로 쓰인다. live 에선 절대 안 한다.
    if dry_run and ref is not None:
        # 업비트가 실제로 들고 있는 건 cash buffer 를 뺀 target 이므로 같은 기준으로 맞춘다
        # (buffer 가 크면 아래 cash buffer drift 재평가가 혼자 재발화한다)
        try:
            _ref_buf = float(state.get('spot_cash_buffer',
                                       state.get('cash_buffer',
                                                 state.get('buffer_pct', CASH_BUFFER_DEFAULT))))
        except (TypeError, ValueError):
            _ref_buf = CASH_BUFFER_DEFAULT
        cur_w_input = apply_cash_buffer({k: v for k, v in ref['last_target_snapshot'].items()
                                         if k != '_ts'}, _ref_buf)
        log(f'  📎 state-ref: drift 평가 보유비중 = 참조 last_target_snapshot'
            f'(cash buffer {_ref_buf*100:.1f}% 반영) 가정 (실잔고 아님)')

    # 엔진 호출
    try:
        result = cle.compute_live_targets(
            state, session, CACHE_DIR, now_utc=now,
            upbit_price_fn=_upbit_ohlcv,
            upbit_status=upbit_status,
            cur_w=cur_w_input or None,
        )
    except Exception as e:
        log(f'❌ 엔진 호출 실패: {_redact(e)}\n{_redact(traceback.format_exc())}')
        _tg(f'❌ 엔진 호출 실패: {_redact(e)}')
        _save_state_unless_dry(state_path, state, dry_run)
        _flush_telegram(dry_run)
        return 2

    # 엔진 결과 진입부 방어 — 상류 규약(‘Cash’)을 신뢰하지 않고 실자금 계층에서 한 번 더 병합
    _eng_odd_cash = [k for k in (result.combined_target or {}) if _is_cash_key(k) and k != 'Cash']
    if _eng_odd_cash:
        log(f'  ⚠ 엔진 combined_target 현금 키 비정규 {_eng_odd_cash} → Cash 로 병합')
    result.combined_target = _norm_cash_map(result.combined_target)
    result.member_targets = {n: _norm_cash_map(t) for n, t in (result.member_targets or {}).items()}

    for a in result.alerts:
        log(f'  [engine] {a}')
        # 카나리 플립 알림은 텔레그램 미전송 (2026-08-20 사용자 요청). 로그·엔진 상태는 그대로.
        if '카나리' in a:
            continue
        _tg(a)

    # Freshness 판정
    if not result.all_fresh:
        fresh_str = ', '.join(f'{k}={"✓" if v else "✗"}' for k, v in result.fresh.items())
        log(f'  ⚠ Freshness 미달 ({fresh_str}) → 리밸런싱 스킵. 상태만 저장.')
        _tg(f'⚠ Freshness 미달: {fresh_str} → 스킵')
        if seeded_now:
            # 이 경로엔 아래 같은 봉 복원이 없다 — 엔진이 써 둔 stale(refill 이전) target 을
            # 저장하면 다음 실행부터 'members 있음'을 이유로 참조를 무시하고 그대로 굳는다
            log('  📎 state-ref 시드: Freshness 미달 → 시드 state 저장 생략 (다음 실행에서 다시 시드)')
        else:
            _save_state_unless_dry(state_path, state, dry_run)
        _flush_telegram(dry_run)
        return 1

    # state-ref: 참조(업비트)가 먼저 돌아 오늘 봉을 이미 처리한 경우 (cron 랜덤 지연으로
    # 절반 정도의 날) 엔진은 같은 봉이라 새 봉 없음으로 답하고 멤버 last_combined(= refill v2
    # 이전 사본)를 돌려준다. dry-run 은 여기서 스킵하면 target 로그가 없어 리포트가 현물 타겟을
    # 못 읽고, live 최초 시드는 그 stale target 을 자체 state 에 굳혀 실주문까지 갈 수 있다.
    _same_bar_ref = bool(ref is not None and not result.any_new_bar and (dry_run or seeded_now))

    # 옵션 Z: cap_defend trigger — cap_ratio < 0.99 면 any_new_bar 우회 (매일 cap 매도 시도)
    _cap_defend_fire = (_spot_cap_ratio is not None and _spot_cap_ratio < (1.0 - CAP_DEFEND_MIN_EXCESS))
    if not result.any_new_bar and not _cap_defend_fire and not _same_bar_ref:
        log('  ℹ 새 봉 없음 (idempotent) → 리밸런싱 스킵.')
        _save_state_unless_dry(state_path, state, dry_run)
        _tg(f'⏸ 새 봉 없음 → 스킵 (다음 봉 닫힘 대기)')
        _flush_telegram(dry_run)
        return 0
    if _cap_defend_fire and not result.any_new_bar:
        log(f'  🛡️ cap_defend trigger: cap_ratio={_spot_cap_ratio:.4f} < {1.0-CAP_DEFEND_MIN_EXCESS:.2f} → any_new_bar 우회 (cap 매도 시도)')

    if _same_bar_ref:
        # 참조(업비트)가 이미 오늘 봉을 처리했다. 엔진은 같은 봉이라 members.last_combined 를 돌려주는데
        # 그 값은 refill v2 이전 사본이다 (엔진은 refill 결과를 last_target_snapshot 에만 반영한다).
        # 업비트가 실제로 쓴 최종 target 은 참조본의 last_target_snapshot / last_member_targets 이므로
        # 그 값으로 맞춘다.
        _ref_final = {k: v for k, v in ref['last_target_snapshot'].items() if k != '_ts'}
        result.combined_target = _norm_cash_map(_ref_final)
        _ref_mt = ref.get('last_member_targets') or {}
        if _ref_mt:   # 없으면 엔진의 같은 봉 멤버 target 을 그대로 둔다 (멤버 줄이 사라지지 않게)
            result.member_targets = {m: _norm_cash_map({k: v for k, v in t.items() if k != '_ts'})
                                     for m, t in _ref_mt.items()}
        if dry_run:
            # 오늘의 drift 판단은 참조가 이미 했으므로 여기서 재발화하지 않는다.
            # (live 는 실잔고 기준 drift 를 그대로 살려둔다 — 시드 첫날 실제 편차를 메워야 한다)
            try:
                result.drift_fire = False
                result.drift_half_turnover = 0.0
            except Exception:
                pass
            log('  📎 state-ref: 참조가 이미 오늘 봉을 처리함 → 참조의 최종 target(refill 반영)으로 같은 봉 비교 진행')
        else:
            # 엔진이 state 에 써 둔 stale(refill 이전) target 을 참조의 최종 target 으로 교정한 뒤
            # 저장한다 — 시드 당일 자체 state 에 굳으면 다음 실행까지 stale 로 주문하게 된다.
            # 엔진은 stale 기준으로 drift 를 평가했고 refill v2 가 members 스냅샷까지 바꿨을 수
            # 있으므로, 참조 members 를 되돌리고 drift 는 실잔고 vs 참조 최종 target 으로 다시 잡는다.
            state['members'] = copy.deepcopy(ref['members'])
            state['last_target_snapshot'] = {**_ref_final, '_ts': cle.to_utc_iso(now)}
            if _ref_mt:
                state['last_member_targets'] = {
                    m: {**{k: v for k, v in t.items() if k != '_ts'}, '_ts': cle.to_utc_iso(now)}
                    for m, t in _ref_mt.items()}
            try:
                _seed_buf = float(state.get('spot_cash_buffer',
                                            state.get('cash_buffer',
                                                      state.get('buffer_pct', CASH_BUFFER_DEFAULT))))
            except (TypeError, ValueError):
                _seed_buf = CASH_BUFFER_DEFAULT
            _ht_ref = (cle.half_turnover(cur_w_input, apply_cash_buffer(_ref_final, _seed_buf))
                       if cur_w_input else 0.0)
            try:
                result.drift_fire = _ht_ref >= result.drift_threshold
                result.drift_half_turnover = _ht_ref
            except Exception:
                pass
            log('  📎 state-ref 시드: 참조가 이미 오늘 봉을 처리함 → 참조의 최종 target 으로 진행 '
                f'(drift 는 실잔고 vs 참조 최종 target 재계산: ht={_ht_ref:.4f})')

    # 멤버/합산 target 로깅
    for mname, mt in result.member_targets.items():
        coins = ', '.join(f'{k}:{v:.1%}' for k, v in mt.items() if k != 'Cash' and v > 0)
        cash_w = result.member_targets.get(mname, {}).get('Cash', 0.0)
        log(f'  {mname} target: {coins or "CASH only"} (cash={cash_w:.1%})')
    combined_coins = ', '.join(f'{k}:{v:.1%}' for k, v in result.combined_target.items() if k != 'Cash' and v > 0)
    combined_cash = result.combined_target.get('Cash', 0.0)
    log(f'  combined target: {combined_coins or "CASH only"} (cash={combined_cash:.1%})')

    # Cash buffer — V24 (2026-05-26): spot_cash_buffer 키 우선, 없으면 legacy cash_buffer
    buffer_pct = float(state.get('spot_cash_buffer', state.get('cash_buffer', state.get('buffer_pct', CASH_BUFFER_DEFAULT))))
    state['spot_cash_buffer'] = buffer_pct
    state['cash_buffer'] = buffer_pct  # backward compat
    state['buffer_pct'] = buffer_pct
    target = apply_cash_buffer(result.combined_target, buffer_pct)
    log(f'  Cash buffer {buffer_pct*100:.1f}% 적용 후 target Cash={target.get("Cash",0)*100:.2f}%')

    # Notional cap — 잔고는 위 단일 스냅샷 재사용 (C2: 재조회로 인한 TOCTOU 제거)
    total_usdt = sum(balance.values())
    _pv_basis_exec = total_usdt if (_spot_cap_ratio is None or _spot_cap_ratio >= 1.0) else (total_usdt * _spot_cap_ratio)
    effective_target = dict(target)
    if total_usdt > 0 and 0 < NOTIONAL_CAP_FRACTION < 1:
        effective_target, gross = apply_notional_cap(target, balance, total_usdt, NOTIONAL_CAP_FRACTION)
        log(f'  Notional cap {NOTIONAL_CAP_FRACTION*100:.0f}% 적용 (gross_delta={gross*100:.1f}%)')

    # 이벤트 트리거 판정 (auto_trade_binance의 rebalancing_needed 패턴 이식)
    # - target이 prev 대비 변하면 rebalancing_needed=True (카나리/스냅 회전/유의 퇴출 등)
    # - 한 번 True면 실제 포지션이 목표에 근접할 때까지 다음 실행에서도 유지
    # - 가격 drift만으로는 여기 안 들어옴 (target 불변 + rebalancing_needed False)
    target_changed = not targets_equal(result.combined_target, prev_combined)
    if target_changed:
        state['rebalancing_needed'] = True
        log(f'  🔔 target 변경 감지 → rebalancing_needed=True. prev={prev_combined}, new={result.combined_target}')

    # cash buffer 반영 — engine 의 combined 은 risky-asset 100% 정규화이므로
    # spot_cash_buffer (실매매가 유지하는 USDT) 만큼 target 스케일 다운 + Cash 추가 후 재평가
    try:
        _spot_buf = float(state.get('spot_cash_buffer', state.get('cash_buffer', CASH_BUFFER_DEFAULT)))
    except (TypeError, ValueError):
        _spot_buf = CASH_BUFFER_DEFAULT
    # state-ref dry-run 은 보유비중 가정에 이미 buffer 가 반영돼 있고, 이 블록의 변환은
    # target 에 Cash 가 이미 있는 경우 apply_cash_buffer 와 달라 혼자 재발화할 수 있다 → 건너뛴다
    if _spot_buf > 0 and cur_w_input and not (dry_run and ref is not None):
        _combined_buf = {}
        _has_cash_in_tgt = any(str(_k).lower() == 'cash' for _k in result.combined_target)
        # canary OFF (Cash=1.0) 인 경우 buffer 적용 skip — 이미 Cash 100%
        _cash_w_in_tgt = sum(float(_v) for _k, _v in result.combined_target.items()
                             if str(_k).lower() == 'cash')
        if _cash_w_in_tgt < 0.99:
            for _k, _v in result.combined_target.items():
                if str(_k).lower() == 'cash':
                    _combined_buf[_k] = float(_v)
                else:
                    _combined_buf[_k] = float(_v) * (1.0 - _spot_buf)
            if not _has_cash_in_tgt:
                _combined_buf['Cash'] = _spot_buf
            _ht_buf = cle.half_turnover(cur_w_input, _combined_buf)
            if _ht_buf >= result.drift_threshold and not result.drift_fire:
                log(f'  🔔 cash buffer 반영 drift 재평가: ht={_ht_buf:.4f} ≥ {result.drift_threshold:.2f} → fire')
                # 엔진 결과 override (dataclass — 일부는 frozen 일 수 있어 try/except)
                try:
                    result.drift_fire = True
                    result.drift_half_turnover = _ht_buf
                except Exception:
                    pass

    # V24 drift 트리거: target 불변이어도 cur_w 와 ht 차이 >= threshold 면 발화
    # crash_cooldown 체크는 엔진이 이미 처리 (canary_on 만 추가 게이트로 가정)
    if not target_changed and result.drift_fire:
        state['rebalancing_needed'] = True
        state['last_rebal_reason'] = 'drift'
        log(f'  🔔 V24 drift 발화 → rebalancing_needed=True. half_turnover={result.drift_half_turnover:.4f} '
            f'>= threshold={result.drift_threshold:.2f}')
        # V24: 정상 운영 알림 silent (Daily Report 09:15 이 통합 보고). 운영자 즉시 알림은 오류만.
        log(f'  ℹ V24 drift 트리거 발화 (silent): ht={result.drift_half_turnover:.3f} ≥ {result.drift_threshold:.2f}')
    elif target_changed:
        state['last_rebal_reason'] = 'snap_or_signal'

    # 옵션 Z: cap_defend trigger (drift/target_changed 와 별개)
    if _cap_defend_fire and not state.get('rebalancing_needed', False):
        state['rebalancing_needed'] = True
        state['last_rebal_reason'] = 'cap_defend'
        log(f'  🛡️ cap_defend trigger → rebalancing_needed=True (cap_ratio={_spot_cap_ratio:.4f})')
    log(f'  V24 debug: schema_version={state.get("schema_version", "N/A")} '
        f'cur_w_count={len(cur_w_input)} ht={result.drift_half_turnover:.4f} '
        f'drift_fire={result.drift_fire} target_changed={target_changed}')

    rebalance_needed = bool(state.get('rebalancing_needed', False))
    if not rebalance_needed:
        log(f'  ℹ target 불변 + rebalancing_needed=False → 스킵. prev={prev_combined}')
        state['last_pv_usdt'] = total_usdt
        _save_state_unless_dry(state_path, state, dry_run)
        # V24: 정상 no-op 보고 silent (Daily Report 09:15 이 통합 보고)
        _flush_telegram(dry_run)
        return 0

    # 실행 가능 최소주문 기준 (minNotional / minQty×price / step×price) + 기존 불가피 dust
    _exec_min_fn = (lambda c: api.executable_min_notional(c, api.last_price(c), market=True))
    _buy_min_fn = (lambda c: api.buy_min_notional(c, api.last_price(c)))
    _known_dust = set(state.get('unexecutable_dust') or [])

    # 실제 잔고 편차 체크 — 목표에 이미 근접해있고 이벤트 흔적(True)만 남았으면 클리어
    if not target_changed and not coin_needs_rebalance(effective_target, balance, total_usdt,
                                                       min_fn=_exec_min_fn,
                                                       buy_min_fn=_buy_min_fn,
                                                       skip_coins=_known_dust):
        state['rebalancing_needed'] = False
        log('  ✅ 포지션이 이미 목표 근접 → rebalancing_needed=False 클리어. 스킵.')
        state['last_pv_usdt'] = total_usdt
        _save_state_unless_dry(state_path, state, dry_run)
        # V24: 정상 no-op 보고 silent
        _flush_telegram(dry_run)
        return 0

    # 사전 알림 (실제 매매가 진행될 때만)
    if state.get('pretrade_alert', True):
        summary = format_target_summary(result.combined_target, result.member_targets)
        delta_preview = format_delta_preview(effective_target, balance, total_usdt)
        universe_sample = ', '.join(result.universe[:8])
        if len(result.universe) > 8:
            universe_sample += f' ... (+{len(result.universe) - 8})'
        # V24: 사전 알림 silent (Daily Report 가 통합)
        log(f'  사전 알림 (silent): {summary[:80]}... / delta {delta_preview[:60]} / universe {result.universe[:5]}')
        flips = [m for m, f in result.canary_flipped.items() if f]
        if flips:
            log(f'  카나리 플립 (silent): {flips}')

    # Delta 매매 (UnknownExecutionError 는 상위로 전파 → health lock + exit 3)
    permanent_block = state.get('permanent_block', [])
    unexecutable = execute_delta(effective_target, api, permanent_block, dry_run,
                                 balance=balance, effective_pv_usdt=_pv_basis_exec,
                                 blocked_coins=blocked_coins)
    if unexecutable:
        log(f'  ℹ 불가피 dust (주문 최소단위 미달) {sorted(unexecutable)} '
            f'→ 리밸 완료 판정에서 제외')
        state['unexecutable_dust'] = sorted(unexecutable)
        state['unexecutable_dust_ts'] = cle.to_utc_iso(now)
    else:
        state.pop('unexecutable_dust', None)
        state.pop('unexecutable_dust_ts', None)

    # 체결 후 잔고 재조회 → 여전히 편차 남으면 다음 실행에서 재시도 (부분체결 대응)
    balance_after = balance
    total_after = total_usdt
    if not dry_run:
        try:
            balance_after = api.get_balance()
        except BalanceIncomplete as e:
            balance_after = {}
            log(f'  ⚠ 체결 후 잔고 평가 불가: {_redact(e)}')
        total_after = sum(balance_after.values()) if balance_after else 0.0
        # 잔고 조회 실패(빈 dict) 또는 total 0 → 판정 보류, 플래그 유지
        if not balance_after or total_after <= 0:
            state['rebalancing_needed'] = True
            log(f'  ⚠ 체결 후 잔고 조회 실패/빈값 → rebalancing_needed=True 유지 (보수적 재시도)')
        else:
            still_needed = coin_needs_rebalance(effective_target, balance_after, total_after,
                                                min_fn=_exec_min_fn,
                                                buy_min_fn=_buy_min_fn,
                                                skip_coins=(_known_dust | unexecutable))
            if still_needed:
                state['rebalancing_needed'] = True
                log(f'  ⏳ 체결 후에도 편차 잔존 → rebalancing_needed 유지. total=${total_after:,.2f}')
            else:
                state['rebalancing_needed'] = False
                log(f'  ✅ 목표 도달 → rebalancing_needed=False. total=${total_after:,.2f}')

    # 상태 저장
    state['last_pv_usdt'] = total_usdt
    _save_state_unless_dry(state_path, state, dry_run)
    log(f'  상태 저장: {STATE_FILE}')

    # V24: 최종 정상 보고 silent. 거래 결과는 Daily Report 09:15 가 통합.
    rebal_done = not bool(state.get('rebalancing_needed', False))
    log(f'  ✅ 거래 완료 (silent). 결과={rebal_done}, 다음 Daily Report 에 반영')
    _flush_telegram(dry_run)
    return 0


# ═══ 진입점 ═══
def main():
    parser = argparse.ArgumentParser(description='Cap Defend 코인 현물 Executor (Binance USDT)')
    parser.add_argument('--dry-run', action='store_true', help='주문 없이 target/delta만 로그+텔레그램')
    parser.add_argument('--wal-mark-resolved', metavar='COID', default=None,
                        help='거래소 주문내역을 직접 확인한 뒤 해당 clientOrderId 를 수동 해소')
    parser.add_argument('--state-ref', metavar='PATH', default=None,
                        help='다른 실행기(업비트 LIVE)의 V24 state 파일을 엔진 상태의 출처로 '
                             '참조한다 (읽기 전용). dry-run: 매 실행 참조. '
                             'live: 자체 state 에 members 가 없을 때 1회 시드.')
    args = parser.parse_args()

    if args.wal_mark_resolved:
        coid = args.wal_mark_resolved
        log(f'✍ 운영자 수동 WAL 해소 요청: {coid}')
        api = BinanceSpotAPI(dry_run=False)
        ok = api.mark_wal_resolved(coid)
        _tg(f'✍ {EXCHANGE_LABEL} WAL 수동 해소 {"성공" if ok else "실패"}: {coid}')
        _flush_telegram(False)
        sys.exit(0 if ok else 2)

    hg = HealthGuard(name='coin_binance')
    lock_reason = hg.is_locked()
    if lock_reason:
        log(f'🔒 health lock 활성 — {lock_reason}. 수동 해제 필요 (rm {hg.lock_file})')
        _tg(f'🔒 {EXCHANGE_LABEL} lock 활성 — 수동 확인 후 rm {hg.lock_file}')
        _flush_telegram(args.dry_run)
        return

    lock_f = None
    try:
        lock_f = open(LOCK_FILE, 'w')
        try:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log('🔒 다른 인스턴스 실행 중 (lock 충돌) → 종료')
            _tg(f'🔒 {EXCHANGE_LABEL} 락 충돌 → 스킵')
            _flush_telegram(args.dry_run)
            return

        rc = run_once(dry_run=args.dry_run, state_ref=args.state_ref)
        if rc == 0 and not args.dry_run:
            hg.record_success()
        sys.exit(rc)
    except SystemExit:
        raise
    except UnknownExecutionError as e:
        log(f'⚠️ UNKNOWN_EXECUTION: {_redact(e)}\n{_redact(traceback.format_exc())}')
        hg.lock(_redact(f'UNKNOWN_EXECUTION: {e}'))
        _tg(f'⚠️ {EXCHANGE_LABEL} UNKNOWN_EXECUTION — 중복 주문 위험. 수동 확인 후 rm {hg.lock_file}: {_redact(e)}')
        _flush_telegram(args.dry_run)
        sys.exit(3)
    except Exception as e:
        log(f'❌ 치명 오류: {_redact(e)}\n{_redact(traceback.format_exc())}')
        streak = hg.record_abort(_redact(e)) if not args.dry_run else 0
        _tg(f'❌ {EXCHANGE_LABEL} 치명 오류 (streak={streak}): {_redact(e)}')
        _flush_telegram(args.dry_run)
        sys.exit(2)
    finally:
        if lock_f is not None:
            try:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
                lock_f.close()
            except Exception:
                pass


if __name__ == '__main__':
    main()
