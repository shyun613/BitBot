#!/usr/bin/env python3
"""일일 운용 리포트 (업비트 현물 / 바낸현물 / 바이낸스 선물).

매일 09:20 KST cron 용. 같은 날 09:05 실행의 세 executor 로그를 파싱해
카나리 상태 / 선정 코인 / 목표 비중 / 실행 결과를 요약하고 텔레그램으로 발송한다.
현물 두 축(업비트 vs 바낸현물)은 같은 전략·같은 자산이라 1:1 로 비교하고,
선물은 전략·자산이 달라 비교 대상이 아니라 결과만 함께 보여준다.

대상 로그 (trade/):
  - executor_coin.log          (업비트 KRW 현물)
  - executor_coin_binance.log  (바이낸스 USDT 현물)
  - binance_trade.log          (바이낸스 USDT-M 선물, V25)

로그 포맷:
  현물: [YYYY-MM-DD HH:MM:SS] [run_id] <message>
  선물: YYYY-MM-DD HH:MM:SS,mmm LEVEL <message>   (python logging 기본 포맷)
  (message 가 여러 줄이면 후속 줄엔 타임스탬프 없음 → 직전 레코드에 이어붙임)

리포트 형태:
  📊 일일 운용 리포트 2026-08-31 (KST 기준)

  [업비트] 09:05:58 (LIVE) ✅ 거래 완료
    타겟: BTC 33.3%, ETH 33.3%, SOL 33.3% (cash=0.0%)
    평가액: 2,100,000원 (+5.0%)

  [바낸현물] 09:05:38 (dry) cycle=09055998 ✅ 거래 완료
    타겟: BTC 33.3%, ETH 33.3%, SOL 33.3% (cash=0.0%)

  [선물] 09:05:32 (dry) run=20260831_090532 ✅ 정상 완료
    타겟: BTC 33.3% (L2), ETH 33.3% (L2), cash 33.4% (refill v2 반영)
    PV: $0.00

  ── 현물 비교 ──
  ✅ 실행 쌍 정합 / ✅ 카나리 일치 / ✅ 코인 집합 일치 / ✅ 비중 일치
  실행 모드: 업비트=LIVE, 바낸현물=dry-run

  실행 모드(dry/LIVE)는 정합 판정 대상이 아니다 — 비교 대상은 전략 산출물(타겟·카나리)이고,
  운영 조합이 '업비트 LIVE + 바낸현물 dry-run' 인 게 정상 상태라 정보로만 표시한다.

  블록의 '카나리:' 줄은 평상시(전 멤버 ON·양쪽 일치)엔 생략하고, OFF·판별 불가·불일치·
  한쪽 미실행일 때만 싣는다. 비교 섹션의 카나리 판정 줄은 항상 나온다.

  '평가액:' 줄은 업비트(실거래 계좌)에만 붙는다. 원금은 trade/report_principal.json
  ({"principal_krw": ..., "last_deposit_check": ...})에 두고 KRW 입금분만 자동 가산한다.
  출금은 조회 권한이 없어 자동 반영되지 않으니 출금 시 이 파일을 수동으로 낮춰야 한다.

날짜 기준:
  서비스 날짜는 KST(Asia/Seoul) 기준 '오늘'. 로그 타임스탬프는 서버 로컬시각(UTC)이므로
  KST 09:05 실행분은 같은 날짜의 UTC 00:05 로 기록된다 (KST 09시 이후 실행에 한해 날짜 일치).
  다른 시간대 실행분을 보려면 --date 로 로그 날짜를 직접 지정한다.

Usage:
  python3 compare_daily_report.py            # 텔레그램 발송
  python3 compare_daily_report.py --stdout   # 표준출력 (테스트)
  python3 compare_daily_report.py --date 2026-08-31 --stdout
  python3 compare_daily_report.py --no-fut --stdout    # 현물 2축만
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

try:
    from zoneinfo import ZoneInfo
except ImportError:  # py<3.9
    ZoneInfo = None  # type: ignore

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRADE_DIR = os.path.join(REPO_DIR, 'trade')
sys.path.insert(0, TRADE_DIR)

from common.notify import send_telegram as _send_tg  # noqa: E402

try:
    from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID  # noqa: E402
except ImportError:
    TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')

try:
    from config import UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY  # noqa: E402
except ImportError:
    UPBIT_ACCESS_KEY = os.environ.get('UPBIT_ACCESS_KEY', '')
    UPBIT_SECRET_KEY = os.environ.get('UPBIT_SECRET_KEY', '')


TG_PREFIX = '비교리포트'
WARN_HEADER = '⚠️ 점검 필요'
KST_TZ = 'Asia/Seoul'
KST_FIXED = timezone(timedelta(hours=9))   # tzdata 없는 환경용 고정 오프셋 fallback
WEIGHT_TOL = 0.01  # 1%p 초과 시 불일치

SIDES = [
    ('업비트', os.path.join(TRADE_DIR, 'executor_coin.log')),
    ('바낸현물', os.path.join(TRADE_DIR, 'executor_coin_binance.log')),
]

# 선물(V25)은 전략·자산이 달라 현물과 1:1 비교하지 않는다. 결과만 함께 싣는다.
FUT_NAME = '선물'
FUT_LOG = os.path.join(TRADE_DIR, 'binance_trade.log')

# 업비트 실계좌 평가액/원금 (업비트 블록에만 싣는다 — 바낸현물·선물은 대상 아님)
UPBIT_API = 'https://api.upbit.com'
UPBIT_HTTP_TIMEOUT = 10
PRINCIPAL_FILE = os.path.join(TRADE_DIR, 'report_principal.json')

# ─── 토큰/URL redaction (m2 — common/notify.py 는 지인 공용 코드라 수정 금지) ───
_TOKEN_RE = re.compile(r'bot\d+:[A-Za-z0-9_\-]+')
# 업비트 JWT 는 예외 메시지/urllib 오류에 헤더째 실려 나올 수 있다
_BEARER_RE = re.compile(r'Bearer [A-Za-z0-9._\-]+')


def _redact(text: str) -> str:
    out = _TOKEN_RE.sub('bot<REDACTED>', str(text))
    out = re.sub(r'(api\.telegram\.org/)[^\s\'"]+', r'\1<REDACTED>', out)
    out = _BEARER_RE.sub('Bearer <REDACTED>', out)
    for secret in (TELEGRAM_BOT_TOKEN, UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY):
        if secret:
            out = out.replace(secret, '<REDACTED>')
    return out


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


class _RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        _redact_record(record)
        return True


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
    """formatter 미설정 핸들러(기본 formatter 경로)까지 커버."""
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
    """logger.addFilter 는 자식 로거(urllib3.connectionpool 등) 레코드에 적용되지 않으므로
    LogRecordFactory 로 프로세스 전역에 적용하고, 핸들러 formatter 도 감싸
    exc_info traceback 텍스트까지 마스킹한다."""
    _prev = logging.getLogRecordFactory()

    def _factory(*args, **kwargs):
        return _redact_record(_prev(*args, **kwargs))

    logging.setLogRecordFactory(_factory)
    _patch_set_formatter()
    _patch_handler_format()
    _filt = _RedactingFilter()
    for name in (None, 'common.notify', 'requests', 'urllib3'):
        lg = logging.getLogger(name) if name else logging.getLogger()
        for h in list(getattr(lg, 'handlers', []) or []):
            _wrap_handler(h, _filt)


_install_redaction()

# ─── 로그 파싱 정규식 ───
RE_LINE = re.compile(r'^\[(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2})\] \[([^\]]*)\] (.*)$')
RE_START = re.compile(r'Executor 시작 \(dry_run=(True|False)')
RE_CANARY = re.compile(r'(\S+) 카나리 (ON|OFF)')
RE_COMBINED_TGT = re.compile(r'^\s*combined target: (.*?) \(cash=([\d.]+)%\)\s*$')
# 'combined target:' 라인과 겹치지 않도록 member 이름에서 combined 제외
RE_MEMBER_TGT = re.compile(r'^\s*(?!combined\b)(\S+) target: (.*?) \(cash=([\d.]+)%\)\s*$')
RE_TOKEN = re.compile(r'([A-Z0-9]+):([\d.]+)%')

SEVERITY = {'ok': 0, 'skip': 1, 'warn': 2, 'error': 3}
PAIR_MAX_GAP_SEC = 30 * 60  # 두 실행 시작 시각 차가 이보다 크면 동일 사이클 쌍으로 보기 어려움
# cron 실행 창 (KST). 래퍼의 랜덤 지연(<60s) + 실행 시간을 감안한 범위.
CRON_WINDOW_KST = (9 * 3600 + 5 * 60, 9 * 3600 + 12 * 60)
LOG_TZ_OFFSET_HOURS = 9   # 로그 타임스탬프(서버 로컬=UTC) → KST 변환

# 시작 로그 없이 끝난 경우의 원인 후보 (M)
ABORT_HINTS: List[Tuple[str, str]] = [
    ('health lock 활성', 'health lock 으로 cron 스킵'),
    ('다른 인스턴스 실행 중', 'flock 충돌로 스킵'),
    ('락 충돌', 'flock 충돌로 스킵'),
    ('치명 오류', '시작 직후 치명 오류'),
]

# (마커, 분류, 라벨) — high-watermark 로 보존한다 (M8: 이후 '거래 완료'가 덮지 못함)
RESULT_MARKERS: List[Tuple[str, str, str]] = [
    # 정상/스킵
    ('거래 완료', 'ok', '거래 완료'),
    ('target 불변', 'ok', 'target 불변 → 스킵'),
    ('포지션이 이미 목표 근접', 'ok', '목표 근접 → 스킵'),
    ('새 봉 없음', 'skip', '새 봉 없음 → 스킵'),
    ('Freshness 미달', 'skip', 'Freshness 미달 → 스킵'),
    ('다른 인스턴스 실행 중', 'skip', '락 충돌'),
    # 경고 (부분 실패 — 이후 '거래 완료'가 덮으면 안 됨)
    ('매도 미완', 'warn', '매도 미완/부분체결'),
    ('매도 robust 종료', 'warn', '매도 robust 실패 종료'),
    ('매도 타임아웃', 'warn', '매도 타임아웃'),
    ('매도 실패', 'warn', '매도 실패'),
    ('매수 실패', 'warn', '매수 실패'),
    ('매수 미접수', 'warn', '매수 미접수'),
    ('매수 미체결 잔존', 'warn', '매수 미체결 잔존'),
    ('매수 최종상태 확인 불가', 'warn', '매수 상태 확인 불가'),
    ('체결 후 잔고 조회 실패', 'warn', '체결 후 잔고 조회 실패'),
    ('편차 잔존', 'warn', '체결 후 편차 잔존'),
    ('취소 실패 미체결 잔존', 'warn', '미체결 취소 실패'),
    ('WAL 미해결', 'error', '미해결 주문 WAL (수동 확인 필요)'),
    ('미해결 주문 WAL 확인 불가', 'error', '미해결 주문 WAL 확인 불가'),
    ('WAL 읽기 실패', 'error', 'WAL 읽기 실패'),
    ('reconcile 실패', 'error', 'WAL reconcile 실패'),
    ('미체결 상태 확인 불가', 'error', '미체결 상태 확인 불가'),
    ('state 손상', 'error', 'state 파일 손상'),
    ('permanent_block', 'warn', 'permanent_block 등록'),
    ('청산 검증', 'ok', '거래정지 청산'),
    # 에러
    ('엔진 호출 실패', 'error', '엔진 호출 실패'),
    ('치명 오류', 'error', '치명 오류'),
    ('청산 실패', 'error', '청산 실패 (fail-closed)'),
    ('잔고 평가 불가', 'error', '잔고 스냅샷 불완전'),
    ('잔고 스냅샷 불완전', 'error', '잔고 스냅샷 불완전'),
    ('미체결 조회 실패', 'error', '미체결 상태 확인 불가'),
    ('UNKNOWN_EXECUTION', 'error', 'UNKNOWN_EXECUTION (health lock)'),
    ('health lock 활성', 'error', 'health lock 활성'),
]

RESULT_ICONS = {'ok': '✅', 'skip': '⏸', 'warn': '⚠️', 'error': '🚨', 'unknown': '⚠️'}

# ─── 선물(binance_trade.log) 파싱 ───
# 포맷: 'YYYY-MM-DD HH:MM:SS,mmm LEVEL message' (python logging 기본, 서버 로컬=UTC)
RE_FUT_LINE = re.compile(r'^(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}),\d{3} ([A-Z]+) (.*)$')
# 블록 경계 마커는 '그 줄 전체'가 마커일 때만 인정한다 (일반 로그가 마커 문구를 인용해도
# 블록이 잘리지 않게). 실물: '=== 바이낸스 선물 매매 시작 (run_id=…) ===' / '=== 완료 (12.0s) ==='
RE_FUT_START = re.compile(r'^=== 바이낸스 선물 매매 시작(?: \(run_id=([^)]*)\))? ===\s*$')
RE_FUT_DONE = re.compile(r'^=== 완료(?: \([\d.]+s\))? ===\s*$')
# '합산: {'BTC': '33.3%', 'ETH': '33.3%'}' / '합산: CASH 100%' / '합산: {}'
RE_FUT_COMBINED = re.compile(r'^합산: (.*?)\s*$')
# '  🔁 V24 refill v2 적용: {'BTC': '33.3%'} | top diffs=[...]'  (refill 후 실제 combined)
RE_FUT_REFILL = re.compile(r'^\s*🔁 V24 refill v2 적용: (.*?) \| top diffs=')
# 'DRY-RUN REBALANCE: {'BTC': '33.3%'}' / 'DRY-RUN REBALANCE: CASH'
RE_FUT_DRYRUN_TGT = re.compile(r'^DRY-RUN REBALANCE: (.*?)\s*$')
# 실거래 전용 최종 확정: "V25 fut: targets={'BTC': 0.333, 'Cash': 0.333} ht=... success=True streak=0"
RE_FUT_V25_TGT = re.compile(r'^V25 fut: targets=(\{.*?\}) ht=')
RE_FUT_V25_SUCCESS = re.compile(r'^V25 fut: .*\bsuccess=(True|False)')
RE_FUT_FLOAT_TOKEN = re.compile(r"'([A-Za-z0-9]+)'\s*:\s*([-+0-9.eE]+)")
# '  D_SMA42 → {'BTC': '33.3%'} (cash=33%)' / '  D_SMA42 → CASH 100% (cash=100%)'
RE_FUT_MEMBER_TGT = re.compile(r'^\s*(\S+) → (.*?) \(cash=([\d.]+)%\)\s*$')
# '  BTC → final L = min(BTC_cap=4, K2=2) = 2'
RE_FUT_LEV = re.compile(r'^\s*([A-Z0-9]+) → final L = .*?=\s*(-?\d+)\s*$')
RE_FUT_PV = re.compile(r'현재 PV: \$(-?[\d,]+(?:\.\d+)?)')
# 실거래 실행이면 리밸런싱 후 PV 가 최종값이다: '리밸런싱 완료: PV $100.00 → $101.00'
RE_FUT_PV_AFTER = re.compile(r'리밸런싱 완료: PV \$[-\d,.]+ → \$(-?[\d,]+(?:\.\d+)?)')
# '  D_SMA42 BTC=$77,682 SMA(42)=$67,928 ratio=1.1436 canary=ON  *** FLIPPED ***'
# '  D_SMA42 weights: canary OFF -> CASH 100%'
RE_FUT_CANARY = re.compile(r'^\s*(\S+)\b.*?\bcanary[= ](ON|OFF)')
RE_FUT_CANARY_KO = re.compile(r'(\S+) 카나리 (ON|OFF)')
RE_FUT_WEIGHT_TOKEN = re.compile(r"'([A-Z0-9]+)'\s*:\s*'([\d.]+)%'")

FUT_ISSUE_LIMIT = 5     # 이슈로 싣는 WARNING/ERROR 줄 최대 개수
FUT_ISSUE_MAXLEN = 120  # 이슈 한 줄 최대 길이
FUT_WEIGHT_SUM_TOL = 0.01   # 비중 합 허용 초과 (로그가 소수 1자리로 반올림해 찍는다)
# 무위험(=CASH only) 을 뜻하는 '정확히 이 문자열' 만 인정한다. 그 외 비정형은 파싱 실패로 본다.
FUT_CASH_SENTINELS = ('{}', 'CASH', 'CASH 100%')

# 최종 타겟 권위 체인 (숫자가 클수록 우선).
# ※ executor 실코드 확인(auto_trade_binance.py:2798,2879,3185,3356):
#   coins_combined 는 2798 에서 refill '전' 값으로 한 번만 만들어지고 3185 의
#   'DRY-RUN REBALANCE' 가 그걸 그대로 재사용한다 → DRY-RUN 줄은 refill 을 못 덮는다.
#   refill 후 실제 combined 를 찍는 줄은 2879 의 '🔁 V24 refill v2 적용' 뿐이고,
#   3356 의 'V25 fut: targets=' 는 `if args.trade:` 안(실거래 전용)이라 최종 확정값이다.
FUT_TGT_COMBINED = 0    # '합산:'            (refill 전)
FUT_TGT_DRYRUN = 1      # 'DRY-RUN REBALANCE:' (합산의 사본 — refill 을 덮으면 안 된다)
FUT_TGT_REFILL = 2      # '🔁 V24 refill v2 적용:' (refill 후 실제 combined)
FUT_TGT_V25 = 3         # 'V25 fut: targets='  (실거래 전용 최종 확정)
FUT_TGT_LABELS = {
    FUT_TGT_COMBINED: '합산',
    FUT_TGT_DRYRUN: 'dry-run 주문계획',
    FUT_TGT_REFILL: 'refill v2 반영',
    FUT_TGT_V25: '실거래 확정',
}

# dry-run 판정 마커 (실코드: --trade 아니면 send_telegram 이 '[DRY] telegram silent:' 로 남는다)
FUT_DRY_HINTS = ('DRY-RUN REBALANCE', '[DRY] telegram silent:')
# 실거래 흔적. '시작 지연:' 은 `if args.trade:` 첫 줄이라 모든 실거래 실행에 반드시 남는다.
FUT_LIVE_HINTS = ('시작 지연:', 'V25 fut: targets=', '리밸런싱 완료', 'ORDER BUY', 'ORDER SELL',
                  'ORDER FAILED', 'ORDER RETRY', '계좌 변경 근거', '  standing: ')

# 시작 로그 없이 끝난 경우의 원인 후보 (실코드 문구 확인: 2614/2716/2736)
FUT_ABORT_HINTS: List[Tuple[str, str]] = [
    ('다른 인스턴스 실행 중', 'flock 충돌로 스킵'),
    ('API key not configured', 'API 키 미설정'),
    ('lock 활성', 'V25 lock 활성 (수동 해제 필요)'),
    ('ABORT', '시작 전 ABORT'),
]

# (마커, 분류, 라벨) — 현물과 같은 high-watermark 방식 (가장 심각한 마커가 최종 판정).
# 문구는 executor 실코드에서 확인한 '로그로 남는' 문자열만 쓴다. 텔레그램 전용 문구
# ('시장 메타 저하', '자동 lock')는 dry-run 에서만 '[DRY] telegram silent:' 로 남으므로
# 실거래에서도 잡히는 로그 문구(market cache / lock 활성)를 함께 둔다.
FUT_RESULT_MARKERS: List[Tuple[str, str, str]] = [
    ('리밸런싱 완료', 'ok', '리밸런싱 완료'),
    ('매매 스킵: rebalancing_needed=false', 'ok', '매매 불필요 → 스킵'),
    # '=== 완료' 는 마커 테이블이 아니라 앵커된 RE_FUT_DONE 으로 판정한다 (R2-3)
    ('포지션 조회 실패', 'warn', '포지션 조회 실패'),
    ('포지션/PV 조회 실패', 'warn', '포지션/PV 조회 실패'),   # '현재 포지션/PV 조회 실패.'
    ('reconciliation 차이', 'warn', 'reconciliation 차이 (체결 미달)'),
    ('시장 메타 저하', 'warn', '시장 메타 저하 상태로 진행'),   # dry-run 텔레그램 에코
    ('market cache 만료', 'warn', '시장 메타 캐시 만료'),
    ('market cache 손상', 'warn', '시장 메타 캐시 손상'),
    ('캐시로 진행', 'warn', '낡은 캐시로 진행 (시장 메타 저하)'),
    ('기준 스냅샷 무효화', 'warn', '무결성 기준 스냅샷 무효화'),
    ('ORDER FAILED', 'error', 'ORDER FAILED (주문 실패)'),
    ('무결성 위반', 'error', 'V25 무결성 위반 (lock 생성)'),
    ('매매 중단', 'error', '매매 중단'),
    ('ABORT', 'error', 'V25 ABORT (매매 차단)'),
    ('lock 활성', 'error', 'V25 lock 활성 (수동 해제 필요)'),
    ('자동 lock', 'error', '자동 lock (수동 해제 필요)'),     # dry-run 텔레그램 에코
    ('치명', 'error', '치명 오류'),
]


class SideResult:
    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path
        self.log_exists = False
        self.ran = False
        self.run_time: Optional[str] = None
        self.dry_run: Optional[bool] = None
        self.canary: Dict[str, str] = {}       # member -> ON/OFF
        self.canary_source = ''                # 'flip 알림' | '타겟 추론'
        self.members: Dict[str, Dict[str, float]] = {}
        self.combined: Dict[str, float] = {}   # 코인 비중 (cash 포함)
        self.result_kind = 'unknown'           # ok | skip | warn | error | unknown
        self.result_label = '기록 없음'
        self.issues: List[str] = []            # warn/error 마커 누적
        self.problems: List[str] = []
        self.start_count = 0                   # 그 날 실행 시작 횟수
        self.abort_hints: List[str] = []       # 시작 로그 없이 끝난 원인 후보
        self.cycle_id: Optional[str] = None    # 래퍼가 주입한 사이클 ID (있는 쪽만)

    # ─── 파싱 ───
    def parse(self, day: str):
        if not os.path.exists(self.path):
            self.problems.append(f'로그 파일 없음: {os.path.basename(self.path)}')
            return
        self.log_exists = True
        # 자정(UTC) 로테이션으로 같은 실행이 파일 경계에서 갈릴 수 있으므로
        # active + 회전 파일을 항상 함께 읽어 시간순으로 병합한다 (m).
        records = _read_records(f'{self.path}.{day}', day) + _read_records(self.path, day)
        records.sort(key=lambda r: r[0])
        if not records:
            self.problems.append(f'{day} 실행 기록 없음')
            return

        # 마지막 "Executor 시작" 이후의 레코드만 사용 (하루 여러 번 실행 대비)
        start_idx = None
        for i, (_, msg, _rid) in enumerate(records):
            if RE_START.search(msg):
                start_idx = i
                self.start_count += 1
        if start_idx is None:
            self.problems.append(f'{day} 실행 시작 로그 없음 (부분 기록)')
            # 시작 전에 종료된 원인 후보 (health lock / flock 충돌 등)
            joined = '\n'.join(m for _, m, _r in records)
            for marker, hint in ABORT_HINTS:
                if marker in joined:
                    self.abort_hints.append(hint)
            recs = records
        else:
            recs = records[start_idx:]
            self.ran = True
            self.run_time = recs[0][0]
            rid = recs[0][2]
            if rid and rid != '--------':
                self.cycle_id = rid
            m = RE_START.search(recs[0][1])
            self.dry_run = (m.group(1) == 'True') if m else None

        for ts, msg, _rid in recs:
            first = msg.split('\n', 1)[0]
            m = RE_CANARY.search(first)
            if m and '카나리 플립' not in first:
                self.canary[m.group(1)] = m.group(2)
                self.canary_source = 'flip 알림'
            m = RE_COMBINED_TGT.match(first)
            if m:
                self.combined = _parse_weights(m.group(1), m.group(2))
                continue
            m = RE_MEMBER_TGT.match(first)
            if m:
                self.members[m.group(1)] = _parse_weights(m.group(2), m.group(3))
                continue

        # 카나리 알림은 flip 때만 찍히므로, 없으면 멤버 타겟에서 추론
        if not self.canary:
            src = self.members or ({'combined': self.combined} if self.combined else {})
            for mname, w in src.items():
                risky = sum(v for k, v in w.items() if k != 'Cash')
                self.canary[mname] = 'ON' if risky > 1e-6 else 'OFF'
            if self.canary:
                self.canary_source = '타겟 추론'

        # 실행 결과 분류 — high-watermark (가장 심각한 마커가 최종 판정, M8)
        best_sev = -1
        for ts, msg, _rid in recs:
            for marker, kind, label in RESULT_MARKERS:
                if marker not in msg:
                    continue
                sev = SEVERITY[kind]
                if kind in ('warn', 'error') and label not in self.issues:
                    self.issues.append(label)
                if sev > best_sev:
                    best_sev = sev
                    self.result_kind, self.result_label = kind, label
        if best_sev < 0 and self.ran:
            self.result_kind, self.result_label = 'unknown', '결과 마커 없음 (미완료?)'

    # ─── 표시 ───
    def coins(self) -> Dict[str, float]:
        return {k: v for k, v in self.combined.items() if k != 'Cash'}

    def canary_overall(self) -> Optional[str]:
        """멤버 키가 서로 다를 때 쓰는 종합 판정 (하나라도 ON 이면 ON)."""
        if not self.canary:
            return None
        return 'ON' if any(v == 'ON' for v in self.canary.values()) else 'OFF'

    def canary_str(self) -> str:
        if not self.canary:
            return '알 수 없음'
        body = ', '.join(f'{m}={s}' for m, s in sorted(self.canary.items()))
        return f'{body} ({self.canary_source})' if self.canary_source else body

    def target_str(self) -> str:
        c = self.coins()
        if not c and not self.combined:
            return '알 수 없음'
        if not c:
            return f'CASH only (cash={self.combined.get("Cash", 0)*100:.1f}%)'
        body = ', '.join(f'{k} {v*100:.1f}%' for k, v in sorted(c.items(), key=lambda kv: -kv[1]))
        return f'{body} (cash={self.combined.get("Cash", 0)*100:.1f}%)'


class FutResult:
    """바이낸스 선물(V25) dry-run/실거래 결과.

    현물과 로그 포맷·전략·자산이 달라 SideResult 와 분리한다(현물 파서는 그대로 둔다).
    비교 대상이 아니라 '그날 선물은 어떻게 돌았나'를 보여주는 용도다.
    """

    def __init__(self, name: str = FUT_NAME, path: str = FUT_LOG):
        self.name = name
        self.path = path
        self.log_exists = False
        self.ran = False
        self.run_time: Optional[str] = None
        self.run_id: Optional[str] = None
        self.dry_run: Optional[bool] = None    # True=dry / False=LIVE / None=불명
        self.canary: Dict[str, str] = {}       # 전략 -> ON/OFF
        self.canary_source = ''                # 'canary 로그' | '타겟 추론'
        self.members: Dict[str, Dict[str, float]] = {}   # 전략별 목표
        self.combined: Dict[str, float] = {}   # 합산 목표 (Cash 포함)
        self.leverage: Dict[str, int] = {}     # 심볼별 최종 L
        self.pv: Optional[float] = None
        self.result_kind = 'unknown'           # ok | warn | error | unknown
        self.result_label = '기록 없음'
        self.issues: List[str] = []            # 실행 블록 내 WARNING/ERROR 줄 (redaction 완료)
        self.issue_omitted = 0                 # 이슈 한도 초과로 생략한 건수
        self.issue_omitted_by_level: Dict[str, int] = {}
        self.problems: List[str] = []
        self.start_count = 0                   # 그 날 실행 시작 횟수
        self.abort_hints: List[str] = []
        self.target_rank = -1                  # combined 를 어느 소스에서 얻었나
        self.target_blocked_rank = -1          # 파싱 실패한 최고 권위 (그 이상만 복구 가능)
        self.target_parse_errors: List[str] = []   # 비정형 타겟 원문 (fail-loud)
        self.mode_source = ''                  # 'DRY 마커' | 'LIVE 마커' | '추론'
        self.outside_note = ''                 # 블록 밖 WARNING/ERROR 요약

    # ─── 파싱 ───
    def parse(self, day: str):
        if not os.path.exists(self.path):
            self.problems.append(f'로그 파일 없음: {os.path.basename(self.path)}')
            return
        self.log_exists = True
        # 선물 로그는 날짜별 로테이션이 없다 (한 파일에 계속 누적).
        records = _read_fut_records(self.path, day)
        if not records:
            self.problems.append(f'{day} 실행 기록 없음')
            return

        # 그 날의 마지막 실행 블록 = 마지막 '매매 시작' ~ 그 뒤 첫 '=== 완료'.
        # 마커는 레코드 첫 줄에만 앵커한다 (멀티라인 본문에 우연히 섞인 문자열 방지).
        starts = [i for i, r in enumerate(records) if RE_FUT_START.match(_first_line(r[2]))]
        self.start_count = len(starts)
        if not starts:
            self.problems.append(f'{day} 실행 시작 로그 없음 (부분 기록)')
            joined = '\n'.join(m for _t, _l, m in records)
            for marker, hint in FUT_ABORT_HINTS:
                if marker in joined:
                    self.abort_hints.append(hint)
            return

        start_idx = starts[-1]
        end_idx = len(records)
        for j in range(start_idx + 1, len(records)):
            if RE_FUT_DONE.match(_first_line(records[j][2])):
                end_idx = j + 1
                break
        recs = records[start_idx:end_idx]
        # 블록이 닫힌 뒤의 기록 = 시작 마커 없는 별도 실행(수동 --report/--status 등).
        # 분류엔 넣지 않고 있다는 사실만 알린다.
        self._note_outside(records[end_idx:])

        self.ran = True
        self.run_time = recs[0][0]
        m = RE_FUT_START.match(_first_line(recs[0][2]))
        self.run_id = (m.group(1) or None) if m else None

        seen_issues: List[Tuple[int, str]] = []   # (심각도 0=ERROR/1=WARNING, 텍스트)
        for _ts, level, msg in recs:
            first = _first_line(msg)
            if level in ('WARNING', 'ERROR', 'CRITICAL'):
                # 원시 로그 줄이 텔레그램 본문에 실리므로 수집 시점에 마스킹한다 (m2 redaction).
                text = _redact(first.strip())
                if len(text) > FUT_ISSUE_MAXLEN:
                    text = text[:FUT_ISSUE_MAXLEN - 1] + '…'
                item = (0 if level in ('ERROR', 'CRITICAL') else 1, f'{level} {text}')
                if item not in seen_issues:
                    seen_issues.append(item)

            mc = RE_FUT_CANARY.match(first)
            if mc is None:
                mk = RE_FUT_CANARY_KO.search(first)
                mc = mk if (mk and '카나리 플립' not in first) else None
            if mc:
                self.canary[mc.group(1)] = mc.group(2)
                self.canary_source = 'canary 로그'

            if self._take_target(first, RE_FUT_COMBINED, FUT_TGT_COMBINED):
                continue
            if self._take_target(first, RE_FUT_REFILL, FUT_TGT_REFILL):
                continue
            if self._take_target(first, RE_FUT_DRYRUN_TGT, FUT_TGT_DRYRUN):
                continue
            if self._take_target(first, RE_FUT_V25_TGT, FUT_TGT_V25, floats=True):
                continue
            mm = RE_FUT_MEMBER_TGT.match(first)
            if mm:
                w = _parse_fut_weights(mm.group(2), mm.group(3))
                if w is None:
                    # cash 까지 함께 보여야 왜 실패했는지 읽힌다 ('{}' + cash=20% 등)
                    self._note_target_error(f'{mm.group(1)} → {mm.group(2)} (cash={mm.group(3)}%)')
                else:
                    self.members[mm.group(1)] = w
                continue
            ml = RE_FUT_LEV.match(first)
            if ml:
                self.leverage[ml.group(1)] = int(ml.group(2))
                continue
            mp = RE_FUT_PV_AFTER.search(first)   # 실거래면 사후 PV 가 최종값
            if mp is None and self.pv is None:
                mp = RE_FUT_PV.search(first)
            if mp:
                try:
                    self.pv = float(mp.group(1).replace(',', ''))
                except ValueError:
                    pass

        # 이슈 절단 — ERROR/CRITICAL 을 먼저 살리고(같은 등급 내 시간순), 생략분은 등급별로 센다
        ordered = [t for _sev, t in sorted(seen_issues, key=lambda it: it[0])]
        self.issues = ordered[:FUT_ISSUE_LIMIT]
        dropped = ordered[FUT_ISSUE_LIMIT:]
        self.issue_omitted = len(dropped)
        for text in dropped:
            lvl = text.split(' ', 1)[0]
            self.issue_omitted_by_level[lvl] = self.issue_omitted_by_level.get(lvl, 0) + 1

        joined = '\n'.join(m for _t, _l, m in recs)
        done = bool(RE_FUT_DONE.match(_first_line(recs[-1][2]))) if recs else False

        # 실행 모드 — dry 마커 > 실거래 흔적 > 추론.
        # (실코드: 실거래는 시작 직후 '시작 지연:' 를 반드시 남긴다. 완주했는데 그 흔적이
        #  하나도 없으면 dry-run 이다 — rebalancing_needed=false 인 조용한 날 대비.)
        if any(h in joined for h in FUT_DRY_HINTS):
            self.dry_run, self.mode_source = True, 'DRY 마커'
        elif any(h in joined for h in FUT_LIVE_HINTS):
            self.dry_run, self.mode_source = False, 'LIVE 마커'
        elif done:
            self.dry_run, self.mode_source = True, '추론'

        # 카나리 로그가 없는 날(상태 불변)엔 목표에서 추론 — 현물과 같은 철학.
        # 단 타겟 파싱이 실패했으면 추론 근거가 없으므로 '알 수 없음' 으로 남긴다.
        if not self.canary and not self.target_parse_errors:
            src = self.members or ({'합산': self.combined} if self.combined else {})
            for mname, w in src.items():
                risky = sum(v for k, v in w.items() if k != 'Cash')
                self.canary[mname] = 'ON' if risky > 1e-6 else 'OFF'
            if self.canary:
                self.canary_source = '타겟 추론'

        # 실행 결과 분류 — high-watermark (가장 심각한 마커가 최종 판정)
        best = [-1]

        def _bump(kind: str, label: str):
            if SEVERITY[kind] > best[0]:
                best[0] = SEVERITY[kind]
                self.result_kind, self.result_label = kind, label

        # 마커도 레코드 첫 줄에만 앵커한다. 멀티라인 본문(트레이스백/이전 로그 재현)에
        # 섞인 '=== 완료' 같은 문자열이 실행 결과를 뒤집으면 안 된다.
        matched: set = set()
        for i, (_ts, _level, msg) in enumerate(recs):
            first = _first_line(msg)
            for marker, kind, label in FUT_RESULT_MARKERS:
                if marker in first:
                    matched.add(i)
                    _bump(kind, label)
        # 완주 판정은 부분일치 마커가 아니라 앵커된 '=== 완료' 행으로만 한다
        # (일반 로그가 그 문구를 인용해도 '정상 완료' 가 되면 안 된다).
        if done:
            matched.add(len(recs) - 1)
            _bump('ok', '정상 완료')
        # 'V25 fut: ... success=False' 는 executor 자신의 최종 판정이라 권위가 있다
        for _ts, _level, msg in recs:
            ms = RE_FUT_V25_SUCCESS.match(_first_line(msg))
            if ms and ms.group(1) == 'False':
                _bump('error', 'V25 success=False (executor 자체 판정)')
        # 마커가 못 잡은 실패가 '정상 완료' 로 남지 않게 레벨로 최소 등급을 올린다.
        # 마커가 이미 분류한 줄은 제외한다 — 의도적 등급(예: 조회 실패는 log.error 지만
        # '거래 없이 스킵' 이라 warn)을 레벨이 덮어버리면 마커 테이블이 무의미해진다.
        unmatched = {l for i, (_t, l, _m) in enumerate(recs) if i not in matched}
        if unmatched & {'ERROR', 'CRITICAL'}:
            _bump('error', '오류 로그 있음 (마커 미분류)')
        elif 'WARNING' in unmatched:
            _bump('warn', '경고 로그 있음 (마커 미분류)')
        if self.target_parse_errors:
            _bump('warn', '타겟 파싱 실패')
        # 모드 불명은 '분류가 있었는데 모드만 모를 때' 만 승격한다.
        # 마커가 아예 없으면(=중간에 끊긴 실행) 아래 unknown 이 더 정확하다.
        if self.dry_run is None and best[0] >= 0:
            _bump('warn', '실행 모드 불명 (dry/LIVE 판별 불가)')
        if best[0] < 0:
            self.result_kind, self.result_label = 'unknown', '결과 마커 없음 (미완료?)'

    # ─── 파싱 보조 ───
    def _take_target(self, first: str, rx, rank: int, floats: bool = False) -> bool:
        """타겟 후보 줄이면 (권위 순위가 더 높거나 같을 때만) combined 를 갱신한다.

        상위 권위 줄이 깨져 있으면 이미 받아둔 하위 권위 값도 못 믿는다 —
        그 값을 그대로 보고하면 '깨진 최종 타겟' 을 정상인 척 싣는 셈이다.
        그래서 실패 rank 이상만 이후 복구할 수 있게 차단점을 남기고 combined 를 비운다.
        """
        m = rx.match(first)
        if not m:
            return False
        raw = m.group(1)
        w = _parse_fut_floats(raw) if floats else _parse_fut_weights(raw)
        if w is None:
            self._note_target_error(raw)
            if rank >= self.target_rank:
                self.combined, self.target_rank = {}, -1
                self.target_blocked_rank = max(self.target_blocked_rank, rank)
            return True
        if rank >= self.target_rank and rank >= self.target_blocked_rank:
            self.combined, self.target_rank = w, rank
        return True

    def _note_target_error(self, raw: str):
        text = _redact((raw or '').strip())
        if len(text) > FUT_ISSUE_MAXLEN:
            text = text[:FUT_ISSUE_MAXLEN - 1] + '…'
        msg = f'타겟 파싱 실패: {text}'
        if msg not in self.target_parse_errors:
            self.target_parse_errors.append(msg)
            self.problems.append(msg)

    def _note_outside(self, tail: List[Tuple[str, str, str]]):
        err = sum(1 for _t, l, _m in tail if l in ('ERROR', 'CRITICAL'))
        wrn = sum(1 for _t, l, _m in tail if l == 'WARNING')
        if not (err or wrn):
            return
        detail = ', '.join(p for p in (f'ERROR {err}' if err else '',
                                       f'WARNING {wrn}' if wrn else '') if p)
        self.outside_note = f'블록 외 로그 {err + wrn}건 ({detail}) — 분류엔 미반영'

    # ─── 표시 ───
    def coins(self) -> Dict[str, float]:
        return {k: v for k, v in self.combined.items() if k != 'Cash'}

    def mode_str(self) -> str:
        if self.dry_run is None:
            return '?'
        base = 'dry' if self.dry_run else 'LIVE'
        return f'{base}·추론' if self.mode_source == '추론' else base

    def target_src_str(self) -> str:
        """타겟 출처 — refill/실거래 확정처럼 '합산'과 달라질 수 있을 때만 붙인다."""
        if self.target_rank in (FUT_TGT_REFILL, FUT_TGT_V25):
            return f' ({FUT_TGT_LABELS[self.target_rank]})'
        return ''

    def canary_str(self) -> str:
        if not self.canary:
            return '알 수 없음'
        body = ', '.join(f'{m}={s}' for m, s in sorted(self.canary.items()))
        return f'{body} ({self.canary_source})' if self.canary_source else body

    def canary_visible(self) -> bool:
        """평상시(전 멤버 ON)엔 숨기고 OFF/판별 불가만 알린다.

        '타겟 추론' 으로 나온 ON 도 매일 뜨는 평상 상태라 숨긴다.
        타겟 파싱 실패로 추론을 못 한 날은 canary 가 비어 '알 수 없음' 이라 표시된다.
        """
        return (not self.canary) or any(v != 'ON' for v in self.canary.values())

    def target_str(self) -> str:
        c = self.coins()
        cash = self.combined.get('Cash', 0.0)
        if not c and not self.combined:
            return '알 수 없음'
        if not c:
            return f'CASH only (cash {cash*100:.1f}%)'
        parts = []
        for k, v in sorted(c.items(), key=lambda kv: -kv[1]):
            lev = self.leverage.get(k)
            parts.append(f'{k} {v*100:.1f}%' + (f' (L{lev})' if lev is not None else ''))
        parts.append(f'cash {cash*100:.1f}%')
        return ', '.join(parts)

    def members_str(self) -> str:
        out = []
        for mname, w in sorted(self.members.items()):
            coins = {k: v for k, v in w.items() if k != 'Cash'}
            body = ', '.join(f'{k} {v*100:.1f}%'
                             for k, v in sorted(coins.items(), key=lambda kv: -kv[1])) or 'CASH only'
            out.append(f'{mname} → {body}')
        return ' | '.join(out)


def _first_line(msg: str) -> str:
    return msg.split('\n', 1)[0]


def _read_fut_records(path: str, day: str) -> List[Tuple[str, str, str]]:
    """선물 로그의 해당 날짜 레코드만 [(HH:MM:SS, LEVEL, message), ...] 로 반환.

    타임스탬프 없는 줄(멀티라인 메시지 / 래퍼가 리다이렉트한 stderr)은 직전 레코드에 붙인다.
    로테이션이 없어 파일이 계속 커지므로 통째로 읽지 않고 한 줄씩 흘려 읽는다.
    """
    out: List[Tuple[str, str, str]] = []
    cur_day = None
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.rstrip('\n').rstrip('\r')
                m = RE_FUT_LINE.match(line)
                if m:
                    cur_day = m.group(1)
                    if cur_day == day:
                        out.append((m.group(2), m.group(3), m.group(4)))
                elif out and cur_day == day:
                    ts, level, msg = out[-1]
                    out[-1] = (ts, level, msg + '\n' + line)
    except OSError:
        return []
    return out


def _entry_count(body: str) -> int:
    """dict 리터럴의 최상위 항목 수. 선물 타겟 값엔 콤마가 없어 단순 분할로 충분하다."""
    inner = body.strip()[1:-1].strip()
    if not inner:
        return 0
    return len([p for p in inner.split(',') if p.strip()])


def _validate_weights(w: Dict[str, float]) -> bool:
    """개별 비중 [0,1] + 합 <= 1+tol. 위반이면 조용히 넘기지 않고 실패로 본다."""
    for v in w.values():
        if not (0.0 <= v <= 1.0):
            return False
    return sum(w.values()) <= 1.0 + FUT_WEIGHT_SUM_TOL


def _parse_fut_weights(body: str, cash_pct: Optional[str] = None) -> Optional[Dict[str, float]]:
    """"{'BTC': '33.3%', 'ETH': '33.3%'}" → {'BTC':0.333, 'ETH':0.333, 'Cash':0.334}

    선물 로그는 종목 비중만 dict 로 찍고 cash 는 합산 라인에 없다 → 1-합 으로 채운다.
    무위험(CASH only)은 '{}' / 'CASH' / 'CASH 100%' 정확히 일치할 때만 인정한다.
    포맷이 조금이라도 어긋나면(따옴표 종류, '%' 누락, 부분 매칭) None 을 돌려
    호출부가 '타겟 파싱 실패' 로 시끄럽게 처리하게 한다 — 조용한 CASH-only 오보 방지.
    """
    raw = (body or '').strip()
    cash = None
    if cash_pct is not None:
        try:
            cash = float(cash_pct) / 100.0
        except (TypeError, ValueError):
            return None
        if not (0.0 <= cash <= 1.0):
            return None
    if raw in FUT_CASH_SENTINELS:
        # 종목이 없다고 찍혔는데 cash 가 100% 가 아니면 두 값이 서로 어긋난다
        if cash is not None and abs(cash - 1.0) > FUT_WEIGHT_SUM_TOL:
            return None
        return {'Cash': 1.0}
    if not (raw.startswith('{') and raw.endswith('}')):
        return None
    w: Dict[str, float] = {}
    for sym, pct in RE_FUT_WEIGHT_TOKEN.findall(raw):
        try:
            w[sym] = float(pct) / 100.0
        except ValueError:
            return None
    # 토큰이 dict 항목을 전부 소비했는지 (부분 매칭이면 남은 종목을 놓친 것)
    if not w or len(w) != _entry_count(raw):
        return None
    if not _validate_weights(w):
        return None
    w['Cash'] = cash if cash is not None else max(0.0, round(1.0 - sum(w.values()), 6))
    # cash 를 넣은 뒤 다시 본다 — 종목합이 통과해도 cash 를 더하면 100% 를 넘길 수 있다
    if not _validate_weights(w):
        return None
    return w


def _parse_fut_floats(body: str) -> Optional[Dict[str, float]]:
    """"{'BTC': 0.333, 'Cash': 0.333}" → 같은 형태. 실거래 'V25 fut: targets=' 전용.

    executor(auto_trade_binance.py:3216)가 CASH 키를 'Cash' 로 바꾸고 1e-4 이하는 버린다.
    """
    raw = (body or '').strip()
    if raw == '{}':
        return {'Cash': 1.0}
    if not (raw.startswith('{') and raw.endswith('}')):
        return None
    w: Dict[str, float] = {}
    for sym, val in RE_FUT_FLOAT_TOKEN.findall(raw):
        try:
            w[sym] = float(val)
        except ValueError:
            return None
    if not w or len(w) != _entry_count(raw):
        return None
    if not _validate_weights(w):
        return None
    if 'Cash' not in w:
        w['Cash'] = max(0.0, round(1.0 - sum(w.values()), 6))
    if not _validate_weights(w):
        return None
    return w


def _read_records(path: str, day: str) -> List[Tuple[str, str, str]]:
    """해당 날짜 레코드만 [(HH:MM:SS, message, run_id), ...] 로 반환. 멀티라인 병합."""
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.read().splitlines()
    except OSError:
        return []
    out: List[Tuple[str, str, str]] = []
    cur_day = None
    for line in lines:
        m = RE_LINE.match(line)
        if m:
            cur_day = m.group(1)
            if cur_day == day:
                out.append((m.group(2), m.group(4), m.group(3)))
        elif out and cur_day == day:
            ts, msg, rid = out[-1]
            out[-1] = (ts, msg + '\n' + line, rid)
    return out


def _in_cron_window(t: Optional[str]) -> bool:
    """로그 시각(서버 로컬=UTC)이 KST cron 창 안인가."""
    try:
        h, m, sec = (int(x) for x in (t or '').split(':'))
    except Exception:
        return False
    kst = ((h + LOG_TZ_OFFSET_HOURS) % 24) * 3600 + m * 60 + sec
    return CRON_WINDOW_KST[0] <= kst <= CRON_WINDOW_KST[1]


def _parse_weights(coins_part: str, cash_pct: str) -> Dict[str, float]:
    """'BTC:33.3%, ETH:33.3%' + cash '0.0' → {'BTC':0.333, 'ETH':0.333, 'Cash':0.0}"""
    w: Dict[str, float] = {}
    if coins_part and 'CASH only' not in coins_part:
        for sym, pct in RE_TOKEN.findall(coins_part):
            w[sym] = float(pct) / 100.0
    try:
        w['Cash'] = float(cash_pct) / 100.0
    except (TypeError, ValueError):
        w['Cash'] = 0.0
    return w


def _time_gap_sec(t1: Optional[str], t2: Optional[str]) -> Optional[int]:
    """'HH:MM:SS' 두 개의 절대 차이(초). 파싱 실패 시 None."""
    def _sec(t):
        try:
            h, m, s = (int(x) for x in t.split(':'))
            return h * 3600 + m * 60 + s
        except Exception:
            return None
    a, b = _sec(t1 or ''), _sec(t2 or '')
    if a is None or b is None:
        return None
    return abs(a - b)


def _service_today() -> str:
    """KST(Asia/Seoul) 기준 오늘 날짜."""
    if ZoneInfo is not None:
        try:
            return datetime.now(ZoneInfo(KST_TZ)).strftime('%Y-%m-%d')
        except Exception:
            pass
    return datetime.now().strftime('%Y-%m-%d')


# ─── 업비트 실계좌 평가액 / 원금 ───
def _kst_tz():
    """KST tzinfo — tzdata 가 없는 환경에서도 UTC+9 고정 오프셋으로 항상 tz-aware."""
    if ZoneInfo is not None:
        try:
            return ZoneInfo(KST_TZ)
        except Exception:
            pass
    return KST_FIXED


def _now_kst() -> datetime:
    """KST(Asia/Seoul) 기준 현재 시각 (항상 tz-aware)."""
    return datetime.now(_kst_tz())


def _parse_iso(s: str) -> Optional[datetime]:
    """ISO8601 → tz-aware datetime. tz 표기가 없으면 KST 로 본다 (업비트 시각 규약)."""
    try:
        dt = datetime.fromisoformat(str(s).replace('Z', '+00:00'))
    except (TypeError, ValueError):
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=_kst_tz())


def _upbit_get(path: str, params: Dict[str, object]) -> Tuple[int, object]:
    """업비트 private GET (JWT 인증). (status, json) 반환.

    private HTTP 는 여기 한 곳에만 둔다 — 테스트는 이 함수만 갈아끼우면 네트워크를 타지 않는다.
    pyupbit 를 쓰지 않는 이유: 그쪽 내부 requests 호출엔 timeout 이 없어, 업비트가 응답을
    안 주면 리포트 cron 이 통째로 매달린다. 여기선 UPBIT_HTTP_TIMEOUT 을 반드시 건다.
    jwt(PyJWT) 는 requirements.txt 에 없지만 pyupbit 의 의존성이라 함께 깔려 있다.
    그래서 import 는 함수 안에서만 한다 (없는 환경에서도 모듈 import 는 살아 있어야 한다).
    """
    import hashlib
    import urllib.parse
    import urllib.request
    import uuid

    import jwt

    query = urllib.parse.urlencode(params, doseq=True)
    payload = {'access_key': UPBIT_ACCESS_KEY, 'nonce': str(uuid.uuid4())}
    if query:
        payload['query_hash'] = hashlib.sha512(query.encode()).hexdigest()
        payload['query_hash_alg'] = 'SHA512'
    token = jwt.encode(payload, UPBIT_SECRET_KEY, algorithm='HS256')
    if isinstance(token, bytes):        # PyJWT 1.x 는 bytes 를 준다
        token = token.decode()
    url = f'{UPBIT_API}{path}' + (f'?{query}' if query else '')
    req = urllib.request.Request(url, headers={'Authorization': f'Bearer {token}'})
    with urllib.request.urlopen(req, timeout=UPBIT_HTTP_TIMEOUT) as resp:
        status = getattr(resp, 'status', None) or resp.getcode()
        return status, json.loads(resp.read().decode('utf-8'))


def _upbit_public_get(path: str, params: Dict[str, object]) -> Tuple[int, object]:
    """업비트 public GET (인증 없음). (status, json) 반환 — private 과 같은 timeout."""
    import urllib.parse
    import urllib.request

    query = urllib.parse.urlencode(params, doseq=True)
    url = f'{UPBIT_API}{path}' + (f'?{query}' if query else '')
    with urllib.request.urlopen(urllib.request.Request(url),
                                timeout=UPBIT_HTTP_TIMEOUT) as resp:
        status = getattr(resp, 'status', None) or resp.getcode()
        return status, json.loads(resp.read().decode('utf-8'))


def _upbit_prices(currencies: List[str]) -> Dict[str, float]:
    """보유 코인 현재가를 배치 1회로 조회한다 (market -> 현재가).

    /v1/ticker 는 markets 를 콤마로 이어 한 번에 물어볼 수 있다 — 심볼 수만큼 때리면
    레이트리밋에 걸린다. 비상장(KRW 마켓 없음) 심볼이 섞이면 응답에서 빠지거나 400 이
    나는데, 어느 쪽이든 '없는 가격'으로 취급한다 (호출부가 0 + 메모로 fail-loud).
    """
    markets = ','.join(f'KRW-{c}' for c in currencies)
    try:
        status, body = _upbit_public_get('/v1/ticker', {'markets': markets})
    except Exception:
        return {}
    if status != 200 or not isinstance(body, list):
        return {}
    out: Dict[str, float] = {}
    for t in body:
        if not isinstance(t, dict):
            continue
        market = str(t.get('market') or '')
        try:
            price = float(t.get('trade_price'))
        except (TypeError, ValueError):
            continue
        if market and price > 0:
            out[market] = price
    return out


def _upbit_account_value() -> Tuple[Optional[float], str]:
    """업비트 실계좌 평가액(KRW). (총액, 메모) — 조회 자체가 실패하면 (None, 사유).

    KRW 잔고는 그대로, 코인은 (balance + locked) × 현재가. 현재가를 못 받은 심볼은
    0 으로 두되 메모에 심볼명을 남긴다 — 조용히 빠지면 등락률이 거짓말이 된다 (fail-loud).
    가격 배치 조회가 통째로 실패해도 KRW 잔고만이라도 돌려주고 심볼을 전부 메모에 싣는다.
    """
    if not (UPBIT_ACCESS_KEY and UPBIT_SECRET_KEY):
        return None, '업비트 키 미설정'
    try:
        status, body = _upbit_get('/v1/accounts', {})
    except Exception as e:
        return None, f'잔고 조회 실패: {_redact(e)[:60]}'
    if status != 200 or not isinstance(body, list):
        return None, f'잔고 조회 실패 (HTTP {status})'

    total, bad = 0.0, []
    holdings: Dict[str, float] = {}
    for b in body:
        if not isinstance(b, dict):
            continue
        cur = str(b.get('currency') or '').upper()
        try:
            amt = float(b.get('balance') or 0) + float(b.get('locked') or 0)
        except (TypeError, ValueError):
            bad.append(cur or '?')
            continue
        if amt <= 0:
            continue
        if cur == 'KRW':
            total += amt
        else:
            holdings[cur] = holdings.get(cur, 0.0) + amt

    if holdings:
        prices = _upbit_prices(sorted(holdings))
        for cur, amt in sorted(holdings.items()):
            price = prices.get(f'KRW-{cur}')
            if not price:
                bad.append(cur)
                continue
            total += amt * price
    err = f'가격 조회 실패: {", ".join(sorted(set(bad)))}' if bad else ''
    return total, err


def _save_principal(data: Dict[str, object]) -> None:
    """원금 상태 원자적 저장 (tmp + os.replace — 프로젝트 상태파일 규칙).

    tmp 이름에 pid 를 붙인다 — 같은 파일을 두 프로세스가 동시에 쓰면 서로의 tmp 를
    덮어써 절반짜리 상태가 replace 될 수 있다.
    """
    tmp = f'{PRINCIPAL_FILE}.tmp.{os.getpid()}'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, PRINCIPAL_FILE)


def _load_principal(now: datetime) -> Tuple[Optional[float], str]:
    """원금(KRW)을 읽고 새 입금분을 반영한다. (원금, 메모) — 못 쓰면 (None, 사유).

    상태파일 trade/report_principal.json:
      {"principal_krw": 2000000.0,
       "last_deposit_check": "2026-08-31T10:00:00+09:00",
       "processed_uuids": ["<그 시각에 이미 반영한 입금 uuid>", ...]}

    워터마크는 '지금 시각'이 아니라 **관측한 입금의 max(done_at || created_at)** 으로만
    전진한다. now 로 밀면, 반영이 늦어 done_at 이 과거인 채 나중에 나타나는 입금이
    영영 창밖으로 밀려 유실된다. 같은 시각에 여러 건이 들어오는 경우를 위해 워터마크
    시각의 입금 uuid 를 processed_uuids 에 남겨 중복 가산을 막는다.
    가산 조건: ts > last, 또는 (ts == last 이고 uuid 가 processed_uuids 에 없음).
    새 입금이 없으면 파일을 건드리지 않는다 (매일 같은 내용으로 쓰지 않는다).
    아직 시각이 오지 않은(now 보다 미래) 입금은 이번엔 세지 않는다 — 워터마크를 미래로
    밀면 그 사이 입금이 통째로 가려진다. 다음 실행에서 정상 반영된다.

    limit=100 은 개인 계좌 입금 빈도상 충분하다 (한 번도 안 돈 사이에 신규 입금이
    100건 넘게 쌓이면 초과분은 유실된다).

    주의: 출금은 조회 권한이 없어 자동 반영되지 않는다 —
          출금했을 땐 report_principal.json 의 principal_krw 를 수동으로 낮춰야 한다.
    """
    if not os.path.exists(PRINCIPAL_FILE):
        return None, '원금 파일 없음(report_principal.json)'
    try:
        with open(PRINCIPAL_FILE, encoding='utf-8') as f:
            data = json.load(f)
        principal = float(data['principal_krw'])
    except Exception as e:
        return None, f'원금 파일 읽기 실패(report_principal.json): {_redact(e)[:60]}'
    if principal <= 0:
        return None, '원금이 0 이하 (report_principal.json)'

    last = _parse_iso(data.get('last_deposit_check'))
    if last is None:
        return principal, '입금동기화 실패 (last_deposit_check 파싱 불가)'
    seen = {str(u) for u in (data.get('processed_uuids') or [])}
    try:
        status, body = _upbit_get('/v1/deposits',
                                  {'currency': 'KRW', 'limit': 100, 'order_by': 'desc'})
    except Exception as e:
        return principal, f'입금동기화 실패: {_redact(e)[:60]}'
    if status != 200 or not isinstance(body, list):
        return principal, f'입금동기화 실패 (HTTP {status})'

    # (ts, uuid, amount) — KRW·ACCEPTED·시각 파싱 성공분만
    deposits: List[Tuple[datetime, str, float]] = []
    for d in body:
        if not isinstance(d, dict):
            continue
        if str(d.get('currency') or '').upper() != 'KRW':
            continue
        if str(d.get('state') or '').upper() != 'ACCEPTED':
            continue
        ts = _parse_iso(d.get('done_at') or d.get('created_at'))
        if ts is None or ts < last or ts > now:
            continue
        try:
            amount = float(d.get('amount') or 0)
        except (TypeError, ValueError):
            continue
        # uuid 가 비면 중복 판정 키가 없어 다음 실행에 또 더해진다 — 대체 키를 만든다
        uid = str(d.get('uuid') or '') or f'{ts.isoformat()}#{amount}'
        deposits.append((ts, uid, amount))

    fresh = [(ts, uid, amt) for ts, uid, amt in deposits
             if ts > last or uid not in seen]
    if not fresh:
        return principal, ''

    principal += sum(amt for _ts, _uid, amt in fresh)
    watermark = max(ts for ts, _uid, _amt in fresh)
    # 워터마크 시각의 입금은 '이미 반영됨'으로 남긴다 (같은 시각 신규 건과 구분하려고)
    processed = {uid for ts, uid, _amt in deposits if ts == watermark and uid}
    if watermark == last:
        processed |= seen

    data['principal_krw'] = principal
    data['last_deposit_check'] = watermark.isoformat()
    data['processed_uuids'] = sorted(processed)
    try:
        _save_principal(data)
    except Exception as e:
        return principal, f'원금 파일 저장 실패: {_redact(e)[:60]}'
    return principal, ''


# ─── 리포트 ───
def _spot_canary_visible(sides: List[SideResult]) -> bool:
    """현물 블록에 카나리 줄을 실을지.

    평상시(양쪽 실행 + 전 멤버 ON + 일치)엔 매일 같은 줄이라 노이즈다 → 숨긴다.
    OFF 가 하나라도 있거나, 판별 불가거나, 불일치거나, 한쪽이 미실행이면
    판단 근거가 필요하므로 양쪽 다 보여준다 (fail-loud).
    """
    a, b = sides[0], sides[1]
    if not (a.ran and b.ran and a.canary and b.canary):
        return True
    if any(v != 'ON' for s in sides for v in s.canary.values()):
        return True
    # 전 멤버 ON 이면 아래는 항상 '일치' 지만, 판정 규칙을 그대로 남겨 둔다
    if set(a.canary) == set(b.canary):
        return any(a.canary[k] != b.canary[k] for k in a.canary)
    return a.canary_overall() != b.canary_overall()


def _mode_word(s: SideResult) -> str:
    """비교 섹션의 실행 모드 표기 — 판정이 아니라 정보 표시용."""
    return '?' if s.dry_run is None else ('dry-run' if s.dry_run else 'LIVE')


def _upbit_value_line(uv: Dict[str, object]) -> Tuple[str, bool]:
    """[업비트] 블록의 '평가액' 줄 (줄, 경고여부).

    uv = {'value': 평가액(KRW) | None, 'pct': 원금대비 등락률(%) | None, 'note': 메모}
    값이나 원금이 없으면 숫자를 지어내지 않고 사유를 싣는다 (fail-loud).
    """
    value, pct = uv.get('value'), uv.get('pct')
    note = str(uv.get('note') or '').strip()
    if value is None:
        return f'  평가액: 조회 실패 — {note or "사유 불명"}', True
    if pct is None:
        return f'  평가액: {value:,.0f}원 (원금 미설정 — {note or "사유 불명"})', True
    if note:   # 값은 나왔지만 일부 심볼 가격 누락 등 — 평가액이 과소일 수 있다
        return f'  평가액: {value:,.0f}원 ({pct:+.1f}%) — {note}', True
    return f'  평가액: {value:,.0f}원 ({pct:+.1f}%)', False


def _fut_lines(fut: 'FutResult') -> Tuple[List[str], bool]:
    """선물 블록 (본문 줄들, 경고여부). 현물과 달리 비교 없이 결과만 싣는다."""
    if not fut.ran:
        reason = '; '.join(fut.problems) or '실행 기록 없음'
        if fut.abort_hints:
            reason += f' (원인 후보: {", ".join(sorted(set(fut.abort_hints)))})'
        elif fut.log_exists:
            reason += ' (원인 후보: cron 미실행 / flock 충돌 / V25 lock — 로그에 흔적 없음)'
        return [f'\n[{fut.name}] ⚠️ 미실행 — {reason}'], True

    warn = fut.result_kind in ('error', 'warn', 'unknown')
    icon = RESULT_ICONS.get(fut.result_kind, '⚠️')
    multi = f' [{fut.start_count}회 실행, 마지막 사용]' if fut.start_count > 1 else ''
    rid = f' run={fut.run_id}' if fut.run_id else ''
    lines = [f'\n[{fut.name}] {fut.run_time} ({fut.mode_str()}){rid} {icon} {fut.result_label}{multi}']
    if fut.canary_visible():   # 평상시(전 멤버 ON)엔 생략 — OFF/판별 불가만 알린다
        lines.append(f'  카나리: {fut.canary_str()}')
    lines.append(f'  타겟: {fut.target_str()}{fut.target_src_str()}')
    if len(fut.members) > 1:   # 앙상블일 때만 (단일 전략이면 합산과 같아 중복)
        lines.append(f'  전략별: {fut.members_str()}')
    if fut.pv is not None:
        lines.append(f'  PV: ${fut.pv:,.2f}')
    for msg in fut.target_parse_errors:
        lines.append(f'  ⚠️ {msg}')
    if fut.issues:
        extra = ''
        if fut.issue_omitted:
            by = ', '.join(f'{k} {v}' for k, v in sorted(fut.issue_omitted_by_level.items()))
            extra = f' 외 {fut.issue_omitted}건({by})'
        lines.append(f'  이슈: {", ".join(fut.issues)}{extra}')
    if fut.outside_note:
        lines.append(f'  참고: {fut.outside_note}')
    return lines, warn


def build_report(day: str, sides: List[SideResult],
                 fut: Optional['FutResult'] = None,
                 upbit_value: Optional[Dict[str, object]] = None) -> Tuple[str, bool]:
    """(본문, 경고여부) 반환.

    sides = 현물 2축(업비트/바낸현물) — 서로 1:1 비교한다.
    fut   = 선물(V25) — 전략·자산이 달라 비교하지 않고 결과만 싣는다 (None 이면 생략).
    upbit_value = {'value': 평가액|None, 'pct': 등락률|None, 'note': 메모} — 업비트 블록에만
            '평가액' 줄을 더한다. None(미주입)이면 줄 자체가 없고 출력은 종전과 완전히 같다.

    블록의 '카나리' 줄은 이상할 때만 싣는다(_spot_canary_visible / FutResult.canary_visible).
    '── 현물 비교 ──' 의 카나리 일치/불일치 판정은 그대로 항상 나온다.
    """
    a, b = sides[0], sides[1]
    warn = False
    lines = [f'📊 일일 운용 리포트 {day} (KST 기준)']
    # 상대편 상태를 알아야 정해지므로 블록 렌더링 전에 한 번만 계산한다
    show_canary = _spot_canary_visible(sides)

    # 실행 모드(dry/LIVE)는 정합 판정 대상이 아니라 경고하지 않는다. 비교 대상은 전략
    # 산출물(타겟·카나리)이고, '업비트 LIVE + 바낸현물 dry-run' 이 정상 운영 조합이다
    # (2026-09-01 사용자 결정). 모드는 블록/비교 섹션에 정보로만 싣는다.

    for s in sides:
        mode = '?' if s.dry_run is None else ('dry' if s.dry_run else 'LIVE')
        if not s.ran:
            warn = True
            reason = '; '.join(s.problems) or '실행 기록 없음'
            if s.abort_hints:
                reason += f' (원인 후보: {", ".join(sorted(set(s.abort_hints)))})'
            elif s.log_exists:
                reason += ' (원인 후보: cron 미실행 / 래퍼 flock / health lock — 로그에 흔적 없음)'
            lines.append(f'\n[{s.name}] ⚠️ 미실행 — {reason}')
            # 미실행일수록 실계좌 상태 확인이 급하다 — 업비트면 평가액은 그래도 싣는다
            if upbit_value is not None and s is a:
                vline, _vwarn = _upbit_value_line(upbit_value)
                lines.append(vline)
            continue
        icon = RESULT_ICONS.get(s.result_kind, '⚠️')
        if s.result_kind in ('error', 'warn', 'unknown'):
            warn = True
        multi = f' [{s.start_count}회 실행, 마지막 사용]' if s.start_count > 1 else ''
        if multi:
            warn = True
        cyc = f' cycle={s.cycle_id}' if s.cycle_id else ''
        lines.append(f'\n[{s.name}] {s.run_time} ({mode}){cyc} {icon} {s.result_label}{multi}')
        if show_canary:
            lines.append(f'  카나리: {s.canary_str()}')
        lines.append(f'  타겟: {s.target_str()}')
        # 실계좌 평가액은 업비트(sides[0]) 블록에만 — 바낸현물/선물은 대상이 아니다
        if upbit_value is not None and s is a:
            vline, vwarn = _upbit_value_line(upbit_value)
            lines.append(vline)
            warn = warn or vwarn
        if s.issues:
            lines.append(f'  이슈: {", ".join(s.issues)}')

    if fut is not None:
        flines, fwarn = _fut_lines(fut)
        lines.extend(flines)
        warn = warn or fwarn

    # 아래는 현물 2축 전용 비교 — 선물은 전략·자산이 달라 비교하지 않는다.
    lines.append('\n── 현물 비교 ──')
    if not (a.ran and b.ran):
        lines.append('⚠️ 한쪽 미실행 → 비교 불가')
        return '\n'.join(lines), True

    # 0) 실행 쌍 신뢰성 (M)
    #    - 양쪽 모두 사이클 ID 가 있으면 ID 일치를 요구 (권위 판정)
    #    - 한쪽이라도 ID 가 없으면(업비트판은 수정 금지라 rid 를 못 찍음)
    #      '양쪽 시작이 cron 창(09:05~09:12 KST) 안 + 시간차 허용범위' 일 때만 ✅
    gap = _time_gap_sec(a.run_time, b.run_time)
    if a.cycle_id and b.cycle_id:
        if a.cycle_id == b.cycle_id:
            lines.append(f'✅ 실행 쌍 정합 (사이클 ID {a.cycle_id} 일치)')
        else:
            lines.append(f'⚠️ 실행 쌍 불일치 — 사이클 ID {a.cycle_id} vs {b.cycle_id}')
            warn = True
    elif gap is None:
        lines.append('⚠️ 실행 시각 파싱 불가 → 실행 쌍 확인 불가')
        warn = True
    else:
        in_win = _in_cron_window(a.run_time) and _in_cron_window(b.run_time)
        if in_win and gap <= PAIR_MAX_GAP_SEC:
            lines.append(f'✅ 실행 쌍 정합 (양쪽 cron 창 내, 시작 시각 차 {gap/60:.0f}분)')
        else:
            why = []
            if not _in_cron_window(a.run_time):
                why.append(f'{a.name} cron 창 밖({a.run_time})')
            if not _in_cron_window(b.run_time):
                why.append(f'{b.name} cron 창 밖({b.run_time})')
            if gap > PAIR_MAX_GAP_SEC:
                why.append(f'시작 시각 차 {gap/60:.0f}분 > {PAIR_MAX_GAP_SEC//60}분')
            lines.append('⚠️ 실행 쌍 확인 불가 — ' + ', '.join(why)
                         + f' (시작 시각 차 {gap/60:.0f}분)')
            warn = True

    # 0-2) 실행 모드 — 정합 판정 아님, 정보 표시만 (모드 불명 warn 은 파싱 단계 M1 이 담당)
    if a.dry_run == b.dry_run and a.dry_run is not None:
        lines.append(f'실행 모드: 양쪽 {_mode_word(a)}')
    else:
        lines.append(f'실행 모드: {a.name}={_mode_word(a)}, {b.name}={_mode_word(b)}')

    # 1) 카나리 — 멤버 키가 같으면 멤버별, 다르면 (추론 경로 차이) 종합 ON/OFF 로 비교
    if not a.canary or not b.canary:
        lines.append('⚠️ 카나리: 한쪽 이상 판별 불가')
        warn = True
    elif set(a.canary) == set(b.canary):
        diffs = [f'{k}: {a.canary[k]} vs {b.canary[k]}'
                 for k in sorted(a.canary) if a.canary[k] != b.canary[k]]
        if diffs:
            lines.append('⚠️ 카나리 불일치 — ' + ', '.join(diffs))
            warn = True
        else:
            lines.append('✅ 카나리 일치')
    else:
        oa, ob = a.canary_overall(), b.canary_overall()
        if oa == ob:
            lines.append(f'✅ 카나리 일치 (종합 {oa}; 멤버 키 상이 — {sorted(a.canary)} vs {sorted(b.canary)})')
        else:
            lines.append(f'⚠️ 카나리 불일치 (종합) — {a.name}={oa}, {b.name}={ob}')
            warn = True

    # 타겟 로그가 없는 실행(freshness 스킵 등)은 코인/비중 비교 불가
    if not a.combined or not b.combined:
        missing = ', '.join(s.name for s in sides if not s.combined)
        lines.append(f'⚠️ combined target 로그 없음/파싱 실패: {missing} → 코인·비중 비교 불가')
        return '\n'.join(lines), True

    # 2) 코인 집합
    ca, cb = set(a.coins()), set(b.coins())
    if ca != cb:
        only_a = ', '.join(sorted(ca - cb)) or '-'
        only_b = ', '.join(sorted(cb - ca)) or '-'
        lines.append(f'⚠️ 코인 집합 불일치 — {a.name}만: {only_a} / {b.name}만: {only_b}')
        warn = True
    elif not ca:
        lines.append('✅ 코인 집합 일치 (양쪽 CASH only)')
    else:
        lines.append(f'✅ 코인 집합 일치 ({len(ca)}종목)')

    # 3) 비중 오차
    wa, wb = a.combined, b.combined
    allk = set(wa) | set(wb)
    gaps = []
    for k in sorted(allk):
        d = abs(wa.get(k, 0.0) - wb.get(k, 0.0))
        if d > WEIGHT_TOL:
            gaps.append(f'{k} {d*100:.1f}%p')
    if gaps:
        lines.append(f'⚠️ 비중 오차 >{WEIGHT_TOL*100:.0f}%p — ' + ', '.join(gaps))
        warn = True
    else:
        maxd = max((abs(wa.get(k, 0.0) - wb.get(k, 0.0)) for k in allk), default=0.0)
        lines.append(f'✅ 비중 일치 (최대 오차 {maxd*100:.2f}%p)')

    return '\n'.join(lines), warn


def _upbit_value_for(day: str) -> Optional[Dict[str, object]]:
    """그 날 리포트에 실을 업비트 평가액 정보 (오늘이 아니면 None → 줄 생략).

    실계좌 조회는 '지금' 값이라 과거 --date 리포트에 붙이면 그 날 상태로 오해된다.
    원금 동기화가 상태파일을 쓰는 것도 오늘치 실행에서만 일어나야 한다.
    외부 API 라 통째로 감싼다 — 여기서 터져도 리포트 본체(로그 파싱분)는 나가야 한다.
    """
    if day != _service_today():
        return None
    try:
        value, verr = _upbit_account_value()
        principal, perr = _load_principal(_now_kst())
        pct = ((value - principal) / principal * 100
               if value is not None and principal else None)
        return {'value': value, 'pct': pct,
                'note': '; '.join(p for p in (verr, perr) if p)}
    except Exception as e:
        return {'value': None, 'pct': None, 'note': _redact(e)[:80]}


def main():
    parser = argparse.ArgumentParser(
        description='일일 운용 리포트 (업비트/바낸현물 비교 + 선물 결과)')
    parser.add_argument('--stdout', action='store_true', help='텔레그램 대신 표준출력')
    parser.add_argument('--date', default=None, help='대상 로그 날짜 YYYY-MM-DD (기본: KST 오늘)')
    parser.add_argument('--no-fut', action='store_true', help='선물 블록 생략')
    args = parser.parse_args()

    day = args.date or _service_today()

    sides = []
    for name, path in SIDES:
        s = SideResult(name, path)
        s.parse(day)
        sides.append(s)

    fut = None
    if not args.no_fut:
        fut = FutResult(FUT_NAME, FUT_LOG)
        fut.parse(day)

    body, warn = build_report(day, sides, fut, _upbit_value_for(day))
    if warn:
        body = WARN_HEADER + '\n' + body

    if args.stdout:
        print(body)
        return

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(body)
        print('\n(텔레그램 토큰/chat_id 없음 → 발송 생략)', file=sys.stderr)
        return
    try:
        _send_tg(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, body, prefix=TG_PREFIX)
    except Exception as e:
        print(f'텔레그램 전송 실패: {_redact(e)}', file=sys.stderr)


if __name__ == '__main__':
    main()
