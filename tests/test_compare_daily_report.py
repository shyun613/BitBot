"""compare_daily_report.py 선물(binance_trade.log) 파싱/리포트 테스트.

실행:
  python3 -m pytest tests/test_compare_daily_report.py -q
  python3 tests/test_compare_daily_report.py          # pytest 없는 환경 fallback

주의: 텔레그램 발송 경로(main / send_telegram)는 절대 타지 않는다.
      파싱/리포트 빌드 함수만 직접 호출하고, common.notify 는 스텁으로 갈아끼운다.
"""
import importlib.util
import json
import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_SCRIPT = os.path.join(_REPO, 'scripts', 'compare_daily_report.py')


def _load_module():
    """common.notify / config 스텁을 먼저 심고 스크립트를 모듈로 로드한다.

    스크립트 top-level 이 common.notify 를 import 하고 그쪽이 requests 를 요구하므로
    스텁 없이는 import 자체가 실패할 수 있다.
    """
    if 'common' not in sys.modules:
        pkg = types.ModuleType('common')
        pkg.__path__ = []  # 패키지처럼 보이게
        sys.modules['common'] = pkg
    notify = types.ModuleType('common.notify')

    def _never_send(*args, **kwargs):  # pragma: no cover - 호출되면 테스트 설계 오류
        raise AssertionError('테스트에서 텔레그램 발송이 호출됐다')

    notify.send_telegram = _never_send
    sys.modules['common.notify'] = notify

    cfg = types.ModuleType('config')
    cfg.TELEGRAM_BOT_TOKEN = 'bot000:STUB-NOT-A-REAL-TOKEN'
    cfg.TELEGRAM_CHAT_ID = '0'
    sys.modules['config'] = cfg

    spec = importlib.util.spec_from_file_location('compare_daily_report_under_test', _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


cdr = _load_module()

DAY = '2026-08-31'

# 2026-08-31 오라클 dry-run cron 실제 블록 (원문 그대로)
SAMPLE_BLOCK = """2026-08-31 09:05:32,177 INFO cash_buffer (fut): 2%
2026-08-31 09:05:32,178 INFO === 바이낸스 선물 매매 시작 (run_id=20260831_090532) ===
2026-08-31 09:05:33,234 INFO universe 갱신: 24개 (cg=40 listed=525) head=['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'SOLUSDT']
2026-08-31 09:05:33,234 INFO 데이터 수집...
2026-08-31 09:05:43,878 INFO BTC D spot override OK (last close=78,508)
2026-08-31 09:05:43,879 INFO 수집 완료: 1h 24개, D 24개 (11.7s)
2026-08-31 09:05:44,057 INFO 현재 PV: $0.00
2026-08-31 09:05:44,058 INFO   D_SMA42 → {'BTC': '33.3%', 'ETH': '33.3%'} (cash=33%)
2026-08-31 09:05:44,058 INFO 합산: {'BTC': '33.3%', 'ETH': '33.3%'}
2026-08-31 09:05:44,058 INFO V24 drift eval: ht=0.6600 threshold=0.03 fire=True enabled=True data_ok=True
2026-08-31 09:05:44,058 INFO refill v2 fut: 시작. strategies=1
2026-08-31 09:05:44,120 INFO   🔁 V24 refill v2 적용: {'BTC': '33.3%', 'ETH': '33.3%'} | top diffs=[]
2026-08-31 09:05:44,120 INFO   BTC_cap: prev_close=$77,682.00 SMA42=$67,927.56 ratio=1.1436 → L=4
2026-08-31 09:05:44,121 INFO   K2[BTC]: prev_close=77682.0000 SMA7=78651.8700 ratio=0.9877 → L=2
2026-08-31 09:05:44,121 INFO   BTC → final L = min(BTC_cap=4, K2=2) = 2
2026-08-31 09:05:44,121 INFO   K2[ETH]: prev_close=2415.5500 SMA7=2464.5857 ratio=0.9801 → L=2
2026-08-31 09:05:44,121 INFO   ETH → final L = min(BTC_cap=4, K2=2) = 2
2026-08-31 09:05:44,121 INFO DRY-RUN REBALANCE: {'BTC': '33.3%', 'ETH': '33.3%'}
2026-08-31 09:05:44,143 INFO === 완료 (12.0s) ===
"""


def _write(tmp_path, text, name='binance_trade.log'):
    p = os.path.join(str(tmp_path), name)
    with open(p, 'w', encoding='utf-8') as f:
        f.write(text)
    return p


def _fut(tmp_path, text):
    r = cdr.FutResult('선물', _write(tmp_path, text))
    r.parse(DAY)
    return r


def _approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


# ─────────────────────────── (a) 실제 블록 파싱 ───────────────────────────
def test_fut_parses_real_dry_run_block(tmp_path):
    r = _fut(tmp_path, SAMPLE_BLOCK)

    assert r.ran is True
    assert r.run_time == '09:05:32'          # 블록 시작 = '매매 시작' 줄
    assert r.run_id == '20260831_090532'
    assert r.start_count == 1
    assert r.dry_run is True and r.mode_str() == 'dry'
    assert r.pv == 0.0

    # 합산 목표: cash 는 1-합 으로 채운다
    assert sorted(r.combined) == ['BTC', 'Cash', 'ETH']
    assert _approx(r.combined['BTC'], 0.333)
    assert _approx(r.combined['ETH'], 0.333)
    assert _approx(r.combined['Cash'], 0.334)

    # 전략별 목표 (cash 는 로그 값 그대로)
    assert set(r.members) == {'D_SMA42'}
    assert _approx(r.members['D_SMA42']['BTC'], 0.333)
    assert _approx(r.members['D_SMA42']['Cash'], 0.33)

    # 심볼별 최종 레버리지 (BTC_cap / K2 중간 줄에 낚이면 안 된다)
    assert r.leverage == {'BTC': 2, 'ETH': 2}

    # 카나리 로그가 없는 날 → 타겟 추론
    assert r.canary == {'D_SMA42': 'ON'}
    assert r.canary_source == '타겟 추론'

    assert (r.result_kind, r.result_label) == ('ok', '정상 완료')
    assert r.issues == []

    tgt = r.target_str()
    assert 'BTC 33.3% (L2)' in tgt and 'ETH 33.3% (L2)' in tgt and 'cash 33.4%' in tgt


def test_fut_explicit_canary_line_wins(tmp_path):
    log = (
        "2026-08-31 09:05:32,178 INFO === 바이낸스 선물 매매 시작 (run_id=R1) ===\n"
        "2026-08-31 09:05:42,137 INFO   D_SMA42 BTC=$77,682 SMA(42)=$67,928 ratio=1.1436"
        " canary=ON  *** FLIPPED ***\n"
        "2026-08-31 09:05:44,058 INFO   D_SMA42 → {'BTC': '33.3%'} (cash=67%)\n"
        "2026-08-31 09:05:44,058 INFO 합산: {'BTC': '33.3%'}\n"
        "2026-08-31 09:05:44,121 INFO DRY-RUN REBALANCE: {'BTC': '33.3%'}\n"
        "2026-08-31 09:05:44,143 INFO === 완료 (12.0s) ===\n"
    )
    r = _fut(tmp_path, log)
    assert r.canary == {'D_SMA42': 'ON'}
    assert r.canary_source == 'canary 로그'


def test_fut_live_mode_detected(tmp_path):
    log = (
        "2026-08-31 00:05:32,178 INFO === 바이낸스 선물 매매 시작 (run_id=R1) ===\n"
        "2026-08-31 00:05:41,000 INFO 현재 PV: $100.00\n"
        "2026-08-31 00:05:44,058 INFO 합산: {'BTC': '33.3%'}\n"
        "2026-08-31 00:05:45,000 INFO ORDER BUY BTCUSDT qty=0.010: NEW\n"
        "2026-08-31 00:05:46,000 INFO 리밸런싱 완료: PV $100.00 → $101.50\n"
        "2026-08-31 00:05:47,000 INFO === 완료 (15.0s) ===\n"
    )
    r = _fut(tmp_path, log)
    assert r.dry_run is False and r.mode_str() == 'LIVE'
    assert (r.result_kind, r.result_label) == ('ok', '리밸런싱 완료')
    assert r.pv == 101.50   # 실거래면 리밸런싱 '후' PV 가 최종값


# ─────────────────── (b) 하루 2회 실행 → 마지막 블록 ────────────────────
def test_fut_uses_last_block_of_the_day(tmp_path):
    earlier = (
        "2026-08-31 02:06:32,478 INFO === 바이낸스 선물 매매 시작 (run_id=20260831_020632) ===\n"
        "2026-08-31 02:06:32,809 WARNING market cache 없음 binance_universe_cache.json\n"
        "2026-08-31 02:06:42,152 INFO   D_SMA42 → {'XRP': '50.0%'} (cash=50%)\n"
        "2026-08-31 02:06:42,152 INFO 합산: {'XRP': '50.0%'}\n"
        "2026-08-31 02:06:42,160 INFO DRY-RUN REBALANCE: {'XRP': '50.0%'}\n"
        "2026-08-31 02:06:42,181 INFO === 완료 (9.7s) ===\n"
    )
    # 전날 기록도 같은 파일에 남아 있다 (로테이션 없음) — 날짜 필터가 걸러야 한다
    prev_day = (
        "2026-08-30 09:05:00,000 INFO === 바이낸스 선물 매매 시작 (run_id=20260830_090500) ===\n"
        "2026-08-30 09:05:01,000 INFO 합산: {'DOGE': '99.0%'}\n"
        "2026-08-30 09:05:02,000 INFO === 완료 (2.0s) ===\n"
    )
    r = _fut(tmp_path, prev_day + earlier + SAMPLE_BLOCK)

    assert r.start_count == 2                      # 그 날 시작 횟수 (전날 제외)
    assert r.run_id == '20260831_090532'           # 마지막 블록
    assert r.run_time == '09:05:32'
    assert sorted(r.coins()) == ['BTC', 'ETH']     # 앞 블록의 XRP 가 섞이면 안 된다
    assert r.issues == []                          # 앞 블록의 WARNING 도 새면 안 된다


# ───────────────────────── (c) CASH only ─────────────────────────
def test_fut_cash_only_empty_dict(tmp_path):
    log = (
        "2026-08-31 00:05:32,178 INFO === 바이낸스 선물 매매 시작 (run_id=R1) ===\n"
        "2026-08-31 00:05:41,207 INFO 현재 PV: $1,234.56\n"
        "2026-08-31 00:05:42,000 INFO   D_SMA42 → CASH 100% (cash=100%)\n"
        "2026-08-31 00:05:42,001 INFO 합산: {}\n"
        "2026-08-31 00:05:42,002 INFO DRY-RUN REBALANCE: CASH\n"
        "2026-08-31 00:05:42,283 INFO === 완료 (10.1s) ===\n"
    )
    r = _fut(tmp_path, log)
    assert r.coins() == {}
    assert _approx(r.combined['Cash'], 1.0)
    assert r.canary == {'D_SMA42': 'OFF'}
    assert r.canary_source == '타겟 추론'
    assert r.pv == 1234.56
    assert r.target_str() == 'CASH only (cash 100.0%)'


def test_fut_cash_only_legacy_text(tmp_path):
    """구형 로그의 '합산: CASH 100%' 도 CASH only 로 읽힌다."""
    log = (
        "2026-08-31 00:05:31,048 INFO === 바이낸스 선물 매매 시작 (run_id=R1) ===\n"
        "2026-08-31 00:05:31,207 INFO   4h_S240_SN120 BTC=$77,377 SMA(240)=$78,016"
        " ratio=0.9918 canary=OFF\n"
        "2026-08-31 00:05:31,207 INFO   4h_S240_SN120 → CASH 100% (cash=100%)\n"
        "2026-08-31 00:05:31,208 INFO 합산: CASH 100%\n"
        "2026-08-31 00:05:31,208 INFO DRY-RUN REBALANCE: CASH\n"
        "2026-08-31 00:05:32,283 INFO === 완료 (22.9s) ===\n"
    )
    r = _fut(tmp_path, log)
    assert r.coins() == {}
    assert _approx(r.combined['Cash'], 1.0)
    assert r.canary == {'4h_S240_SN120': 'OFF'}
    assert r.canary_source == 'canary 로그'


# ──────────── (d) WARNING/ERROR 수집 + high-watermark 분류 ────────────
def test_fut_collects_issues_and_classifies_error(tmp_path):
    log = (
        "2026-08-31 00:05:32,178 INFO === 바이낸스 선물 매매 시작 (run_id=R1) ===\n"
        "2026-08-31 00:05:32,809 WARNING market cache 없음 binance_universe_cache.json\n"
        "2026-08-31 00:05:32,810 WARNING market cache 없음 binance_universe_cache.json\n"
        "2026-08-31 00:05:33,000 INFO 수집 완료: 1h 24개, D 24개 (0.8s)\n"
        "2026-08-31 00:05:33,100 ERROR BTC 데이터 누락! 매매 중단. 이전 포지션 유지.\n"
        "2026-08-31 00:05:33,200 INFO === 완료 (1.0s) ===\n"
    )
    r = _fut(tmp_path, log)

    # '=== 완료'(ok) 가 뒤에 와도 error 가 유지된다 (high-watermark)
    assert (r.result_kind, r.result_label) == ('error', '매매 중단')
    assert len(r.issues) == 2                      # 중복 제거
    assert r.issues[0].startswith('ERROR BTC 데이터 누락')       # ERROR 우선 (M3)
    assert r.issues[1].startswith('WARNING market cache 없음')
    assert r.issue_omitted == 0


def test_fut_issue_dedup_limit_and_truncation(tmp_path):
    long_tail = 'x' * 200
    lines = ["2026-08-31 00:05:32,178 INFO === 바이낸스 선물 매매 시작 (run_id=R1) ===\n"]
    for i in range(7):
        lines.append(f"2026-08-31 00:05:33,{100+i:03d} WARNING 경고{i} {long_tail}\n")
    lines.append("2026-08-31 00:05:34,000 INFO === 완료 (2.0s) ===\n")
    r = _fut(tmp_path, ''.join(lines))

    assert len(r.issues) == cdr.FUT_ISSUE_LIMIT == 5
    assert r.issue_omitted == 2
    body = r.issues[0][len('WARNING '):]
    assert len(body) == cdr.FUT_ISSUE_MAXLEN and body.endswith('…')


def test_fut_abort_and_lock_are_errors(tmp_path):
    log = (
        "2026-08-31 00:05:32,178 INFO === 바이낸스 선물 매매 시작 (run_id=R1) ===\n"
        "2026-08-31 00:05:32,300 ERROR V25 ABORT: lock 활성 — abort_streak=3. 수동 해제 필요\n"
    )
    r = _fut(tmp_path, log)
    assert r.result_kind == 'error'
    assert 'ABORT' in r.result_label


def test_fut_warn_marker_position_query_failure(tmp_path):
    log = (
        "2026-08-31 00:05:32,178 INFO === 바이낸스 선물 매매 시작 (run_id=R1) ===\n"
        "2026-08-31 00:05:33,000 ERROR 현재 포지션/PV 조회 실패. 이번 실행은 거래 없이 스킵.\n"
    )
    r = _fut(tmp_path, log)
    assert (r.result_kind, r.result_label) == ('warn', '포지션/PV 조회 실패')

    # 리밸런싱 후 조회 실패도 같은 warn 등급
    r2 = _fut(tmp_path, (
        "2026-08-31 00:05:32,178 INFO === 바이낸스 선물 매매 시작 (run_id=R1) ===\n"
        "2026-08-31 00:05:40,000 ERROR 리밸런싱 후 포지션 조회 실패. kill-switch/미달 판정 없이 종료.\n"
    ))
    assert (r2.result_kind, r2.result_label) == ('warn', '포지션 조회 실패')


def test_fut_started_but_never_finished_is_unknown(tmp_path):
    log = (
        "2026-08-31 00:05:32,178 INFO === 바이낸스 선물 매매 시작 (run_id=R1) ===\n"
        "2026-08-31 00:05:33,234 INFO 데이터 수집...\n"
    )
    r = _fut(tmp_path, log)
    assert r.ran is True
    assert r.result_kind == 'unknown'
    assert '결과 마커 없음' in r.result_label


# ──────────────────────── (e) 미실행 처리 ────────────────────────
def test_fut_missing_log_file(tmp_path):
    r = cdr.FutResult('선물', os.path.join(str(tmp_path), 'nope.log'))
    r.parse(DAY)
    assert r.ran is False and r.log_exists is False
    assert any('로그 파일 없음' in p for p in r.problems)
    lines, warn = cdr._fut_lines(r)
    assert warn is True and '미실행' in lines[0]


def test_fut_no_record_for_the_day(tmp_path):
    r = _fut(tmp_path, "2026-08-30 09:05:00,000 INFO === 바이낸스 선물 매매 시작 (run_id=R0) ===\n")
    assert r.ran is False and r.log_exists is True
    assert any('실행 기록 없음' in p for p in r.problems)


def test_fut_partial_record_without_start(tmp_path):
    r = _fut(tmp_path, "2026-08-31 00:05:32,300 ERROR V25 ABORT: lock 활성 — 수동 해제 필요\n")
    assert r.ran is False
    assert any('실행 시작 로그 없음' in p for p in r.problems)
    assert r.abort_hints  # 원인 후보를 남긴다


# ──────────────────── (f) build_report 통합 ────────────────────
SPOT_LOG = """[{day} 00:05:10] [{rid}] Executor 시작 (dry_run={dry})
{canary}[{day} 00:05:11] [{rid}] combined target: {tgt} (cash={cash}%)
[{day} 00:05:20] [{rid}] 거래 완료
"""
SPOT_SIDES = (('업비트', 'executor_coin.log'), ('바낸현물', 'executor_coin_binance.log'))


def _spot_canary(state):
    """'D_SMA42 카나리 ON|OFF' 한 줄 (SideResult 가 명시 상태로 읽는다)."""
    return f'[{DAY} 00:05:10] [0005abcd] D_SMA42 카나리 {state}\n' if state else ''


def _spot_sides(tmp_path, canary_a=None, canary_b=None,
                tgt='BTC:33.3%, ETH:33.3%', cash='33.4', dry=(True, True)):
    """dry=(업비트, 바낸현물) — False 면 'Executor 시작 (dry_run=False)' = LIVE 로 파싱된다."""
    sides = []
    for (name, fname), state, dry_run in zip(SPOT_SIDES, (canary_a, canary_b), dry):
        body = SPOT_LOG.format(day=DAY, rid='0005abcd', canary=_spot_canary(state),
                               tgt=tgt, cash=cash, dry=dry_run)
        path = _write(tmp_path, body, fname)
        s = cdr.SideResult(name, path)
        s.parse(DAY)
        sides.append(s)
    return sides


def test_build_report_has_three_sections_and_spot_compare(tmp_path):
    sides = _spot_sides(tmp_path)
    fut = _fut(tmp_path, SAMPLE_BLOCK)
    body, warn = cdr.build_report(DAY, sides, fut)

    assert body.startswith('📊 일일 운용 리포트 2026-08-31 (KST 기준)')
    assert '[업비트]' in body and '[바낸현물]' in body and '[선물]' in body
    assert '\n── 현물 비교 ──' in body        # 헤더 변경
    assert '\n── 비교 ──' not in body         # 옛 헤더는 남지 않는다
    # 현물 비교 로직은 그대로 — 선물이 끼어들지 않는다
    assert '✅ 실행 쌍 정합' in body
    assert '✅ 코인 집합 일치' in body
    assert '✅ 비중 일치' in body
    # 선물 블록 내용
    assert 'run=20260831_090532' in body
    assert 'BTC 33.3% (L2)' in body
    assert 'PV: $0.00' in body
    assert warn is False


def test_build_report_warns_when_fut_errors(tmp_path):
    sides = _spot_sides(tmp_path)
    fut = _fut(tmp_path, (
        "2026-08-31 00:05:32,178 INFO === 바이낸스 선물 매매 시작 (run_id=R1) ===\n"
        "2026-08-31 00:05:33,100 ERROR BTC 데이터 누락! 매매 중단. 이전 포지션 유지.\n"
    ))
    body, warn = cdr.build_report(DAY, sides, fut)
    assert warn is True
    assert '🚨 매매 중단' in body

    rendered = (cdr.WARN_HEADER + '\n' + body) if warn else body   # main() 과 같은 조립
    assert rendered.startswith('⚠️ 점검 필요')


def test_build_report_warns_when_fut_missing(tmp_path):
    sides = _spot_sides(tmp_path)
    fut = cdr.FutResult('선물', os.path.join(str(tmp_path), 'nope.log'))
    fut.parse(DAY)
    body, warn = cdr.build_report(DAY, sides, fut)
    assert warn is True
    assert '[선물] ⚠️ 미실행' in body


def test_build_report_without_fut_is_backward_compatible(tmp_path):
    sides = _spot_sides(tmp_path)
    body, warn = cdr.build_report(DAY, sides)
    assert '[선물]' not in body
    assert '\n── 현물 비교 ──' in body
    assert warn is False


# ──────────────────── (g) 멀티라인 메시지 ────────────────────
def test_fut_multiline_message_is_joined(tmp_path):
    log = (
        "2026-08-31 00:05:32,178 INFO === 바이낸스 선물 매매 시작 (run_id=R1) ===\n"
        "2026-08-31 00:05:33,000 ERROR 치명 오류: Traceback (most recent call last):\n"
        '  File "auto_trade_binance.py", line 3210, in main\n'
        "    execute_rebalance(client, combined, pv)\n"
        "ValueError: boom\n"
        "2026-08-31 00:05:34,000 INFO === 완료 (2.0s) ===\n"
    )
    path = _write(tmp_path, log)
    recs = cdr._read_fut_records(path, DAY)
    assert len(recs) == 3                                  # 3 레코드로 병합
    ts, level, msg = recs[1]
    assert (ts, level) == ('00:05:33', 'ERROR')
    assert 'ValueError: boom' in msg and msg.count('\n') == 3

    r = cdr.FutResult('선물', path)
    r.parse(DAY)
    assert (r.result_kind, r.result_label) == ('error', '치명 오류')
    # 이슈는 첫 줄만 싣는다 (traceback 본문으로 리포트가 넘치면 안 된다)
    assert r.issues == ['ERROR 치명 오류: Traceback (most recent call last):']


# ═══════════ Codex 리뷰 라운드 1 반영분 (C1~m2) ═══════════

# ── C1: 최종 타겟 권위 체인 ──
def test_c1_refill_overrides_stale_combined_and_dryrun(tmp_path):
    """refill 로 종목이 교체된 날 stale 타겟(합산/DRY-RUN)을 보고하면 안 된다.

    executor 실코드(auto_trade_binance.py:2798,3185): coins_combined 는 refill '전'에
    한 번 만들어지고 'DRY-RUN REBALANCE' 가 그걸 그대로 재사용한다 → DRY-RUN 줄은
    refill 을 덮으면 안 된다(코디네이터 초안의 순서와 반대).
    """
    log = (
        "2026-08-31 00:05:32,178 INFO === 바이낸스 선물 매매 시작 (run_id=R1) ===\n"
        "2026-08-31 00:05:44,058 INFO   D_SMA42 → {'BTC': '33.3%', 'ETH': '33.3%'} (cash=33%)\n"
        "2026-08-31 00:05:44,058 INFO 합산: {'BTC': '33.3%', 'ETH': '33.3%'}\n"
        "2026-08-31 00:05:44,120 INFO   🔁 V24 refill v2 적용: {'BTC': '33.3%', 'SOL': '33.3%'}"
        " | top diffs=[('SOL', '+33.3%'), ('ETH', '-33.3%')]\n"
        "2026-08-31 00:05:44,121 INFO   BTC → final L = min(BTC_cap=4, K2=2) = 2\n"
        "2026-08-31 00:05:44,121 INFO   SOL → final L = min(BTC_cap=4, K2=3) = 3\n"
        # stale 사본 — refill 결과를 덮으면 안 된다
        "2026-08-31 00:05:44,121 INFO DRY-RUN REBALANCE: {'BTC': '33.3%', 'ETH': '33.3%'}\n"
        "2026-08-31 00:05:44,143 INFO === 완료 (12.0s) ===\n"
    )
    r = _fut(tmp_path, log)
    assert sorted(r.coins()) == ['BTC', 'SOL']
    assert 'ETH' not in r.combined
    assert r.target_rank == cdr.FUT_TGT_REFILL
    assert r.target_src_str() == ' (refill v2 반영)'
    assert 'SOL 33.3% (L3)' in r.target_str()

    lines, warn = cdr._fut_lines(r)
    body = '\n'.join(lines)
    assert 'SOL' in body and 'ETH' not in body
    assert warn is False


def test_c1_v25_fut_targets_is_final_authority(tmp_path):
    """실거래 전용 'V25 fut: targets=' (executor:3216/3356, float dict) 가 최종 확정값."""
    log = (
        "2026-08-31 00:05:32,178 INFO === 바이낸스 선물 매매 시작 (run_id=R1) ===\n"
        "2026-08-31 00:05:32,200 INFO 시작 지연: 17s (크론 동시충돌 완화)\n"
        "2026-08-31 00:05:44,058 INFO 합산: {'BTC': '33.3%', 'ETH': '33.3%'}\n"
        "2026-08-31 00:05:44,120 INFO   🔁 V24 refill v2 적용: {'BTC': '33.3%', 'SOL': '33.3%'}"
        " | top diffs=[]\n"
        "2026-08-31 00:05:46,000 INFO 리밸런싱 완료: PV $100.00 → $101.00\n"
        "2026-08-31 00:05:47,000 INFO V25 fut: targets={'BTC': 0.5, 'Cash': 0.5} ht=0.1000"
        " fire=True pv=$101.00 success=True streak=0\n"
        "2026-08-31 00:05:47,100 INFO === 완료 (15.0s) ===\n"
    )
    r = _fut(tmp_path, log)
    assert r.target_rank == cdr.FUT_TGT_V25
    assert sorted(r.coins()) == ['BTC']
    assert _approx(r.combined['BTC'], 0.5) and _approx(r.combined['Cash'], 0.5)
    assert r.target_src_str() == ' (실거래 확정)'
    assert r.dry_run is False


def test_c1_dryrun_line_used_when_combined_missing(tmp_path):
    """'합산:' 이 없을 때는 DRY-RUN 줄이라도 쓴다 (권위는 낮지만 유일한 근거)."""
    log = (
        "2026-08-31 00:05:32,178 INFO === 바이낸스 선물 매매 시작 (run_id=R1) ===\n"
        "2026-08-31 00:05:44,121 INFO DRY-RUN REBALANCE: {'BTC': '50.0%'}\n"
        "2026-08-31 00:05:44,143 INFO === 완료 (12.0s) ===\n"
    )
    r = _fut(tmp_path, log)
    assert r.target_rank == cdr.FUT_TGT_DRYRUN
    assert _approx(r.combined['BTC'], 0.5)
    assert r.target_src_str() == ''      # 합산과 같은 값이라 출처 표기 안 함


# ── C2: 결과 분류 보강 ──
def test_c2_order_failed_survives_later_success_markers(tmp_path):
    log = (
        "2026-08-31 00:05:32,178 INFO === 바이낸스 선물 매매 시작 (run_id=R1) ===\n"
        "2026-08-31 00:05:32,200 INFO 시작 지연: 5s (크론 동시충돌 완화)\n"
        "2026-08-31 00:05:45,000 ERROR ORDER FAILED BUY BTCUSDT qty=0.010: APIError(code=-2010)\n"
        "2026-08-31 00:05:46,000 INFO 리밸런싱 완료: PV $100.00 → $99.00\n"
        "2026-08-31 00:05:47,000 INFO === 완료 (15.0s) ===\n"
    )
    r = _fut(tmp_path, log)
    assert (r.result_kind, r.result_label) == ('error', 'ORDER FAILED (주문 실패)')


def test_c2_integrity_violation_and_reconciliation(tmp_path):
    integ = _fut(tmp_path, (
        "2026-08-31 00:05:32,178 INFO === 바이낸스 선물 매매 시작 (run_id=R1) ===\n"
        "2026-08-31 00:05:33,000 ERROR 🔒 V25 무결성 위반 — 매매 차단\n"
        "  - BTCUSDT qty 0.010 → 0.004\n"
        "2026-08-31 00:05:34,000 INFO === 완료 (2.0s) ===\n"
    ))
    assert integ.result_kind == 'error'
    assert '무결성 위반' in integ.result_label

    recon = _fut(tmp_path, (
        "2026-08-31 00:05:32,178 INFO === 바이낸스 선물 매매 시작 (run_id=R1) ===\n"
        "2026-08-31 00:05:32,200 INFO 시작 지연: 5s (크론 동시충돌 완화)\n"
        "2026-08-31 00:05:33,000 WARNING ⚠ V25 reconciliation 차이:\n"
        "  - BTCUSDT notional 100.0 vs 80.0\n"
        "2026-08-31 00:05:34,000 INFO === 완료 (2.0s) ===\n"
    ))
    assert (recon.result_kind, recon.result_label) == ('warn', 'reconciliation 차이 (체결 미달)')


def test_c2_v25_success_false_is_error(tmp_path):
    log = (
        "2026-08-31 00:05:32,178 INFO === 바이낸스 선물 매매 시작 (run_id=R1) ===\n"
        "2026-08-31 00:05:32,200 INFO 시작 지연: 5s (크론 동시충돌 완화)\n"
        "2026-08-31 00:05:47,000 INFO V25 fut: targets={'Cash': 1.0} ht=0.0 fire=False"
        " pv=$100.00 success=False streak=1\n"
        "2026-08-31 00:05:47,100 INFO === 완료 (15.0s) ===\n"
    )
    r = _fut(tmp_path, log)
    assert r.result_kind == 'error'
    assert 'success=False' in r.result_label


def test_c2_unmatched_error_level_promotes_result(tmp_path):
    """마커가 모르는 ERROR 줄이 있으면 '정상 완료' 로 남지 않는다."""
    log = (
        "2026-08-31 00:05:32,178 INFO === 바이낸스 선물 매매 시작 (run_id=R1) ===\n"
        "2026-08-31 00:05:33,000 ERROR 알 수 없는 내부 오류 zzz\n"
        "2026-08-31 00:05:34,000 INFO DRY-RUN REBALANCE: CASH\n"
        "2026-08-31 00:05:34,100 INFO === 완료 (2.0s) ===\n"
    )
    r = _fut(tmp_path, log)
    assert (r.result_kind, r.result_label) == ('error', '오류 로그 있음 (마커 미분류)')

    # 반대로 마커가 의도적으로 warn 으로 분류한 줄은 레벨(ERROR)이 덮지 않는다
    r2 = _fut(tmp_path, (
        "2026-08-31 00:05:32,178 INFO === 바이낸스 선물 매매 시작 (run_id=R1) ===\n"
        "2026-08-31 00:05:33,000 ERROR 현재 포지션/PV 조회 실패. 이번 실행은 거래 없이 스킵.\n"
    ))
    assert r2.result_kind == 'warn'


# ── C3: weights 파싱 fail-loud ──
def test_c3_double_quoted_dict_fails_loud(tmp_path):
    log = (
        "2026-08-31 00:05:32,178 INFO === 바이낸스 선물 매매 시작 (run_id=R1) ===\n"
        '2026-08-31 00:05:44,058 INFO 합산: {"BTC": "33.3%"}\n'
        "2026-08-31 00:05:44,143 INFO === 완료 (12.0s) ===\n"
    )
    r = _fut(tmp_path, log)
    assert r.combined == {}                       # 조용히 CASH-only 로 만들지 않는다
    assert r.target_str() == '알 수 없음'
    assert r.canary_str() == '알 수 없음'         # 근거 없는 추론 금지
    assert any('타겟 파싱 실패' in p for p in r.problems)
    assert r.result_kind == 'warn'
    body = '\n'.join(cdr._fut_lines(r)[0])
    assert '⚠️ 타겟 파싱 실패: {"BTC": "33.3%"}' in body      # 원문을 그대로 보여준다


def test_c3_missing_percent_sign_fails_loud(tmp_path):
    log = (
        "2026-08-31 00:05:32,178 INFO === 바이낸스 선물 매매 시작 (run_id=R1) ===\n"
        "2026-08-31 00:05:44,058 INFO 합산: {'BTC': '33.3'}\n"
        "2026-08-31 00:05:44,143 INFO === 완료 (12.0s) ===\n"
    )
    r = _fut(tmp_path, log)
    assert r.combined == {}
    assert r.result_kind == 'warn'
    assert any('타겟 파싱 실패' in p for p in r.problems)


def test_c3_partial_token_match_fails_loud(tmp_path):
    """항목 2개 중 1개만 토큰으로 잡히면(부분 매칭) 종목을 놓친 것이다."""
    assert cdr._parse_fut_weights("{'BTC': '33.3%', 'ETH': 'n/a'}") is None
    assert cdr._parse_fut_weights("{'BTC': '150.0%'}") is None          # 범위 위반
    assert cdr._parse_fut_weights("{'BTC': '60.0%', 'ETH': '60.0%'}") is None  # 합 > 1
    assert cdr._parse_fut_weights('{}') == {'Cash': 1.0}
    assert cdr._parse_fut_weights('CASH 100%') == {'Cash': 1.0}
    assert cdr._parse_fut_weights('CASH') == {'Cash': 1.0}
    assert cdr._parse_fut_weights('CASH 100') is None
    assert cdr._parse_fut_floats("{'BTC': 0.5, 'Cash': 0.5}")['BTC'] == 0.5
    assert cdr._parse_fut_floats("{'BTC': abc}") is None


# ── M1: 블록 경계 ──
def test_m1_block_closes_at_done_marker(tmp_path):
    """'=== 완료' 뒤의 별도 실행(수동 --report 등) 로그는 블록 분류에 안 섞인다."""
    log = SAMPLE_BLOCK + (
        "2026-08-31 10:00:00,000 ERROR 리포트 생성 중 포지션 조회 실패\n"
        "2026-08-31 10:00:01,000 WARNING 뭔가 경고\n"
    )
    r = _fut(tmp_path, log)
    assert (r.result_kind, r.result_label) == ('ok', '정상 완료')   # 블록은 그대로 ok
    assert r.issues == []
    assert r.outside_note.startswith('블록 외 로그 2건')
    assert 'ERROR 1' in r.outside_note and 'WARNING 1' in r.outside_note
    body = '\n'.join(cdr._fut_lines(r)[0])
    assert '블록 외 로그 2건' in body


def test_m1_start_marker_anchored_to_first_line(tmp_path):
    """멀티라인 본문에 시작/완료 문자열이 섞여도 블록 경계로 오인하지 않는다."""
    log = (
        "2026-08-31 00:05:32,178 INFO === 바이낸스 선물 매매 시작 (run_id=R1) ===\n"
        "2026-08-31 00:05:33,000 ERROR 예외 발생\n"
        "  이전 로그 재현: === 바이낸스 선물 매매 시작 (run_id=FAKE) ===\n"
        "  이전 로그 재현: === 완료 (0.1s) ===\n"
        "2026-08-31 00:05:34,000 INFO DRY-RUN REBALANCE: CASH\n"
        "2026-08-31 00:05:34,100 INFO === 완료 (2.0s) ===\n"
    )
    r = _fut(tmp_path, log)
    assert r.start_count == 1
    assert r.run_id == 'R1'
    assert r.outside_note == ''          # 진짜 완료는 마지막 줄 하나뿐
    assert r.result_kind == 'error'      # 마커 미분류 ERROR 승격


# ── M2: redaction ──
def test_m2_issue_lines_are_redacted(tmp_path):
    log = (
        "2026-08-31 00:05:32,178 INFO === 바이낸스 선물 매매 시작 (run_id=R1) ===\n"
        "2026-08-31 00:05:33,000 WARNING 텔레그램 실패 bot123456:ABC-token"
        " https://api.telegram.org/bot123456:ABC-token/sendMessage\n"
        "2026-08-31 00:05:34,000 INFO DRY-RUN REBALANCE: CASH\n"
        "2026-08-31 00:05:34,100 INFO === 완료 (2.0s) ===\n"
    )
    r = _fut(tmp_path, log)
    assert len(r.issues) == 1
    assert 'bot<REDACTED>' in r.issues[0]
    assert 'ABC-token' not in r.issues[0]
    assert 'bot123456' not in '\n'.join(cdr._fut_lines(r)[0])


def test_m2_target_parse_error_is_redacted(tmp_path):
    r = _fut(tmp_path, (
        "2026-08-31 00:05:32,178 INFO === 바이낸스 선물 매매 시작 (run_id=R1) ===\n"
        '2026-08-31 00:05:44,058 INFO 합산: {"BTC": "33.3%"} bot999999:SECRET\n'
        "2026-08-31 00:05:44,143 INFO === 완료 (12.0s) ===\n"
    ))
    joined = '\n'.join(r.problems)
    assert 'SECRET' not in joined and 'bot<REDACTED>' in joined


# ── M3: 이슈 절단 우선순위 ──
def test_m3_errors_survive_issue_truncation(tmp_path):
    lines = ["2026-08-31 00:05:32,178 INFO === 바이낸스 선물 매매 시작 (run_id=R1) ===\n"]
    for i in range(5):
        lines.append(f"2026-08-31 00:05:33,{100+i:03d} WARNING 경고{i} 어쩌구\n")
    lines.append("2026-08-31 00:05:33,900 ERROR 알 수 없는 내부 오류 zzz\n")
    lines.append("2026-08-31 00:05:34,000 INFO DRY-RUN REBALANCE: CASH\n")
    lines.append("2026-08-31 00:05:34,100 INFO === 완료 (2.0s) ===\n")
    r = _fut(tmp_path, ''.join(lines))

    assert len(r.issues) == 5
    assert r.issues[0].startswith('ERROR 알 수 없는 내부 오류')   # ERROR 먼저 보존
    assert all(x.startswith('WARNING') for x in r.issues[1:])
    assert [x for x in r.issues if x.startswith('WARNING')] == [
        f'WARNING 경고{i} 어쩌구' for i in range(4)]              # 같은 등급 내 시간순
    assert r.issue_omitted == 1
    assert r.issue_omitted_by_level == {'WARNING': 1}
    assert r.result_kind == 'error'                              # 분류는 절단 전 전체 기준
    assert '외 1건(WARNING 1)' in '\n'.join(cdr._fut_lines(r)[0])


# ── m1: 힌트 보강 / 모드 판정 ──
def test_m1_abort_hints_from_executor_wordings(tmp_path):
    lock = _fut(tmp_path, "2026-08-31 00:05:00,000 WARNING 다른 인스턴스 실행 중, 종료\n")
    assert lock.ran is False
    assert 'flock 충돌로 스킵' in lock.abort_hints

    key = _fut(tmp_path, "2026-08-31 00:05:00,000 ERROR API key not configured\n")
    assert 'API 키 미설정' in key.abort_hints


def test_m1_quiet_dry_run_without_dryrun_line(tmp_path):
    """rebalancing_needed=false 인 dry-run 은 'DRY-RUN REBALANCE' 를 안 찍는다.

    실코드(2963): 그 경로는 '매매 스킵: rebalancing_needed=false' 만 남긴다.
    실거래라면 '시작 지연:' 이 반드시 있으므로, 그게 없이 완주했으면 dry 로 본다.
    """
    log = (
        "2026-08-31 00:05:32,178 INFO === 바이낸스 선물 매매 시작 (run_id=R1) ===\n"
        "2026-08-31 00:05:44,058 INFO 합산: {'BTC': '33.3%'}\n"
        "2026-08-31 00:05:44,100 INFO 매매 스킵: rebalancing_needed=false\n"
        "2026-08-31 00:05:44,143 INFO === 완료 (12.0s) ===\n"
    )
    r = _fut(tmp_path, log)
    assert r.dry_run is True and r.mode_source == '추론'
    assert r.mode_str() == 'dry·추론'
    assert (r.result_kind, r.result_label) == ('ok', '매매 불필요 → 스킵')

    live = _fut(tmp_path, log.replace(
        "2026-08-31 00:05:44,058 INFO 합산:",
        "2026-08-31 00:05:32,200 INFO 시작 지연: 12s (크론 동시충돌 완화)\n"
        "2026-08-31 00:05:44,058 INFO 합산:"))
    assert live.dry_run is False and live.mode_str() == 'LIVE'


def test_m1_unknown_mode_promotes_to_warn(tmp_path):
    """분류는 됐는데 dry/LIVE 판별이 안 되면 최소 warn."""
    log = (
        "2026-08-31 00:05:32,178 INFO === 바이낸스 선물 매매 시작 (run_id=R1) ===\n"
        "2026-08-31 00:05:33,000 INFO 리밸런싱 완료: PV $1.00 → $1.00\n"
    )
    r = _fut(tmp_path, log)          # '=== 완료' 없음 → 추론도 불가
    assert r.dry_run is False        # '리밸런싱 완료' 는 LIVE 흔적
    r2 = _fut(tmp_path, (
        "2026-08-31 00:05:32,178 INFO === 바이낸스 선물 매매 시작 (run_id=R1) ===\n"
        "2026-08-31 00:05:33,000 INFO 매매 스킵: rebalancing_needed=false\n"
    ))
    assert r2.dry_run is None and r2.mode_str() == '?'
    assert r2.result_kind == 'warn' and '모드 불명' in r2.result_label


# ── m2: 스트리밍 리더 ──
def test_m2_reader_streams_and_keeps_semantics(tmp_path):
    """파일을 통째로 읽지 않고 라인 순회 — 결과는 동일해야 한다."""
    import inspect
    src = inspect.getsource(cdr._read_fut_records)
    assert 'read().splitlines()' not in src
    assert 'for line in f' in src

    path = _write(tmp_path, SAMPLE_BLOCK * 3)
    recs = cdr._read_fut_records(path, DAY)
    assert len(recs) == 19 * 3
    assert recs[0][1] == 'INFO'


# ═══════════ Codex 리뷰 라운드 2 반영분 (R2-1~R2-3) ═══════════

# ── R2-1: 상위 권위 파싱 실패는 하위 권위 값을 무효화한다 ──
def test_r2_1_broken_higher_rank_invalidates_accepted_target(tmp_path):
    log = (
        "2026-08-31 00:05:32,178 INFO === 바이낸스 선물 매매 시작 (run_id=R1) ===\n"
        "2026-08-31 00:05:32,200 INFO 시작 지연: 5s (크론 동시충돌 완화)\n"
        "2026-08-31 00:05:44,058 INFO 합산: {'BTC': '33.3%', 'ETH': '33.3%'}\n"
        "2026-08-31 00:05:44,120 INFO   🔁 V24 refill v2 적용: {'BTC': '33.3%', 'SOL': '33.3%'}"
        " | top diffs=[]\n"
        # 최종 확정 줄이 깨졌다 → refill 값도 '최종' 이라 믿을 수 없다
        "2026-08-31 00:05:47,000 INFO V25 fut: targets={\"BTC\": 0.5} ht=0.1000 fire=True"
        " pv=$1.00 success=True streak=0\n"
        "2026-08-31 00:05:47,100 INFO === 완료 (15.0s) ===\n"
    )
    r = _fut(tmp_path, log)
    assert r.combined == {}
    assert r.target_rank == -1
    assert r.target_blocked_rank == cdr.FUT_TGT_V25
    assert r.target_str() == '알 수 없음'
    assert r.result_kind == 'warn'
    assert any('타겟 파싱 실패' in p for p in r.problems)
    assert cdr._fut_lines(r)[1] is True


def test_r2_1_lower_rank_failure_recovers_from_higher_rank(tmp_path):
    """깨진 합산(rank0) → 정상 refill(rank2) 이면 refill 값으로 복구된다."""
    log = (
        "2026-08-31 00:05:32,178 INFO === 바이낸스 선물 매매 시작 (run_id=R1) ===\n"
        '2026-08-31 00:05:44,058 INFO 합산: {"BTC": "33.3%"}\n'
        "2026-08-31 00:05:44,120 INFO   🔁 V24 refill v2 적용: {'BTC': '33.3%', 'SOL': '33.3%'}"
        " | top diffs=[]\n"
        "2026-08-31 00:05:44,121 INFO DRY-RUN REBALANCE: {'BTC': '33.3%'}\n"
        "2026-08-31 00:05:44,143 INFO === 완료 (12.0s) ===\n"
    )
    r = _fut(tmp_path, log)
    assert sorted(r.coins()) == ['BTC', 'SOL']       # refill 로 복구
    assert r.target_rank == cdr.FUT_TGT_REFILL
    assert r.result_kind == 'warn'                   # 실패 사실은 남는다
    assert any('타겟 파싱 실패' in p for p in r.problems)


def test_r2_1_same_rank_later_good_line_recovers(tmp_path):
    """같은 rank 의 뒤 정상 줄은 복구할 수 있다 (합산이 두 번 찍힌 경우)."""
    log = (
        "2026-08-31 00:05:32,178 INFO === 바이낸스 선물 매매 시작 (run_id=R1) ===\n"
        "2026-08-31 00:05:44,058 INFO 합산: {'BTC': '33.3'}\n"
        "2026-08-31 00:05:44,059 INFO 합산: {'BTC': '33.3%'}\n"
        "2026-08-31 00:05:44,143 INFO === 완료 (12.0s) ===\n"
    )
    r = _fut(tmp_path, log)
    assert sorted(r.coins()) == ['BTC']
    assert r.target_rank == cdr.FUT_TGT_COMBINED


def test_r2_1_lower_rank_failure_after_good_higher_rank_keeps_it(tmp_path):
    """이미 상위 권위 값을 받은 뒤 하위 권위 줄이 깨져도 상위 값은 유지된다."""
    log = (
        "2026-08-31 00:05:32,178 INFO === 바이낸스 선물 매매 시작 (run_id=R1) ===\n"
        "2026-08-31 00:05:44,120 INFO   🔁 V24 refill v2 적용: {'BTC': '33.3%', 'SOL': '33.3%'}"
        " | top diffs=[]\n"
        '2026-08-31 00:05:44,121 INFO DRY-RUN REBALANCE: {"BTC": "33.3%"}\n'
        "2026-08-31 00:05:44,143 INFO === 완료 (12.0s) ===\n"
    )
    r = _fut(tmp_path, log)
    assert sorted(r.coins()) == ['BTC', 'SOL']
    assert r.target_rank == cdr.FUT_TGT_REFILL
    assert r.result_kind == 'warn'


# ── R2-2: 전략별 타겟은 cash 추가 후 재검증 ──
def test_r2_2_member_cash_makes_sum_over_100(tmp_path):
    log = (
        "2026-08-31 00:05:32,178 INFO === 바이낸스 선물 매매 시작 (run_id=R1) ===\n"
        "2026-08-31 00:05:44,058 INFO   D_SMA42 → {'BTC': '80.0%'} (cash=80%)\n"
        "2026-08-31 00:05:44,059 INFO 합산: {'BTC': '80.0%'}\n"
        "2026-08-31 00:05:44,143 INFO === 완료 (12.0s) ===\n"
    )
    r = _fut(tmp_path, log)
    assert r.members == {}                                  # 전략 타겟은 버린다
    assert any('타겟 파싱 실패' in p for p in r.problems)
    assert 'D_SMA42 → {\'BTC\': \'80.0%\'} (cash=80%)' in '\n'.join(r.problems)
    assert r.result_kind == 'warn'
    assert r.canary_str() == '알 수 없음'                    # 추론 금지
    assert cdr._parse_fut_weights("{'BTC': '80.0%'}", '80') is None


def test_r2_2_empty_dict_with_non_100_cash(tmp_path):
    log = (
        "2026-08-31 00:05:32,178 INFO === 바이낸스 선물 매매 시작 (run_id=R1) ===\n"
        "2026-08-31 00:05:44,058 INFO   D_SMA42 → {} (cash=20%)\n"
        "2026-08-31 00:05:44,143 INFO === 완료 (12.0s) ===\n"
    )
    r = _fut(tmp_path, log)
    assert r.members == {}                                  # 조용한 Cash=100% 금지
    assert any('cash=20%' in p for p in r.problems)
    assert r.result_kind == 'warn'
    assert cdr._parse_fut_weights('{}', '20') is None
    assert cdr._parse_fut_weights('CASH 100%', '20') is None
    # 실물 정상 케이스는 그대로 통과해야 한다
    assert cdr._parse_fut_weights('CASH 100%', '100') == {'Cash': 1.0}
    assert cdr._parse_fut_weights("{'BTC': '33.3%', 'ETH': '33.3%'}", '33')['Cash'] == 0.33


# ── R2-3: 시작/완료 마커 전체행 앵커 ──
def test_r2_3_quoted_markers_in_normal_log_line(tmp_path):
    """일반 로그 첫 줄이 마커 문구를 인용해도 블록 경계/완주 판정이 흔들리면 안 된다."""
    log = (
        "2026-08-31 00:05:32,178 INFO === 바이낸스 선물 매매 시작 (run_id=R1) ===\n"
        "2026-08-31 00:05:33,000 WARNING 이전 실행 인용: '=== 바이낸스 선물 매매 시작"
        " (run_id=OLD) ===' 재확인 필요\n"
        "2026-08-31 00:05:33,500 INFO 참고: 지난 실행은 '=== 완료 (9.9s) ===' 로 끝났다\n"
        "2026-08-31 00:05:34,000 INFO DRY-RUN REBALANCE: CASH\n"
        "2026-08-31 00:05:34,100 INFO === 완료 (2.0s) ===\n"
    )
    r = _fut(tmp_path, log)
    assert r.start_count == 1                 # 인용된 시작 문구는 세지 않는다
    assert r.run_id == 'R1'
    assert r.run_time == '00:05:32'
    assert r.outside_note == ''               # 인용된 완료로 블록이 잘리지 않는다
    assert r.coins() == {}                    # 블록 끝까지 읽어 DRY-RUN 줄을 봤다
    assert r.dry_run is True
    # 인용 완료 줄은 '정상 완료' 판정 근거가 아니다 — 진짜 완료 행이 근거
    assert r.result_kind == 'warn'            # WARNING 승격 (마커 미분류)
    assert len(r.issues) == 1


def test_r2_3_incomplete_block_not_closed_by_quoted_done(tmp_path):
    """완료 문구가 인용만 됐고 진짜 완료 행이 없으면 미완료(unknown)로 남는다."""
    log = (
        "2026-08-31 00:05:32,178 INFO === 바이낸스 선물 매매 시작 (run_id=R1) ===\n"
        "2026-08-31 00:05:33,500 INFO 참고: '=== 완료 (9.9s) ===' 형식으로 끝난다\n"
    )
    r = _fut(tmp_path, log)
    assert r.ran is True
    assert r.result_kind == 'unknown'
    assert '결과 마커 없음' in r.result_label
    assert r.outside_note == ''


def test_r2_3_real_marker_rows_still_match():
    assert cdr.RE_FUT_START.match('=== 바이낸스 선물 매매 시작 (run_id=20260831_090532) ===')
    assert cdr.RE_FUT_START.match('=== 바이낸스 선물 매매 시작 ===')
    assert cdr.RE_FUT_DONE.match('=== 완료 (12.0s) ===')
    assert cdr.RE_FUT_DONE.match('=== 완료 ===')
    assert not cdr.RE_FUT_DONE.match("인용: '=== 완료 (12.0s) ==='")
    assert not cdr.RE_FUT_START.match('앞말 === 바이낸스 선물 매매 시작 (run_id=X) ===')


# ═══════════ 카나리 줄 조건부 표시 (평상시 숨김, 이상 시 표시) ═══════════
def test_canary_line_hidden_when_all_on_and_matching(tmp_path):
    """양쪽 실행 + 전 멤버 ON + 일치 → 블록에선 숨기고, 비교 판정은 그대로 낸다."""
    sides = _spot_sides(tmp_path)                       # 타겟 추론 → 양쪽 ON
    fut = _fut(tmp_path, SAMPLE_BLOCK)                  # 타겟 추론 → ON
    body, warn = cdr.build_report(DAY, sides, fut)

    assert '  카나리:' not in body                       # 세 블록 모두 생략
    assert '✅ 카나리 일치' in body                       # 비교 로직은 그대로
    assert cdr._spot_canary_visible(sides) is False
    assert fut.canary_visible() is False
    assert warn is False
    # 숨겨도 판정 근거 자체는 남아 있다
    assert sides[0].canary == {'combined': 'ON'} and fut.canary == {'D_SMA42': 'ON'}


def test_canary_line_shown_when_off(tmp_path):
    """OFF 인 날은 양쪽 블록에 표시 (일치해도 평상 상태가 아니다)."""
    sides = _spot_sides(tmp_path, canary_a='OFF', canary_b='OFF')
    body, _warn = cdr.build_report(DAY, sides)

    assert body.count('  카나리: D_SMA42=OFF') == 2
    assert '✅ 카나리 일치' in body
    assert cdr._spot_canary_visible(sides) is True


def test_canary_line_shown_when_sides_disagree(tmp_path):
    """불일치면 양쪽 블록 모두 표시 — 어느 쪽이 어떤 상태였는지 봐야 한다."""
    sides = _spot_sides(tmp_path, canary_a='ON', canary_b='OFF')
    body, warn = cdr.build_report(DAY, sides)

    assert '[업비트]' in body and '  카나리: D_SMA42=ON' in body
    assert '  카나리: D_SMA42=OFF' in body
    assert body.count('  카나리:') == 2
    assert '⚠️ 카나리 불일치' in body
    assert warn is True
    assert cdr._spot_canary_visible(sides) is True


def test_canary_line_shown_when_undeterminable(tmp_path):
    """한쪽이 판별 불가('알 수 없음')면 양쪽 표시."""
    skip_log = (f'[{DAY} 00:05:10] [0005abcd] Executor 시작 (dry_run=True)\n'
                f'[{DAY} 00:05:12] [0005abcd] 새 봉 없음\n')
    sides = _spot_sides(tmp_path)
    path = _write(tmp_path, skip_log, 'executor_coin_binance.log')
    b = cdr.SideResult('바낸현물', path)
    b.parse(DAY)
    sides[1] = b
    assert b.canary == {}

    body, warn = cdr.build_report(DAY, sides)
    assert body.count('  카나리:') == 2
    assert '  카나리: 알 수 없음' in body
    assert '⚠️ 카나리: 한쪽 이상 판별 불가' in body
    assert warn is True


def test_canary_line_shown_when_one_side_missing(tmp_path):
    """한쪽 미실행이면 실행된 쪽은 근거를 보여준다 (비교가 불가하므로)."""
    sides = _spot_sides(tmp_path)
    missing = cdr.SideResult('바낸현물', os.path.join(str(tmp_path), 'nope.log'))
    missing.parse(DAY)
    sides[1] = missing

    body, warn = cdr.build_report(DAY, sides)
    assert body.count('  카나리:') == 1               # 미실행 블록엔 애초에 줄이 없다
    assert '[바낸현물] ⚠️ 미실행' in body
    assert warn is True
    assert cdr._spot_canary_visible(sides) is True


def test_fut_canary_line_shown_when_off_or_unknown(tmp_path):
    off = _fut(tmp_path, (
        "2026-08-31 00:05:31,048 INFO === 바이낸스 선물 매매 시작 (run_id=R1) ===\n"
        "2026-08-31 00:05:31,207 INFO   D_SMA42 BTC=$77,377 SMA(42)=$78,016"
        " ratio=0.9918 canary=OFF\n"
        "2026-08-31 00:05:31,208 INFO 합산: CASH 100%\n"
        "2026-08-31 00:05:31,208 INFO DRY-RUN REBALANCE: CASH\n"
        "2026-08-31 00:05:32,283 INFO === 완료 (22.9s) ===\n"
    ))
    assert off.canary_visible() is True
    assert '  카나리: D_SMA42=OFF (canary 로그)' in '\n'.join(cdr._fut_lines(off)[0])

    # 타겟 파싱 실패로 추론을 못 한 날 → '알 수 없음' 을 보여준다
    broken = _fut(tmp_path, (
        "2026-08-31 00:05:32,178 INFO === 바이낸스 선물 매매 시작 (run_id=R1) ===\n"
        '2026-08-31 00:05:44,058 INFO 합산: {"BTC": "33.3%"}\n'
        "2026-08-31 00:05:44,143 INFO === 완료 (12.0s) ===\n"
    ))
    assert broken.canary == {} and broken.canary_visible() is True
    assert '  카나리: 알 수 없음' in '\n'.join(cdr._fut_lines(broken)[0])


def test_fut_canary_line_hidden_when_all_on(tmp_path):
    """'타겟 추론' ON 도 매일 뜨는 평상 상태라 숨긴다."""
    r = _fut(tmp_path, SAMPLE_BLOCK)
    assert r.canary == {'D_SMA42': 'ON'} and r.canary_source == '타겟 추론'
    assert r.canary_visible() is False
    body = '\n'.join(cdr._fut_lines(r)[0])
    assert '  카나리:' not in body
    assert '  타겟: BTC 33.3% (L2)' in body            # 나머지 줄은 그대로


# ══════ 실행 모드(dry/LIVE)는 정합 판정 대상이 아니다 (2026-09-01 결정) ══════
def _mode_lines(body):
    return [ln for ln in body.split('\n') if ln.startswith('실행 모드')]


def test_mode_upbit_live_vs_binance_dry_is_not_a_warning(tmp_path):
    """운영 조합(업비트 LIVE + 바낸현물 dry) — 경고 없이 정보 줄만, 정합 판정은 그대로."""
    sides = _spot_sides(tmp_path, dry=(False, True))
    assert sides[0].dry_run is False and sides[1].dry_run is True
    body, warn = cdr.build_report(DAY, sides)

    assert 'LIVE 혼입' not in body                     # 옛 경고는 사라졌다
    assert _mode_lines(body) == ['실행 모드: 업비트=LIVE, 바낸현물=dry-run']
    assert '⚠️' not in body                            # 모드 차이로 어떤 경고도 뜨지 않는다
    assert not body.startswith('⚠️')                   # 옛 코드가 맨 앞에 끼워 넣던 줄
    assert warn is False
    # 실제 비교 대상(전략 산출물)은 그대로 판정된다
    assert '✅ 실행 쌍 정합' in body
    assert '✅ 카나리 일치' in body
    assert '✅ 코인 집합 일치' in body
    assert '✅ 비중 일치' in body
    assert '[업비트] 00:05:10 (LIVE)' in body          # 블록엔 모드가 계속 보인다


def test_mode_both_dry_is_a_single_neutral_line(tmp_path):
    """양쪽 같은 모드여도 ✅ 판정을 주지 않는다 — 중립 한 줄."""
    sides = _spot_sides(tmp_path)
    body, warn = cdr.build_report(DAY, sides)

    assert _mode_lines(body) == ['실행 모드: 양쪽 dry-run']
    assert '✅ 실행 모드 일치' not in body
    assert warn is False


def test_mode_upbit_dry_vs_binance_live_is_not_a_warning(tmp_path):
    """역조합(업비트 dry + 바낸현물 LIVE)도 방향과 무관하게 정보 줄 하나뿐이다."""
    sides = _spot_sides(tmp_path, dry=(True, False))
    assert sides[0].dry_run is True and sides[1].dry_run is False
    body, warn = cdr.build_report(DAY, sides)

    assert _mode_lines(body) == ['실행 모드: 업비트=dry-run, 바낸현물=LIVE']
    assert '⚠️' not in body
    assert warn is False


def test_mode_both_live_is_a_single_neutral_line(tmp_path):
    """양쪽 LIVE 여도 판정하지 않는다 — 'LIVE 혼입' 경고는 완전히 사라졌다."""
    sides = _spot_sides(tmp_path, dry=(False, False))
    body, warn = cdr.build_report(DAY, sides)

    assert _mode_lines(body) == ['실행 모드: 양쪽 LIVE']
    assert 'LIVE 혼입' not in body
    assert warn is False


def test_mode_unknown_side_shows_question_mark_without_warning(tmp_path):
    """모드 불명(None)은 '?' 로만 표시 — 이 줄 때문에 warn 이 서지 않는다.

    SideResult 는 RE_START 로 블록을 잡으므로 ran=True 면서 dry_run=None 이 나올 수 없다.
    파싱 품질 경고(FutResult 의 '실행 모드 불명')는 result_label 경로가 따로 담당하므로,
    여기서는 비교 섹션의 표시 분기만 직접 값을 넣어 확인한다.
    """
    sides = _spot_sides(tmp_path)
    sides[1].dry_run = None
    body, warn = cdr.build_report(DAY, sides)

    assert _mode_lines(body) == ['실행 모드: 업비트=dry-run, 바낸현물=?']
    assert '⚠️' not in body
    assert warn is False
    assert cdr._mode_word(sides[0]) == 'dry-run' and cdr._mode_word(sides[1]) == '?'


# ══════ 업비트 실계좌 평가액 + 원금 대비 등락률 (네트워크 전면 mock) ══════
def _swap(**kw):
    """cdr 모듈 속성 임시 교체 — pytest monkeypatch 없이도 도는 fallback 러너 대비."""
    old = {k: getattr(cdr, k) for k in kw}
    for k, v in kw.items():
        setattr(cdr, k, v)
    return old


def _restore(old):
    for k, v in old.items():
        setattr(cdr, k, v)


def _fake_http(accounts=None, ticker=None, deposits=None, calls=None):
    """(_upbit_get, _upbit_public_get) 대역 — 네트워크 없이 (status, json) 을 돌려준다.

    accounts/ticker/deposits 에 (status, body) 튜플이나 Exception 을 넣는다.
    호출 인자는 calls 리스트에 (path, params) 로 쌓여 배치 조회 여부까지 검증한다.
    """
    routes = {'/v1/accounts': accounts, '/v1/ticker': ticker, '/v1/deposits': deposits}

    def _serve(path, params):
        if calls is not None:
            calls.append((path, dict(params)))
        got = routes.get(path)
        if isinstance(got, Exception):
            raise got
        if got is None:
            raise AssertionError(f'예상 못한 호출: {path}')
        return got

    return _serve, _serve


NOW = '2026-09-01T09:20:00+09:00'


def _principal_file(tmp_path, principal=2000000.0, last='2026-08-30T00:00:00+09:00',
                    processed=None):
    path = os.path.join(str(tmp_path), 'report_principal.json')
    state = {'principal_krw': principal, 'last_deposit_check': last}
    if processed is not None:
        state['processed_uuids'] = processed
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(state, f)
    return path


def _deposit(uuid, done_at, amount, state='ACCEPTED', currency='KRW'):
    return {'uuid': uuid, 'currency': currency, 'state': state,
            'amount': str(amount), 'done_at': done_at}


def _read_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


# ── 렌더링 ──
def test_upbit_value_line_rendered_in_upbit_block_only(tmp_path):
    """정상 케이스: 천단위 콤마 + 부호 + 소수 1자리, warn 없음. 업비트 블록에만 붙는다."""
    sides = _spot_sides(tmp_path, dry=(False, True))
    fut = _fut(tmp_path, SAMPLE_BLOCK)
    body, warn = cdr.build_report(DAY, sides, fut,
                                  {'value': 2100000.0, 'pct': 5.0, 'note': ''})

    assert '  평가액: 2,100,000원 (+5.0%)' in body
    assert body.count('평가액:') == 1                  # 바낸현물·선물 블록엔 없다
    upbit_block = body.split('[바낸현물]')[0]
    assert '평가액: 2,100,000원 (+5.0%)' in upbit_block
    assert warn is False


def test_upbit_value_line_negative_pct(tmp_path):
    """손실이면 부호가 그대로 -, 역시 warn 아님 (평가액은 판정 대상이 아니다)."""
    sides = _spot_sides(tmp_path)
    body, warn = cdr.build_report(DAY, sides, None,
                                  {'value': 1899000.0, 'pct': -5.14, 'note': ''})

    assert '  평가액: 1,899,000원 (-5.1%)' in body
    assert warn is False


def test_upbit_value_without_principal_warns(tmp_path):
    """평가액은 있는데 원금을 못 읽으면 등락률을 지어내지 않고 사유를 싣는다 (warn)."""
    sides = _spot_sides(tmp_path)
    body, warn = cdr.build_report(DAY, sides, None,
                                  {'value': 1000000.0, 'pct': None,
                                   'note': '원금 파일 없음(report_principal.json)'})

    assert '  평가액: 1,000,000원 (원금 미설정 — 원금 파일 없음(report_principal.json))' in body
    assert warn is True


def test_upbit_value_fetch_failure_warns(tmp_path):
    """조회 자체가 실패하면 숫자 없이 사유만 (warn)."""
    sides = _spot_sides(tmp_path)
    body, warn = cdr.build_report(DAY, sides, None,
                                  {'value': None, 'pct': None, 'note': '업비트 키 미설정'})

    assert '  평가액: 조회 실패 — 업비트 키 미설정' in body
    assert warn is True


def test_upbit_value_partial_price_failure_warns(tmp_path):
    """가격 누락 심볼이 있으면 평가액이 과소일 수 있으니 메모를 붙이고 warn (fail-loud)."""
    sides = _spot_sides(tmp_path)
    body, warn = cdr.build_report(DAY, sides, None,
                                  {'value': 500000.0, 'pct': 1.0,
                                   'note': '가격 조회 실패: XRP'})

    assert '  평가액: 500,000원 (+1.0%) — 가격 조회 실패: XRP' in body
    assert warn is True


def test_upbit_value_absent_keeps_report_identical(tmp_path):
    """미주입(None)이면 줄 자체가 없고 기존 출력과 완전히 동일하다."""
    sides = _spot_sides(tmp_path)
    fut = _fut(tmp_path, SAMPLE_BLOCK)
    base, base_warn = cdr.build_report(DAY, sides, fut)
    same, same_warn = cdr.build_report(DAY, sides, fut, None)

    assert '평가액' not in base
    assert same == base and same_warn == base_warn is False


# ── 평가액 조회 (pyupbit 없이 REST 직접 — timeout 이 걸린 경로) ──
def test_upbit_account_value_sums_krw_and_coins(tmp_path):
    """KRW 는 그대로, 코인은 (balance+locked)×현재가. 가격 없는 심볼은 0 + 메모."""
    calls = []
    priv, pub = _fake_http(
        accounts=(200, [
            {'currency': 'KRW', 'balance': '100000.0', 'locked': '0.0'},
            {'currency': 'BTC', 'balance': '0.01', 'locked': '0.005'},
            {'currency': 'XRP', 'balance': '100.0', 'locked': '0.0'},   # ticker 응답에 없음
            {'currency': 'DOGE', 'balance': '0.0', 'locked': '0.0'},    # 잔고 0 → 무시
        ]),
        ticker=(200, [{'market': 'KRW-BTC', 'trade_price': 100000000.0}]),
        calls=calls,
    )
    old = _swap(UPBIT_ACCESS_KEY='ak', UPBIT_SECRET_KEY='sk',
                _upbit_get=priv, _upbit_public_get=pub)
    try:
        value, err = cdr._upbit_account_value()
    finally:
        _restore(old)

    assert _approx(value, 100000.0 + 0.015 * 100000000.0)   # 1,600,000
    assert err == '가격 조회 실패: XRP'
    # 현재가는 심볼마다가 아니라 배치 1회 (레이트리밋)
    assert calls == [('/v1/accounts', {}),
                     ('/v1/ticker', {'markets': 'KRW-BTC,KRW-XRP'})]


def test_upbit_account_value_without_keys(tmp_path):
    old = _swap(UPBIT_ACCESS_KEY='', UPBIT_SECRET_KEY='')
    try:
        value, err = cdr._upbit_account_value()
    finally:
        _restore(old)
    assert value is None and err == '업비트 키 미설정'


def test_upbit_account_value_rejects_error_status(tmp_path):
    """잔고 API 가 오류 상태를 주면 0원으로 착각하지 않고 조회 실패로 둔다."""
    priv, pub = _fake_http(accounts=(401, {'error': {'message': 'invalid_access_key'}}))
    old = _swap(UPBIT_ACCESS_KEY='ak', UPBIT_SECRET_KEY='sk',
                _upbit_get=priv, _upbit_public_get=pub)
    try:
        value, err = cdr._upbit_account_value()
    finally:
        _restore(old)
    assert value is None and '잔고 조회 실패 (HTTP 401)' in err


def test_upbit_account_value_price_batch_failure_keeps_krw(tmp_path):
    """ticker 배치가 통째로 죽어도 KRW 만이라도 돌려주고 심볼을 전부 메모에 싣는다."""
    priv, pub = _fake_http(
        accounts=(200, [
            {'currency': 'KRW', 'balance': '250000.0', 'locked': '0.0'},
            {'currency': 'BTC', 'balance': '0.01', 'locked': '0.0'},
            {'currency': 'ETH', 'balance': '1.0', 'locked': '0.0'},
        ]),
        ticker=RuntimeError('timeout'),
    )
    old = _swap(UPBIT_ACCESS_KEY='ak', UPBIT_SECRET_KEY='sk',
                _upbit_get=priv, _upbit_public_get=pub)
    try:
        value, err = cdr._upbit_account_value()
    finally:
        _restore(old)
    assert _approx(value, 250000.0) and err == '가격 조회 실패: BTC, ETH'


# ── 원금 (report_principal.json) ──
def test_principal_adds_only_new_accepted_krw_deposits(tmp_path):
    """last_deposit_check 이후 · ACCEPTED · KRW 입금만 가산한다."""
    path = _principal_file(tmp_path)
    calls = []
    priv, _pub = _fake_http(deposits=(200, [
        _deposit('u-new', '2026-08-31T10:00:00+09:00', 500000.0),            # 가산
        _deposit('u-old', '2026-08-29T10:00:00+09:00', 900000.0),            # 체크시각 이전
        _deposit('u-pend', '2026-08-31T11:00:00+09:00', 700000.0,
                 state='PROCESSING'),                                         # 미확정
        _deposit('u-btc', '2026-08-31T12:00:00+09:00', 1.0, currency='BTC'),  # KRW 아님
    ]), calls=calls)

    old = _swap(PRINCIPAL_FILE=path, _upbit_get=priv)
    try:
        principal, err = cdr._load_principal(cdr._parse_iso(NOW))
    finally:
        _restore(old)

    assert _approx(principal, 2500000.0) and err == ''
    assert calls == [('/v1/deposits',
                      {'currency': 'KRW', 'limit': 100, 'order_by': 'desc'})]
    saved = _read_json(path)
    assert _approx(saved['principal_krw'], 2500000.0)
    assert not os.path.exists(f'{path}.tmp.{os.getpid()}')   # 원자적 저장 흔적 없음


def test_principal_watermark_advances_to_latest_deposit_not_now(tmp_path):
    """워터마크는 now 가 아니라 관측한 입금의 max(done_at) 으로만 전진한다.

    now 로 밀면 반영이 늦어 done_at 이 과거인 채 나중에 나타나는 입금이 유실된다.
    """
    path = _principal_file(tmp_path)
    priv, _pub = _fake_http(deposits=(200, [
        _deposit('u-a', '2026-08-31T10:00:00+09:00', 100000.0),
        _deposit('u-b', '2026-08-31T18:30:00+09:00', 200000.0),   # 최신
    ]))
    old = _swap(PRINCIPAL_FILE=path, _upbit_get=priv)
    try:
        principal, err = cdr._load_principal(cdr._parse_iso(NOW))
    finally:
        _restore(old)

    saved = _read_json(path)
    assert _approx(principal, 2300000.0) and err == ''
    assert saved['last_deposit_check'] == '2026-08-31T18:30:00+09:00'
    assert saved['last_deposit_check'] != NOW
    assert saved['processed_uuids'] == ['u-b']       # 워터마크 시각 건만 기록


def test_principal_no_new_deposit_leaves_file_untouched(tmp_path):
    """새 입금이 없으면 상태파일을 아예 건드리지 않는다 (매일 같은 내용 재기록 금지)."""
    path = _principal_file(tmp_path, last='2026-08-31T10:00:00+09:00',
                           processed=['u-a'])
    before = _read_json(path)
    mtime = os.path.getmtime(path)
    priv, _pub = _fake_http(deposits=(200, [
        _deposit('u-a', '2026-08-31T10:00:00+09:00', 100000.0),   # 이미 반영분
        _deposit('u-old', '2026-08-20T10:00:00+09:00', 500000.0),
    ]))
    old = _swap(PRINCIPAL_FILE=path, _upbit_get=priv)
    try:
        principal, err = cdr._load_principal(cdr._parse_iso(NOW))
    finally:
        _restore(old)

    assert _approx(principal, 2000000.0) and err == ''
    assert _read_json(path) == before
    assert os.path.getmtime(path) == mtime


def test_principal_same_timestamp_new_uuid_is_added_once(tmp_path):
    """워터마크와 같은 시각의 신규 입금은 가산하되, 다음 실행에서 또 더하지 않는다."""
    path = _principal_file(tmp_path, last='2026-08-31T10:00:00+09:00',
                           processed=['u-a'])
    batch = (200, [
        _deposit('u-a', '2026-08-31T10:00:00+09:00', 100000.0),   # 이미 반영
        _deposit('u-b', '2026-08-31T10:00:00+09:00', 300000.0),   # 같은 시각 신규
    ])
    priv, _pub = _fake_http(deposits=batch)
    old = _swap(PRINCIPAL_FILE=path, _upbit_get=priv)
    try:
        first, _e1 = cdr._load_principal(cdr._parse_iso(NOW))
        saved = _read_json(path)
        second, _e2 = cdr._load_principal(cdr._parse_iso(NOW))   # 같은 응답 재조회
    finally:
        _restore(old)

    assert _approx(first, 2300000.0)
    assert saved['last_deposit_check'] == '2026-08-31T10:00:00+09:00'
    assert saved['processed_uuids'] == ['u-a', 'u-b']
    assert _approx(second, 2300000.0)          # 중복 가산 없음


def test_principal_ignores_future_dated_deposit(tmp_path):
    """now 보다 미래인 입금은 이번엔 세지 않는다 — 워터마크를 미래로 밀면 안 된다."""
    path = _principal_file(tmp_path)
    priv, _pub = _fake_http(deposits=(200, [
        _deposit('u-future', '2026-09-02T10:00:00+09:00', 400000.0),
    ]))
    old = _swap(PRINCIPAL_FILE=path, _upbit_get=priv)
    try:
        principal, err = cdr._load_principal(cdr._parse_iso(NOW))
    finally:
        _restore(old)

    assert _approx(principal, 2000000.0) and err == ''
    assert _read_json(path)['last_deposit_check'] == '2026-08-30T00:00:00+09:00'


def test_principal_missing_file(tmp_path):
    old = _swap(PRINCIPAL_FILE=os.path.join(str(tmp_path), 'nope.json'))
    try:
        principal, err = cdr._load_principal(cdr._parse_iso(NOW))
    finally:
        _restore(old)
    assert principal is None and err == '원금 파일 없음(report_principal.json)'


def test_principal_kept_when_deposit_api_fails(tmp_path):
    """입금 API 실패 시 저장된 원금은 그대로 쓰고, 체크시각은 밀지 않는다."""
    path = _principal_file(tmp_path)

    def boom(p, params):
        raise RuntimeError('timeout')

    old = _swap(PRINCIPAL_FILE=path, _upbit_get=boom)
    try:
        principal, err = cdr._load_principal(cdr._parse_iso(NOW))
    finally:
        _restore(old)

    assert _approx(principal, 2000000.0) and '입금동기화 실패' in err
    saved = _read_json(path)
    assert saved['last_deposit_check'] == '2026-08-30T00:00:00+09:00'   # 미갱신
    assert _approx(saved['principal_krw'], 2000000.0)


def test_principal_kept_when_deposit_api_returns_error_status(tmp_path):
    path = _principal_file(tmp_path)
    old = _swap(PRINCIPAL_FILE=path,
                _upbit_get=lambda p, params: (401, {'error': 'invalid_access_key'}))
    try:
        principal, err = cdr._load_principal(cdr._parse_iso(NOW))
    finally:
        _restore(old)

    assert _approx(principal, 2000000.0) and '입금동기화 실패 (HTTP 401)' in err
    assert _read_json(path)['last_deposit_check'] == '2026-08-30T00:00:00+09:00'


def test_principal_zero_is_rejected(tmp_path):
    """원금 0 이면 등락률을 못 낸다 — 0 나눗셈 대신 사유를 돌려준다."""
    path = _principal_file(tmp_path, principal=0.0)
    old = _swap(PRINCIPAL_FILE=path,
                _upbit_get=lambda p, params: (200, []))
    try:
        principal, err = cdr._load_principal(cdr._parse_iso(NOW))
    finally:
        _restore(old)
    assert principal is None and '0 이하' in err


# ── 미실행 블록 / 과거 날짜 / redaction ──
def test_upbit_value_shown_even_when_upbit_did_not_run(tmp_path):
    """업비트가 미실행이어도 평가액은 싣는다 — 그럴 때일수록 실계좌 확인이 급하다."""
    sides = _spot_sides(tmp_path)
    missing = cdr.SideResult('업비트', os.path.join(str(tmp_path), 'nope.log'))
    missing.parse(DAY)
    sides[0] = missing

    body, warn = cdr.build_report(DAY, sides, None,
                                  {'value': 2100000.0, 'pct': 5.0, 'note': ''})
    block = body.split('[바낸현물]')[0]
    assert '[업비트] ⚠️ 미실행' in block
    assert '  평가액: 2,100,000원 (+5.0%)' in block
    assert warn is True                      # 미실행 자체의 warn 은 그대로


def test_upbit_value_skipped_for_past_date(tmp_path):
    """--date 로 과거를 볼 땐 '지금' 잔고를 붙이지 않는다 (원금 동기화도 안 돈다)."""
    called = []
    old = _swap(_upbit_account_value=lambda: called.append('v') or (1.0, ''),
                _load_principal=lambda now: called.append('p') or (1.0, ''))
    try:
        past = cdr._upbit_value_for('2020-01-01')
        today = cdr._upbit_value_for(cdr._service_today())
    finally:
        _restore(old)

    assert past is None and called == ['v', 'p']      # 과거 조회는 호출 자체가 없다
    assert today == {'value': 1.0, 'pct': 0.0, 'note': ''}


def test_upbit_value_for_swallows_api_errors(tmp_path):
    """외부 API 가 터져도 리포트 본체는 나가야 한다 — note 만 남기고 값은 None."""
    def boom():
        raise RuntimeError('업비트 5xx')

    old = _swap(_upbit_account_value=boom)
    try:
        uv = cdr._upbit_value_for(cdr._service_today())
    finally:
        _restore(old)
    assert uv['value'] is None and uv['pct'] is None and '업비트 5xx' in uv['note']


def test_redact_masks_upbit_keys_and_bearer(tmp_path):
    """업비트 키/JWT 가 예외 메시지로 새 나가지 않는다 (텔레그램 마스킹은 그대로)."""
    old = _swap(UPBIT_ACCESS_KEY='AK-SECRET-1234', UPBIT_SECRET_KEY='SK-SECRET-5678')
    try:
        out = cdr._redact('HTTP 401 key=AK-SECRET-1234 sec=SK-SECRET-5678 '
                          'Authorization: Bearer eyJhbG.ciOiJI-UzI1NiJ9 '
                          'bot123456:AAH-fake_token')
    finally:
        _restore(old)

    assert 'AK-SECRET-1234' not in out and 'SK-SECRET-5678' not in out
    assert 'eyJhbG.ciOiJI-UzI1NiJ9' not in out and 'Bearer <REDACTED>' in out
    assert 'bot<REDACTED>' in out and 'AAH-fake_token' not in out


def test_now_kst_is_always_tz_aware(tmp_path):
    """ZoneInfo 가 없어도 UTC+9 고정 오프셋 — naive 로 새면 입금 시각 비교가 터진다."""
    assert cdr._now_kst().tzinfo is not None
    old = _swap(ZoneInfo=None)
    try:
        now = cdr._now_kst()
        parsed = cdr._parse_iso('2026-09-01T09:20:00')      # tz 표기 없는 입력
    finally:
        _restore(old)
    assert now.utcoffset() == cdr.KST_FIXED.utcoffset(None)
    assert parsed is not None and parsed.utcoffset() == cdr.KST_FIXED.utcoffset(None)


# ──────────────────── state-ref (현물 dry-run 이 업비트 state 를 참조) ────────────────────
SPOT_STATE_REF = ("📎 state-ref: trade_state.json 참조 (dry-run, schema=V24, "
                  "members=['D_SMA42'], bar_counter=1234, "
                  "last_bar=2026-08-30T00:00:00Z, snapshots=7)")
SPOT_STATE_REF_FAIL = ('🚨 state-ref 참조 실패: 참조 파일 없음: trade_state.json '
                       '→ fail-closed (fresh 초기화로 대체하지 않음)')
# 참조 줄 뒤에 오는 들여쓴 정보줄 — 파일명을 덮어쓰면 안 된다 (실물 로그 순서 그대로)
SPOT_STATE_REF_DRIFT = '  📎 state-ref: drift 평가 보유비중 = 참조 last_target_snapshot 가정 (실잔고 아님)'


def _spot_sides_with(tmp_path, extra_bn=(), dry=(True, True)):
    """바낸현물 쪽에만 로그 줄을 더한 현물 2축 (그 외는 _spot_sides 와 동일한 블록)."""
    sides = []
    for (name, fname), dry_run in zip(SPOT_SIDES, dry):
        body = SPOT_LOG.format(day=DAY, rid='0005abcd', canary='',
                               tgt='BTC:33.3%, ETH:33.3%', cash='33.4', dry=dry_run)
        if name == '바낸현물' and extra_bn:
            head, rest = body.split('\n', 1)
            extra = ''.join(f'[{DAY} 00:05:10] [0005abcd] {m}\n' for m in extra_bn)
            body = head + '\n' + extra + rest
        s = cdr.SideResult(name, _write(tmp_path, body, fname))
        s.parse(DAY)
        sides.append(s)
    return sides


def test_spot_state_ref_line_is_parsed_and_shown(tmp_path):
    """참조 파일명을 읽어 모드 표기·안내줄에 싣는다 (경고는 아니다)."""
    sides = _spot_sides_with(tmp_path, extra_bn=[SPOT_STATE_REF, SPOT_STATE_REF_DRIFT],
                             dry=(False, True))
    assert sides[1].state_ref == 'trade_state.json', sides[1].state_ref
    assert sides[0].state_ref is None
    body, warn = cdr.build_report(DAY, sides)

    assert '[바낸현물] 00:05:10 (dry(state-ref))' in body, body
    assert '실행 모드: 업비트=LIVE, 바낸현물=dry-run(state-ref)' in body, body
    assert ('ℹ 바낸현물 는 trade_state.json 를 state-ref 로 참조 → '
            '타겟 일치는 파이프라인 검증이며 신호 독립 비교가 아님') in body, body
    # 같은 블록에서 state-ref 줄만 뺀 경우와 warn 판정이 같아야 한다 (정보 표시일 뿐)
    _, base_warn = cdr.build_report(DAY, _spot_sides_with(tmp_path, dry=(False, True)))
    assert warn == base_warn, (warn, base_warn)


def test_spot_state_ref_failure_is_error(tmp_path):
    """참조 실패는 fail-closed 종료 — '거래 완료' 마커가 있어도 error 로 남는다."""
    sides = _spot_sides_with(tmp_path, extra_bn=[SPOT_STATE_REF_FAIL])
    s = sides[1]
    assert s.result_kind == 'error', (s.result_kind, s.result_label)
    assert 'state-ref' in s.result_label, s.result_label
    assert s.state_ref is None, s.state_ref      # 실패 라인을 참조 파일명으로 오인하지 않는다
    body, warn = cdr.build_report(DAY, sides)
    assert warn is True
    assert 'state-ref 참조 실패' in body, body


def test_spot_state_ref_filename_with_space_is_captured(tmp_path):
    """파일명에 공백이 있어도 통째로 잡는다 (뒤따르는 '참조'/'→' 까지가 경계)."""
    line = ("📎 state-ref: my trade state.json 참조 (dry-run, schema=V24, "
            "members=['D_SMA42'], bar_counter=1234, last_bar=2026-08-30T00:00:00Z, snapshots=7)")
    sides = _spot_sides_with(tmp_path, extra_bn=[line, SPOT_STATE_REF_DRIFT])
    assert sides[1].state_ref == 'my trade state.json', sides[1].state_ref
    body, _warn = cdr.build_report(DAY, sides)
    assert 'ℹ 바낸현물 는 my trade state.json 를 state-ref 로 참조' in body, body


def test_spot_state_ref_failure_reason_keeps_state_ref_label(tmp_path):
    """실패 사유가 'state … 손상' 을 담아도 라벨은 state-ref 쪽이 남는다 (마커 우선순위)."""
    for reason in ('참조 state JSON 손상: Expecting value', '참조 state 손상 의심'):
        sides = _spot_sides_with(
            tmp_path, extra_bn=[f'🚨 state-ref 참조 실패: {reason} → fail-closed'])
        s = sides[1]
        assert s.result_kind == 'error', (reason, s.result_kind)
        assert s.result_label == 'state-ref 참조 실패 (fail-closed)', (reason, s.result_label)


def test_spot_report_without_state_ref_is_unchanged(tmp_path):
    """state-ref 줄이 없으면 출력은 종전과 완전히 같다."""
    sides = _spot_sides(tmp_path)
    assert all(s.state_ref is None for s in sides)
    body, warn = cdr.build_report(DAY, sides)
    assert 'state-ref' not in body and 'ℹ' not in body, body
    assert '[바낸현물] 00:05:10 (dry)' in body, body
    assert '실행 모드: 양쪽 dry-run' in body, body
    assert warn is False


# ─── pytest 없는 환경(오라클 .venv)용 최소 러너 — pytest 스타일은 그대로 ───
if __name__ == '__main__':  # pragma: no cover
    import tempfile
    import traceback

    fails = 0
    for _name, _fn in sorted(globals().items()):
        if not (_name.startswith('test_') and callable(_fn)):
            continue
        with tempfile.TemporaryDirectory() as _tmp:
            try:
                _fn(_tmp) if _fn.__code__.co_argcount else _fn()
                print(f'  PASS  {_name}')
            except Exception:
                fails += 1
                print(f'  FAIL  {_name}')
                traceback.print_exc()
    print('ALL PASS' if not fails else f'{fails} FAILED')
    sys.exit(1 if fails else 0)
