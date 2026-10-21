"""Fail-open supervision of an already systemd-owned guest action server.

systemd remains the parent: unlike polling an unrelated PID, it retains the
actual wait status after the server exits. Never adopt a desktop/session unit
or restart a healthy process to install supervision. Unsupported launch
arrangements are explicitly left unchanged.
"""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass
from typing import Any, Callable, Literal, Sequence

from lop_osworld_v2_adapter import guest_disk

#: The action server's listening port. The guest's control server and the action
#: server share :5000 on this AMI (``guest_loss.HEALTH_PROBE_SOURCE`` reads the
#: same port as 0x1388), and the layout probe needs it to identify the process
#: that owns the listening socket.
ACTION_SERVER_PORT = 5000

#: ONE shared deadline for the whole layout capture, matching the console
#: capture discipline in ``guest_loss``: the diagnosis must never outlive the
#: episode's setup path. The guest script carries a shorter internal deadline so
#: it can still print partial evidence before the host abandons it.
LAYOUT_BUDGET_S = 4.0

#: The guest's OWN subprocess deadline for the probe, and the reason the socket
#: deadline is not ``LAYOUT_BUDGET_S``: ``guest_deadline_for`` subtracts the
#: 10 s transport margin, so a 4 s socket timeout asks the guest to finish in
#: 1 s -- less than the script's 3 s internal deadline, which would cut a
#: loaded guest off before it could print anything. The provider derives the
#: socket timeout from this value; ``_run_bounded`` keeps the 4 s HOST budget
#: authoritative regardless of how long the guest side is allowed.
LAYOUT_GUEST_DEADLINE_S = 3.5

#: Refusal strings for the socket-ownership checks the install adds. They are
#: fixed vocabulary raised by ``INSTALL_SOURCE`` and, like every other refusal
#: tied to the guest's process layout, they trigger the read-only layout
#: capture when they fire.
SOCKET_SCAN_INCOMPLETE = "action-server socket scan was incomplete"
NO_LISTENING_SOCKET = "action server has no listening socket"
SOCKET_OWNER_UNREACHABLE = "socket owner is outside the action-server service"

#: Refusals decided by what the guest's units look like. The install command's
#: other refusals (journal policy, an existing hook, a oneshot service) already
#: name a property of a discovered unit, so they need no process layout. This is
#: a frozenset of the fixed, guest-authored-never reason strings from
#: ``INSTALL_SOURCE``; a type name or arbitrary text never matches.
LAYOUT_REFUSALS = frozenset(
    {
        "unsupported action-server command",
        "unsupported action-server ancestry",
        "action server has no private service",
        "unsupported service name",
        "service does not exclusively own action server",
        SOCKET_SCAN_INCOMPLETE,
        NO_LISTENING_SOCKET,
        SOCKET_OWNER_UNREACHABLE,
    }
)

#: The ``KillMode`` the supervisor drop-in sets (``INSTALL_SOURCE`` writes
#: exactly ``KillMode=process`` and re-asserts it on re-entry). This is the
#: mode the drop-in WOULD install, not the unit's current effective mode -- the
#: drop-in is absent on the refusal path that triggers the layout capture.
#: Recorded beside the cgroup comparison because membership is NOT survival
#: under this mode: systemd signals only the MainPID, so every other process
#: survives a restart whether or not it shares the unit's cgroup. The
#: membership fact still decides whether an explicit cgroup-wide kill can
#: reach such a holder.
SUPERVISOR_KILL_MODE = "process"

#: The explicit restart's internal budget, in the guest. The SCRIPT deadline is
#: one second inside ``RESTART_GUEST_DEADLINE_S``: the guest's subprocess budget
#: starts at launch, while the script's clock starts a moment later, so the two
#: cannot share one number (the layout path documents the same requirement in
#: the other direction). ``RESTART_KILL_GRACE_S`` is the bounded wait for every
#: recorded holder to release the socket; ``RESTART_VERIFY_S`` is the bounded
#: wait for a FRESH listener after ``systemctl start``. Their sum is the script
#: deadline, leaving the systemctl calls to fit inside the same clock.
RESTART_KILL_GRACE_S = 2.0
RESTART_VERIFY_S = 4.0
RESTART_SCRIPT_DEADLINE_S = 6.0
RESTART_GUEST_DEADLINE_S = 7.0

#: The restart runs as a TRANSIENT SERVICE, not a scope: the target unit's
#: process is the one serving our HTTP request, so a restart running in its
#: cgroup would kill its own executor and its own response channel. The service
#: writes its verdict here and the host reads it from a LATER command, once the
#: action server is back. The token ties a result to the dispatch that asked for
#: it, so a stale file from an earlier restart cannot be read as this one.
RESTART_RESULT_PATH = "/var/lib/lop-action-supervisor/restart-result.json"

#: The restart script on disk. The host dispatches it with ``python3 -c``;
#: the watchdog dispatches the SAME file with ``systemd-run ... --wait``, so
#: the all-process restart has one implementation and two dispatchers.
RESTART_SCRIPT_PATH = "/var/lib/lop-action-supervisor/restart.py"

#: The guest-local watchdog: a SEPARATE unit, in its own cgroup, that can bring
#: the action server back when the host cannot. The host's recovery channel is
#: ``/execute`` on the action server itself, so it is unavailable exactly when
#: the server is down; systemd's own restart policy covers the first three
#: deaths (``StartLimitBurst=3``) and then leaves the unit ``failed`` for the
#: rest of the hour. This unit is the path back that does not run through the
#: thing that died.
#:
#: THE START LIMIT STAYS AT 3. Raising the burst would make a crash loop rarer
#: and its diagnosis harder without giving the service a recovery path that
#: survives it -- it only postpones the same wall. The honest fix is an
#: independent actor that can ``reset-failed`` the wall away.
WATCHDOG_UNIT = "lop-action-watchdog.service"
WATCHDOG_RECORD_PATH = "/var/lib/lop-action-supervisor/watchdog.json"

#: Bounds shared by the script and the host-side sanitizer. The interval is
#: deliberately slower than systemd's ``RestartSec=1``: the watchdog must never
#: race a restart systemd is already performing. The absent-listener grace is
#: longer than a normal action-server start, so a slow boot is observed, never
#: interrupted.
WATCHDOG_INTERVAL_S = 10.0
WATCHDOG_MIN_INTERVAL_S = 30.0
WATCHDOG_WINDOW_S = 3600.0
WATCHDOG_MAX_ATTEMPTS = 5
#: A dispatch that provably changed nothing (``no-effect``) draws on a smaller
#: budget than a real repair: a permanently broken unit must not spend a later
#: real failure's five attempts. Mirrors ``MAX_INEFFECTIVE`` in the script.
WATCHDOG_MAX_INEFFECTIVE = 2
WATCHDOG_ABSENT_GRACE_S = 30.0
WATCHDOG_EVENT_CAP = 8

#: Positional contract between ``restart_service_argv`` and ``RESTART_SOURCE``.
#: CPython's ``-c`` puts the command string where argv[0] would be and the
#: first real argument at argv[1], so the script unpacks ``sys.argv[1:]`` in
#: exactly this order. One tuple, not two literals that can drift.
RESTART_ARGUMENT_ORDER = ("unit", "port", "result_path", "token")
RESTART_DISPATCH_TIMEOUT_S = 15.0
RESTART_RESULT_TIMEOUT_S = 5.0
RESTART_RESULT_INTERVAL_S = 0.25
RESTART_RESULT_BUDGET_S = 20.0

#: The install's socket scan must fit inside the install command's 5 s guest
#: deadline (``guest_deadline_for(15)``). One second is several times what the
#: ~200-process AMI needs, and leaves the existing systemctl calls their room;
#: an incomplete scan refuses rather than spending the whole budget.
INSTALL_SCAN_DEADLINE_S = 1.0


class GenerationMismatch(ValueError):
    """systemd observed a restart that the guest capture history cannot prove."""


class SupervisorOutputUnparseable(ValueError):
    """The captured guest output carried no JSON payload to parse.

    WHY A NAMED FAILURE. The setup evidence records only the exception's TYPE
    name, never its text, because the text can carry a password prompt. A bare
    ``JSONDecodeError`` said nothing about WHAT failed, so the episode read as
    "supervision unavailable" with no way to tell a guest that printed noise
    from a guest that printed nothing. This name is fixed vocabulary and never
    contains guest text. It subclasses ``ValueError`` because that is what the
    ``json`` module raised here before, so no caller changes meaning.
    """


@dataclass(frozen=True)
class ReceiptResult:
    """Only a complete verified read may authorize input on an existing binding."""

    state: Literal["new_restart", "verified_none", "read_failed"]
    recovery_exhausted: bool = False


# ExecStopPost gets the old process's wait status BEFORE Restart replaces it.
# Keep only four bounded records, including the terminal crash-loop failure.
CAPTURE_SOURCE = r"""
import json, os, pathlib, subprocess, sys, time
unit = sys.argv[1]
root = pathlib.Path('/var/lib/lop-action-supervisor')
path = root / (unit + '.json')
record = {'restart': time.time_ns(), 'unit': unit,
          'exit_code': os.environ.get('EXIT_CODE', 'unknown'),
          'exit_status': os.environ.get('EXIT_STATUS', 'unknown'),
          'service_result': os.environ.get('SERVICE_RESULT', 'unknown'),
          'stderr_tail': '', 'journal_tail_status': 'pending'}
try:
    history = json.loads(path.read_text())
    records = history['records']
    captured = history['captured_exits']
    if not isinstance(records, list) or type(captured) is not int or captured < 0:
        raise ValueError('invalid capture history')
except (OSError, ValueError, KeyError, TypeError):
    # Never invent a new baseline after corruption: the report must retain a
    # detectable disagreement with systemd, not bless a restarted process.
    history, records, captured = {}, [], 0
records = (records + [record])[-4:]
history.update(schema_version=1, records=records, captured_exits=captured + 1)
def persist():
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(history))
    tmp.replace(path)
# The exit status is authoritative and cheap; persist it BEFORE asking a
# potentially overloaded journal. Even a timed-out journal must not hide death.
persist()
try:
    p = subprocess.run(['journalctl', '--no-pager', '-o', 'cat', '-n', '30', '-u', unit],
        capture_output=True, text=True, timeout=2, check=True)
    record['stderr_tail'] = p.stdout[-4096:]
    record['journal_tail_status'] = 'ok'
except Exception:
    record['journal_tail_status'] = 'unavailable'
persist()
"""

#: The guest-local recovery path. Runs as its own service in its own cgroup, so
#: it is still alive when the target unit is ``failed`` and the host has no
#: transport to dispatch a restart through.
#:
#: THE TRIGGER TABLE IS THE WHOLE DESIGN. ``systemctl start`` is a no-op on a
#: unit that is already active, and ``reset-failed`` on a non-failed unit only
#: clears bookkeeping, so an intervention aimed at an ``active`` unit would
#: change nothing while spending an attempt and recording success -- false
#: evidence, and a cap burned three times over before a real ``failed`` state
#: arrives. Each state therefore gets exactly the action that can change it:
#:
#:   failed/inactive, no listener -> reset-failed + start   (attempt)
#:   failed/inactive, listener that does not answer
#:                                -> verified all-process restart (attempt)
#:   active/activating/... with a listener that does not answer
#:                                -> verified restart only after TWO consecutive
#:                                   observations (attempt on the second)
#:   any state, listener that answers -> none; observed as ``serving``
#:   active/activating/reloading/deactivating, no listener
#:                                -> none; observed as ``no-listener`` only
#:
#: WHY THE ACTIVE WEDGE IS REPAIRED HERE. The host's only channel is the same
#: listener: its recovery probe fails against a wedged server and
#: ``_restart_supervisor_unit`` runs only after a SUCCESSFUL probe, so a
#: permanent wedge is repaired by neither side. Two consecutive unresponsive
#: observations are required before acting, so a single slow response is never
#: mistaken for death.
#:
#: THE VERDICT IS DERIVED FROM THE PAYLOAD, NEVER FROM A FIELD THE SCRIPT DOES
#: NOT WRITE. ``RESTART_SOURCE`` reports observations (pids, start ticks,
#: ancestry, receipt); only the host's ``restart_record`` turns them into a
#: ``status``. This watchdog therefore applies the same rule guest-side --
#: complete pre/post scans, post owners that postdate the kill and descend from
#: the new MainPID, a postdating MainPID and a non-null receipt -- instead of
#: reading a ``status`` key that does not exist.
#:
#: OBSERVATIONS ARE NOT ATTEMPTS. A state the watchdog cannot repair is
#: recorded in ``last_observation`` (fixed words) without touching
#: ``attempts``, so a dead-but-blocked server is distinguishable from a
#: healthy one in the host's evidence instead of both reading ``attempts: 0``.
#: A dispatch that provably changed nothing (``no-effect``) draws on a smaller
#: budget (2/hour) than real repairs, so a permanently broken unit cannot
#: starve a later real failure of its five attempts.
#:
#: RATE LIMITS. One intervention per ``MIN_INTERVAL_S`` (never race systemd's
#: own ``RestartSec=1`` attempt), at most ``MAX_ATTEMPTS`` per rolling
#: ``WINDOW_S`` -- the same hour systemd uses -- and at most
#: ``MAX_INEFFECTIVE`` no-effect dispatches per window. ``capped`` is
#: recomputed from the window every cycle, so it clears when the hour rolls.
#:
#: THE RECORD IS NUMBERS AND FIXED WORDS ONLY: no unit name, no argv, no
#: environment, no guest text. Attempt ticks are monotonic microseconds, the
#: same clock as systemd's ``*TimestampMonotonic`` properties. The event ring
#: is bounded because the file is rewritten on every intervention. A process
#: restart of the WATCHDOG seeds its counters from that same file: the unit's
#: own ``Restart=always`` must not turn ``attempts`` back to zero and hand the
#: crash loop a fresh cap. Monotonic stamps do not survive a BOOT, so a stamp
#: in the future is ignored rather than trusted.
#:
#: FAIL-OPEN. A defect here must never fail an episode: every cycle is caught,
#: the loop continues, and a record it cannot write is simply missing evidence.
WATCHDOG_SOURCE = r"""
import json, pathlib, socket, subprocess, sys, time

unit, port, record_path, restart_script, result_path = (
    sys.argv[1], int(sys.argv[2]), pathlib.Path(sys.argv[3]),
    sys.argv[4], pathlib.Path(sys.argv[5]))
INTERVAL_S = 10.0
MIN_INTERVAL_S = 30.0
WINDOW_S = 3600.0
MAX_ATTEMPTS = 5
MAX_INEFFECTIVE = 2
ABSENT_GRACE_S = 30.0
EVENT_CAP = 8
PROBE_TIMEOUT_S = 2.0
START_VERIFY_S = 9.0
RESTART_TIMEOUT_S = 15.0

TRIGGERS = ('failed', 'inactive', 'unresponsive')
ACTIONS = ('reset-failed+start', 'verified-restart')
OUTCOMES = ('started', 'start-failed', 'reset-failed-error', 'no-effect',
            'restarted', 'restart-failed', 'throttled', 'capped')
OBSERVATIONS = ('none', 'serving', 'no-listener', 'unresponsive', 'unknown')

absent_since = None
unresponsive_streak = 0
#: The throttle compares against the last DISPATCH's clock value, not the
#: record's microsecond stamp: a legitimate intervention at monotonic 0 would
#: otherwise read as "never attempted" and defeat the minimum interval.
last_attempt = None
recent = []
ineffective = []
state = {'attempts': 0, 'intervened': False, 'capped': False,
         'last_attempt_us': 0, 'last_trigger': 'none',
         'last_action': 'none', 'last_outcome': 'none',
         'last_observation': 'none', 'events': []}

def monotonic_us():
    return int(time.monotonic() * 1000000)

def run(*args):
    subprocess.check_output(args, text=True, timeout=5)

def listening():
    for name in ('/proc/net/tcp', '/proc/net/tcp6'):
        try:
            lines = pathlib.Path(name).read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            if len(fields) < 4 or fields[3] != '0A':
                continue
            try:
                if int(fields[1].rsplit(':', 1)[1], 16) == port:
                    return True
            except (IndexError, ValueError):
                continue
    return False

def answers():
    # A LISTEN socket is not a serving server, and an exit code is not proof of
    # one either. The rule is deliberately one-sided: ANY complete HTTP status
    # line means the server is answering (a 404/500 is still a live app, and
    # this probe is not a health endpoint), so a slow-but-alive server is never
    # restarted. Only silence -- connect refused, or no status line inside the
    # bound -- is "unresponsive". Both families are tried because the scan
    # above accepts a v6-only listener.
    connection = None
    for host in ('127.0.0.1', '::1'):
        try:
            connection = socket.create_connection((host, port), PROBE_TIMEOUT_S)
            break
        except OSError:
            continue
    if connection is None:
        return False
    try:
        connection.sendall(b'GET / HTTP/1.0\r\nHost: localhost\r\n\r\n')
        response = b''
        while b'\r\n' not in response and len(response) < 256:
            chunk = connection.recv(64)
            if not chunk:
                break
            response += chunk
        return response.startswith(b'HTTP/')
    except OSError:
        return False
    finally:
        connection.close()

def active_state():
    output = subprocess.check_output(
        ['systemctl', 'show', unit, '-p', 'ActiveState'], text=True, timeout=3)
    for line in output.splitlines():
        if line.startswith('ActiveState='):
            return line[len('ActiveState='):].strip()
    return 'unknown'

def persist():
    document = dict(state)
    document['schema_version'] = 1
    document['events'] = state['events'][-EVENT_CAP:]
    tmp = record_path.with_suffix('.tmp')
    tmp.write_text(json.dumps(document))
    tmp.replace(record_path)

def seed():
    # A restarted watchdog inherits the counters it already spent. Only this
    # file's own fixed vocabulary is copied forward. A stamp in the FUTURE is
    # dropped: CLOCK_MONOTONIC restarts at boot, so a record written before a
    # reboot otherwise throttles every cycle until the wall clock catches up.
    global last_attempt
    try:
        previous = json.loads(record_path.read_text())
    except (OSError, ValueError):
        return
    if not isinstance(previous, dict) or previous.get('schema_version') != 1:
        return
    now = time.monotonic()
    events = previous.get('events')
    if isinstance(events, list):
        for event in events[-EVENT_CAP:]:
            if not isinstance(event, dict):
                continue
            stamp, trigger = event.get('us'), event.get('trigger')
            action, outcome = event.get('action'), event.get('outcome')
            if (type(stamp) is not int or stamp <= 0 or trigger not in TRIGGERS
                    or action not in ACTIONS or outcome not in OUTCOMES):
                continue
            age = now - stamp / 1000000.0
            if age < 0:
                continue
            state['events'] = (state['events'] + [{
                'us': stamp, 'trigger': trigger,
                'action': action, 'outcome': outcome}])[-EVENT_CAP:]
            if age < WINDOW_S:
                recent.append(stamp / 1000000.0)
                if outcome == 'no-effect':
                    ineffective.append(stamp / 1000000.0)
    recent.sort()
    ineffective.sort()
    if recent:
        last_attempt = recent[-1]
        state['last_attempt_us'] = int(recent[-1] * 1000000)
    attempts = previous.get('attempts')
    if type(attempts) is int and 0 <= attempts <= MAX_ATTEMPTS * 1000:
        state['attempts'] = max(attempts, len(state['events']))
    for key, allowed in (('last_trigger', TRIGGERS), ('last_action', ACTIONS),
                         ('last_outcome', OUTCOMES),
                         ('last_observation', OBSERVATIONS)):
        value = previous.get(key)
        if value in allowed:
            state[key] = value
    if state['last_attempt_us'] and not state['last_trigger'] == 'none':
        state['intervened'] = True
    state['capped'] = (len(recent) >= MAX_ATTEMPTS
                       or len(ineffective) >= MAX_INEFFECTIVE)

def postdated(owners, new_main, new_main_start, kill_ticks):
    # The host's freshness rule, derived guest-side from the raw payload.
    if kill_ticks <= 0 or new_main <= 0 or new_main_start < kill_ticks:
        return False
    for owner in owners:
        if not isinstance(owner, dict):
            return False
        start = owner.get('start_ticks')
        pid = owner.get('pid')
        ancestors = owner.get('ancestors')
        if type(pid) is not int or type(start) is not int:
            return False
        if start < kill_ticks or not isinstance(ancestors, list):
            return False
        if pid != new_main and new_main not in ancestors:
            return False
    return True

def verified_verdict(verdict, token):
    # The host's restart_record rule, applied to the script's raw payload.
    # RESTART_SOURCE writes observations; ``status`` is a HOST-derived field.
    # Reading one here would accept a key the producer never emits, so every
    # real repair would read as a failure. This mirrors the derivation instead,
    # with ONE deliberate difference: the receipt must exist and carry a
    # generation, but its counters are the host's reconciliation against a
    # LIVE systemd. reset-failed zeroes NRestarts, so after a start-limit
    # recovery those counters can never reconcile; the watchdog's claim is
    # about the SERVER (a new MainPID and a fresh listener), which is exactly
    # the fact this path exists to establish.
    if not isinstance(verdict, dict) or verdict.get('schema_version') != 1:
        return False
    if verdict.get('token') != token:
        return False
    if (verdict.get('pre_scan_complete') is not True
            or verdict.get('post_scan_complete') is not True):
        return False
    pre = verdict.get('pre_restart_pids')
    post = verdict.get('post_restart_pids')
    owners = verdict.get('post_restart_owners')
    if not isinstance(pre, list) or not isinstance(post, list) or not isinstance(owners, list):
        return False
    if verdict.get('pre_restart_count') != len(pre):
        return False
    if verdict.get('post_restart_count') != len(post):
        return False
    if not post or [owner.get('pid') if isinstance(owner, dict) else None
                    for owner in owners] != post:
        return False
    new_main = verdict.get('new_main_pid')
    new_main_start = verdict.get('new_main_start_ticks')
    kill_ticks = verdict.get('kill_uptime_ticks')
    if (type(new_main) is not int or type(new_main_start) is not int
            or type(kill_ticks) is not int):
        return False
    if not postdated(owners, new_main, new_main_start, kill_ticks):
        return False
    receipt = verdict.get('receipt')
    if not isinstance(receipt, dict) or receipt.get('schema_version') != 1:
        return False
    records = receipt.get('records')
    if not isinstance(records, list) or not records:
        return False
    latest = records[-1]
    return isinstance(latest, dict) and type(latest.get('restart')) is int

def start_outcome():
    # ``systemctl start`` exits 0 when the JOB completes, even if the service
    # exited the moment it started (measured on systemd 255: five `started`
    # outcomes for a unit with a missing ExecStart target that never served).
    # So the exit code proves nothing; the outcome is only ``started`` when a
    # bounded poll sees BOTH an active unit and a listener. ``no-effect`` is
    # the honest word for a job that completed and changed nothing.
    try:
        run('systemctl', 'reset-failed', unit)
    except Exception:
        return 'reset-failed-error'
    try:
        run('systemctl', 'start', unit)
    except Exception:
        return 'start-failed'
    deadline = time.monotonic() + START_VERIFY_S
    while True:
        if listening() and active_state() == 'active':
            return 'started'
        if time.monotonic() >= deadline:
            return 'no-effect'
        time.sleep(0.5)

def verified_restart():
    token = monotonic_us()
    try:
        subprocess.check_output(
            ['systemd-run', '--quiet', '--collect', '--wait',
             '--unit=lop-watchdog-restart-%d' % token,
             'python3', restart_script, unit, str(port),
             str(result_path), str(token)],
            text=True, timeout=RESTART_TIMEOUT_S)
    except Exception:
        # The dispatch may lose its exit status; the verdict FILE, token-scoped,
        # is the only claim accepted either way.
        pass
    try:
        verdict = json.loads(result_path.read_text())
    except (OSError, ValueError):
        return 'restart-failed'
    if isinstance(verdict, dict) and verdict.get('start_limit_cleared') is False:
        # The latch was never cleared, so the start could not have been
        # accepted: a distinct outcome from a restart that ran and failed.
        return 'reset-failed-error'
    return 'restarted' if verified_verdict(verdict, token) else 'restart-failed'

def intervene(now, trigger, action):
    global last_attempt
    last_attempt = now
    state['attempts'] += 1
    recent.append(now)
    if action == 'reset-failed+start':
        outcome = start_outcome()
    else:
        outcome = verified_restart()
    if outcome == 'no-effect':
        ineffective.append(now)
    event = {'us': monotonic_us(), 'trigger': trigger,
             'action': action, 'outcome': outcome}
    state.update(intervened=True, last_attempt_us=event['us'],
                 last_trigger=trigger, last_action=action,
                 last_outcome=outcome,
                 events=(state['events'] + [event])[-EVENT_CAP:])
    state['capped'] = (len(recent) >= MAX_ATTEMPTS
                       or len(ineffective) >= MAX_INEFFECTIVE)
    persist()

def suppressed(outcome):
    if state['last_outcome'] == outcome:
        return
    state['last_outcome'] = outcome
    persist()

def observe(word):
    if state['last_observation'] == word:
        return
    state['last_observation'] = word
    persist()

def cycle():
    global absent_since, unresponsive_streak
    now = time.monotonic()
    recent[:] = [stamp for stamp in recent if now - stamp < WINDOW_S]
    ineffective[:] = [stamp for stamp in ineffective if now - stamp < WINDOW_S]
    capped = (len(recent) >= MAX_ATTEMPTS
              or len(ineffective) >= MAX_INEFFECTIVE)
    if state['capped'] != capped:
        state['capped'] = capped
        persist()
    if listening():
        absent_since = None
        if answers():
            # A serving port is the whole healthy-path shortcut: no unit query,
            # no verb, not even a state read.
            unresponsive_streak = 0
            observe('serving')
            return
        unresponsive_streak += 1
        observe('unresponsive')
        observed = active_state()
        if observed not in ('failed', 'inactive') and unresponsive_streak < 2:
            # One slow response is not death. The second consecutive silence is
            # a wedged server the host cannot reach either.
            return
        trigger, action = 'unresponsive', 'verified-restart'
    else:
        unresponsive_streak = 0
        observed = active_state()
        if observed in ('failed', 'inactive'):
            observe('no-listener')
            trigger = observed
            action = 'reset-failed+start'
        elif observed in ('active', 'activating', 'reloading', 'deactivating'):
            # A slow boot is observed, never interrupted: ``start`` would be a
            # no-op or a merge into systemd's own pending job.
            if absent_since is None:
                absent_since = now
            if now - absent_since >= ABSENT_GRACE_S:
                observe('no-listener')
            return
        else:
            observe('unknown')
            return
    if capped:
        suppressed('capped')
        return
    if last_attempt is not None and now - last_attempt < MIN_INTERVAL_S:
        suppressed('throttled')
        return
    intervene(now, trigger, action)

# One record before the first observation: "attempts: 0" is evidence the
# watchdog is alive, and a missing file is NOT the same answer.
try:
    seed()
    persist()
except Exception:
    pass

while True:
    try:
        cycle()
    except Exception:
        pass
    time.sleep(INTERVAL_S)
"""

# Only the action server's private service is eligible. A shared user/session
# scope must never gain a restart policy because it owns the GUI. Inspect only
# our HTTP command's ancestors, never arbitrary process or task paths.
INSTALL_SOURCE = r"""
import errno, json, os, pathlib, re, subprocess, sys, time

# The socket scan, not the ancestry walk, decides ownership. Bounded so a
# loaded guest cannot turn an install into an unbounded privileged probe; an
# incomplete scan refuses rather than claiming an owner set it did not see.
SCAN_DEADLINE = time.monotonic() + 1.0

def run(*args):
    return subprocess.check_output(args, text=True, timeout=3)

def listening_inodes(port):
    inodes = set()
    readable = False
    for name in ('/proc/net/tcp', '/proc/net/tcp6'):
        try:
            lines = pathlib.Path(name).read_text().splitlines()[1:]
        except OSError:
            continue
        readable = True
        for line in lines:
            fields = line.split()
            if len(fields) < 10 or fields[3] != '0A':
                continue
            try:
                if int(fields[1].rsplit(':', 1)[1], 16) != port:
                    continue
                inodes.add(int(fields[9]))
            except (IndexError, ValueError):
                continue
    return inodes, readable

def socket_owners(port):
    inodes, readable = listening_inodes(port)
    if not readable:
        return [], False
    try:
        entries = list(pathlib.Path('/proc').iterdir())
    except OSError:
        return [], False
    owners = set()
    for entry in entries:
        if time.monotonic() > SCAN_DEADLINE:
            return sorted(owners), False
        if not entry.name.isdigit():
            continue
        try:
            fds = list(entry.joinpath('fd').iterdir())
        except OSError as error:
            # A vanished process took its sockets with it. ANY other failure
            # (typically EACCES) means a LIVE process whose fds we could not
            # inspect; reporting the scan complete would omit a holder and
            # authorize a restart that cannot reap it.
            if error.errno in (errno.ENOENT, errno.ESRCH):
                continue
            return sorted(owners), False
        for fd in fds:
            if time.monotonic() > SCAN_DEADLINE:
                return sorted(owners), False
            try:
                target = os.readlink(fd)
            except OSError as error:
                if error.errno in (errno.ENOENT, errno.ESRCH):
                    continue
                return sorted(owners), False
            link = re.fullmatch(r'socket:\[(\d+)\]', target)
            if link and int(link[1]) in inodes:
                owners.add(int(entry.name))
                break
    return sorted(owners), True

def cgroup_of(pid):
    text = pathlib.Path('/proc/%d/cgroup' % pid).read_text()
    return text.split('::', 1)[1].strip() if '::' in text else ''

port = int(sys.argv[2])
pid = os.getppid()
for _ in range(8):
    proc = pathlib.Path('/proc/%d' % pid)
    argv = proc.joinpath('cmdline').read_bytes().split(b'\0')
    if any(a.endswith(b'/server/main.py') or a == b'main.py' for a in argv):
        break
    pid = int(re.search(r'^PPid:\s+(\d+)', proc.joinpath('status').read_text(), re.M)[1])
    if pid <= 1:
        raise RuntimeError('unsupported action-server command')
else:
    raise RuntimeError('unsupported action-server ancestry')
cgroup = proc.joinpath('cgroup').read_text()
units = re.findall(r'/([^/\n]+\.service)(?:/|$)', cgroup, re.M)
if len(set(units)) != 1:
    raise RuntimeError('action server has no private service')
unit = units[0]
if not re.fullmatch(r'[A-Za-z0-9_.-]+\.service', unit):
    raise RuntimeError('unsupported service name')
props = dict(line.split('=', 1) for line in run('systemctl', 'show', unit,
    '-p', 'MainPID', '-p', 'ExecStart', '-p', 'ExecStartPre', '-p', 'ExecStartPost',
    '-p', 'ExecStop', '-p', 'ExecStopPost', '-p', 'StandardError', '-p', 'StandardOutput',
    '-p', 'Type', '-p', 'LoadState', '-p', 'ControlGroup', '-p', 'NRestarts',
    '-p', 'ActiveEnterTimestampMonotonic'
    ).splitlines())
# THE UNIT'S MainPID MUST BE ONE OF THE SOCKET'S OWNERS, not the single owner.
# The AMI's real layout is MainPID 1637 plus a forked 1887 both holding :5000;
# requiring the ancestry pid to equal MainPID refused that healthy case. What
# is still refused is an owner the unit's cgroup-wide kill could not reach: a
# holder outside the service's control group survives even an explicit kill
# and can keep answering with stale screen state.
if 'main.py' not in props.get('ExecStart', ''):
    raise RuntimeError('service does not exclusively own action server')
if not props.get('MainPID', '').isdigit() or int(props['MainPID']) <= 0:
    raise RuntimeError('service does not exclusively own action server')
main_pid = int(props['MainPID'])
control_group = props.get('ControlGroup', '')
if not control_group:
    raise RuntimeError('action server has no private service')
owners, complete = socket_owners(port)
if not complete:
    raise RuntimeError('action-server socket scan was incomplete')
if not owners:
    raise RuntimeError('action server has no listening socket')
if main_pid not in owners:
    raise RuntimeError('service does not exclusively own action server')
prefix = control_group.rstrip('/') + '/'
for owner in owners:
    if owner == main_pid:
        continue
    try:
        owner_cgroup = cgroup_of(owner)
    except OSError:
        owner_cgroup = ''
    if not owner_cgroup or not (owner_cgroup == control_group or
                                owner_cgroup.startswith(prefix)):
        raise RuntimeError('socket owner is outside the action-server service')
if props.get('StandardError') != 'journal' and not (
        props.get('StandardError') == 'inherit' and props.get('StandardOutput') == 'journal'):
    raise RuntimeError('existing server stderr is not journalled')
if props.get('Type') == 'oneshot':
    raise RuntimeError('oneshot action-server service cannot be supervised')
if props.get('LoadState') != 'loaded':
    raise RuntimeError('action-server service is not loaded')
if (not props.get('NRestarts', '').isdigit() or
        not props.get('ActiveEnterTimestampMonotonic', '').isdigit() or
        int(props['ActiveEnterTimestampMonotonic']) <= 0):
    raise RuntimeError('systemd restart generation is unavailable')
root = pathlib.Path('/var/lib/lop-action-supervisor')
capture = root / 'capture.py'
path = pathlib.Path('/etc/systemd/system') / (unit + '.d') / '90-lop-supervisor.conf'
# The server commonly runs as the desktop user; only this fixed, root-owned
# capture helper needs to bypass User= to write its private diagnostic record.
hook = '+/usr/bin/python3 ' + str(capture) + ' ' + unit
# No ExecStart replacement or process signals. KillMode=process prevents
# systemd from killing the desktop applications a server command spawned.
content = ('[Unit]\nStartLimitIntervalSec=3600\nStartLimitBurst=3\n'
           '[Service]\nRestart=always\nRestartSec=1\nKillMode=process\n'
           'ExecStopPost=' + hook + '\n')
created = not path.exists()
# Set only when THIS attempt issued the start: a unit that was already
# running is not ours to stop, and one we started must not survive a refusal.
watchdog_started = False
# Every file THIS attempt created is removed again on a refusal, and only
# those: a later install of a changed version must not trip over half of an
# earlier refused one, and a pre-existing file is never ours to delete.
capture_created = watchdog_script_created = restart_script_created = False
receipt_created = False
if not created:
    if path.read_text() != content or capture.read_text() != sys.argv[1]:
        raise RuntimeError('supervisor installation conflict')
else:
    if any(props.get(k) for k in ('ExecStartPre', 'ExecStartPost', 'ExecStop', 'ExecStopPost')):
        raise RuntimeError('service hooks could mutate desktop state')
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if capture.exists() and capture.read_text() != sys.argv[1]:
        raise RuntimeError('supervisor capture conflict')
    capture_created = not capture.exists()
    capture.write_text(sys.argv[1])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as out:
        out.write(content)
# THE WATCHDOG IS A SEPARATE UNIT, and that is the whole mechanism: it must
# outlive the unit it repairs, so it can never share its cgroup. ``Restart=always``
# is its own, and its start limit is DISABLED: this is the recovery of last
# resort, and a watchdog that latches ``failed`` is the one failure nothing
# else in the system can repair. Its restart cost is bounded by RestartSec.
watchdog_path = pathlib.Path('/etc/systemd/system') / 'lop-action-watchdog.service'
watchdog_script = root / 'watchdog.py'
# The watchdog dispatches the SAME restart script the host uses (argv[4]), as
# the same transient service. One implementation of the all-process restart,
# two dispatchers: a bare ``start`` here would fail to bind beside a survivor.
restart_script = root / 'restart.py'
watchdog_content = ('[Unit]\nDescription=Local Operator action-server watchdog\n'
                    'StartLimitIntervalSec=0\n'
                    '[Service]\nType=simple\n'
                    'ExecStart=/usr/bin/python3 ' + str(watchdog_script) + ' '
                    + unit + ' ' + str(port) + ' '
                    + str(root / 'watchdog.json') + ' '
                    + str(restart_script) + ' '
                    + str(root / 'restart-result.json') + '\n'
                    'Restart=always\nRestartSec=5\n'
                    '[Install]\nWantedBy=multi-user.target\n')
watchdog_created = not watchdog_path.exists()
if not watchdog_created:
    if (watchdog_path.read_text() != watchdog_content
            or watchdog_script.read_text() != sys.argv[3]
            or restart_script.read_text() != sys.argv[4]):
        raise RuntimeError('watchdog installation conflict')
else:
    if watchdog_script.exists() and watchdog_script.read_text() != sys.argv[3]:
        raise RuntimeError('watchdog script conflict')
    if restart_script.exists() and restart_script.read_text() != sys.argv[4]:
        raise RuntimeError('watchdog restart script conflict')
    watchdog_script_created = not watchdog_script.exists()
    restart_script_created = not restart_script.exists()
    watchdog_script.write_text(sys.argv[3])
    restart_script.write_text(sys.argv[4])
    with watchdog_path.open('x') as out:
        out.write(watchdog_content)
# An initialized empty envelope is distinct from a missing/corrupt receipt.
# Never let a failed read become evidence that no restart happened.
receipt = root / (unit + '.json')
try:
    if not receipt.exists():
        receipt_created = True
        with receipt.open('x') as out:
            json.dump({'schema_version': 1, 'records': [], 'captured_exits': 0,
                       'baseline_nrestarts': int(props['NRestarts']),
                       'baseline_active_enter': int(props['ActiveEnterTimestampMonotonic'])}, out)
    if created or watchdog_created:
        run('systemctl', 'daemon-reload')
    effective = dict(line.split('=', 1) for line in run('systemctl', 'show', unit,
        '-p', 'Type', '-p', 'LoadState', '-p', 'Restart', '-p', 'KillMode',
        '-p', 'ExecStopPost', '-p', 'ProtectSystem', '-p', 'ReadOnlyPaths',
        '-p', 'InaccessiblePaths', '-p', 'RootDirectory', '-p', 'RootImage',
        '-p', 'StartLimitBurst', '-p', 'StartLimitIntervalUSec', '-p', 'RestartUSec',
        '-p', 'RestartPreventExitStatus', '-p', 'ExitType'
        ).splitlines())
    if effective.get('Type') == 'oneshot':
        raise RuntimeError('oneshot action-server service cannot be supervised')
    if effective.get('LoadState') != 'loaded':
        raise RuntimeError('action-server service is not loaded')
    if effective.get('Restart') != 'always':
        raise RuntimeError('effective Restart is not always')
    # Refuse rather than clear existing exclusions: their original intent is
    # unknown, and Restart=always cannot override this native policy veto.
    if effective.get('RestartPreventExitStatus', '').strip():
        raise RuntimeError('unit has RestartPreventExitStatus exclusions')
    if effective.get('ExitType', 'main') != 'main':
        raise RuntimeError('unit uses ExitType=cgroup')
    if (effective.get('KillMode') != 'process' or
            effective.get('StartLimitBurst') != '3' or
            effective.get('StartLimitIntervalUSec') != '1h' or
            effective.get('RestartUSec') != '1s'):
        raise RuntimeError('effective supervisor restart policy was overridden')
    if str(capture) + ' ' + unit not in effective.get('ExecStopPost', ''):
        raise RuntimeError('effective supervisor capture hook is missing')
    if (effective.get('ProtectSystem', 'no') not in ('no', 'false') or
            any(effective.get(k) for k in ('ReadOnlyPaths', 'InaccessiblePaths',
                                          'RootDirectory', 'RootImage'))):
        raise RuntimeError('service sandbox may block supervisor capture')
    # THE WATCHDOG MUST PROVE IT IS NOT INSIDE THE UNIT IT REPAIRS. A unit that
    # shares the server's cgroup dies with the server -- exactly the failure
    # this mechanism exists to survive -- so a wrong placement is refused, not
    # installed as decoration. A stopped watchdog is likewise repaired here:
    # install is the only step that can still reach the guest to do it.
    run('systemctl', 'enable', 'lop-action-watchdog.service')
    run('systemctl', 'start', 'lop-action-watchdog.service')
    watchdog_started = True
    watchdog = dict(line.split('=', 1) for line in run('systemctl', 'show',
        'lop-action-watchdog.service', '-p', 'LoadState', '-p', 'ActiveState',
        '-p', 'ControlGroup').splitlines())
    if watchdog.get('LoadState') != 'loaded':
        raise RuntimeError('watchdog unit is not loaded')
    if watchdog.get('ActiveState') != 'active':
        raise RuntimeError('watchdog unit is not active')
    watchdog_group = watchdog.get('ControlGroup', '')
    if not watchdog_group:
        raise RuntimeError('watchdog unit has no control group')
    if (watchdog_group == control_group or
            watchdog_group.startswith(control_group.rstrip('/') + '/')):
        raise RuntimeError('watchdog unit shares the action-server cgroup')
except Exception:
    # Roll back only the units this attempt created, never a pre-existing
    # installation. Re-entry MUST verify effective config, not trust bytes.
    # A watchdog that was STARTED before the refusal keeps repairing the
    # server while the host records ``installed: false``, so stop and disable
    # it before the file goes away. The VERBS are not the proof: a bounded
    # state read must show it gone, and a survivor is killed. Every rollback
    # step is best-effort: a failure here must not replace the refusal.
    if watchdog_started:
        for verb in ('disable', 'stop'):
            try:
                run('systemctl', verb, 'lop-action-watchdog.service')
            except Exception:
                pass
        try:
            left = dict(line.split('=', 1) for line in run('systemctl', 'show',
                'lop-action-watchdog.service', '-p', 'ActiveState',
                '-p', 'ControlGroup').splitlines())
            if left.get('ActiveState') == 'active' or left.get('ControlGroup'):
                run('systemctl', 'kill', '--kill-whom=all', 'lop-action-watchdog.service')
        except Exception:
            pass
    for owned in (
        (capture_created, capture),
        (watchdog_script_created, watchdog_script),
        (restart_script_created, restart_script),
        (receipt_created, receipt),
    ):
        if owned[0]:
            try:
                owned[1].unlink()
            except OSError:
                pass
    if created:
        path.unlink()
    if watchdog_created:
        watchdog_path.unlink()
    if created or watchdog_created:
        try:
            run('systemctl', 'daemon-reload')
        except Exception:
            pass
    raise
print(json.dumps({'unit': unit, 'installed': True,
                  'watchdog': {'installed': True, 'status': 'active',
                               'cgroup_distinct': True}}))
"""

#: One explicit, all-process restart followed by a proved-fresh listener.
#: ``KillMode=process`` makes systemd's own restart signal only the MainPID, so
#: a second holder of the action port survives it and can keep answering with
#: stale screen state -- a false success, worse than no supervision. This script
#: therefore signals EVERY process in the unit's cgroup, waits (bounded) for the
#: recorded holders to release the socket, starts the unit, and reports the
#: pre/post owner sets, each post owner's ancestry and start time, the kill
#: stamp and the new MainPID. The host derives freshness: a post-restart
#: listener is acceptable only when it started at or after the kill AND is the
#: new MainPID or one of its descendants, so a worker forked by a dying holder
#: cannot pass as fresh. The receipt envelope is completed here (the on-disk
#: file has five keys; NRestarts/ActiveEnter are report-only) so the host can
#: consume the generation this restart produced instead of bouncing again.
#:
#: THE RESULT IS A FILE, NOT THIS PROCESS'S STDOUT. The script runs as a
#: TRANSIENT SERVICE (``systemd-run``, never ``--scope``) because the target
#: unit is the process serving the HTTP request that started it; a scope would
#: place the script in the same cgroup and the all-process kill would reap its
#: own executor and response channel. The service writes the verdict to
#: ``RESTART_RESULT_PATH`` and the host reads it from a later command once the
#: action server answers again.
RESTART_SOURCE = r"""
import errno, json, os, pathlib, re, subprocess, sys, time

# ``python3 -c <source> a b c`` gives sys.argv == ['-c', 'a', 'b', 'c']:
# the source occupies argv[0], so the arguments start at argv[1]. The builder
# (``restart_service_argv``) appends them in RESTART_ARGUMENT_ORDER.
unit, port, result_path, token = sys.argv[1:5]
UNIT = unit
PORT = int(port)
RESULT = pathlib.Path(result_path)
TOKEN = int(token)
DEADLINE = time.monotonic() + 6.0
KILL_GRACE = 2.0
VERIFY = 4.0
RECEIPT = pathlib.Path('/var/lib/lop-action-supervisor') / (UNIT + '.json')
CLK_TCK = int(os.sysconf('SC_CLK_TCK'))

def run(*args):
    left = DEADLINE - time.monotonic()
    if left <= 0.2:
        raise RuntimeError('restart budget exhausted')
    return subprocess.check_output(args, text=True, timeout=min(2.0, left))

def listening_inodes():
    inodes = set()
    readable = False
    for name in ('/proc/net/tcp', '/proc/net/tcp6'):
        try:
            lines = pathlib.Path(name).read_text().splitlines()[1:]
        except OSError:
            continue
        readable = True
        for line in lines:
            fields = line.split()
            if len(fields) < 10 or fields[3] != '0A':
                continue
            try:
                if int(fields[1].rsplit(':', 1)[1], 16) != PORT:
                    continue
                inodes.add(int(fields[9]))
            except (IndexError, ValueError):
                continue
    return inodes, readable

def owners():
    inodes, readable = listening_inodes()
    if not readable:
        return [], False
    try:
        entries = list(pathlib.Path('/proc').iterdir())
    except OSError:
        return [], False
    found = set()
    for entry in entries:
        if time.monotonic() > DEADLINE:
            return sorted(found), False
        if not entry.name.isdigit():
            continue
        try:
            fds = list(entry.joinpath('fd').iterdir())
        except OSError as error:
            # ENOENT/ESRCH: the process vanished, taking its sockets with it.
            # Any other failure is a LIVE process we could not inspect; the
            # scan is incomplete, never "this holder does not exist".
            if error.errno in (errno.ENOENT, errno.ESRCH):
                continue
            return sorted(found), False
        for fd in fds:
            if time.monotonic() > DEADLINE:
                return sorted(found), False
            try:
                target = os.readlink(fd)
            except OSError as error:
                if error.errno in (errno.ENOENT, errno.ESRCH):
                    continue
                return sorted(found), False
            link = re.fullmatch(r'socket:\[(\d+)\]', target)
            if link and int(link[1]) in inodes:
                found.add(int(entry.name))
                break
    return sorted(found), True

def stat_of(pid):
    text = pathlib.Path('/proc/%d/stat' % pid).read_text()
    fields = text[text.rfind(')') + 2:].split()
    return int(fields[1]), int(fields[19])

def uptime_ticks():
    return int(float(pathlib.Path('/proc/uptime').read_text().split()[0]) * CLK_TCK)

def ancestors(pid):
    chain = []
    for _ in range(8):
        if pid <= 1:
            break
        try:
            parent, _ = stat_of(pid)
        except OSError:
            break
        chain.append(pid)
        pid = parent
    return chain

def describe(pids):
    records = []
    complete = True
    for pid in pids:
        try:
            ppid, start = stat_of(pid)
        except OSError:
            complete = False
            continue
        records.append({'pid': pid, 'ppid': ppid, 'start_ticks': start,
                        'ancestors': ancestors(ppid)})
    return records, complete

def main_pid():
    for line in run('systemctl', 'show', UNIT, '-p', 'MainPID').splitlines():
        if line.startswith('MainPID=') and line[8:].isdigit():
            return int(line[8:])
    return 0

def acceptable(pid, new_main, new_main_start):
    # The same rule the host derives: postdates the kill AND descends from it.
    if kill_ticks <= 0 or new_main <= 0 or new_main_start < kill_ticks:
        return False
    try:
        ppid, start = stat_of(pid)
    except OSError:
        return False
    if start < kill_ticks:
        return False
    return pid == new_main or new_main in ancestors(ppid)

# A result left by an earlier dispatch must not be read as this one.
try:
    RESULT.unlink()
except OSError:
    pass

# An EMPTY pre set is not itself unprovable: freshness below is decided by
# start time and ancestry, so a holder this scan could not see is still
# refused. What must never be claimed is COMPLETENESS after an unreadable
# process, which is why a non-vanishing OSError marks the scan incomplete and
# this path refuses BEFORE the disruptive kill.
pre, pre_complete = owners()
start_limit_cleared = True
kill_ticks = 0
post = []
post_complete = False
owner_records = []
detail_complete = False
new_main = 0
new_main_start = 0
receipt = None
if pre_complete:
    # CLEAR THE START-LIMIT LATCH BEFORE THE START, but ONLY on a failed unit.
    # A unit that hit StartLimitBurst refuses `start` with "Start request
    # repeated too quickly" and stays failed; reset-failed is the only
    # operation that clears the latch and its rate-limit counters (measured on
    # systemd 255). It goes AFTER the pre scan, so the evidence still names the
    # holders, and BEFORE the kill, so the kill stamp still describes the
    # signal that follows.
    #
    # THE CONDITION IS LOAD-BEARING TOO: reset-failed also zeroes NRestarts on
    # a LIVE unit (measured), which would break the receipt invariant the host
    # reconciles (systemd_nrestarts - baseline == captured_exits) on every
    # ordinary host restart. A unit that is not failed has no latch to clear.
    try:
        props = dict(line.split('=', 1) for line in run(
            'systemctl', 'show', UNIT, '-p', 'ActiveState').splitlines())
        unit_state = props.get('ActiveState', '')
    except Exception:
        unit_state = ''
    if unit_state == 'failed':
        try:
            run('systemctl', 'reset-failed', UNIT)
        except Exception:
            start_limit_cleared = False
    # The stamp is taken IMMEDIATELY before the signal. Freshness is then a
    # start time at or after it, not merely a pid absent from the pre set: a
    # holder that forks a worker while dying produces a pid the pre scan could
    # not have seen, and that pid serves pre-restart state.
    kill_ticks = uptime_ticks()
    try:
        run('systemctl', 'kill', '--kill-whom=all', UNIT)
    except Exception:
        pass
    until = min(time.monotonic() + KILL_GRACE, DEADLINE)
    while time.monotonic() < until:
        current, _ = owners()
        if not (set(pre) & set(current)):
            break
        time.sleep(0.1)
    try:
        run('systemctl', 'start', UNIT)
    except Exception:
        pass
    # post_complete starts False: a loop that never runs must not claim a scan
    # it did not perform.
    post, post_complete = [], False
    # Reserve room for the two reads after the loop (MainPID and the
    # receipt): a verification that consumes the whole budget would report
    # new_main_pid=0 and lose the evidence that explains the refusal.
    until = min(time.monotonic() + VERIFY, DEADLINE - 0.8)
    last_main_poll = -1.0
    while time.monotonic() < until:
        post, post_complete = owners()
        now = time.monotonic()
        if now - last_main_poll >= 0.5:
            try:
                new_main = main_pid()
            except Exception:
                new_main = 0
            new_main_start = 0
            if new_main > 0:
                try:
                    _, new_main_start = stat_of(new_main)
                except OSError:
                    new_main_start = 0
            last_main_poll = now
        # ACCEPTANCE IN THE LOOP, not the round-1 set difference. A dying
        # holder can fork a worker the pre scan never saw; stopping on that
        # pid would report the stale holder as the outcome while a genuine
        # new listener was still binding. Only a postdating descendant of the
        # new MainPID ends the poll early; anything else keeps waiting.
        if (post_complete and post and new_main > 0 and
                new_main_start >= kill_ticks and
                all(acceptable(pid, new_main, new_main_start) for pid in post)):
            break
        time.sleep(0.1)
    owner_records, detail_complete = describe(post[:8])
    try:
        new_main = main_pid()
    except Exception:
        new_main = 0
    if new_main > 0:
        try:
            _, new_main_start = stat_of(new_main)
        except OSError:
            new_main_start = 0
    # The capture file has FIVE keys (the installer writes it); NRestarts and
    # ActiveEnterTimestampMonotonic are added only in REPORT_SOURCE, in memory.
    # Query them here too, or the host's receipt validation can never pass.
    try:
        receipt = json.loads(RECEIPT.read_text())
        props = dict(line.split('=', 1) for line in run(
            'systemctl', 'show', UNIT, '-p', 'NRestarts',
            '-p', 'ActiveEnterTimestampMonotonic').splitlines())
        receipt['systemd_nrestarts'] = int(props['NRestarts'])
        receipt['systemd_active_enter'] = int(props['ActiveEnterTimestampMonotonic'])
    except Exception:
        receipt = None
payload = {
    'schema_version': 1,
    'token': TOKEN,
    'unit': UNIT,
    'pre_restart_pids': pre[:8],
    'pre_restart_count': len(pre),
    'pre_scan_complete': pre_complete,
    'start_limit_cleared': start_limit_cleared,
    'kill_uptime_ticks': kill_ticks,
    'post_restart_pids': post[:8],
    'post_restart_count': len(post),
    'post_scan_complete': post_complete and detail_complete,
    'post_restart_owners': owner_records,
    'new_main_pid': new_main,
    'new_main_start_ticks': new_main_start,
    'receipt': receipt,
}
tmp = RESULT.with_suffix('.tmp')
tmp.write_text(json.dumps(payload))
tmp.replace(RESULT)
"""

RESTART_RESULT_SOURCE = r"""
import pathlib, sys
path = pathlib.Path('/var/lib/lop-action-supervisor/restart-result.json')
try:
    text = path.read_text()
except OSError:
    raise SystemExit(1)
if len(text) > 262144:
    raise SystemExit(1)
sys.stdout.write(text)
"""

#: Read-only process/unit topology behind a supervisor refusal. This exists
#: because the refusal said only that the service does not exclusively own the
#: server; the next run must say WHO does. It prints one fixed-schema document:
#: numeric ids, unit names, process ``comm`` and cgroup paths. No environment, no
#: argv of any other process, no free-form guest text -- the host-side
#: ``layout_record`` whitelists these keys again, and the provider scrubs the
#: result, because a guest can print anything.
LAYOUT_SOURCE = r"""
import json, os, pathlib, re, subprocess, sys, time

DEADLINE = time.monotonic() + 3.0
PORT = int(sys.argv[1])
PROPS = ('MainPID', 'ExecStart', 'Type', 'ControlGroup', 'SubState', 'LoadState')
CONTROL = re.compile(r'[\x00-\x1f\x7f]')


def left():
    return max(0.1, DEADLINE - time.monotonic())


def text(name):
    return pathlib.Path(name).read_text()


def run(*args):
    return subprocess.check_output(args, text=True, timeout=min(1.5, left()))


def clean(value, limit=64):
    return CONTROL.sub('', value)[:limit]


def status(pid):
    fields = {}
    for line in text('/proc/%d/status' % pid).splitlines():
        key, _, value = line.partition(':')
        if key in ('Name', 'PPid'):
            fields[key] = value.strip()
    return clean(fields['Name'], 32), int(fields['PPid'])


def cgroup_of(pid):
    paths = []
    for line in text('/proc/%d/cgroup' % pid).splitlines():
        parts = line.split(':', 2)
        if len(parts) == 3:
            if line.startswith('0::'):
                return parts[2]
            paths.append(parts[2])
    return paths[-1] if paths else ''


def owner_record(pid):
    # One entry per observed socket owner -- INCLUDING one /proc cannot be read.
    # Dropping an unreadable owner would leave the host deriving "every other
    # owner shares the unit cgroup" from a list it never saw the gap in.
    record = {'pid': pid, 'ppid': None, 'comm': '', 'cgroup': None}
    if time.monotonic() > DEADLINE:
        return record
    try:
        record['comm'], record['ppid'] = status(pid)
    except (OSError, KeyError, ValueError):
        pass
    try:
        record['cgroup'] = cgroup_of(pid)
    except OSError:
        pass
    return record


def units_in(cgroup):
    found = []
    for name in re.findall(r'/([^/\n]+\.(?:service|scope))(?:/|$)', cgroup):
        if name not in found:
            found.append(name)
    return found


def listening_inodes(port):
    inodes = set()
    for name in ('/proc/net/tcp', '/proc/net/tcp6'):
        try:
            lines = text(name).splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            if len(fields) < 10 or fields[3] != '0A':
                continue
            try:
                if int(fields[1].rsplit(':', 1)[1], 16) != port:
                    continue
            except (IndexError, ValueError):
                continue
            inodes.add(fields[9])
    return inodes


def inode_pids(inodes):
    pids = set()
    if not inodes:
        return [], True
    wanted = {'socket:[' + inode + ']' for inode in inodes}
    complete = True
    for entry in pathlib.Path('/proc').iterdir():
        if time.monotonic() > DEADLINE:
            complete = False
            break
        if not entry.name.isdigit():
            continue
        try:
            for fd in entry.joinpath('fd').iterdir():
                try:
                    if os.readlink(fd) in wanted:
                        pids.add(int(entry.name))
                        break
                except OSError:
                    continue
        except OSError:
            continue
    return sorted(pids), complete


def ancestry(pid):
    chain, seen = [], set()
    while pid > 0 and pid not in seen and len(chain) < 16 and time.monotonic() <= DEADLINE:
        seen.add(pid)
        try:
            name, ppid = status(pid)
        except (OSError, KeyError, ValueError):
            break
        chain.append({'pid': pid, 'ppid': ppid, 'comm': name})
        pid = ppid
    return chain


def server_ancestor():
    # Same walk as the install command. It is a FALLBACK only: the socket owner
    # is the server even when the install could not classify it.
    pid = os.getppid()
    for _ in range(8):
        try:
            argv = pathlib.Path('/proc/%d/cmdline' % pid).read_bytes().split(b'\0')
        except OSError:
            return None
        if any(a.endswith(b'/server/main.py') or a == b'main.py' for a in argv):
            return pid
        try:
            _, ppid = status(pid)
        except (OSError, KeyError, ValueError):
            return None
        if ppid <= 1:
            return None
        pid = ppid
    return None


def unit_record(unit):
    record = {'unit': unit, 'main_pid': None, 'exec_start': '', 'type': 'other',
              'control_group': '', 'sub_state': 'other', 'load_state': 'unavailable'}
    try:
        args = ['systemctl', 'show', unit]
        for prop in PROPS:
            args += ['-p', prop]
        output = run(*args)
        props = dict(line.split('=', 1) for line in output.splitlines() if '=' in line)
        if props.get('MainPID', '').isdigit():
            record['main_pid'] = int(props['MainPID'])
        record['exec_start'] = clean(props.get('ExecStart', ''), 512)
        record['control_group'] = clean(props.get('ControlGroup', ''), 512)
        record['type'] = clean(props.get('Type', ''), 32) or 'other'
        record['sub_state'] = clean(props.get('SubState', ''), 32) or 'other'
        record['load_state'] = clean(props.get('LoadState', ''), 32) or 'unavailable'
    except Exception:
        # A unit that cannot be queried is still evidence: keep its name.
        pass
    return record


socket_pids, socket_scan_complete = inode_pids(listening_inodes(PORT))
# The PRE-cap count and the scan flag are what let the host refuse a universal
# claim over a list this capture shortened: a ninth holder in another cgroup
# must produce "undetermined", never "every other owner shares".
socket_owner_count = len(socket_pids)
# Same cap as the printed pid list: the detail cannot outgrow the evidence it
# annotates. A pid whose reads hit the shared deadline still gets its entry
# with an unknown cgroup, so the host can say "undetermined", never "shares".
socket_owners = [owner_record(pid) for pid in socket_pids[:8]]
fallback = server_ancestor()
pid = socket_pids[0] if socket_pids else fallback
chain = ancestry(pid) if pid else []
server = None
names = []
for member in chain:
    try:
        cgroup = cgroup_of(member['pid'])
    except OSError:
        continue
    for name in units_in(cgroup):
        if name not in names:
            names.append(name)
if pid and chain:
    try:
        server = {'pid': pid, 'ppid': chain[0]['ppid'], 'comm': chain[0]['comm'],
                  'cgroup': cgroup_of(pid)}
    except OSError:
        pass
units = []
for name in names[:8]:
    # A call already in flight is capped by min(1.5, left()); starting one
    # at the deadline could still run past the body timeout, so gate here.
    if time.monotonic() > DEADLINE:
        break
    units.append(unit_record(name))
# Only raw observations leave the guest. Derived claims (which pid owns the
# socket, whether every MainPID agrees) are recomputed host-side from these,
# so a guest cannot assert a contradiction into the evidence file.
print(json.dumps({'schema_version': 1, 'socket_pids': socket_pids[:8], 'server': server,
                  'ancestry': chain, 'units': units, 'socket_owners': socket_owners,
                  'socket_owner_count': socket_owner_count,
                  'socket_scan_complete': socket_scan_complete}))
"""

REPORT_SOURCE = r"""
import json, pathlib, subprocess, sys
unit = sys.argv[1]
# This view exists even when our capture helper cannot write (ENOSPC/EROFS).
# Read systemd FIRST: a concurrent restart then makes the envelope disagree or
# the HTTP read fail, rather than authorizing input from a stale empty history.
props = dict(line.split('=', 1) for line in subprocess.check_output(
    ['systemctl', 'show', unit, '-p', 'NRestarts', '-p', 'ActiveEnterTimestampMonotonic'],
    text=True, timeout=2).splitlines())
path = pathlib.Path('/var/lib/lop-action-supervisor') / (unit + '.json')
history = json.loads(path.read_text())
history['systemd_nrestarts'] = int(props['NRestarts'])
history['systemd_active_enter'] = int(props['ActiveEnterTimestampMonotonic'])
# The watchdog's own record rides in the SAME privileged read: a separate
# command would add a round trip to every action, and the episode needs this
# evidence precisely when the host is already reading the receipt.
# A missing or corrupt record must never fail the receipt read -- the record is
# additive evidence, and empty-because-absent is not the same as a read failure.
watchdog_path = pathlib.Path('/var/lib/lop-action-supervisor') / 'watchdog.json'
try:
    watchdog_text = watchdog_path.read_text()
except OSError:
    # No file at all: the key is OMITTED, which the host reads as ``absent``.
    pass
else:
    try:
        history['watchdog'] = json.loads(watchdog_text)
    except ValueError:
        # A corrupt file is NOT an absent one; ``{}`` is a present-but-unusable
        # document, which the host sanitizes to ``unparseable``.
        history['watchdog'] = {}
print(json.dumps(history))
"""


def _privileged_fragment(fragment: str, password: str) -> Sequence[str]:
    # Reuse preparation's non-interactive escalation ladder. Nothing logs its
    # argv or error output, which may contain the supplied password.
    return ["bash", "-c", guest_disk.escalation_script(fragment, password)]


def _privileged(source: str, args: Sequence[str], password: str) -> Sequence[str]:
    return _privileged_fragment(shlex.join(["python3", "-c", source, *args]), password)


def install_command(password: str) -> Sequence[str]:
    # Report capability failures as successful command execution, so the sudo
    # ladder does not mistake an unsupported service for failed authentication.
    # RuntimeError messages above are fixed vocabulary, never guest-authored.
    #
    # ``sys.argv`` in the wrapper: [1] is the capture helper's source, [2] the
    # action port, [3] the watchdog's, [4] the restart script the watchdog
    # dispatches. Every script is data, never interpolated into a shell
    # fragment, so no guest-authored text can join the command.
    wrapper = (
        "import json\ntry:\n    exec(" + repr(INSTALL_SOURCE) + ")\n"
        "except RuntimeError as error:\n"
        "    print(json.dumps({'installed': False, 'reason': str(error)}))\n"
        "except Exception as error:\n"
        "    print(json.dumps({'installed': False, 'reason': type(error).__name__}))\n"
    )
    return _privileged(
        wrapper,
        [CAPTURE_SOURCE, str(ACTION_SERVER_PORT), WATCHDOG_SOURCE, RESTART_SOURCE],
        password,
    )


def report_command(unit: str, password: str) -> Sequence[str]:
    return _privileged(REPORT_SOURCE, [unit], password)


def restart_command(
    unit: str, password: str, token: int, port: int = ACTION_SERVER_PORT
) -> Sequence[str]:
    """Dispatch one privileged, bounded, all-process restart of the unit.

    Explicit rather than a bare ``systemctl restart`` because ``KillMode=process``
    leaves every non-MainPID holder alive; ``--kill-whom=all`` is the one
    operation that reaches them, and freshness is verified after it.

    ``systemd-run`` (a transient SERVICE) is load-bearing: the unit being
    restarted is the process serving this very request, so a ``--scope`` unit
    would sit in the same cgroup and the kill would reap its own executor and
    response channel. The service removes any old result file, restarts, and
    writes its verdict to ``RESTART_RESULT_PATH``; this command only dispatches
    it and may legitimately lose its HTTP response when the kill lands.
    """
    return _privileged_fragment(shlex.join(restart_service_argv(unit, port, token)), password)


def restart_service_argv(unit: str, port: int, token: int) -> list[str]:
    """The exact argv systemd-run execs, in RESTART_ARGUMENT_ORDER.

    ``--collect`` and the transient-unit name keep the restart OUT of the
    target cgroup; ``--scope`` here would place it back inside and let the
    all-process kill reap its own executor (round-2 blocker 1).
    """
    name = "lop-supervisor-restart-%d" % token
    return [
        "systemd-run",
        "--quiet",
        "--collect",
        "--unit=" + name,
        "python3",
        "-c",
        RESTART_SOURCE,
        unit,
        str(port),
        RESTART_RESULT_PATH,
        str(token),
    ]


def restart_result_command(password: str) -> Sequence[str]:
    """Read the detached restart's result file, never the restart itself."""
    return _privileged(RESTART_RESULT_SOURCE, [], password)


def layout_command(password: str, port: int = ACTION_SERVER_PORT) -> Sequence[str]:
    """One read-only, privileged probe of the layout behind a refusal.

    Privileged for the same reason the install step is: rooting the guest is
    what lets the socket-owner scan read another user's ``/proc/<pid>/fd``.
    The escalation ladder is the existing one, so a stale password fails here
    the same way it does in the install step -- and the caller treats any
    failure as absent evidence, never as a changed refusal.
    """
    return _privileged(LAYOUT_SOURCE, [str(port)], password)


#: Bounds on guest-controlled text. Every recovery attempt scans the capture,
#: so an adversarial guest must not be able to buy unbounded work: before these
#: existed, a brace-dense 1 MB capture measured ~11 s against a sub-millisecond
#: old parser. Real captures are ~20 KB (four 4 KB journal tails plus the
#: envelope), so both bounds leave years of headroom and fail closed past them.
#: The candidate count is TOP-LEVEL decode attempts, not raw ``{`` characters:
#: a decoded document is skipped whole, so braces inside its string values (a
#: journal tail is the common case) never spend the budget.
_MAX_CAPTURE_CHARS = 256 * 1024
_MAX_PAYLOAD_CANDIDATES = 32


def _ends_a_document(output: str, end: int) -> bool:
    """True when the decoded value is not a MEMBER of an enclosing container.

    A member is followed by its container's own delimiter (``,``, ``]`` or
    ``}``), so an array-wrapped ``[{"installed": true, ...}]`` and an object
    nested in a larger object are not top-level documents and must not be read
    as one. A real payload is followed by a line break, the end of the capture,
    or ordinary trailing noise, so it passes here.

    This is deliberately a RIGHT boundary only. Real ``sudo -S`` writes its
    prompt WITHOUT a trailing newline when input is a pipe, so the genuine
    capture is ``[sudo] password for user: {"unit": ..., "installed": true}``
    on ONE line; requiring a line start would refuse the very payload this
    module exists to read.
    """

    cursor = end
    while cursor < len(output) and output[cursor] in " \t\r\n":
        cursor += 1
    return cursor >= len(output) or output[cursor] not in ",]}"


def parse_output(output: str, accept: Callable[[object], bool]) -> object:
    """Extract the guest's expected JSON payload from a captured escalation output.

    WHY A SCAN, NOT A FIXED OFFSET. The escalation ladder captures the
    command's combined stdout+stderr and prefixes one metadata line
    (``escalation=agentless>supplied``), but the captured text is more than
    the payload: ``sudo -S`` writes its ``[sudo] password for user:`` prompt
    to stderr, and a guest MOTD, locale notice or warning lands the same way.
    Reading a fixed offset after the first newline therefore handed
    ``json.loads`` a prompt line plus the payload and raised
    ``JSONDecodeError``, so the supervisor never installed and the episode ran
    unsupervised. Scanning for the positions that decode as a JSON object is
    identical for the clean case and tolerates a preamble, a prompt line, a
    payload spanning several lines and trailing noise.

    WHY THE CALLER PASSES THE SHAPE, AND WHY THE LAST MATCH WINS. A bare
    "first object that decodes" accepts ANY object the preamble happens to
    carry, so a shadow that looks like a receipt would be read instead of the
    real envelope and report a restart as "no restart" -- the silent loss
    this module exists to prevent. The caller therefore states what its own
    command prints (``is_install_payload`` / ``is_receipt_payload``), and only
    objects of that shape count. Among those, the LAST wins: the ladder prints
    the command's own output last (``guest_disk.escalation_script`` carries the
    invariant next to that print), so its payload is the final document while
    sudo's prompt and guest warnings land before it.

    WHY ONLY TOP-LEVEL DOCUMENTS, AND WHY THE BOUNDARY IS ON THE RIGHT. Decoding
    an inner object and accepting it would let a guest-chosen fragment inside an
    array or a larger object stand in for the payload -- e.g.
    ``[{"installed": true, "unit": "x.service"}]`` would install a unit the
    action server does not own, where the old parser returned the list and the
    caller refused it. What distinguishes a payload is that nothing encloses it,
    and that shows up AFTER it: a member is followed by its container's
    delimiter. So a document is a candidate when it ends a document, which also
    accepts the genuine same-line ``sudo`` capture. A left rule would reject
    that capture, because real sudo writes its prompt without a newline.

    Only ``SupervisorOutputUnparseable`` (a ``ValueError``) escapes for an
    extraction failure: no expected top-level document, a capture past
    ``_MAX_CAPTURE_CHARS``, more than ``_MAX_PAYLOAD_CANDIDATES`` candidates,
    or nesting deep enough to raise ``RecursionError`` in the decoder. Guest
    text is never quoted into the message, because it can carry a password
    prompt.
    """

    if len(output) > _MAX_CAPTURE_CHARS:
        raise SupervisorOutputUnparseable("supervisor output exceeded the parser size bound")
    decoder = json.JSONDecoder()
    found = False
    match: object = None
    candidates = 0
    index = output.find("{")
    while index != -1:
        candidates += 1
        if candidates > _MAX_PAYLOAD_CANDIDATES:
            # Fail closed rather than keep buying work for a guest that keeps
            # printing brace-led text.
            raise SupervisorOutputUnparseable(
                "supervisor output exceeded the parser candidate bound"
            )
        try:
            value, end = decoder.raw_decode(output, index)
        except (ValueError, RecursionError):
            # Not a JSON object here, or nested past the interpreter's limit:
            # neither is a payload, and both must stay inside the named error.
            index = output.find("{", index + 1)
            continue
        if _ends_a_document(output, end) and accept(value):
            found, match = True, value
        # Skip the whole decoded document: a nested member is never a top-level
        # document, and braces inside its strings must not spend the budget.
        index = output.find("{", end)
    if not found:
        raise SupervisorOutputUnparseable("supervisor output contained no JSON object")
    return match


def is_install_payload(value: object) -> bool:
    """The install command prints either the installed unit or a refusal reason.

    Deliberately shape-only: the caller's own validation still decides whether
    an ``installed`` answer is a usable unit, so a near-miss payload reaches it
    instead of being skipped in favour of an unrelated object.
    """

    return isinstance(value, dict) and isinstance(value.get("installed"), bool)


def is_layout_payload(value: object) -> bool:
    """Shape-only test for the layout probe's document.

    The version marker keeps a future schema from being read as this one; the
    detailed whitelist lives in ``layout_record`` so malformed fields are
    dropped instead of refusing the whole capture.
    """

    return (
        isinstance(value, dict)
        and value.get("schema_version") == 1
        and isinstance(value.get("units"), list)
    )


#: Host-side fixed vocabulary for guest-reported fields. Anything outside these
#: sets becomes a neutral word, so a guest that prints arbitrary text cannot
#: smuggle it into the evidence file through a field the adapter expects to be
#: an enum.
_LAYOUT_STATUSES = frozenset({"ok", "timeout", "error", "unparseable", "withheld"})
_LAYOUT_UNIT_TYPES = frozenset(
    {"simple", "exec", "forking", "oneshot", "dbus", "notify", "notify-reload", "idle", "scope"}
)
_LAYOUT_LOAD_STATES = frozenset(
    {"loaded", "not-found", "bad-setting", "error", "merged", "masked", "unavailable"}
)
_LAYOUT_UNIT_NAME = re.compile(r"[A-Za-z0-9_.:@-]{1,256}\.(?:service|scope)")
_LAYOUT_WORD = re.compile(r"[a-z][a-z-]{0,31}")
_LAYOUT_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_LAYOUT_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "port",
        "pid_source",
        "socket_pids",
        "socket_owners",
        "socket_owner_count",
        "socket_scan_complete",
        "server",
        "ancestry",
        "units",
        "socket_owned_by_unit",
        "other_socket_owners_share_unit_cgroup",
        "supervisor_dropin_kill_mode",
    }
)


def empty_layout(status: str) -> dict[str, Any]:
    """A layout record that says only how the CAPTURE fared.

    ``timeout``/``error``/``unparseable`` describe the diagnostic, not the
    guest: the absence of a layout is never evidence that the guest has no
    units. ``withheld`` means the scrub could not vouch for the capture.
    """

    return {"schema_version": 1, "status": status if status in _LAYOUT_STATUSES else "error"}


def _layout_text(value: object, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    return _LAYOUT_CONTROL.sub("", value)[:limit]


def _layout_pid(value: object) -> int | None:
    # ``type(...) is int`` (not isinstance) so ``True`` is not read as PID 1.
    return value if type(value) is int and value > 0 else None


def _server_unit(
    server: dict[str, Any] | None, units: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """The unit that owns the action server, or ``None`` when it is ambiguous.

    MainPID is the anchor the install path uses, so it wins. A stale MainPID
    (a unit whose MainPID is not the observed server) falls back to the one
    unit whose ControlGroup IS the server's cgroup; two such units cannot
    both own it, so the answer stays undetermined. Equality, not prefix
    matching: a nested sub-cgroup is not what the install path compares.
    """

    if server is None:
        return None
    by_pid = [unit for unit in units if unit["main_pid"] == server["pid"]]
    if by_pid:
        if len({unit["control_group"] for unit in by_pid}) > 1:
            return None
        return by_pid[0]
    if server["cgroup"]:
        by_cgroup = [unit for unit in units if unit["control_group"] == server["cgroup"]]
        if len(by_cgroup) == 1:
            return by_cgroup[0]
    return None


def _owners_share_unit_cgroup(
    server: dict[str, Any] | None,
    sockets: list[int],
    owners: list[dict[str, Any]],
    units: list[dict[str, Any]],
    owner_count: int | None,
    scan_complete: bool | None,
) -> bool | None:
    """Three-state answer: do all NON-server socket owners share the unit cgroup?

    This is CGROUP MEMBERSHIP ONLY, and it does not imply survival. The
    supervisor's drop-in sets ``KillMode=process`` (``SUPERVISOR_KILL_MODE``),
    so a restart signals only the MainPID and every other process survives it
    whether or not it shares the unit's cgroup. What the fact decides is
    whether an explicit cgroup-wide kill (``systemctl kill --kill-whom=all``)
    can reach the holder: same cgroup yes, different cgroup no. A same-cgroup
    survivor therefore needs that explicit kill, or a post-restart check that
    the listener owner is the new MainPID, before supervision may claim
    recovery.

    ``true`` when every other owner's cgroup is readable and equals the owning
    unit's ControlGroup -- including when there is no other owner, which is
    vacuously true (there is no sibling to survive). ``false`` when any
    readable other owner differs. ``null`` when the question cannot be
    answered: no observed socket, no server record, no single owning unit, an
    empty unit ControlGroup, an owner missing from ``socket_owners``, an
    unreadable owner cgroup, a scan that stopped early, a pre-cap owner count
    larger than the recorded list, a missing completeness signal, or one pid
    recorded twice with differing cgroups. ``true`` is never derived from
    missing or truncated data; a known difference still answers ``false`` when
    another owner is unreadable, because the universal claim is already
    refuted.
    """

    if not sockets or server is None:
        return None
    # A universal claim over a list this capture shortened is not evidence.
    # The guest reports the count BEFORE its eight-pid cap and whether the
    # /proc scan reached its end, so a hidden ninth holder cannot be invisible.
    if scan_complete is not True or owner_count is None or owner_count != len(sockets):
        return None
    unit = _server_unit(server, units)
    if unit is None or not unit["control_group"]:
        return None
    by_pid: dict[int, dict[str, Any]] = {}
    for owner in owners:
        seen = by_pid.get(owner["pid"])
        if seen is not None and seen["cgroup"] != owner["cgroup"]:
            # Two entries for one pid that disagree make the evidence
            # self-contradictory; order must not decide the answer.
            return None
        by_pid.setdefault(owner["pid"], owner)
    verdict: bool | None = True
    for pid in sockets:
        if pid == server["pid"]:
            continue
        owner = by_pid.get(pid)
        if owner is None or not owner["cgroup"]:
            verdict = None
            continue
        if owner["cgroup"] != unit["control_group"]:
            return False
    return verdict


def layout_record(output: str, port: int = ACTION_SERVER_PORT) -> dict[str, Any]:
    """Parse and whitelist one layout capture.

    Every field is re-validated on this side, because the guest could print any
    document it likes. Unknown keys are dropped, strings are bounded and
    stripped of control characters, and enum-shaped fields fall back to a
    neutral word. Derived claims (``pid_source``, both socket-owner
    comparisons, the probed port) are recomputed from RAW fields and the
    host's own request,
    never taken from the guest: a capture that says ``socket_owned_by_unit``
    is false beside a unit MainPID that is missing must not be able to assert a
    negative it never observed. Raises ``SupervisorOutputUnparseable`` when no
    layout document is present; callers turn that into
    ``empty_layout('unparseable')``.
    """

    value = parse_output(output, is_layout_payload)
    if not isinstance(value, dict):
        raise ValueError("invalid supervisor layout")
    units: list[dict[str, Any]] = []
    for item in value.get("units") or []:
        if not isinstance(item, dict):
            continue
        unit = item.get("unit")
        if not isinstance(unit, str) or not _LAYOUT_UNIT_NAME.fullmatch(unit):
            continue
        sub_state = item.get("sub_state")
        units.append(
            {
                "unit": unit,
                "main_pid": _layout_pid(item.get("main_pid")),
                "exec_start": _layout_text(item.get("exec_start"), 512),
                "type": item.get("type") if item.get("type") in _LAYOUT_UNIT_TYPES else "other",
                "control_group": _layout_text(item.get("control_group"), 512),
                "sub_state": (
                    sub_state
                    if isinstance(sub_state, str) and _LAYOUT_WORD.fullmatch(sub_state)
                    else "other"
                ),
                "load_state": (
                    item.get("load_state")
                    if item.get("load_state") in _LAYOUT_LOAD_STATES
                    else "unavailable"
                ),
            }
        )
    ancestry: list[dict[str, Any]] = []
    for item in (value.get("ancestry") or [])[:16]:
        if not isinstance(item, dict):
            continue
        pid = _layout_pid(item.get("pid"))
        if pid is None:
            continue
        ancestry.append(
            {
                "pid": pid,
                "ppid": (
                    item.get("ppid")
                    if type(item.get("ppid")) is int and item["ppid"] >= 0
                    else None
                ),
                "comm": _layout_text(item.get("comm"), 64),
            }
        )
    server = value.get("server")
    server_record = None
    if isinstance(server, dict):
        pid = _layout_pid(server.get("pid"))
        if pid is not None:
            server_record = {
                "pid": pid,
                "ppid": (
                    server.get("ppid")
                    if type(server.get("ppid")) is int and server["ppid"] >= 0
                    else None
                ),
                "comm": _layout_text(server.get("comm"), 64),
                "cgroup": _layout_text(server.get("cgroup"), 512),
            }
    # ``socket_owners`` is raw per-owner evidence, not a derived claim: each
    # entry is kept even when its cgroup could not be read (``null``), so the
    # host can tell "shares" from "could not determine".
    owners: list[dict[str, Any]] = []
    for item in (value.get("socket_owners") or [])[:8]:
        if not isinstance(item, dict):
            continue
        pid = _layout_pid(item.get("pid"))
        if pid is None:
            continue
        owners.append(
            {
                "pid": pid,
                "ppid": (
                    item.get("ppid")
                    if type(item.get("ppid")) is int and item["ppid"] >= 0
                    else None
                ),
                "comm": _layout_text(item.get("comm"), 64),
                # An empty or non-string cgroup is unreadable, never "equal".
                "cgroup": _layout_text(item.get("cgroup"), 512) or None,
            }
        )
    # Cap BEFORE deriving: a claim must be reproducible from the units the
    # record actually carries, not from a list it truncated afterwards.
    units = units[:8]
    sockets = [pid for pid in (value.get("socket_pids") or [])[:8] if _layout_pid(pid) is not None]
    # ``pid_source`` is a statement about this capture, not a guest opinion.
    # "socket" needs the recorded server proc to be one of the observed socket
    # owners; "ancestry" needs no socket owner AND an ancestry whose head is
    # that proc -- a lone server record is not ancestry evidence.
    if server_record is not None and sockets and server_record["pid"] in sockets:
        pid_source: str | None = "socket"
    elif (
        server_record is not None
        and not sockets
        and ancestry
        and ancestry[0]["pid"] == server_record["pid"]
    ):
        pid_source = "ancestry"
    else:
        pid_source = None
    # Three-state ``socket_owned_by_unit``: true when ANY observed socket owner
    # is a unit's MainPID (a listener owned by one unit is what supervision
    # needs), false only when every MainPID was observed and none owns a
    # socket, and null while an unobserved MainPID could still be an owner.
    owned: bool | None = None
    if sockets and units:
        known = [unit["main_pid"] for unit in units if unit["main_pid"] is not None]
        if any(socket in known for socket in sockets):
            owned = True
        elif len(known) == len(units):
            owned = False
    owners = owners[:8]
    # Raw completeness signals, validated like every other guest field:
    # ``False`` means the /proc scan stopped early; ``None`` means this capture
    # did not say. Either makes a universal owner claim undetermined.
    count_value = value.get("socket_owner_count")
    owner_count = count_value if type(count_value) is int and count_value >= 0 else None
    scan_value = value.get("socket_scan_complete")
    scan_complete = True if scan_value is True else (False if scan_value is False else None)
    share = _owners_share_unit_cgroup(
        server_record, sockets, owners, units, owner_count, scan_complete
    )
    record: dict[str, Any] = {
        "schema_version": 1,
        "status": "ok",
        "port": port if type(port) is int and 0 < port < 65536 else ACTION_SERVER_PORT,
        "pid_source": pid_source,
        "socket_pids": sockets,
        "socket_owners": owners,
        "socket_owner_count": owner_count,
        "socket_scan_complete": scan_complete,
        "server": server_record,
        "ancestry": ancestry,
        "units": units,
        "socket_owned_by_unit": owned,
        "other_socket_owners_share_unit_cgroup": share,
        # Host-side, never guest input: the KillMode the supervisor drop-in
        # would install -- NOT the unit's current effective mode, which on this
        # refusal path has no drop-in at all. It sits beside the membership
        # answer so a reader cannot mistake "same cgroup" for "a restart reaps
        # it".
        "supervisor_dropin_kill_mode": SUPERVISOR_KILL_MODE,
    }
    # The frozen key set is the contract the provider writes; keeping it here
    # means a future guest field cannot appear in the evidence unannounced.
    assert set(record) <= _LAYOUT_KEYS
    return record


def is_receipt_payload(value: object) -> bool:
    """The report command prints the capture envelope, whose schema is versioned.

    This mirrors the first checks in ``receipt_records``; the generation and
    record validation stay there, so an envelope-shaped object is still
    checked before any input is authorized.
    """

    return (
        isinstance(value, dict)
        and value.get("schema_version") == 1
        and isinstance(value.get("records"), list)
    )


def receipt_records(output: str) -> list[dict[str, Any]]:
    """An empty initialized envelope proves absence; empty/malformed output does not."""
    return _receipt_records_value(parse_output(output, is_receipt_payload))


def _receipt_records_value(value: object) -> list[dict[str, Any]]:
    """Validate one decoded envelope; shared with the restart's nested receipt."""
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("invalid supervisor receipt")
    records = value.get("records")
    if not isinstance(records, list):
        raise ValueError("invalid supervisor records")
    counters = ("baseline_nrestarts", "captured_exits", "systemd_nrestarts")
    timestamps = ("baseline_active_enter", "systemd_active_enter")
    if any(type(value.get(key)) is not int or value[key] < 0 for key in counters):
        raise ValueError("invalid systemd restart counters")
    if any(type(value.get(key)) is not int or value[key] <= 0 for key in timestamps):
        raise ValueError("invalid systemd activation timestamps")
    captured = value["captured_exits"]
    # NRestarts counts systemd's automatic restart jobs, independently of our
    # helper. A successful report of an old empty file cannot hide a failed
    # first write. Timestamp comparison also catches starts with no counter
    # advance (e.g. a manual restart/reset), which cannot be safely correlated.
    if (
        value["systemd_nrestarts"] - value["baseline_nrestarts"] != captured
        or len(records) != min(captured, 4)
        or (value["systemd_active_enter"] == value["baseline_active_enter"]) != (captured == 0)
    ):
        raise GenerationMismatch("systemd generation and capture history disagree")
    for record in records:
        if (
            not isinstance(record, dict)
            or type(record.get("restart")) is not int
            or record["restart"] <= 0
            or any(
                not isinstance(record.get(key), str)
                for key in ("exit_code", "exit_status", "service_result", "stderr_tail")
            )
        ):
            raise ValueError("invalid supervisor record")
    return records


#: Fixed vocabulary for the watchdog record, restated host-side and kept in
#: lockstep with the literals in ``WATCHDOG_SOURCE``. A guest that
#: prints arbitrary text cannot smuggle it into the evidence through a field
#: the adapter reads as an enum; anything outside these sets becomes neutral.
_WATCHDOG_TRIGGERS = frozenset({"none", "failed", "inactive", "unresponsive"})
_WATCHDOG_ACTIONS = frozenset({"none", "reset-failed+start", "verified-restart"})
_WATCHDOG_OUTCOMES = frozenset(
    {
        "none",
        "started",
        "start-failed",
        "reset-failed-error",
        "no-effect",
        "restarted",
        "restart-failed",
        "throttled",
        "capped",
    }
)
#: Observations are NOT attempts: a state the watchdog cannot repair is
#: recorded here without spending the cap, so a dead-but-blocked server is
#: distinguishable from a healthy one instead of both reading ``attempts: 0``.
_WATCHDOG_OBSERVATIONS = frozenset({"none", "serving", "no-listener", "unresponsive", "unknown"})
#: ``_watchdog_count`` gates counters; ``_watchdog_ticks`` gates the monotonic
#: stamps, whose natural range is microseconds since boot (a long-lived guest
#: reaches 1e11), so they cannot share the counter bound.
_MAX_WATCHDOG_COUNT = 1_000_000
_MAX_WATCHDOG_TICKS = 1 << 62


def empty_watchdog(status: str) -> dict[str, Any]:
    """A watchdog record that says only how the READ fared.

    ``absent`` (no record yet), ``unparseable`` (a record we cannot vouch for)
    and ``withheld`` (a scrub refused it) stay distinct, so a missing record is
    never read as "the watchdog never acted".
    """

    return {
        "schema_version": 1,
        "status": status,
        "attempts": 0,
        "intervened": False,
        "capped": False,
        "last_attempt_us": 0,
        "last_trigger": "none",
        "last_action": "none",
        "last_outcome": "none",
        "last_observation": "none",
        "events": [],
    }


def _watchdog_count(value: object) -> int:
    return value if type(value) is int and 0 <= value <= _MAX_WATCHDOG_COUNT else 0


def _watchdog_ticks(value: object) -> int:
    return value if type(value) is int and 0 <= value <= _MAX_WATCHDOG_TICKS else 0


def _watchdog_word(value: object, vocabulary: frozenset[str]) -> str:
    return value if isinstance(value, str) and value in vocabulary else "none"


def _watchdog_value(value: object) -> dict[str, Any]:
    """Sanitize one guest watchdog document; never raises.

    Numbers are the only values carried through, plus the three fixed-vocabulary
    words and the bounded event ring. Note what is absent by construction: unit
    name, port, argv, environment or any other guest text.
    """

    if not isinstance(value, dict) or value.get("schema_version") != 1:
        # A JSON ``null`` is a RECORD THAT EXISTS but cannot be used; only a
        # missing key means no record at all.
        return empty_watchdog("unparseable")
    events: list[dict[str, Any]] = []
    raw_events = value.get("events")
    if isinstance(raw_events, list):
        for event in raw_events[-WATCHDOG_EVENT_CAP:]:
            if not isinstance(event, dict):
                continue
            events.append(
                {
                    "us": _watchdog_ticks(event.get("us")),
                    "trigger": _watchdog_word(event.get("trigger"), _WATCHDOG_TRIGGERS),
                    "action": _watchdog_word(event.get("action"), _WATCHDOG_ACTIONS),
                    "outcome": _watchdog_word(event.get("outcome"), _WATCHDOG_OUTCOMES),
                }
            )
    return {
        "schema_version": 1,
        "status": "ok",
        "attempts": _watchdog_count(value.get("attempts")),
        "intervened": value.get("intervened") is True,
        "capped": value.get("capped") is True,
        "last_attempt_us": _watchdog_ticks(value.get("last_attempt_us")),
        "last_trigger": _watchdog_word(value.get("last_trigger"), _WATCHDOG_TRIGGERS),
        "last_action": _watchdog_word(value.get("last_action"), _WATCHDOG_ACTIONS),
        "last_outcome": _watchdog_word(value.get("last_outcome"), _WATCHDOG_OUTCOMES),
        "last_observation": _watchdog_word(value.get("last_observation"), _WATCHDOG_OBSERVATIONS),
        "events": events,
    }


def receipt_report(output: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Both halves of ONE capture read: the records and the watchdog summary.

    One parse, so the action path does not decode a large capture twice. The
    receipt's own validation is untouched; the watchdog half is additive
    evidence that authorizes nothing, so a malformed record can never fail a
    recovery that the receipt itself proved.
    """

    value = parse_output(output, is_receipt_payload)
    if not isinstance(value, dict):
        raise ValueError("invalid supervisor receipt")
    if "watchdog" not in value:
        return _receipt_records_value(value), empty_watchdog("absent")
    return _receipt_records_value(value), _watchdog_value(value["watchdog"])


#: Every status a restart verification can report. Fixed vocabulary only: no
#: guest text reaches the evidence or an exception through this field.
RESTART_STATUSES = frozenset(
    {
        "verified",
        "stale_listener",
        "verification_timeout",
        "incomplete_scan",
        "receipt_unavailable",
        "command_failed",
        "result_unavailable",
    }
)

#: One fixed cause per status. The failure evidence names WHY this restart was
#: not accepted, so a stale listener and a missing result do not both read as
#: the same sentence. Never guest text.
RESTART_CAUSES = {
    "stale_listener": "post-restart listener predates the restart",
    "verification_timeout": "no listener appeared before the verification deadline",
    "incomplete_scan": "socket owner scan was incomplete",
    "receipt_unavailable": "post-restart receipt could not be reconciled",
    "command_failed": "restart dispatch failed",
    "result_unavailable": "restart result was not readable",
}


def restart_cause(status: str) -> str:
    return RESTART_CAUSES.get(status, "restart verification failed")


#: Evidence is bounded like the layout record: the pre/post owner sets are
#: capped, and a pre-cap count above the cap makes the scan incomplete. An
#: incomplete set cannot prove anything about the holders it did not see.
RESTART_OWNER_CAP = 8


def is_restart_payload(value: object) -> bool:
    return (
        isinstance(value, dict)
        and value.get("schema_version") == 1
        and type(value.get("token")) is int
        and value["token"] > 0
        and isinstance(value.get("unit"), str)
        and isinstance(value.get("pre_restart_pids"), list)
        and isinstance(value.get("post_restart_pids"), list)
        and isinstance(value.get("post_restart_owners"), list)
    )


def listener_postdates_restart(
    owners: Sequence[dict[str, Any]],
    new_main_pid: int,
    new_main_start_ticks: int,
    kill_uptime_ticks: int,
) -> bool:
    """True only when every listener STARTED at/after the kill and is the new
    MainPID or one of its descendants.

    Set difference alone is not freshness: a holder that forks a worker while
    dying creates a pid the pre scan could not have seen, and that worker
    serves pre-restart state. The kill stamp (``/proc/uptime`` in clock ticks,
    the same base as ``/proc/<pid>/stat`` start times) orders the processes;
    the reported ancestor chain ties the listener to the process systemd just
    started. Same-tick equality is allowed because ancestry still gates it.
    """
    if (
        not owners
        or new_main_pid <= 0
        or kill_uptime_ticks <= 0
        or new_main_start_ticks < kill_uptime_ticks
    ):
        return False
    for owner in owners:
        if owner["start_ticks"] < kill_uptime_ticks:
            return False
        if owner["pid"] != new_main_pid and new_main_pid not in owner["ancestors"]:
            return False
    return True


def empty_restart(status: str = "command_failed") -> dict[str, Any]:
    """A verification that produced no usable guest payload."""
    return {
        "schema_version": 1,
        "status": status if status in RESTART_STATUSES else "command_failed",
        "unit": None,
        "token": None,
        "pre_restart_pids": [],
        "post_restart_pids": [],
        "post_restart_owners": [],
        "post_restart_owner": None,
        "kill_uptime_ticks": 0,
        "new_main_pid": None,
        "new_main_start_ticks": 0,
        "listener_is_new_main_pid": None,
        "fresh": False,
        "latest_restart": None,
    }


def _restart_pids(value: object) -> list[int]:
    if not isinstance(value, list):
        raise ValueError("invalid restart owner list")
    pids: list[int] = []
    for item in value:
        if type(item) is not int or item <= 0:
            raise ValueError("invalid restart owner pid")
        if item not in pids:
            pids.append(item)
    return pids[:RESTART_OWNER_CAP]


def _restart_owners(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("invalid restart owner detail")
    owners: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("invalid restart owner detail")
        pid = item.get("pid")
        ppid = item.get("ppid")
        start = item.get("start_ticks")
        chain = item.get("ancestors")
        if type(pid) is not int or pid <= 0:
            raise ValueError("invalid restart owner pid")
        if type(ppid) is not int or ppid < 0:
            raise ValueError("invalid restart owner ppid")
        if type(start) is not int or start < 0:
            raise ValueError("invalid restart owner start time")
        if not isinstance(chain, list) or not all(type(pid_) is int and pid_ > 0 for pid_ in chain):
            raise ValueError("invalid restart ancestor chain")
        owners.append(
            {
                "pid": pid,
                "ppid": ppid,
                "start_ticks": start,
                "ancestors": chain[:RESTART_OWNER_CAP],
            }
        )
        if len(owners) >= RESTART_OWNER_CAP:
            break
    return owners


def restart_record(output: str) -> dict[str, Any]:
    """Whitelist one detached restart document and DERIVE freshness here.

    The guest reports what it observed (numeric ids, start ticks, ancestor
    chains and the receipt envelope); it does not get to say whether the
    listener is fresh. ``fresh`` is computed from the kill stamp, the new
    MainPID's start time and each owner's ancestry, and the status is chosen
    from the fixed vocabulary. A receipt the host cannot reconcile makes the
    restart unrecoverable even when the listener is fresh: the generation could
    not be consumed, so the next poll would bounce again.
    """
    value = parse_output(output, is_restart_payload)
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("invalid restart record")
    unit = value.get("unit")
    if not isinstance(unit, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+\.service", unit):
        raise ValueError("invalid restart unit")
    pre = _restart_pids(value.get("pre_restart_pids"))
    post = _restart_pids(value.get("post_restart_pids"))
    owners = _restart_owners(value.get("post_restart_owners"))
    pre_count = value.get("pre_restart_count")
    post_count = value.get("post_restart_count")
    if type(pre_count) is not int or pre_count < len(pre):
        raise ValueError("invalid pre-restart count")
    if type(post_count) is not int or post_count < len(post):
        raise ValueError("invalid post-restart count")
    if type(value.get("pre_scan_complete")) is not bool:
        raise ValueError("invalid pre-restart scan flag")
    if type(value.get("post_scan_complete")) is not bool:
        raise ValueError("invalid post-restart scan flag")
    new_main = value.get("new_main_pid")
    if type(new_main) is not int or new_main < 0:
        raise ValueError("invalid restart MainPID")
    new_main_start = value.get("new_main_start_ticks")
    if type(new_main_start) is not int or new_main_start < 0:
        raise ValueError("invalid restart MainPID start time")
    kill_ticks = value.get("kill_uptime_ticks")
    if type(kill_ticks) is not int or kill_ticks < 0:
        raise ValueError("invalid restart kill stamp")
    try:
        records = _receipt_records_value(value.get("receipt"))
        latest = records[-1]["restart"] if records else None
    except (ValueError, GenerationMismatch):
        records, latest = [], None
    incomplete = (
        value["pre_scan_complete"] is not True
        or value["post_scan_complete"] is not True
        or pre_count != len(pre)
        or post_count != len(post)
        or [owner["pid"] for owner in owners] != post
    )
    fresh = (
        False
        if incomplete
        else listener_postdates_restart(owners, new_main, new_main_start, kill_ticks)
    )
    if incomplete:
        status = "incomplete_scan"
    elif not post:
        status = "verification_timeout"
    elif not fresh:
        status = "stale_listener"
    elif latest is None:
        status = "receipt_unavailable"
    else:
        status = "verified"
    owner = post[0] if post else None
    return {
        "schema_version": 1,
        "status": status,
        "unit": unit,
        "token": value["token"],
        "pre_restart_pids": pre,
        "post_restart_pids": post,
        "post_restart_owners": owners,
        "post_restart_owner": owner,
        "kill_uptime_ticks": kill_ticks,
        "new_main_pid": new_main or None,
        "new_main_start_ticks": new_main_start,
        "listener_is_new_main_pid": (owner == new_main) if owner and new_main else None,
        "fresh": fresh,
        "latest_restart": latest,
    }


def installed_unit(output: str) -> str:
    value = parse_output(output, is_install_payload)
    if not isinstance(value, dict) or value.get("installed") is not True:
        raise ValueError("invalid supervisor installation")
    unit = value.get("unit")
    if not isinstance(unit, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+\.service", unit):
        raise ValueError("invalid supervisor unit")
    return unit


def watchdog_install_status(value: object) -> dict[str, Any] | None:
    """The install's watchdog proof, accepted only in its exact fixed shape.

    The guest refuses to install when its watchdog is not active or shares the
    server's cgroup, so an ``installed: true`` payload should always carry
    this. It is still re-derived host-side: a guest that omits or malforms the
    proof yields ``None`` (no claim) instead of a fabricated success.
    """

    if not isinstance(value, dict):
        return None
    watchdog = value.get("watchdog")
    if not isinstance(watchdog, dict):
        return None
    if (
        watchdog.get("installed") is not True
        or watchdog.get("status") != "active"
        or watchdog.get("cgroup_distinct") is not True
    ):
        return None
    return {"installed": True, "status": "active", "cgroup_distinct": True}
