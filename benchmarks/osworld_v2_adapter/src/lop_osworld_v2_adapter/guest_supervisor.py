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

# Only the action server's private service is eligible. A shared user/session
# scope must never gain a restart policy because it owns the GUI. Inspect only
# our HTTP command's ancestors, never arbitrary process or task paths.
INSTALL_SOURCE = r"""
import json, os, pathlib, re, subprocess, sys

def run(*args):
    return subprocess.check_output(args, text=True, timeout=3)

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
    '-p', 'Type', '-p', 'LoadState', '-p', 'NRestarts', '-p', 'ActiveEnterTimestampMonotonic'
    ).splitlines())
if props.get('MainPID') != str(pid) or 'main.py' not in props.get('ExecStart', ''):
    raise RuntimeError('service does not exclusively own action server')
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
if not created:
    if path.read_text() != content or capture.read_text() != sys.argv[1]:
        raise RuntimeError('supervisor installation conflict')
else:
    if any(props.get(k) for k in ('ExecStartPre', 'ExecStartPost', 'ExecStop', 'ExecStopPost')):
        raise RuntimeError('service hooks could mutate desktop state')
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if capture.exists() and capture.read_text() != sys.argv[1]:
        raise RuntimeError('supervisor capture conflict')
    capture.write_text(sys.argv[1])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as out:
        out.write(content)
# An initialized empty envelope is distinct from a missing/corrupt receipt.
# Never let a failed read become evidence that no restart happened.
receipt = root / (unit + '.json')
try:
    if not receipt.exists():
        with receipt.open('x') as out:
            json.dump({'schema_version': 1, 'records': [], 'captured_exits': 0,
                       'baseline_nrestarts': int(props['NRestarts']),
                       'baseline_active_enter': int(props['ActiveEnterTimestampMonotonic'])}, out)
    if created:
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
except Exception:
    # Roll back only the drop-in this attempt created, never a pre-existing
    # installation. Re-entry MUST verify effective config, not trust bytes.
    if created:
        path.unlink()
        run('systemctl', 'daemon-reload')
    raise
print(json.dumps({'unit': unit, 'installed': True}))
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
print(json.dumps(history))
"""


def _privileged(source: str, args: Sequence[str], password: str) -> Sequence[str]:
    # Reuse preparation's non-interactive escalation ladder. Nothing logs its
    # argv or error output, which may contain the supplied password.
    fragment = shlex.join(["python3", "-c", source, *args])
    return ["bash", "-c", guest_disk.escalation_script(fragment, password)]


def install_command(password: str) -> Sequence[str]:
    # Report capability failures as successful command execution, so the sudo
    # ladder does not mistake an unsupported service for failed authentication.
    # RuntimeError messages above are fixed vocabulary, never guest-authored.
    wrapper = (
        "import json\ntry:\n    exec(" + repr(INSTALL_SOURCE) + ")\n"
        "except RuntimeError as error:\n"
        "    print(json.dumps({'installed': False, 'reason': str(error)}))\n"
        "except Exception as error:\n"
        "    print(json.dumps({'installed': False, 'reason': type(error).__name__}))\n"
    )
    return _privileged(wrapper, [CAPTURE_SOURCE], password)


def report_command(unit: str, password: str) -> Sequence[str]:
    return _privileged(REPORT_SOURCE, [unit], password)


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
    value = parse_output(output, is_receipt_payload)
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


def installed_unit(output: str) -> str:
    value = parse_output(output, is_install_payload)
    if not isinstance(value, dict) or value.get("installed") is not True:
        raise ValueError("invalid supervisor installation")
    unit = value.get("unit")
    if not isinstance(unit, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+\.service", unit):
        raise ValueError("invalid supervisor unit")
    return unit
