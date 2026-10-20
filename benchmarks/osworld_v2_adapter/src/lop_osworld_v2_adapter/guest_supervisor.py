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
