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
from typing import Any, Literal, Sequence

from lop_osworld_v2_adapter import guest_disk


class GenerationMismatch(ValueError):
    """systemd observed a restart that the guest capture history cannot prove."""


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


def parse_output(output: str) -> object:
    # The escalation ladder prints one metadata line before the command output.
    return json.loads(output.split("\n", 1)[1])


def receipt_records(output: str) -> list[dict[str, Any]]:
    """An empty initialized envelope proves absence; empty/malformed output does not."""
    value = parse_output(output)
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
    value = parse_output(output)
    if not isinstance(value, dict) or value.get("installed") is not True:
        raise ValueError("invalid supervisor installation")
    unit = value.get("unit")
    if not isinstance(unit, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+\.service", unit):
        raise ValueError("invalid supervisor unit")
    return unit
