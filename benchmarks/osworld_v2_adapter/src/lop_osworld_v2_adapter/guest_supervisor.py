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
from typing import Sequence

from lop_osworld_v2_adapter import guest_disk

# ExecStopPost gets the old process's wait status BEFORE Restart replaces it.
# Keep only four bounded records, including the terminal crash-loop failure.
CAPTURE_SOURCE = r"""
import json, os, pathlib, subprocess, sys, time
unit = sys.argv[1]
root = pathlib.Path('/var/lib/lop-action-supervisor')
path = root / (unit + '.json')
p = subprocess.run(['journalctl', '--no-pager', '-o', 'cat', '-n', '30', '-u', unit],
    capture_output=True, text=True, timeout=2)
record = {'restart': time.time_ns(), 'unit': unit,
          'exit_code': os.environ.get('EXIT_CODE', 'unknown'),
          'exit_status': os.environ.get('EXIT_STATUS', 'unknown'),
          'service_result': os.environ.get('SERVICE_RESULT', 'unknown'),
          'stderr_tail': p.stdout[-4096:]}
try:
    records = json.loads(path.read_text())
except (OSError, ValueError):
    records = []
records = (records + [record])[-4:]
tmp = path.with_suffix('.tmp')
tmp.write_text(json.dumps(records))
tmp.replace(path)
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
    '-p', 'ExecStop', '-p', 'ExecStopPost', '-p', 'StandardError', '-p', 'StandardOutput'
    ).splitlines())
if props.get('MainPID') != str(pid) or 'main.py' not in props.get('ExecStart', ''):
    raise RuntimeError('service does not exclusively own action server')
if props.get('StandardError') != 'journal' and not (
        props.get('StandardError') == 'inherit' and props.get('StandardOutput') == 'journal'):
    raise RuntimeError('existing server stderr is not journalled')
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
if path.exists():
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
    try:
        run('systemctl', 'daemon-reload')
    except Exception:
        path.unlink()
        run('systemctl', 'daemon-reload')
        raise
print(json.dumps({'unit': unit, 'installed': True}))
"""

REPORT_SOURCE = r"""
import pathlib, sys
print((pathlib.Path('/var/lib/lop-action-supervisor') / (sys.argv[1] + '.json')).read_text())
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


def installed_unit(output: str) -> str:
    value = parse_output(output)
    if not isinstance(value, dict) or value.get("installed") is not True:
        raise ValueError("invalid supervisor installation")
    unit = value.get("unit")
    if not isinstance(unit, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+\.service", unit):
        raise ValueError("invalid supervisor unit")
    return unit
