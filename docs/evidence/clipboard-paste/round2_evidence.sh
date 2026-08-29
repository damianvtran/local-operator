#!/bin/sh
# Round-2 remediation evidence: the F2/U6 regression (a real hung tool), the
# U3/U7 latency, and the D8/D9/D10/D11 fixes. Real processes and a real
# pasteboard driven through the real OperatorApp.
#
# Reproduce with:  sh docs/evidence/clipboard-paste/round2_evidence.sh
set -u
REPO="${REPO:-/tmp/lop-clipboard-paste}"
cd "$REPO"

echo "############ F2 / U6 (blocker) — a REAL hung tool is bounded ############"
echo "The round-2 regression: a blocking stdout.read() ran BEFORE the deadline"
echo "check, so a tool holding stdout open never reached it. 15s X11 / 12s macOS"
echo "measured against a 2.0s budget, child orphaned."
echo
echo "load average: $(uptime | sed 's/.*load averages*: //')"
.venv/bin/python -c "
import subprocess, sys, time
sys.path.insert(0, '$REPO')
import local_operator.clipboard as cb

MARK = 'lo-clip-evidence-probe'
for label, argv, kw in [
    ('hung tool, bounded read', ['/bin/sh', '-c', f': {MARK}; sleep 30'], {'max_bytes': 4096}),
    ('hung tool, unbounded read', ['/bin/sh', '-c', f': {MARK}; sleep 30'], {}),
    ('answers then hangs', ['/bin/sh', '-c', f': {MARK}; printf xx; sleep 30'], {'max_bytes': 4096}),
]:
    d = cb._Deadline(2.0)
    t0 = time.perf_counter()
    out = cb._run(argv, d, **kw)
    elapsed = time.perf_counter() - t0
    time.sleep(0.3)
    left = subprocess.run(['pgrep', '-f', MARK], capture_output=True, text=True, check=False)
    print('  %-26s returned %.2fs (budget 2.0s) -> %r  orphans: %s'
          % (label, elapsed, out, left.stdout.strip() or 'none'))

# The control: the bound must not break the healthy path.
d = cb._Deadline(2.0)
print('  %-26s -> %r' % ('healthy tool', cb._run(['/bin/echo', '-n', 'hello'], d)))
"

echo
echo "############ U3 / U7 (major/minor) — latency, measured at the composer ############"
echo "Round 2 measured 4-9s to a marker and 4203ms for a whitespace paste. The"
echo "cost was osascript SPAWNS: 2-4s of wall time each against ~0.25s of CPU,"
echo "and the miss path paid two of them. One script now answers both shapes,"
echo "and a copied-whitespace paste does not consult the clipboard at all."
echo
screencapture -c -x 2>/dev/null
sleep 1
.venv/bin/python -c "
import asyncio, sys, time
sys.path.insert(0, '$REPO')
from textual import events
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.toast import Toast
from tests.unit.tui.test_app_pilot import FakeSession, _factory

async def measure(payload, label):
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        end = time.monotonic() + 10
        while time.monotonic() < end and app._session is None:
            await pilot.pause()
        ed = app.query_one(Editor); ed.focus(); await pilot.pause()
        t0 = time.perf_counter()
        app.post_message(events.Paste(payload))
        for _ in range(400):
            await pilot.pause()
            if ed.referenced_images() or app.query_one(Toast).message or (payload and ed.text):
                break
        print('  %-30s %6.0f ms  -> %r' % (label, (time.perf_counter()-t0)*1000, ed.text[:38]))

async def main():
    await measure('hello world', 'TEXT paste (control)')
    await measure('    ', 'WHITESPACE paste (was 4203ms)')
    await measure('\t', 'TAB paste (was 3850ms)')
    await measure('', 'EMPTY -> screenshot (was 4-9s)')
asyncio.run(main())
"

echo
echo "############ D9 (minor) — copied whitespace is no longer traded for an image ############"
.venv/bin/python -c "
import asyncio, io, sys, time
sys.path.insert(0, '$REPO')
from PIL import Image
from textual import events
from local_operator.clipboard import ClipboardContents, ClipboardImage
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets import editor as em
from local_operator.tui.widgets.editor import Editor
from tests.unit.tui.test_app_pilot import FakeSession, _factory

b = io.BytesIO(); Image.new('RGB', (1568, 200), (30, 30, 40)).save(b, 'PNG')
# An image IS readable on the clipboard for every case below.
em.read_clipboard = lambda *a, **k: ClipboardContents(image=ClipboardImage(b.getvalue(), 'image/png'))

async def run(payload, label):
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        end = time.monotonic() + 10
        while time.monotonic() < end and app._session is None:
            await pilot.pause()
        ed = app.query_one(Editor); ed.focus(); await pilot.pause()
        ed.insert('def handler():' + chr(10))
        app.post_message(events.Paste(payload))
        for _ in range(150):
            await pilot.pause()
            if ed.referenced_images():
                break
        kept = ed.text.endswith(payload)
        print('  %-14s pasted=%-8r -> %-44r whitespace kept: %s'
              % (label, payload, ed.text, kept))

async def main():
    await run('    ', 'four spaces')
    await run(chr(9), 'tab')
    await run('', 'EMPTY signal')
asyncio.run(main())
"

echo
echo "############ D8 (major) — the path route retires a held notice ############"
echo "The notice tells the user to paste a file path. They do, it works, and the"
echo "card used to come back afterwards to say it did not."
.venv/bin/python -c "
import asyncio, io, sys, tempfile, time
from pathlib import Path
sys.path.insert(0, '$REPO')
from PIL import Image
from textual import events
from local_operator.clipboard import ClipboardContents
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets import editor as em
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.toast import Toast, TOAST_FAILURE_MS
from tests.unit.tui.test_app_pilot import FakeSession, _factory

em.read_clipboard = lambda *a, **k: ClipboardContents(refused_remote=True)

async def main():
    with tempfile.TemporaryDirectory() as td:
        shot = Path(td) / 'shot.png'
        Image.new('RGB', (400, 100), (30, 30, 40)).save(shot)
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            end = time.monotonic() + 10
            while time.monotonic() < end and app._session is None:
                await pilot.pause()
            ed = app.query_one(Editor); ed.focus(); await pilot.pause()
            t = app.query_one(Toast)
            t.show('MCP github failed: command not found: gh', duration_ms=TOAST_FAILURE_MS)
            await pilot.pause()
            print('  1. MCP failure holds the slot')
            app.post_message(events.Paste(''))
            for _ in range(150):
                await pilot.pause()
                if t._deferred:
                    break
            print('  2. Cmd+V over SSH  deferred=%r' % (t._deferred[0] if t._deferred else None))
            app.post_message(events.Paste(str(shot)))
            for _ in range(200):
                await pilot.pause()
                if ed.referenced_images():
                    break
            print('  3. user pastes the FILE PATH, as instructed')
            print('       editor  =%r' % ed.text)
            print('       deferred=%r   <- retired' % (t._deferred[0] if t._deferred else None))
            t.dismiss_toast()
            for _ in range(8):
                await pilot.pause()
            print('  4. MCP card expires -> showing=%r display=%s' % (t.message, t.display))
asyncio.run(main())
"

echo
echo "############ D10 / D11 (minor) — message accuracy and toast rank ############"
.venv/bin/python -c "
import asyncio, sys, time
sys.path.insert(0, '$REPO')
from textual import events
from local_operator.clipboard import ClipboardContents, ClipboardImage
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets import editor as em
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.toast import Toast
from tests.unit.tui.test_app_pilot import FakeSession, _factory

CASES = [
    ('empty / text-only clipboard', ClipboardContents()),
    ('image found, unattachable', ClipboardContents(image=ClipboardImage(b'\x00nope', 'image/png'))),
    ('file copied, unreadable', ClipboardContents(paths=('/nonexistent/x.txt',))),
    ('over SSH', ClipboardContents(refused_remote=True)),
]

async def main():
    for label, contents in CASES:
        em.read_clipboard = lambda *a, _c=contents, **k: _c
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            end = time.monotonic() + 10
            while time.monotonic() < end and app._session is None:
                await pilot.pause()
            app.query_one(Editor).focus(); await pilot.pause()
            app.post_message(events.Paste(''))
            for _ in range(200):
                await pilot.pause()
                if app.query_one(Toast).message:
                    break
            t = app.query_one(Toast)
            print('  %-28s actionable=%-5s %r' % (label, t._actionable, t.message))
asyncio.run(main())
"
