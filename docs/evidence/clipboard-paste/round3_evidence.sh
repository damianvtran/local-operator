#!/bin/sh
# Round-3 remediation evidence: F4 (the fourth hang shape), D12 (route
# asymmetry and copy), D13 (a showing card surviving its answer).
#
# Reproduce with:  sh docs/evidence/clipboard-paste/round3_evidence.sh
set -u
REPO="${REPO:-/tmp/lop-clipboard-paste}"
cd "$REPO"

echo "############ F4 (major) — direct child exits, descendant holds the pipe ############"
echo "poll() returns 0, so a kill gated on it was skipped; the reader stayed"
echo "blocked and Popen.__exit__ deadlocked the MAIN thread on stdout.close()."
echo "Measured before the fix: never returned after 20s on a 2.0s budget."
echo
echo "load average: $(uptime | sed 's/.*load averages*: //')"
.venv/bin/python -c "
import faulthandler, subprocess, sys, time
sys.path.insert(0, '$REPO')
import local_operator.clipboard as cb

MARK = 'lo-f4-evidence'
faulthandler.dump_traceback_later(20, exit=True)
d = cb._Deadline(2.0)
t0 = time.perf_counter()
out = cb._run(['/bin/sh', '-c', f'sleep 600 & : {MARK}; exit 0'], d, max_bytes=4096)
print('  returned %.2fs (budget 2.0s) -> %r' % (time.perf_counter() - t0, out))
faulthandler.cancel_dump_traceback_later()
time.sleep(0.3)
left = subprocess.run(['pgrep', '-f', 'sleep 600'], capture_output=True, text=True, check=False)
print('  survivors: %s' % (left.stdout.strip() or 'none'))
"
pkill -f "sleep 600" 2>/dev/null || true

echo
echo "############ D12 (major) — one image, both routes, same answer ############"
echo "The clipboard route bounds before the attachment cap (U1's fix); the path"
echo "route refused at stat() against that cap, so Finder Cmd+C rejected a"
echo "screenshot Cmd+V attached, and blamed its format."
.venv/bin/python -c "
import asyncio, io, random, sys, tempfile, time
from pathlib import Path
sys.path.insert(0, '$REPO')
from PIL import Image
from textual import events
from local_operator.clipboard import ClipboardContents, ClipboardImage
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets import editor as em
from local_operator.tui.widgets.editor import Editor, MAX_ATTACHMENT_BYTES
from local_operator.tui.widgets.toast import Toast
from tests.unit.tui.test_app_pilot import FakeSession, _factory

# High-entropy: a flat-colour PNG compresses under the cap by accident.
random.seed(5)
im = Image.new('RGB', (2600, 900))
im.putdata([(random.randrange(256), random.randrange(256), random.randrange(256))
            for _ in range(2600 * 900)])
buf = io.BytesIO(); im.save(buf, 'PNG'); data = buf.getvalue()
tmp = Path(tempfile.mkdtemp()) / 'screenshot.png'; tmp.write_bytes(data)
print('  source PNG: %.1f MB (attachment cap %.1f MB)' % (len(data)/1048576, MAX_ATTACHMENT_BYTES/1048576))

async def drive(label, contents):
    em.read_clipboard = lambda *a, _c=contents, **k: _c
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        end = time.monotonic() + 10
        while time.monotonic() < end and app._session is None:
            await pilot.pause()
        ed = app.query_one(Editor); ed.focus(); await pilot.pause()
        app.post_message(events.Paste(''))
        for _ in range(300):
            await pilot.pause()
            if ed.referenced_images() or app.query_one(Toast).message:
                break
        print('  %-26s editor=%-26r toast=%r'
              % (label, ed.text[:24], app.query_one(Toast).message))

async def main():
    await drive('Cmd+V (clipboard bytes)', ClipboardContents(image=ClipboardImage(data, 'image/png')))
    await drive('Finder Cmd+C (file URL)', ClipboardContents(paths=(str(tmp),)))
asyncio.run(main())
"

echo
echo "############ D12 — the five causes that reach the file notice ############"
.venv/bin/python -c "
import asyncio, sys, tempfile, time
from pathlib import Path
sys.path.insert(0, '$REPO')
from PIL import Image
from textual import events
from local_operator.clipboard import ClipboardContents
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets import editor as em
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.toast import Toast
from tests.unit.tui.test_app_pilot import FakeSession, _factory

td = Path(tempfile.mkdtemp())
txt = td / 'notes.txt'; txt.write_text('hello')
good = td / 'ok.png'; Image.new('RGB', (40, 20), (9, 9, 9)).save(good)
heic = td / 'photo.heic'; heic.write_bytes(b'\x00\x00\x00\x18ftypheic' + b'\x00' * 128)

CASES = [('A non-image .txt', (str(txt),)),
         ('C HEIC', (str(heic),)),
         ('D good PNG + .txt', (str(good), str(txt))),
         ('E path does not exist', ('/nope/missing.png',)),
         ('F control: small PNG', (str(good),))]

async def main():
    for label, paths in CASES:
        em.read_clipboard = lambda *a, _p=paths, **k: ClipboardContents(paths=_p)
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            end = time.monotonic() + 10
            while time.monotonic() < end and app._session is None:
                await pilot.pause()
            ed = app.query_one(Editor); ed.focus(); await pilot.pause()
            app.post_message(events.Paste(''))
            for _ in range(250):
                await pilot.pause()
                if ed.referenced_images() or app.query_one(Toast).message:
                    break
            print('  %-24s -> %-22r %s'
                  % (label, ed.text[:20], app.query_one(Toast).message or '(no notice)'))
asyncio.run(main())
"

echo
echo "############ D13 (minor) — a SHOWING card is retired by the paste answering it ############"
.venv/bin/python -c "
import asyncio, sys, tempfile, time
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

shot = Path(tempfile.mkdtemp()) / 's.png'
Image.new('RGB', (400, 100), (30, 30, 40)).save(shot)
em.read_clipboard = lambda *a, **k: ClipboardContents(refused_remote=True)

async def main():
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        end = time.monotonic() + 10
        while time.monotonic() < end and app._session is None:
            await pilot.pause()
        ed = app.query_one(Editor); ed.focus(); await pilot.pause()
        t = app.query_one(Toast)
        app.post_message(events.Paste(''))
        for _ in range(200):
            await pilot.pause()
            if t.message:
                break
        print('  SHOWING case')
        print('    raised   %r' % t.message)
        app.post_message(events.Paste(str(shot)))
        for _ in range(200):
            await pilot.pause()
            if ed.referenced_images():
                break
        for _ in range(6):
            await pilot.pause()
        print('    attached editor=%r' % ed.text)
        print('    showing  %r display=%s' % (t.message, t.display))
        print('  HELD case (D3/D8 must stay closed)')
        t.show('MCP github failed: command not found: gh', duration_ms=TOAST_FAILURE_MS)
        await pilot.pause()
        app.post_message(events.Paste(''))
        for _ in range(200):
            await pilot.pause()
            if t._deferred:
                break
        print('    deferred %r' % (t._deferred[0] if t._deferred else None))
        app.post_message(events.Paste(str(shot)))
        for _ in range(250):
            await pilot.pause()
            if len(ed.referenced_images()) > 1:
                break
        print('    deferred %r   <- retired' % (t._deferred[0] if t._deferred else None))
asyncio.run(main())
"
