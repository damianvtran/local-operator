#!/bin/sh
# Round-1 remediation evidence: the blocker (U1) and the regressions (F1/D1,
# D3, D2/U2) verified against real pasteboard state and the real OperatorApp.
#
# Reproduce with:  sh docs/evidence/clipboard-paste/round1_evidence.sh
set -u
REPO="${REPO:-/tmp/lop-clipboard-paste}"
FIX="${FIX:-/tmp/lop372}"
cd "$REPO"

echo "############ U1 (blocker) — a REAL Retina screenshot now attaches ############"
echo "The gesture: screencapture -c, which is exactly what Cmd+Shift+Ctrl+4 runs."
echo
# A dense full-screen-sized capture: a flat-colour screen compresses to a few KB
# and would sit under the old cliff by accident, which is why the first round of
# testing missed this. This is what a screen full of text and windows looks like
# to the PNG encoder.
osascript -e "set the clipboard to (read (POSIX file \"$FIX/dense.png\") as «class PNGf»)"
sleep 1
echo "  clipboard info: $(osascript -e 'clipboard info' | cut -c1-40)"
echo "  pbpaste bytes : $(pbpaste | wc -c | tr -d ' ')"
.venv/bin/python -c "
import asyncio, base64, sys, time
sys.path.insert(0, '$REPO')
from textual import events
from local_operator.clipboard import read_clipboard, MAX_CLIPBOARD_READ_BYTES
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.editor import Editor, MAX_ATTACHMENT_BYTES
from local_operator.tui.widgets.toast import Toast
from tests.unit.tui.test_app_pilot import FakeSession, _factory

raw = read_clipboard()
print('  pasteboard PNG        : %.2f MB' % (len(raw.image.data) / 1048576))
print('  attachment cap        : %.0f MB   <- what the OLD code applied to the read' % (MAX_ATTACHMENT_BYTES / 1048576))
print('  ingest ceiling        : %.0f MB   <- what bounds the read now' % (MAX_CLIPBOARD_READ_BYTES / 1048576))
old = read_clipboard(MAX_ATTACHMENT_BYTES)
print('  read at the OLD cap   : %s   <- the U1 blocker, reproduced' % old.image)

async def main():
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        end = time.monotonic() + 10
        while time.monotonic() < end and app._session is None:
            await pilot.pause()
        ed = app.query_one(Editor); ed.focus(); await pilot.pause()
        t0 = time.perf_counter()
        app.post_message(events.Paste(''))
        for _ in range(300):
            await pilot.pause()
            if ed.referenced_images() or app.query_one(Toast).message:
                break
        print()
        print('  COMPOSER after Cmd+V (%.2fs): %r' % (time.perf_counter() - t0, ed.text))
        img = ed.referenced_images()[0]
        data = base64.b64decode(img.data)
        print('  attached              : %.2f MB, under the attachment cap: %s'
              % (len(data) / 1048576, len(data) <= MAX_ATTACHMENT_BYTES))
        print('  toast                 : %r' % app.query_one(Toast).message)
        ed.text = ed.text + 'what is this?'
        await pilot.press('enter')
        for _ in range(80):
            await pilot.pause()
            if app._session.prompts:
                break
        print('  images sent to model  :', len(app._session.prompt_images[-1]))
asyncio.run(main())
"

echo
echo "############ F1 / D1 (blocker) — pasted whitespace is inserted again ############"
printf 'plain text, no image' | pbcopy
sleep 1
.venv/bin/python -c "
import asyncio, sys, time
sys.path.insert(0, '$REPO')
from textual import events
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.editor import Editor
from tests.unit.tui.test_app_pilot import FakeSession, _factory

CASES = [('    ', 'four-space indent'), ('  ', 'two-space indent'), (' ', 'single space'),
         ('\t', 'tab'), ('\n', 'blank line'), ('\n\n', 'two blank lines'), ('', 'EMPTY (the real signal)')]

async def main():
    for payload, label in CASES:
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            end = time.monotonic() + 10
            while time.monotonic() < end and app._session is None:
                await pilot.pause()
            ed = app.query_one(Editor); ed.focus(); await pilot.pause()
            ed.insert('X')
            app.post_message(events.Paste(payload))
            for _ in range(60):
                await pilot.pause()
            ok = ed.text == 'X' + payload
            print('  %-24s pasted=%-12r -> buffer=%-14r %s'
                  % (label, payload, ed.text, 'OK' if ok else 'DISCARDED'))
asyncio.run(main())
"

echo
echo "############ D2 / U2 (major) — the message states the outcome, not a diagnosis ############"
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
    ('clipboard genuinely empty', ClipboardContents()),
    ('text-only clipboard', ClipboardContents()),
    ('image found, unattachable', ClipboardContents(image=ClipboardImage(b'\x00not an image', 'image/png'))),
    ('over SSH, image IS present', ClipboardContents(refused_remote=True)),
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
            ed = app.query_one(Editor); ed.focus(); await pilot.pause()
            app.post_message(events.Paste(''))
            for _ in range(80):
                await pilot.pause()
                if app.query_one(Toast).message:
                    break
            print('  %-28s -> %r' % (label, app.query_one(Toast).message))
asyncio.run(main())
"

echo
echo "############ D3 (major) — a held notice cannot be replayed after it is false ############"
.venv/bin/python -c "
import asyncio, io, sys, time
sys.path.insert(0, '$REPO')
from PIL import Image
from textual import events
from local_operator.clipboard import ClipboardContents, ClipboardImage
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets import editor as em
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.toast import Toast, TOAST_FAILURE_MS
from tests.unit.tui.test_app_pilot import FakeSession, _factory

def png(w=1568, h=200):
    b = io.BytesIO(); Image.new('RGB', (w, h), (30, 30, 40)).save(b, 'PNG'); return b.getvalue()

state = {'image': None}
em.read_clipboard = lambda *a, **k: ClipboardContents(image=state['image'])

async def main():
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
        print('  1. MCP failure claims the slot  showing=%r' % t.message)
        app.post_message(events.Paste(''))
        for _ in range(80):
            await pilot.pause()
        print('  2. empty paste -> notice DEFERRED')
        print('       showing =%r' % t.message)
        print('       deferred=%r' % (None if t._deferred is None else t._deferred[0],))
        state['image'] = ClipboardImage(png(), 'image/png')
        app.post_message(events.Paste(''))
        for _ in range(200):
            await pilot.pause()
            if ed.referenced_images():
                break
        print('  3. screenshot pasted -> ATTACHES')
        print('       editor  =%r' % ed.text)
        print('       deferred=%r   <- retired by the attach' % (None if t._deferred is None else t._deferred[0],))
        t.dismiss_toast()
        for _ in range(8):
            await pilot.pause()
        print('  4. MCP toast expires')
        print('       showing =%r  display=%s   <- nothing stale surfaces' % (t.message, t.display))
asyncio.run(main())
"

echo
echo "############ F2 (major) — the whole read is bounded, not each subprocess ############"
.venv/bin/python -c "
import sys, time
sys.path.insert(0, '$REPO')
import local_operator.clipboard as cb

class Hanging:
    def __init__(self, *a, **k):
        self.stdout, self.stdin, self.returncode = None, None, 1
        self._t = k.get('__t')
    def __enter__(self): return self
    def __exit__(self, *e): return None
    def wait(self, timeout=None):
        budgets.append(timeout)
        time.sleep(max(timeout or 0, 0))
        raise cb.subprocess.TimeoutExpired('x', timeout or 0)
    def poll(self): return None
    def kill(self): pass

budgets = []
cb.subprocess.Popen = lambda argv, **k: Hanging()
cb.shutil.which = lambda name: '/usr/bin/' + name
t0 = time.perf_counter()
cb.read_clipboard(platform='linux', env={'DISPLAY': ':0'})
print('  X11, every xclip hangs, 4 MIME types tried')
print('    total blocking time : %.2fs (cap is %.1fs)' % (time.perf_counter() - t0, cb.CLIPBOARD_TIMEOUT_S))
print('    budget per spawn    : %s' % ['%.2f' % b for b in budgets if b is not None])
print('    non-increasing      : %s  <- each call gets the REMAINING budget' % (budgets == sorted(budgets, reverse=True)))
"
