#!/bin/sh
# Real macOS pasteboard evidence for issue #372, driven through the real
# OperatorApp (the one that loads the stylesheet), not a mocked backend.
#
# Reproduce with:  sh docs/evidence/clipboard-paste/macos_evidence.sh
set -u
REPO="${REPO:-/tmp/lop-clipboard-paste}"
FIX="${FIX:-/tmp/lop372}"
cd "$REPO"

drive() {
  .venv/bin/python -c "
import asyncio, sys, base64, time
sys.path.insert(0, '$REPO')
from textual import events
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.toast import Toast
from tests.unit.tui.test_app_pilot import FakeSession, _factory

async def main():
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and app._session is None:
            await pilot.pause()
        ed = app.query_one(Editor); ed.focus(); await pilot.pause()
        t0 = time.perf_counter()
        app.post_message(events.Paste(''))
        for _ in range(80):
            await pilot.pause()
            if ed.referenced_images():
                break
        elapsed = (time.perf_counter() - t0) * 1000
        print('  composer text :', repr(ed.text))
        if ed.referenced_images():
            img = ed.referenced_images()[0]
            raw = base64.b64decode(img.data)
            print('  attached      :', img.mime_type, len(raw), 'bytes, magic', raw[:8])
            print('  elapsed       : %.0f ms' % elapsed)
            ed.text = ed.text + 'what is in this image?'
            await pilot.press('enter')
            for _ in range(80):
                await pilot.pause()
                if app._session.prompts:
                    break
            if app._session.prompts:
                print('  prompt sent   :', repr(app._session.prompts[-1][:52]))
                print('  images sent   :', len(app._session.prompt_images[-1]))
        else:
            print('  attached      : nothing')
            t = app.query_one(Toast)
            print('  toast visible :', t.display)
            print('  toast message :', repr(t.message))
asyncio.run(main())
"
}

echo "############ 1. PNG on the pasteboard (native screenshot shape) ############"
osascript -e "set the clipboard to (read (POSIX file \"$FIX/shot.png\") as «class PNGf»)"
sleep 1
echo "  clipboard info: $(osascript -e 'clipboard info')"
echo "  pbpaste bytes : $(pbpaste | wc -c | tr -d ' ')  <- the terminal has no text to send"
drive

echo
echo "############ 2. TIFF only, no PNG flavor (Preview-style copy) ############"
osascript <<EOF >/dev/null
use framework "AppKit"
use scripting additions
set pb to current application's NSPasteboard's generalPasteboard()
pb's clearContents()
set d to current application's NSData's dataWithContentsOfFile:"$FIX/shot.tiff"
pb's setData:d forType:"public.tiff"
EOF
sleep 1
echo "  clipboard info: $(osascript -e 'clipboard info')"
drive

echo
echo "############ 3. Finder Cmd+C (public.file-url only) ############"
osascript -e "set the clipboard to (POSIX file \"$FIX/shot.png\")" >/dev/null
sleep 1
echo "  clipboard info: $(osascript -e 'clipboard info')"
echo "  pbpaste bytes : $(pbpaste | wc -c | tr -d ' ')"
drive

echo
echo "############ 4. NEGATIVE: text-only clipboard ############"
printf 'some ordinary text' | pbcopy
sleep 1
echo "  clipboard info: $(osascript -e 'clipboard info')"
drive

echo
echo "############ 5. NEGATIVE: SSH session, image still on the pasteboard ############"
osascript -e "set the clipboard to (read (POSIX file \"$FIX/shot.png\") as «class PNGf»)" >/dev/null
sleep 1
echo "  clipboard info: $(osascript -e 'clipboard info')"
SSH_CONNECTION='10.0.0.1 51234 10.0.0.2 22' drive
