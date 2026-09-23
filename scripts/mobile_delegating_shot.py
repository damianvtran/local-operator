"""Capture the phone's session list out of headless Chrome over CDP.

Run:  PYTHONPATH=. .venv/bin/python scripts/mobile_delegating_shot.py OUTDIR

Starts ``scripts/mobile_delegating_fixture.py`` (the REAL bundle, summaries built
by the daemon's own ``_merge_summaries``), then photographs the list at the two
phone viewports a screenshot is actually judged on. ``Chrome``/``Page`` are
reused from ``scripts/mobile_overflow_capture.py``: one throwaway
``--headless=new`` browser per run, its own profile, ``--use-mock-keychain``, and
a teardown that asserts no helper survived — all of which that module already
got right, and a second copy of it here would be a second thing to keep right.

WHAT THE NUMBERS BESIDE THE FRAME ARE FOR. The chip is ``shrink-0``, so the row's
title is what pays for a wide count. The geometry dump reports each row's title
start x and the chip's box, so "the count did not move the title" is a measured
statement rather than a claim about a PNG — the invariant the unread mark's own
round (D2) established for this slot.
"""

from __future__ import annotations

import json
import secrets
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from scripts.mobile_overflow_capture import Chrome, Page

VIEWPORTS = [(390, 844), (360, 640)]

GEOMETRY_JS = """
(() => {
  const rows = [...document.querySelectorAll('button')].map((b) => {
    const title = b.querySelector('.truncate.font-medium') || b.querySelector('.truncate');
    const slot = b.querySelector('.size-3');
    const chip = [...b.querySelectorAll('span')].find((s) =>
      /subagent|queued/.test(s.textContent||''));
    return {
      text: (b.textContent||'').trim().slice(0, 60),
      titleLeft: title ? Math.round(title.getBoundingClientRect().left) : null,
      titleRight: title ? Math.round(title.getBoundingClientRect().right) : null,
      markLeft: slot ? Math.round(slot.getBoundingClientRect().left) : null,
      markWidth: slot ? Math.round(slot.getBoundingClientRect().width) : null,
      markChildren: slot ? slot.children.length : null,
      chip: chip ? (chip.textContent||'').trim() : null,
      chipLeft: chip ? Math.round(chip.getBoundingClientRect().left) : null,
      chipRight: chip ? Math.round(chip.getBoundingClientRect().right) : null,
      rowRight: Math.round(b.getBoundingClientRect().right),
    };
  });
  return JSON.stringify({viewport: [innerWidth, innerHeight], rows});
})()
"""


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def main() -> None:
    outdir = Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    # A THROWAWAY PASSWORD, GENERATED PER RUN AND NEVER PRINTED. The fixture daemon
    # is bound to loopback and lived for one capture, so a shared constant bought
    # nothing and cost something: a credential-shaped literal sitting in a script
    # that a reader (or an agent) has to open to understand the harness. The value
    # is passed to the fixture as an argument and typed into the login form below,
    # so it never appears in this file, in the fixture, or in any output.
    password = secrets.token_urlsafe(16)
    fixture = subprocess.Popen(
        [sys.executable, "scripts/mobile_delegating_fixture.py", str(port), password],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    chrome = Chrome()
    report: dict[str, Any] = {}
    try:
        # Wait for the fixture to answer before pointing a browser at it: an
        # immediate connect gives a refused socket, which reads like a broken
        # bundle rather than a race.
        import urllib.request

        deadline = time.time() + 45
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"{base}/login", timeout=2):
                    break
            except Exception:  # noqa: BLE001  -- not up yet
                time.sleep(0.4)
        else:
            raise RuntimeError("fixture never served /login")

        page = Page(chrome.target_ws())
        for width, height in VIEWPORTS:
            page.metrics(width, height)
            page.goto(f"{base}/login")
            page.js(
                "(() => { const f = document.querySelector('form');"
                f" f.password.value = {password!r}; f.submit(); return true; }})()"
            )
            time.sleep(2.0)
            page.goto(f"{base}/#/")
            time.sleep(2.0)
            name = f"delegating-{width}x{height}"
            page.shot(outdir / f"{name}.png")
            report[name] = json.loads(page.js(GEOMETRY_JS))
            print(f"{name}: {json.dumps(report[name]['rows'], indent=1)}", flush=True)

            # THE TAP FRAMES. Tap a row, read the header the view lands on, go back
            # to the list for the next one. The back step is a hash route the app
            # itself spells (``#/``), not a bare origin — see ``_tap_js``.
            for label, prefix in TAP_SESSIONS.items():
                view = f"{prefix}-view-{width}x{height}"
                report[view] = {"tap": json.loads(page.js(_tap_js(label)))}
                time.sleep(2.0)
                report[view].update(json.loads(page.js(ROSTER_JS)))
                page.shot(outdir / f"{view}.png")
                print(f"{view}: {report[view]}", flush=True)
                # The child rows, for the two sessions whose children are NOT all
                # running: that is where the parked child's own treatment lives.
                if prefix in ("mixed", "parked"):
                    expanded = f"{prefix}-roster-{width}x{height}"
                    report[expanded] = {"tap": json.loads(page.js(EXPAND_JS))}
                    time.sleep(1.0)
                    report[expanded].update(json.loads(page.js(GLYPH_COUNT_JS)))
                    page.shot(outdir / f"{expanded}.png")
                    print(f"{expanded}: {report[expanded]}", flush=True)
                page.goto(f"{base}/#/")
                time.sleep(1.5)
        page.close()
    finally:
        chrome.close()
        fixture.terminate()
        try:
            fixture.wait(timeout=10)
        except subprocess.TimeoutExpired:
            fixture.kill()
    (outdir / "delegating-mobile-geometry.json").write_text(json.dumps(report, indent=2))
    for width, height in VIEWPORTS:
        left = outdir / f"delegating-{width}x{height}.png"
        for prefix in TAP_SESSIONS.values():
            right = outdir / f"{prefix}-view-{width}x{height}.png"
            if left.exists() and right.exists():
                _compose(left, right, outdir / f"{prefix}-list-vs-view-{width}x{height}.png")
                print(f"composed {prefix}-list-vs-view-{width}x{height}.png", flush=True)
    print("shots:", ", ".join(f"{key}.png" for key in report))


def _compose(left: Path, right: Path, out: Path) -> None:
    """One image carrying both numbers — the list chip and the view's roster header.

    Two frames would need a reader to hold the first in their head while looking at
    the second, and this finding is exactly about a pair of numbers being compared.
    Pillow is already a dependency of this repo's capture path (``visual_capture``).
    """
    from PIL import Image

    a = Image.open(left)
    b = Image.open(right)
    canvas = Image.new("RGB", (a.width + b.width, max(a.height, b.height)), (10, 10, 10))
    canvas.paste(a, (0, 0))
    canvas.paste(b, (a.width, 0))
    canvas.save(out)


#: The sessions whose SESSION VIEW the rig taps into, keyed by the prefix used for
#: their frames: ``{row label the list shows: frame prefix}``. Three shapes, because
#: the view is where the two counts have to agree with the list and with each
#: other (UX round 3): a parent with one child spending and one waiting, a parent
#: with nothing running at all, and a parent whose children have children.
TAP_SESSIONS: dict[str, str] = {
    "Parent with one running, one waiting": "mixed",
    "Parent with everything parked": "parked",
    "Parent with nested descendants": "nested",
}


def _tap_js(label: str) -> str:
    """Tap the list row for ``label`` — the REAL gesture, not a route assignment.

    The route is ``#/s/<id>`` and the app owns the hash, so a ``goto`` at a URL the
    router does not spell (``#/session/<id>``) is a same-document no-op that leaves
    the capture sitting on the list (measured, design round 2). A click is also
    what the finding is about: what ONE TAP changes.
    """
    return (
        "(() => {"
        " const nodes = Array.from("
        'document.querySelectorAll(\'a,button,[role="link"],[role="button"]\'));'
        f" const hit = nodes.find((n) => (n.textContent || '').includes({label!r}));"
        " if (!hit) return JSON.stringify({ tapped: false });"
        " hit.click();"
        " return JSON.stringify({ tapped: true, tag: hit.tagName });"
        "})()"
    )


#: The roster header's own numbers: the fraction span (``subagents-panel.tsx``
#: renders ``{running}/{direct.length} running``) and the parked addend, read from
#: the view's text so the frame's own words are recorded rather than inferred.
ROSTER_JS = r"""
(() => {
  const spans = Array.from(document.querySelectorAll('span'));
  const hit = spans.find((s) => /^\d+\/\d+ running$/.test((s.textContent || '').trim()));
  const header = hit && hit.parentElement ? hit.parentElement.textContent.trim() : null;
  const body = document.body.textContent || '';
  const waiting = /(\d+) queued/.exec(body);
  return JSON.stringify({
    roster: hit ? hit.textContent.trim() : null,
    header,
    queued: waiting ? waiting[0] : null,
  });
})()
"""


#: Expand the roster whose header was just read, so the child rows themselves are in
#: the frame: the waiting child's own treatment is half of UX round 3's finding (it
#: must not be the spinner), and a collapsed panel never draws it.
EXPAND_JS = r"""
(() => {
  const hit = Array.from(document.querySelectorAll('span')).find((s) =>
    /^\d+\/\d+ running$/.test((s.textContent || '').trim()));
  const button = hit ? hit.closest('button') : null;
  if (!button) return JSON.stringify({ expanded: false });
  button.click();
  return JSON.stringify({ expanded: true });
})()
"""

#: The two status glyphs a parked child must not be confused with, counted off the
#: expanded roster's own text.
GLYPH_COUNT_JS = r"""
(() => {
  const text = document.body.textContent || '';
  return JSON.stringify({
    waiting: (text.match(/…/g) || []).length,
    spinning: (text.match(/⟳/g) || []).length,
  });
})()
"""


if __name__ == "__main__":
    main()
