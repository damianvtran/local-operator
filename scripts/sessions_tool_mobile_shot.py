"""Capture the phone's sessions-tool surfaces out of headless Chrome over CDP.

Run:  PYTHONPATH=. .venv/bin/python scripts/sessions_tool_mobile_shot.py OUTDIR

Starts ``scripts/sessions_tool_mobile_fixture.py`` (the REAL bundle, summaries
built by the daemon's own merge and the projection's own ``_summarize_args``),
then photographs two surfaces at the two phone viewports a screenshot is
actually judged on:

* the session LIST — where the agent-opened chip lands;
* the session VIEW of the seeded tool ledger — where the tool-row summary
  renders (the row is state glyph + monospace name + summary).

``Chrome``/``Page`` are reused from ``scripts/mobile_overflow_capture.py``: one
throwaway ``--headless=new`` browser per run, its own profile,
``--use-mock-keychain``, and a teardown that asserts no helper survived — all
of which that module already got right, and a second copy here would be a
second thing to keep right.

WHAT THE NUMBERS BESIDE THE FRAME ARE FOR. The chip is ``shrink-0``, so the
row's title pays for it; the geometry dump reports each row's title edges and
its right-cluster spans, so "the mark costs the title a known number of pixels"
is a measured statement rather than a claim about a PNG — the invariant the
delegated-work chip's own round established for this slot.

Run the SAME rig against a base checkout for the before frame; the fixture's
summaries and chips are then produced by that tree's own code by construction.
"""

from __future__ import annotations

import json
import secrets
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from scripts.mobile_overflow_capture import Chrome, Page

VIEWPORTS = [(390, 844), (360, 640)]

#: The row whose session view carries the tool ledger, and the label the tap
#: matches it by (the fixture's own title).
TOOLS_ROW_LABEL = "Sessions tool run"

#: Every list row's title geometry and its right-cluster words: the chip's text
#: and box, so before (no chip) and after (chip) are comparable numbers, not
#: just pixels.
GEOMETRY_JS = r"""
(() => {
  const rows = [...document.querySelectorAll('button')].map((b) => {
    const title = b.querySelector('.truncate.font-medium') || b.querySelector('.truncate');
    return {
      text: (b.textContent || '').trim().slice(0, 90),
      titleLeft: title ? Math.round(title.getBoundingClientRect().left) : null,
      titleRight: title ? Math.round(title.getBoundingClientRect().right) : null,
      rowRight: Math.round(b.getBoundingClientRect().right),
      spans: [...b.querySelectorAll('span')].map((s) => (s.textContent || '').trim()),
    };
  });
  return JSON.stringify({viewport: [innerWidth, innerHeight], rows});
})()
"""

#: The view's tool rows, read off the button text so the frame's own words are
#: recorded rather than inferred.
VIEW_JS = r"""
(() => {
  const buttons = [...document.querySelectorAll('button')]
    .map((b) => (b.textContent || '').trim())
    .filter((t) => t.includes('sessions'));
  return JSON.stringify({toolRows: buttons});
})()
"""


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _login(page: Page, base: str, password: str) -> None:
    page.goto(f"{base}/login")
    page.js(
        "(() => { const f = document.querySelector('form');"
        f" f.password.value = {password!r}; f.submit(); return true; }})()"
    )
    time.sleep(2.0)


def _tap_js(label: str) -> str:
    """Tap the list row for ``label`` — the REAL gesture, not a route assignment.

    The route is ``#/s/<id>`` and the app owns the hash, so a ``goto`` at a URL
    the router does not spell is a same-document no-op that would leave the
    capture sitting on the list. A click is also what a reader does.
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


def main() -> None:
    outdir = Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    # A THROWAWAY PASSWORD, GENERATED PER RUN AND NEVER PRINTED (the shared rule
    # in ``mobile_overflow_capture``: a reusable constant is a credential).
    password = secrets.token_urlsafe(16)
    fixture = subprocess.Popen(
        [sys.executable, "scripts/sessions_tool_mobile_fixture.py", str(port), password],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    chrome = Chrome()
    report: dict[str, Any] = {}
    try:
        for _ in range(40):
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
            _login(page, base, password)
            page.goto(f"{base}/#/")
            time.sleep(2.0)
            name = f"list-{width}x{height}"
            page.shot(outdir / f"{name}.png")
            report[name] = json.loads(page.js(GEOMETRY_JS))
            print(f"{name}: {json.dumps(report[name]['rows'], indent=1)}", flush=True)

            view = f"view-{width}x{height}"
            report[view] = {"tap": json.loads(page.js(_tap_js(TOOLS_ROW_LABEL)))}
            time.sleep(2.0)
            report[view].update(json.loads(page.js(VIEW_JS)))
            page.shot(outdir / f"{view}.png")
            print(f"{view}: {json.dumps(report[view], indent=1)}", flush=True)
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
    (outdir / "sessions-mobile-geometry.json").write_text(json.dumps(report, indent=2))
    print("shots:", ", ".join(sorted(p.name for p in outdir.glob("*.png"))))


if __name__ == "__main__":
    main()
