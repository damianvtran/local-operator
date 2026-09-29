"""Capture the session screen's spend/context row out of headless Chrome over CDP.

Run:  PYTHONPATH=. .venv/bin/python scripts/mobile_glance_shot.py OUTDIR

Starts ``scripts/mobile_glance_fixture.py`` (the REAL bundle over synthetic
projections carrying the spend/context block), taps into each of the three
session states, and photographs the session screen at 390x844 @ dpr 2 — the
phone viewport phase 1 is judged on — plus a geometry dump: the status row's box
and each cell's box and text, the transcript scroller's box, and the composer's
box. On a base-commit bundle the row does not exist, and the dump records
``status: null`` with a lower transcript top — so "the row costs one text line
and overlaps nothing" is a measured statement (row height, transcript-top delta)
rather than a claim about a PNG.

``Chrome``/``Page`` are reused from ``scripts/mobile_overflow_capture.py``: one
throwaway ``--headless=new`` browser per run, its own profile,
``--use-mock-keychain``, and a teardown that asserts no helper survived — all of
which that module already got right, and a second copy here would be a second
thing to keep right.

The taps are the REAL gesture: the app owns the hash route, and a ``goto`` at a
URL the router does not spell is a same-document no-op that leaves the capture
sitting on the list (measured, design round 2 on the delegating rig) — so a row
is clicked, exactly as a thumb would.
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

VIEWPORT = (390, 844)

#: The sessions to photograph, keyed by the row label the list shows. The value
#: is the prefix used for the frame's file name.
TAP_SESSIONS: dict[str, str] = {
    "Spend + context glance": "glance",
    "Floor and estimate": "glance-floor",
    "Unpriceable reading": "glance-unknown",
}

GEOMETRY_JS = r"""
(() => {
  const box = (el) => {
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return {
      top: Math.round(r.top), bottom: Math.round(r.bottom),
      left: Math.round(r.left), right: Math.round(r.right),
      width: Math.round(r.width), height: Math.round(r.height),
    };
  };
  const text = (el) => (el && el.textContent ? el.textContent.trim() : null);
  const row = document.querySelector('[data-testid="session-status"]');
  const spend = document.querySelector('[data-testid="session-status-spend"]');
  const context = document.querySelector('[data-testid="session-status-context"]');
  const scroller = document.querySelector('.lo-scroll');
  const composer = document.querySelector('textarea');
  return JSON.stringify({
    viewport: [window.innerWidth, window.innerHeight],
    status: box(row),
    spend: { box: box(spend), text: text(spend) },
    context: { box: box(context), text: text(context) },
    transcript: box(scroller),
    composer: box(composer),
    bodyScrollH: document.body.scrollHeight,
  });
})()
"""


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _tap_js(label: str) -> str:
    """Tap the list row whose text the label names — the REAL gesture."""
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
    # A THROWAWAY PASSWORD, GENERATED PER RUN AND NEVER PRINTED — the same
    # contract the sibling rigs document: passed to the fixture as an argument
    # and typed into the login form below, so it appears in no file and in no
    # output.
    password = secrets.token_urlsafe(16)
    fixture = subprocess.Popen(
        [sys.executable, "scripts/mobile_glance_fixture.py", str(port), password],
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
        width, height = VIEWPORT
        page.metrics(width, height)
        page.goto(f"{base}/login")
        page.js(
            "(() => { const f = document.querySelector('form');"
            f" f.password.value = {password!r}; f.submit(); return true; }})()"
        )
        time.sleep(2.0)
        page.goto(f"{base}/#/")
        time.sleep(2.0)
        for label, prefix in TAP_SESSIONS.items():
            frame = f"{prefix}-{width}x{height}"
            report[frame] = {"tap": json.loads(page.js(_tap_js(label)))}
            time.sleep(2.0)
            report[frame].update(json.loads(page.js(GEOMETRY_JS)))
            page.shot(outdir / f"{frame}.png")
            print(f"{frame}: {json.dumps(report[frame], indent=1)}", flush=True)
            # Back to the list for the next session — the route the app spells.
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
    (outdir / "glance-mobile-geometry.json").write_text(json.dumps(report, indent=2))
    print("shots:", ", ".join(f"{key}.png" for key in report))


if __name__ == "__main__":
    main()
