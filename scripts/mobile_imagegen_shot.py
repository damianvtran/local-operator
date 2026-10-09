"""Capture the image-generation card out of headless Chrome over CDP.

Run:  PYTHONPATH=. .venv/bin/python scripts/mobile_imagegen_shot.py OUTDIR

Starts ``scripts/mobile_imagegen_fixture.py`` (the REAL bundle over synthetic
projections carrying every card state), taps into each session the way a thumb
does, and photographs the card at 390x844 and 360x780 at dpr 2 — the two phone
viewports this surface is judged on — plus a geometry dump per frame: the
card's box, the row button's, the tile's, the progress bar's (with
``aria-valuenow``), the cancel control's (against the 44px touch floor), the
state lines' text, the transcript scroller's box and the body's scroll width.
So "the control is a 44px target" and "nothing overflows sideways" are
measured statements rather than claims about a PNG.

ONE EXTRA GESTURE: on the running session the script presses the card's own
Cancel and photographs the ``cancelling…`` hold — the state the frozen
contract says must never be skipped optimistically — and takes a second
running frame ~450 ms later so the tile's shimmer is evidenced as MOTION (the
two frames differ) rather than asserted from one still.

``Chrome``/``Page`` are reused from ``scripts/mobile_overflow_capture.py``:
one throwaway ``--headless=new`` browser per run, its own profile,
``--use-mock-keychain``, and a teardown that asserts no helper survived — all
of which that module already got right, and a second copy here would be a
second thing to keep right. The taps are the REAL gesture for the same reason
the sibling rigs give: the app owns its routes, and a click is what a thumb
does.
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

#: Both widths the card is judged at — the glance rig's own pair.
VIEWPORTS = ((390, 844), (360, 780))

#: The sessions to photograph, keyed by the row label the list shows. The value
#: is the prefix used for the frame's file name. Order is the reading order of
#: the state machine. The running row ALSO yields the press-driven
#: ``cancelling-<vw>`` frame below; ``cancelling-wire`` is the feed-driven hold
#: (``stage: "cancelling"`` with no click) and ``mid-walk`` the
#: ``stage: None`` failure beat — the two canonical inputs added with the wire
#: freeze, photographed rather than argued from the adapter tests alone.
TAP_SESSIONS: dict[str, str] = {
    "Image gen queued": "queued",
    "Image gen queue position": "queued-pos",
    "Image gen running": "running",
    "Image gen cancelling": "cancelling-wire",
    "Image gen progress": "running-detail",
    "Image gen mid-walk failure": "mid-walk",
    "Image gen done": "done",
    "Image gen failed": "failed",
    "Image gen already finished": "finished",
    "Image gen interrupted": "cancelled",
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
  const card = document.querySelector('[data-testid="image-gen-card"]');
  const buttons = Array.from(document.querySelectorAll('button'));
  const byText = (label) =>
    buttons.find((b) => (b.textContent || '').trim() === label) || null;
  const tile = document.querySelector('.lo-gen-tile');
  const bar = document.querySelector('[role="progressbar"]');
  const scroller = document.querySelector('.lo-scroll');
  // Every <p> inside the card, in DOM order — the state lines each state
  // renders (queued / image ready / the failed sentence) — PLUS the
  // cancelling hold, which renders as a <span> and is collected by the test
  // id the card ships: a `p`-only query named the hold in this dump's
  // comment while being structurally unable to see it, which is exactly the
  // instrument that reports nothing as if it had checked (review round 1,
  // F3).
  const stateLines = card
    ? Array.from(card.querySelectorAll('p')).map((p) => text(p))
    : [];
  const hold = card ? card.querySelector('[data-testid="image-gen-hold"]') : null;
  if (hold) stateLines.push(text(hold));
  return JSON.stringify({
    card: box(card),
    row: box(card ? card.querySelector('button') : null),
    tile: box(tile),
    bar: bar
      ? {
          box: box(bar),
          valueNow: bar.getAttribute('aria-valuenow'),
          label: bar.getAttribute('aria-label'),
        }
      : null,
    cancel: box(byText('cancel')),
    restart: box(byText('restart')),
    stateLines,
    imageCount: card ? card.querySelectorAll('img').length : 0,
    scroller: box(scroller),
    bodyScrollW: document.body.scrollWidth,
    innerW: window.innerWidth,
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


def _click_button_js(label: str) -> str:
    """Press the in-card control whose text the label names — the REAL gesture."""
    return (
        "(() => {"
        " const hit = Array.from(document.querySelectorAll('button'))"
        f".find((n) => (n.textContent || '').trim() === {label!r});"
        " if (!hit) return JSON.stringify({ clicked: false });"
        " hit.click();"
        " return JSON.stringify({ clicked: true });"
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
        [sys.executable, "scripts/mobile_imagegen_fixture.py", str(port), password],
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
        page.metrics(*VIEWPORTS[0])
        page.goto(f"{base}/login")
        page.js(
            "(() => { const f = document.querySelector('form');"
            f" f.password.value = {password!r}; f.submit(); return true; }})()"
        )
        time.sleep(2.0)
        for vw, vh in VIEWPORTS:
            page.metrics(vw, vh)
            page.goto(f"{base}/#/")
            time.sleep(1.5)
            for label, prefix in TAP_SESSIONS.items():
                frame = f"{prefix}-{vw}x{vh}"
                report[frame] = {"tap": json.loads(page.js(_tap_js(label)))}
                time.sleep(2.0)
                report[frame].update(json.loads(page.js(GEOMETRY_JS)))
                first = outdir / f"{frame}.png"
                page.shot(first)
                if prefix == "running":
                    # Frame B exists to show the tile MOVES: a shimmer is
                    # motion, and one still can only show a phase. The flag
                    # below is the WHOLE-frame comparison — any animation in
                    # the viewport moves it — so the tile's own region is
                    # pixel-checked separately in the round's evidence notes
                    # (crop the tile box from the two frames and count changed
                    # pixels against a control region).
                    time.sleep(0.45)
                    second = outdir / f"{frame}-b.png"
                    page.shot(second)
                    report[frame]["frames_differ"] = first.read_bytes() != second.read_bytes()
                    # The cancel press, held in the fixture (no confirmation
                    # ever lands), photographed as the state under test.
                    hold = f"cancelling-{vw}x{vh}"
                    report[hold] = {"click": json.loads(page.js(_click_button_js("cancel")))}
                    time.sleep(0.8)
                    report[hold].update(json.loads(page.js(GEOMETRY_JS)))
                    page.shot(outdir / f"{hold}.png")
                    print(f"{hold}: {json.dumps(report[hold], indent=1)}", flush=True)
                print(f"{frame}: {json.dumps(report[frame], indent=1)}", flush=True)
                # Back to the list for the next session — the route the app spells.
                page.goto(f"{base}/#/")
                time.sleep(1.5)
        page.close()
    finally:
        # TEARDOWN IS BEST-EFFORT PER RESOURCE, and the first failure is
        # re-raised after both are attempted (review round 1, F1). The Chrome
        # teardown fails CLOSED — it raises when a helper outlived the run —
        # and a bare `chrome.close()` line skipped the fixture's own teardown
        # below it on exactly that alert path, stranding a live daemon
        # (loopback-only and isolated, but still a process nobody collects).
        # Each resource is reclaimed whatever the other did.
        failure: Exception | None = None
        try:
            chrome.close()
        except Exception as exc:  # noqa: BLE001 — re-raised after both claims
            failure = exc
        try:
            fixture.terminate()
            try:
                fixture.wait(timeout=10)
            except subprocess.TimeoutExpired:
                fixture.kill()
        except Exception as exc:  # noqa: BLE001
            if failure is None:
                failure = exc
            else:
                print(f"teardown: second failure: {exc!r}", file=sys.stderr)
        if failure is not None:
            raise failure
    (outdir / "imagegen-mobile-geometry.json").write_text(json.dumps(report, indent=2))
    print("shots:", ", ".join(f"{key}.png" for key in report))


if __name__ == "__main__":
    main()
