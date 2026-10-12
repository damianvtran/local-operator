"""Capture the quiet-group bar states out of headless Chrome over CDP.

Run:  PYTHONPATH=. .venv/bin/python scripts/mobile_quietgroups_shot.py OUTDIR

Starts ``scripts/mobile_quietgroups_fixture.py`` (the real daemon over synthetic
projections carrying the bar states), taps into each session the way a thumb
does, and photographs the bar at 390x844 and 360x780 at dpr 2 — the two phone
viewports this surface is judged on — plus a geometry dump per frame: the bar's
box, its text and ``aria-expanded``, the expansion's receipt-row count, the
transcript scroller's box and scroll offsets, and the body's scroll width. So
"the bar is a real target", "the count is a minimum only where it claims to
be" and "nothing overflows sideways" are measured statements rather than
claims about a PNG.

TWO EXTRA GESTURES, both REAL finger drags — never a ``scrollTop`` assignment,
so a capture a thumb could not reproduce cannot pass: on the resolved session
the script drags to the top, which trips the transcript's auto-load, and waits
for the empty history answer to settle ``hasMore`` false before photographing
``Peer messages · 2``; on the expanded session it presses the bar itself and
photographs the receipts with their tool rows.

THE FIRST RUN'S LESSON, kept because it cost a frame: the very first tap raced
the list's own render after login and photographed the bare list. Every tap now
retries a miss (``_tap_row``) — a miss is almost always a NOT-YET, not a wrong
name.

``Chrome``/``Page`` are reused from ``scripts/mobile_overflow_capture.py``:
one throwaway ``--headless=new`` browser per run, its own profile,
``--use-mock-keychain``, and a teardown that asserts no helper survived — all
of which that module already got right, and a second copy here would be a
second thing to keep right.
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

#: Both widths the bar is judged at — the other phone rigs' own pair.
VIEWPORTS = ((390, 844), (360, 780))

#: The sessions to photograph, keyed by the row label the list shows. The value
#: is the prefix used for the frame's file name. Order is the reading order of
#: the states. ``collapsed-resolved`` adds the drag-to-top gesture before its
#: frame (see the module doc); ``expanded`` adds the bar press.
TAP_SESSIONS: dict[str, str] = {
    "Quiet group collapsed": "collapsed-minimum",
    "Quiet group resolved": "collapsed-resolved",
    "Quiet group expanded": "expanded",
    "Quiet group mixed": "mixed",
    "Quiet group wake only": "wake-only",
    "Quiet group single": "single-receipt",
}

#: The quiet-group bar's family words, longest first so ``Peer messages`` wins
#: over ``Messages`` in the alternation — the same copy table
#: ``lib/quiet-groups`` ships (this matcher must not be a second spelling).
_BAR_RE = r"/^(Peer messages|Wake messages|Monitor messages|Job results|Messages)/"

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
  const buttons = Array.from(document.querySelectorAll('button'));
  // The bar's button carries the chevron glyph INSIDE it (disclosure.tsx), so
  // an anchored match on raw textContent would never see the family word —
  // strip the two glyphs (the only ▸/▾ the app ships) before testing.
  const buttonText = (b) => (b.textContent || '').replace(/[▸▾]/g, '').trim();
  const bar = buttons.find((b) => __BAR_RE__.test(buttonText(b))) || null;
  const scroller = document.querySelector('.lo-scroll');
  // The expansion's rows, counted by the anchor every transcript row carries:
  // the Disclosure mounts children only while open, so a closed bar cannot
  // report receipts it is not showing. The bar's parent is the Disclosure
  // root (the button is its direct child, disclosure.tsx).
  const receipts = bar ? bar.parentElement.querySelectorAll('[data-completion-anchor]').length : 0;
  return JSON.stringify({
    bar: bar
      ? { box: box(bar), text: text(bar), expanded: bar.getAttribute('aria-expanded') }
      : null,
    receipts,
    scroller: box(scroller),
    scrollTop: scroller ? Math.round(scroller.scrollTop) : null,
    scrollMax: scroller ? Math.round(scroller.scrollHeight - scroller.clientHeight) : null,
    bodyScrollW: document.body.scrollWidth,
    innerW: window.innerWidth,
  });
})()
""".replace("__BAR_RE__", _BAR_RE)


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


def _press_bar_js() -> str:
    """Press the bar itself — the real tap, and the only way the expansion is
    produced."""
    return (
        "(() => {"
        " const hit = Array.from(document.querySelectorAll('button'))"
        f".find((b) => {_BAR_RE}.test((b.textContent || '').replace(/[▸▾]/g, '').trim()));"
        " if (!hit) return JSON.stringify({ pressed: false });"
        " hit.click();"
        " return JSON.stringify({ pressed: true, text: hit.textContent.trim() });"
        "})()"
    )


def _scroll_state_js() -> str:
    """The transcript scroller's offsets — the drag loop's own instrument, so
    'we are at the top' is read, not assumed."""
    return (
        "(() => {"
        " const el = document.querySelector('.lo-scroll');"
        " if (!el) return JSON.stringify(null);"
        " const r = el.getBoundingClientRect();"
        " return JSON.stringify({"
        "   top: Math.round(el.scrollTop),"
        "   max: Math.round(el.scrollHeight - el.clientHeight),"
        "   cx: Math.round(r.left + r.width / 2),"
        "   cy: Math.round(r.top + r.height / 2),"
        " });"
        "})()"
    )


def _drag_to_top(page: Page, vw: int, vh: int) -> dict[str, Any]:
    """Drag to the transcript's top with REAL finger drags, bounded.

    The finger starts high in the viewport and travels DOWN (content moves
    down = earlier rows revealed), staying on-screen the whole way. The loop
    stops at the top, when the scroller proves immovable (two rounds without
    progress — a fitting list, where the auto-load cannot fire), or after the
    round bound, and returns its own final reading either way: a frame whose
    dump says ``top: 480`` is a frame the desk can see was never at the top.
    """
    state: dict[str, Any] = {"top": None, "max": None}
    still = 0
    for _ in range(24):
        raw = page.js(_scroll_state_js())
        state = json.loads(raw) if raw else {"top": None, "max": None}
        if state.get("top") is None or state["top"] <= 8 or (state.get("max") or 0) <= 0:
            break
        cx = int(state.get("cx") or vw // 2)
        # The finger must END on-screen: start high enough in the scroller
        # that the whole 560px downward travel fits inside the viewport.
        cy = int(state.get("cy") or vh // 2)
        page.swipe(cx, max(100, min(cy, vh - 600)), -560)
        after = json.loads(page.js(_scroll_state_js()) or "null")
        if after and after.get("top", 0) >= state.get("top", 0):
            still += 1
            if still >= 2:
                state = after
                break
        else:
            still = 0
        state = after or state
    return state


def _tap_row(page: Page, label: str, base: str, tries: int = 4) -> dict[str, Any]:
    """Tap the row, waiting out the list's own first render.

    The first capture run photographed the bare list for its very first frame:
    the tap raced the list render after login, so the row it wanted did not
    exist yet. A miss is almost always a NOT-YET, not a wrong name — so a miss
    re-enters the list and tries again, bounded.
    """
    result: dict[str, Any] = {"tapped": False}
    for attempt in range(tries):
        if attempt:
            page.goto(f"{base}/#/")
            time.sleep(1.5)
        result = json.loads(page.js(_tap_js(label)))
        if result.get("tapped"):
            return result
        time.sleep(1.0)
    return result


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
        [sys.executable, "scripts/mobile_quietgroups_fixture.py", str(port), password],
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
                report[frame] = {"tap": _tap_row(page, label, base)}
                time.sleep(2.0)
                if prefix == "collapsed-resolved":
                    # The real drag; its final scroll reading proves whether
                    # the frame actually sits at the top.
                    report[frame]["scroll"] = _drag_to_top(page, vw, vh)
                    # The auto-load's empty answer settles hasMore; give the
                    # re-render a beat longer than a tap's.
                    time.sleep(1.6)
                if prefix == "expanded":
                    report[frame]["press"] = json.loads(page.js(_press_bar_js()))
                    time.sleep(0.8)
                report[frame].update(json.loads(page.js(GEOMETRY_JS)))
                page.shot(outdir / f"{frame}.png")
                print(f"{frame}: {json.dumps(report[frame], indent=1)}", flush=True)
                # Back to the list for the next session — the route the app spells.
                page.goto(f"{base}/#/")
                time.sleep(1.5)
        page.close()
    finally:
        # TEARDOWN IS BEST-EFFORT PER RESOURCE, and the first failure is
        # re-raised after both are attempted — the same contract the imagegen
        # rig carries (review round 1, F1): the Chrome teardown fails CLOSED
        # (it raises when a helper outlived the run) and a bare `chrome.close()`
        # line would skip the fixture's teardown on exactly that alert path,
        # stranding a live daemon. Each resource is reclaimed whatever the
        # other did.
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
    (outdir / "quietgroups-mobile-geometry.json").write_text(json.dumps(report, indent=2))
    print("shots:", ", ".join(f"{key}.png" for key in report))


if __name__ == "__main__":
    main()
