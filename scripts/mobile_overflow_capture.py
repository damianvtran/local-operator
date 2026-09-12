"""Capture the fixture's phone surfaces out of headless Chrome over CDP.

Run:  .venv/bin/python scripts/mobile_overflow_capture.py <port> <outdir> <label>

Follows AGENTS.md §6 exactly and for the reasons recorded there: a unique
``mktemp`` profile under /tmp (never the operator's), ``--remote-debugging-port=0``
with the real port read from ``DevToolsActivePort`` (a fixed port silently drives
another session's Chrome), ``--headless=new`` (a headful window steals the
operator's focus — the 2026-09-07 incident), the viewport set by
``Emulation.setDeviceMetricsOverride`` rather than ``--window-size`` (which clamps
at a 500px floor and yields a dimension nobody set), ``start_new_session=True`` so
the survivors of a killed harness are addressable as a process group, and an
``atexit`` sweep that ASSERTS zero leftovers rather than trusting the terminate.

Each capture also records the geometry behind the frame (§4): the scroll extent
and client height of the ask card's scroller and of the layout column, so the
stills show the symptom and the numbers show the cause.

**Every scroll here is a real touch gesture**, and that is load-bearing rather
than fastidious. An earlier version of this script reached the tail of a long
list by assigning ``el.scrollTop = el.scrollHeight``, which a script may do on
any element — including one inside ``overflow: hidden`` — and a finger may not.
It therefore photographed regions as reachable that no user could reach, and
two review rounds paid for it: a design pass was masked by exactly this and had
to be redone, and a UX round wrote its own driver rather than trust this one.
The harness now drives ``Input.dispatchTouchEvent`` drags, and after the
gestures are exhausted it ASSERTS that nothing further moves by assignment —
exiting non-zero if anything does. A capture that cannot be reproduced by a
thumb is not evidence.
"""

from __future__ import annotations

import atexit
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

from websockets.sync.client import connect

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
VIEWPORTS = [(390, 844), (360, 780)]
PASSWORD = "overflow-demo"


class Chrome:
    """A throwaway headless Chrome owned by this process group."""

    def __init__(self) -> None:
        self.profile = Path(tempfile.mkdtemp(prefix="lo-harness-askcap."))
        self.proc = subprocess.Popen(
            [
                CHROME,
                "--headless=new",
                f"--user-data-dir={self.profile}",
                "--remote-debugging-port=0",
                "--use-mock-keychain",
                "--password-store=basic",
                "--no-first-run",
                "--no-default-browser-check",
                "about:blank",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        atexit.register(self.close)
        # Chrome writes DevToolsActivePort asynchronously: an immediate read gets
        # an empty file and the connect fails intermittently, which reads like a
        # flaky harness rather than a race.
        port_file = self.profile / "DevToolsActivePort"
        deadline = time.time() + 30
        while time.time() < deadline:
            if port_file.exists() and port_file.stat().st_size > 0:
                break
            time.sleep(0.2)
        else:
            raise RuntimeError("Chrome never wrote DevToolsActivePort")
        self.port = int(port_file.read_text().splitlines()[0])

    def target_ws(self) -> str:
        # Reuse the about:blank target Chrome already opened rather than asking
        # for a new one: `/json/new` requires PUT on current Chrome (152 answers
        # a GET with 405), and one page target is all this harness needs.
        deadline = time.time() + 20
        while time.time() < deadline:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/json/list") as r:
                targets = [t for t in json.load(r) if t.get("type") == "page"]
            if targets:
                return targets[0]["webSocketDebuggerUrl"]
            time.sleep(0.3)
        raise RuntimeError("Chrome exposed no page target")

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
        time.sleep(1.5)
        # Scoped to OUR unique profile prefix, never `pkill -f "Google Chrome"`:
        # SIGTERM to the browser pid alone left 0-5 helpers behind across
        # measured runs, so the count is asserted rather than assumed.
        subprocess.run(["pkill", "-f", str(self.profile)], capture_output=True)
        time.sleep(0.5)
        left = subprocess.run(
            ["pgrep", "-f", str(self.profile)], capture_output=True, text=True
        ).stdout.split()
        subprocess.run(["rm", "-rf", str(self.profile)], check=False)
        if left:
            raise RuntimeError(f"leaked {len(left)} Chrome processes: {left}")


class Page:
    def __init__(self, ws_url: str) -> None:
        self.ws = connect(ws_url, max_size=80 * 1024 * 1024)
        self.n = 0
        self.send("Page.enable")
        self.send("Runtime.enable")

    def send(self, method: str, **params: Any) -> dict[str, Any]:
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self.n:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    def metrics(self, width: int, height: int) -> None:
        self.send(
            "Emulation.setDeviceMetricsOverride",
            width=width,
            height=height,
            deviceScaleFactor=2,
            mobile=True,
        )

    def goto(self, url: str) -> None:
        self.send("Page.navigate", url=url)
        time.sleep(2.0)

    def js(self, expression: str) -> Any:
        result = self.send(
            "Runtime.evaluate", expression=expression, awaitPromise=True, returnByValue=True
        )
        return result.get("result", {}).get("value")

    def shot(self, path: Path) -> None:
        data = self.send("Page.captureScreenshot", format="png")["data"]
        import base64

        path.write_bytes(base64.b64decode(data))

    def swipe(self, x: int, y: int, dy: int, steps: int = 12) -> None:
        """One finger drag through REAL touch events.

        The distinction this method exists for: a script may assign
        ``el.scrollTop`` on any element, including one inside
        ``overflow: hidden``, and a finger may not. A harness that scrolls by
        assignment therefore reports a clipped, gesture-proof region as fine —
        which is exactly what happened here. Two review rounds hit it: the
        design round's first pass "recovered" a card that a thumb could not
        reach, and the UX round had to write its own driver to see the defect at
        all. So the harness dispatches what the phone dispatches: touchStart, a
        run of touchMove, touchEnd. Whatever does not move under this does not
        move for the user either.

        `dy` is the CONTENT displacement (positive scrolls down toward the
        tail), so the finger travels in the opposite direction.
        """
        self.send(
            "Input.dispatchTouchEvent",
            type="touchStart",
            touchPoints=[{"x": x, "y": y}],
        )
        for i in range(1, steps + 1):
            self.send(
                "Input.dispatchTouchEvent",
                type="touchMove",
                touchPoints=[{"x": x, "y": y - round(dy * i / steps)}],
            )
            time.sleep(0.012)
        self.send(
            "Input.dispatchTouchEvent",
            type="touchEnd",
            touchPoints=[],
        )
        # Let momentum and any smooth-scroll settle before the next measurement;
        # reading immediately catches the scroller mid-flight and understates it.
        time.sleep(0.35)

    def close(self) -> None:
        self.ws.close()


# The ask card is the block pinned above the composer, found by the test id the
# component ships. That id is deliberate shipped markup rather than a
# harness-only hook: `.border-accent`, the previous selector, is also applied by
# the composer on drag-over and by the new-session screen on selection, so a
# class query silently measures the wrong node once either state renders.
GEOMETRY_JS = """
(() => {
  const col = document.querySelector('div.h-dvh') || document.querySelector('[class*="h-dvh"]');
  // By test id, not `.border-accent`: composer.tsx (drag-over) and
  // new-session.tsx (selection) apply that class too.
  const card = document.querySelector('[data-testid="pending-card"]');
  const scrollers = [...document.querySelectorAll('.lo-scroll')].map((el) => ({
    cls: el.className.slice(0, 60),
    scrollH: el.scrollHeight,
    clientH: el.clientHeight,
    scrollable: el.scrollHeight > el.clientHeight + 1,
  }));
  // Scoped to the CARD: `.border-l-accent` is also the transcript's message
  // bubble, so an unscoped query measures a chat message as though it were an
  // ask option.
  const opts = card ? [...card.querySelectorAll('.border-l-accent')] : [];
  const last = opts[opts.length - 1];
  const lastRect = last ? last.getBoundingClientRect() : null;
  return JSON.stringify({
    viewport: [window.innerWidth, window.innerHeight],
    column: col ? { clientH: col.clientHeight, scrollH: col.scrollHeight } : null,
    card: card
      ? {
          rectTop: Math.round(card.getBoundingClientRect().top),
          rectBottom: Math.round(card.getBoundingClientRect().bottom),
          scrollH: card.scrollHeight,
          clientH: card.clientHeight,
        }
      : null,
    scrollers,
    optionCount: opts.length,
    lastOption: lastRect
      ? {
          label: last.textContent.slice(0, 12),
          top: Math.round(lastRect.top),
          bottom: Math.round(lastRect.bottom),
          // "Reachable" is the real question: is the last option's box inside
          // the viewport after scrolling every scroller to its end?
          insideViewport: lastRect.bottom <= window.innerHeight + 1 && lastRect.top >= -1,
        }
      : null,
    composerVisible: (() => {
      const ta = document.querySelector('textarea');
      if (!ta) return null;
      const r = ta.getBoundingClientRect();
      return r.bottom <= window.innerHeight + 1 && r.top >= 0;
    })(),
  });
})()
"""

# Where a real finger would land to scroll a given element: the centre of its
# visible intersection with the viewport, which is the only part a thumb can
# reach. An element whose visible slice is empty gets no gesture at all — that
# is the D1 state, and reporting it as unreachable is the point.
TOUCH_TARGET_JS = """
(() => {
  const out = [];
  for (const el of document.querySelectorAll('.lo-scroll, [class*="overflow-y-auto"]')) {
    const r = el.getBoundingClientRect();
    const top = Math.max(r.top, 0);
    const bottom = Math.min(r.bottom, window.innerHeight);
    if (bottom - top < 8 || r.width < 8) continue;
    out.push({
      x: Math.round(Math.min(Math.max(r.left + r.width / 2, 1), window.innerWidth - 1)),
      y: Math.round((top + bottom) / 2),
      before: el.scrollTop,
      max: el.scrollHeight - el.clientHeight,
      cls: el.className.slice(0, 60),
    });
  }
  return JSON.stringify(out);
})()
"""

SCROLLER_STATE_JS = """
(() => JSON.stringify(
  [...document.querySelectorAll('.lo-scroll, [class*="overflow-y-auto"]')].map((el) => ({
    top: el.scrollTop,
    max: el.scrollHeight - el.clientHeight,
  })),
))()
"""


def gesture_scroll_all(page: Page, rounds: int = 14) -> list[dict[str, Any]]:
    """Drive every visible scroller to its end with real finger drags.

    Returns one record per scroller that still has content below its fold after
    the gestures, i.e. every region a user cannot reach by swiping. An empty
    list is the property the evidence actually needs: "the tail is reachable".
    """
    for _ in range(rounds):
        targets = json.loads(page.js(TOUCH_TARGET_JS))
        moved = False
        for t in targets:
            if t["before"] >= t["max"]:
                continue
            page.swipe(t["x"], t["y"], 320)
            moved = True
        if not moved:
            break
    return [s for s in json.loads(page.js(SCROLLER_STATE_JS)) if s["top"] < s["max"] - 2]


def assert_gesture_honest(page: Page, key: str) -> list[str]:
    """Fail loudly when a region is reachable by script but not by finger.

    This is the check that would have caught the stacked-panels defect the
    first time. After the real gestures above have done all they can, it asks
    the DOM to scroll everything by assignment — the thing a script may do and
    a thumb may not — and reports any element that moves. Anything in this list
    is a region the harness could only "reach" by cheating, which is precisely
    the false pass this script used to hand back.

    Deliberately a report rather than a silent repair: the old behaviour was to
    assign `scrollTop` first and photograph the result, so the frame showed a
    reachable card that the user could never have produced.
    """
    cheated = json.loads(page.js("""
(() => {
  const moved = [];
  const sel = '.lo-scroll, [class*="overflow-y-auto"], [class*="h-dvh"]';
  for (const el of document.querySelectorAll(sel)) {
    const before = el.scrollTop;
    el.scrollTop = el.scrollHeight;
    if (el.scrollTop > before + 2) {
      moved.push({ cls: el.className.slice(0, 60), before, after: el.scrollTop });
    }
    el.scrollTop = before;
  }
  return JSON.stringify(moved);
})()
"""))
    problems = [
        f"{key}: {m['cls']!r} moved {m['before']}→{m['after']} by scrollTop "
        "assignment AFTER real gestures were exhausted — reachable by script, "
        "not by finger"
        for m in cheated
    ]
    for line in problems:
        print(f"  UNREACHABLE  {line}", file=sys.stderr)
    return problems


def main() -> None:
    port, outdir, label = int(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    outdir.mkdir(parents=True, exist_ok=True)
    base = f"http://127.0.0.1:{port}"
    chrome = Chrome()
    report: dict[str, dict[str, Any]] = {}
    unreachable: list[str] = []
    try:
        page = Page(chrome.target_ws())
        for width, height in VIEWPORTS:
            page.metrics(width, height)
            vp = f"{width}x{height}"
            # Log in once per viewport: the cookie is per profile, and a
            # re-navigation to /login after an auth cookie exists just redirects.
            page.goto(f"{base}/login")
            page.js(
                "(() => { const f = document.querySelector('form');"
                f" f.password.value = {PASSWORD!r}; f.submit(); return true; }})()"
            )
            time.sleep(2.0)
            for session, expand in (
                ("roster", False),
                ("roster", True),
                ("ask-long", False),
                ("approval", False),
                ("ask-free", False),
            ):
                page.goto(f"{base}/#/s/{session}")
                time.sleep(1.5)
                name = f"{label}-{vp}-{session}"
                if expand:
                    # Tap the roster header, so the expanded body's cap and
                    # internal scrolling are captured too, not just the
                    # collapsed default.
                    page.js(
                        "(() => { const b = [...document.querySelectorAll('button')]"
                        ".find((x) => /subagents/.test(x.textContent));"
                        " if (b) b.click(); return true; })()"
                    )
                    time.sleep(0.8)
                    name = f"{label}-{vp}-{session}-expanded"
                page.shot(outdir / f"{name}-top.png")
                report[f"{name}-top"] = json.loads(page.js(GEOMETRY_JS))
                # Real finger drags, never `scrollTop = scrollHeight`: see
                # Page.swipe. What does not move here does not move for a user.
                stranded = gesture_scroll_all(page)
                page.shot(outdir / f"{name}-bottom.png")
                geo = json.loads(page.js(GEOMETRY_JS))
                geo["strandedAfterGestures"] = stranded
                report[f"{name}-bottom"] = geo
                unreachable.extend(assert_gesture_honest(page, f"{name}-bottom"))
        page.close()
    finally:
        chrome.close()
    (outdir / f"{label}-geometry.json").write_text(json.dumps(report, indent=2))
    for key, geo in report.items():
        last = geo.get("lastOption")
        print(
            f"{key:<46} vp={geo['viewport']}"
            f" card={geo['card'] and (geo['card']['rectTop'], geo['card']['rectBottom'])}"
            f" lastOpt={last and (last['top'], last['bottom'], last['insideViewport'])}"
            f" composer={geo['composerVisible']}"
            f" scrollable={[s['scrollable'] for s in geo['scrollers']]}"
        )

    # A non-zero exit rather than a note in the log. This harness exists to
    # produce review evidence, and evidence that quietly reports a
    # finger-unreachable region as fine is worse than no evidence — it is what
    # let the stacked-panels defect through a capture round. If this fires,
    # something on the captured screen can only be reached by a script.
    if unreachable:
        print(
            f"\nFAIL: {len(unreachable)} region(s) reachable only by scrollTop "
            "assignment, not by a real gesture:",
            file=sys.stderr,
        )
        for line in unreachable:
            print(f"  - {line}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
