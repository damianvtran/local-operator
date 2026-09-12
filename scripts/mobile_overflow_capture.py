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

    def close(self) -> None:
        self.ws.close()


# The ask card is the accent-bordered block pinned above the composer. The probe
# finds it structurally (its accent border class) rather than by a test id, so it
# measures the shipped markup and not a hook added for the harness.
GEOMETRY_JS = """
(() => {
  const col = document.querySelector('div.h-dvh') || document.querySelector('[class*="h-dvh"]');
  const card = document.querySelector('.border-accent');
  const scrollers = [...document.querySelectorAll('.lo-scroll')].map((el) => ({
    cls: el.className.slice(0, 60),
    scrollH: el.scrollHeight,
    clientH: el.clientHeight,
    scrollable: el.scrollHeight > el.clientHeight + 1,
  }));
  const opts = [...document.querySelectorAll('.border-l-accent')];
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

SCROLL_ALL_JS = """
(() => {
  // Drive every scroller to its end, the gesture a user makes to reach the
  // tail of a long option list. Whatever stays off screen after this is
  // genuinely unreachable, not merely un-scrolled.
  for (const el of document.querySelectorAll('.lo-scroll, [class*="overflow-y-auto"]')) {
    el.scrollTop = el.scrollHeight;
  }
  document.documentElement.scrollTop = document.documentElement.scrollHeight;
  return true;
})()
"""


def main() -> None:
    port, outdir, label = int(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    outdir.mkdir(parents=True, exist_ok=True)
    base = f"http://127.0.0.1:{port}"
    chrome = Chrome()
    report: dict[str, dict[str, Any]] = {}
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
                page.js(SCROLL_ALL_JS)
                time.sleep(0.6)
                page.shot(outdir / f"{name}-bottom.png")
                report[f"{name}-bottom"] = json.loads(page.js(GEOMETRY_JS))
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


if __name__ == "__main__":
    main()
