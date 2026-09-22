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
        page.close()
    finally:
        chrome.close()
        fixture.terminate()
        try:
            fixture.wait(timeout=10)
        except subprocess.TimeoutExpired:
            fixture.kill()
    (outdir / "delegating-mobile-geometry.json").write_text(json.dumps(report, indent=2))
    print("shots:", ", ".join(f"{key}.png" for key in report))


if __name__ == "__main__":
    main()
