"""Capture the phone's Team Bridge tool rows out of headless Chrome over CDP.

Run:  PYTHONPATH=. .venv/bin/python scripts/bridge_tool_mobile_shot.py OUTDIR

Starts ``scripts/bridge_tool_mobile_fixture.py`` (the REAL bundle; rows built
by the projection's own ``_summarize_args`` / ``_tool_row_details``), opens its
one session and photographs, at 320/360/390 and in the product's light and dark
palettes:

* ``group`` - the condensed tool groups, folded: the count line where bridge
  posts and reads split into `team message(s)` / `team action(s)` beside lop's
  own ``send``;
* ``group-open`` - the earlier turn's group opened: bridge rows and ``send``
  side by side in ONE group;
* ``collapsed`` - the latest turn's four bridge rows (post, DM, refusal,
  inbox-only send), one line each;
* ``expanded`` - the DM row opened: headline, markdown body, outcome sentence.

``Chrome``/``Page`` come from ``scripts/mobile_overflow_capture.py`` (one
throwaway ``--headless=new`` browser, own profile, ``--use-mock-keychain``,
process-group teardown that asserts nothing survived). The fixture imports
``scripts.probe_isolation`` first, so HOME/config are a temp dir and every
``CMUX_*`` variable is gone before any app code runs.

Beside each frame the GEOMETRY of every tool row is dumped (row width vs the
document's scroll width, each lane's right edge), so "nothing overflows at
320" is a measured number, not a reading of a PNG. Run the SAME rig from a base
checkout for the before frames.
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

WIDTHS = [320, 360, 390]
HEIGHT = 780
THEMES = {"dark": "localOperatorDark", "light": "localOperatorLight"}
SESSION_LABEL = "Bridge tool run"

GEOMETRY_JS = r"""
(() => {
  const rows = [...document.querySelectorAll('button')]
    .filter((b) => /bridge|send/.test(b.textContent || ''))
    .map((b) => {
      const r = b.getBoundingClientRect();
      return {
        text: (b.textContent || '').trim().slice(0, 110),
        left: Math.round(r.left), right: Math.round(r.right),
        overflowX: b.scrollWidth > b.clientWidth + 1,
        lanes: [...b.children].map((c) => Math.round(c.getBoundingClientRect().right)),
      };
    });
  const doc = document.scrollingElement;
  return JSON.stringify({
    viewport: innerWidth,
    docScrollWidth: doc.scrollWidth,
    docClientWidth: doc.clientWidth,
    bcvInHeadlines: rows.filter((r) => /\b(?:bcv|bcmsg|acc|ep)_/.test(r.text)).length,
    rows,
  });
})()
"""


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _click_text_js(needle: str, *, selector: str = "button", last: bool = False) -> str:
    pick = "hits[hits.length - 1]" if last else "hits[0]"
    return (
        "(() => {"
        f" const hits = [...document.querySelectorAll({selector!r})]"
        f".filter((n) => (n.textContent || '').includes({needle!r}));"
        f" const hit = {pick};"
        " if (!hit) return false; hit.scrollIntoView({block: 'center'}); hit.click(); return true;"
        "})()"
    )


def _open_session(page: Page, base: str, password: str, theme: str) -> None:
    page.goto(f"{base}/login")
    page.js(
        "(() => { const f = document.querySelector('form');"
        f" f.password.value = {password!r}; f.submit(); return true; }})()"
    )
    time.sleep(2.0)
    page.js(f"localStorage.setItem('lo-mobile-theme', {theme!r}); true")
    # A hash change is same-document, so `initTheme` would not run again and
    # every frame would keep the first palette; a reload re-applies it.
    page.goto(f"{base}/#/")
    page.send("Page.reload", ignoreCache=True)
    time.sleep(2.0)
    time.sleep(1.5)
    page.js(_click_text_js(SESSION_LABEL, selector='a,button,[role="link"],[role="button"]'))
    time.sleep(2.0)


def main() -> None:
    outdir = Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    # Generated per run and never printed (the rule in mobile_overflow_capture).
    password = secrets.token_urlsafe(16)
    fixture = subprocess.Popen(
        [sys.executable, "scripts/bridge_tool_mobile_fixture.py", str(port), password],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    chrome = Chrome()
    report: dict[str, Any] = {}
    try:
        for _ in range(60):
            try:
                with urllib.request.urlopen(f"{base}/login", timeout=2):
                    break
            except Exception:  # noqa: BLE001  -- not up yet
                if fixture.poll() is not None:
                    raise RuntimeError(
                        f"fixture exited: {fixture.stdout.read() if fixture.stdout else ''}"
                    ) from None
                time.sleep(0.4)
        else:
            raise RuntimeError("fixture never served /login")

        page = Page(chrome.target_ws())
        for tone, theme in THEMES.items():
            for width in WIDTHS:
                page.metrics(width, HEIGHT)
                _open_session(page, base, password, theme)
                # The fixture session is settled, so BOTH turns condense behind
                # their turn bars (the product's own first paint). Open the bars
                # to reach the tool groups - the gesture a reader makes.
                page.js(
                    "(() => { document.querySelectorAll('[data-condense-header]')"
                    ".forEach((h) => h.click()); return true; })()"
                )
                time.sleep(0.8)

                # group: each fold's header with its count line, still folded.
                page.js(
                    "(() => { const g = document.querySelector('[data-condense-fold]');"
                    " if (g) g.scrollIntoView({block: 'center'}); return !!g; })()"
                )
                time.sleep(0.5)
                name = f"group-{tone}-{width}"
                page.shot(outdir / f"{name}.png")
                report[name] = json.loads(page.js(GEOMETRY_JS))
                report[name]["foldHeaders"] = page.js(
                    "JSON.stringify([...document.querySelectorAll("
                    "'[data-condense-fold] > button, [data-condense-fold] button:first-of-type')]"
                    ".map((b) => (b.textContent || '').trim()))"
                )

                # Open both folds: every bridge row, one line each.
                page.js(
                    "(() => { document.querySelectorAll('[data-condense-fold]')"
                    ".forEach((g) => g.querySelector('button')?.click()); return true; })()"
                )
                time.sleep(0.8)
                name = f"group-open-{tone}-{width}"
                page.js(
                    "(() => { const g = document.querySelector('[data-condense-fold]');"
                    " if (g) g.scrollIntoView({block: 'start'}); return !!g; })()"
                )
                time.sleep(0.5)
                page.shot(outdir / f"{name}.png")
                report[name] = json.loads(page.js(GEOMETRY_JS))

                # collapsed: the live turn's four rows (post, dm, refusal, inbox).
                page.js(
                    "(() => { const g ="
                    " [...document.querySelectorAll('[data-condense-fold]')].pop();"
                    " if (g) g.scrollIntoView({block: 'start'}); return !!g; })()"
                )
                time.sleep(0.5)
                name = f"collapsed-{tone}-{width}"
                page.shot(outdir / f"{name}.png")
                report[name] = json.loads(page.js(GEOMETRY_JS))

                # expanded: the DM row (markdown body, Conversation, Outcome).
                # The DM row: a bridge row whose summary names the dm op. Matched
                # on both spellings - the generic `op=dm` before the sibling
                # lane's localized summary lands, and its `dm · …` after.
                page.js(
                    "(() => { const b = [...document.querySelectorAll('button')]"
                    ".filter((n) => /bridge/.test(n.textContent || '')"
                    " && /op=dm|DM|awaiting acceptance/.test(n.textContent || '')).pop();"
                    " if (!b) return false; b.scrollIntoView({block: 'start'}); b.click();"
                    " return true; })()"
                )
                time.sleep(0.8)
                # Bring the expansion's outcome sentence above the
                # composer: the frame must show them, not just the body's top.
                page.js(
                    "(() => { const d = document.querySelector('[data-bridge-outcome-sentence]');"
                    " if (d) d.scrollIntoView({block: 'center'}); return !!d; })()"
                )
                time.sleep(0.5)
                name = f"expanded-{tone}-{width}"
                page.shot(outdir / f"{name}.png")
                report[name] = json.loads(page.js(GEOMETRY_JS))
                print(f"{tone} {width}: ok", flush=True)
        page.close()
    finally:
        chrome.close()
        fixture.terminate()
        try:
            fixture.wait(timeout=10)
        except subprocess.TimeoutExpired:
            fixture.kill()
    (outdir / "bridge-mobile-geometry.json").write_text(json.dumps(report, indent=2))
    print("shots:", ", ".join(sorted(p.name for p in outdir.glob("*.png"))))


if __name__ == "__main__":
    main()
