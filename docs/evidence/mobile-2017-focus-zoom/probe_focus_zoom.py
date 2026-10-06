#!/usr/bin/env python3
"""Focus-behaviour probe for issue #2017 (reproduction only, untracked).

Companion to capture_focus_zoom.py. It focuses each field class on the REAL
built SPA and reads `visualViewport.scale` immediately before and after the
focus, in BOTH modes, to demonstrate exactly what this host's Chromium does and
does not do:

* Chromium (headless, this host) does NOT implement WebKit iOS's
  `_zoomToFocusRect:` — so focusing a 12-16px field never changes the scale, and
  the zoom leg of #2017 is NOT reproducible here;
* the fit-scale PRECONDITION wide view creates IS reproduced (scale settles at
  screen.width / 512 at load and stays there through focus).

Fields probed: composer (16px), the pending-card secret input (14px), the
model-sheet search (14px), the directory-sheet path (12px), list search (14px)
— the composer and the extremes for the rest.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import scripts.probe_isolation  # noqa: F401  -- must be the first local import
from scripts.mobile_overflow_capture import Chrome, Page, fixture_password

PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 4317
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("captures")
OUT.mkdir(parents=True, exist_ok=True)
BASE = f"http://127.0.0.1:{PORT}"

report: dict = {}


def log(msg: str) -> None:
    print(msg, flush=True)


def scale_of(page: Page) -> dict:
    # returnByValue: the object arrives as a Python dict already.
    return page.js("""
(() => ({
  meta: (document.querySelector('meta[name="viewport"]') || {}).content || null,
  fitScale: document.documentElement.style.getPropertyValue('--lo-fit-scale'),
  scale: window.visualViewport ? Number(window.visualViewport.scale.toFixed(4)) : null,
  vvTop: window.visualViewport ? window.visualViewport.offsetTop : null,
  focused: document.activeElement ? (document.activeElement.tagName + '[' + (document.activeElement.getAttribute('placeholder') || document.activeElement.id || '') + ']') : null,
}))()
""")


def focus_probe(page: Page, key: str, selector: str) -> dict:
    before = scale_of(page)
    result = page.js(f"""
(() => {{
  const el = document.querySelector({selector!r});
  if (!el) return "missing";
  el.focus();
  return "focused";
}})()
""")
    time.sleep(0.6)
    after = scale_of(page)
    report[key] = {"selector": selector, "focus": result, "before": before, "after": after}
    log(f"[focus] {key}: {result} scale {before['scale']} -> {after['scale']} "
        f"(meta {after['meta']!r}, active {after['focused']!r})")
    return after


def served_assets(page: Page) -> dict:
    """The build the fixture actually served — recorded IN the report.

    WHY: a report that names its build only in the README cannot be checked
    against the file itself, and a stale scan once passed as fresh (round-2
    review). These filenames are content hashes, so recording them here makes
    every report self-identifying.
    """
    refs = page.js(
        "[...document.querySelectorAll('link[rel=stylesheet],script[src]')]"
        ".map((e) => e.getAttribute('href') || e.getAttribute('src'))"
    )
    assets = [str(r).rsplit("/", 1)[-1] for r in (refs or [])]
    return {
        "served_css": sorted({a for a in assets if a.endswith(".css")}),
        "served_js": sorted({a for a in assets if a.endswith(".js")}),
    }


def main() -> None:
    chrome = Chrome()
    try:
        page = Page(chrome.target_ws())
        page.metrics(390, 844)
        # login
        page.goto(f"{BASE}/login")
        page.js(
            "(() => { const f = document.querySelector('form');"
            f" if (!f || !f.password) return 'no-form';"
            f" f.password.value = {fixture_password()!r}; f.submit(); return 'submitted'; }})()"
        )
        time.sleep(2.0)

        # --- default mode: focus composer + pending secret input ---
        page.goto(f"{BASE}/#/s/ask-free")
        time.sleep(1.5)
        report["provenance"] = served_assets(page)
        log(f"[provenance] served: {json.dumps(report['provenance'])}")
        focus_probe(page, "wideoff-composer-16px", 'textarea')
        focus_probe(page, "wideoff-pending-secret-14px", '[data-testid="pending-card"] input')

        # --- wide mode: toggle on the list footer, reload at the session ---
        page.goto(f"{BASE}/#/")
        time.sleep(1.0)
        page.js("""
(() => {
  const b = document.querySelector('button[aria-label="wide view"]');
  if (!b) return "missing";
  b.click(); return "clicked";
})()
""")
        page.goto(f"{BASE}/#/s/ask-free")
        time.sleep(0.8)
        page.send("Page.reload")
        time.sleep(2.5)
        log(f"[state] wide at-load: {json.dumps(scale_of(page))}")
        focus_probe(page, "wideon-composer-16px", 'textarea')
        focus_probe(page, "wideon-pending-secret-14px", '[data-testid="pending-card"] input')
        # model sheet search (14px) — open, focus, close
        page.js("""
(() => {
  const b = [...document.querySelectorAll('button')]
    .find((x) => { const t = (x.textContent || '').trim(); return t === 'model' || t === 'fixture'; });
  if (!b) return "missing";
  b.click(); return "clicked";
})()
""")
        time.sleep(0.8)
        focus_probe(page, "wideon-model-search-14px", 'input[placeholder="filter models"]')
        page.js("""
(() => {
  const b = document.querySelector('[role="dialog"] button[aria-label="close sheet"]');
  if (b) { b.click(); return "closed"; }
  return "no-sheet";
})()
""")
        time.sleep(0.6)
        # directory sheet path (12px)
        page.js("""
(() => {
  const b = document.querySelector('button[aria-label^="working directory"]');
  if (!b) return "missing";
  b.click(); return "clicked";
})()
""")
        time.sleep(0.8)
        focus_probe(page, "wideon-directory-path-12px", 'input[placeholder="or type another path…"]')
        page.close()
    finally:
        chrome.close()
    (OUT / "focus-behavior-report.json").write_text(json.dumps(report, indent=2))
    log(f"[done] report: {OUT / 'focus-behavior-report.json'}")


if __name__ == "__main__":
    main()
