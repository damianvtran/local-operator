#!/usr/bin/env python3
"""Debug probe: why does st-second's ask card render no input? (reproduction only)"""

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


def main() -> None:
    chrome = Chrome()
    try:
        page = Page(chrome.target_ws())
        page.metrics(390, 844)
        page.goto(f"{BASE}/login")
        page.js(
            "(() => { const f = document.querySelector('form');"
            f" if (!f || !f.password) return 'no-form';"
            f" f.password.value = {fixture_password()!r}; f.submit(); return 'submitted'; }})()"
        )
        time.sleep(2.0)
        page.goto(f"{BASE}/#/s/asks-stacked")
        time.sleep(1.5)
        page.js("""
(() => {
  const b = document.querySelector('[data-testid="ask-dock"]');
  if (b) b.click();
  return true;
})()
""")
        time.sleep(1.2)
        dump = page.js("""
(() => {
  const card = document.querySelector('[data-ask-id="st-second"]');
  if (!card) return JSON.stringify({missing: true});
  const inputs = card.querySelectorAll('input, textarea, select');
  const buttons = [...card.querySelectorAll('button')].map((b) => b.textContent.trim());
  return JSON.stringify({
    text: card.textContent.replace(/\\s+/g, ' ').slice(0, 400),
    inputCount: inputs.length,
    buttons,
    html: card.innerHTML.replace(/\\s+/g, ' ').slice(0, 1500),
  });
})()
""")
        print(json.dumps(json.loads(dump), indent=2), flush=True)
        # scroll the card into view for a frame (inspection only)
        page.js("""
(() => {
  const card = document.querySelector('[data-ask-id="st-second"]');
  if (card) card.scrollIntoView({block: 'center'});
  return true;
})()
""")
        time.sleep(0.8)
        page.shot(OUT / "st-second-card.png")
        print(f"frame: {OUT / 'st-second-card.png'}", flush=True)
        page.close()
    finally:
        chrome.close()


if __name__ == "__main__":
    main()
