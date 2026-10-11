"""Capture the phone's session-status SPEND surfaces out of headless Chrome over CDP.

Run:  .venv/bin/python scripts/mobile_spend_capture.py <port> <outdir> <label>

Sibling of ``mobile_asks_capture.py`` and deliberately a separate script rather
than a branch inside it: that harness walks the ASK surfaces (a queue, a sheet,
an answer flow), and PR-2's surfaces are about the session-status row's two
readings — which of the published-spend states is mounted, what the total says,
and what the breakdown line spells. The driver — ``Chrome``/``Page`` and the
per-run password contract — is IMPORTED from ``mobile_overflow_capture``, so
there is one headless-Chrome launch recipe and one login path.

AGENTS.md §6 is followed by the imported driver: a unique profile, a
``--remote-debugging-port=0`` with the real port read from
``DevToolsActivePort``, ``--headless=new`` (a headful window steals the
operator's focus), the viewport set by ``Emulation.setDeviceMetricsOverride``,
``start_new_session=True`` so the browser is one process GROUP, and an
``atexit`` sweep that ASSERTS zero leftovers. One browser instance for the
whole run (the rig-hygiene rule).

WHAT EACH FRAME IS FOR (PR-2, design §5.1/§6):

* ``-spend-populated`` — the published object POPULATED: named chips with their
  own amounts, one ``$—`` for an amount that could not be stated, and one
  ``sub-equiv`` marker so subscription quota is never read as billed dollars.
* ``-spend-untracked`` — ``tracked: false``: the ``channels not tracked``
  disclosure beside a marked total — never an implied ``$0`` of channel spend.
* ``-spend-unknown``   — the unknown total with billed tokens: ``$—`` on line
  one, and an unstated row beside it.
* ``-light`` is kept to the populated state: the theme switch's effect on the
  new line, checked once rather than per state.

Geometry is recorded beside every frame (the status row's rect, the breakdown
line's presence and rect, the chips' text), because the stills show the symptom
and the numbers show the cause: a chip line pushed under the viewport by the
header would be invisible in a screenshot and obvious in the rects.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.mobile_overflow_capture import Chrome, Page, fixture_password  # noqa: E402

VIEWPORTS = [(390, 844), (360, 780)]

#: What the two readings currently say. ``textContent`` rather than a parsed
#: number: the spellings (`≥$1.053`, `$—`, `sub-equiv`) ARE the evidence, so
#: the probe reports the DOM's own words instead of re-deriving them.
SPEND_PROBE = """
(() => {
  const status = document.querySelector('[data-testid="session-status"]');
  const spend = document.querySelector('[data-testid="session-status-spend"]');
  const channels = document.querySelector('[data-testid="session-status-channels"]');
  const rect = (el) => {
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return {top: Math.round(r.top), bottom: Math.round(r.bottom), h: Math.round(r.height)};
  };
  const text = (el) => (el ? el.textContent.trim() : null);
  return JSON.stringify({
    viewport: [window.innerWidth, window.innerHeight],
    status: rect(status),
    spend: rect(spend),
    spendText: text(spend),
    channelsPresent: Boolean(channels),
    channels: rect(channels),
    chips: [...document.querySelectorAll('[data-testid="session-status-channel"]')]
      .map((c) => c.textContent.trim()),
    untracked: text(document.querySelector('[data-testid="session-status-channels-untracked"]')),
    more: channels ? (channels.textContent.match(/\\+\\d+ more/) || [null])[0] : null,
    statusText: text(status),
    // The whole line's visibility rule, measured rather than asserted: a line
    // that exists but sits below the fold is a defect a still cannot show.
    channelsInViewport: channels
      ? (() => { const r = channels.getBoundingClientRect();
          return r.top >= 0 && r.bottom <= window.innerHeight + 1; })()
      : null,
  });
})()
"""

#: The fixture sessions, each with the one line its frame exists to show.
SESSIONS = [
    ("spend-populated", "named chips, one $— row, one sub-equiv chip"),
    ("spend-untracked", "the untracked disclosure beside a marked total"),
    ("spend-unknown", "the unpriceable total ($—) with a billed signal"),
    ("spend-capped", "more rows than chips: `· +N more`, named channels kept"),
    ("spend-untracked-exact", "marker-less and exact: NO disclosure (M-4)"),
    ("spend-recovered", "marker-less with recovered rows: chips AND the notice"),
]

LIGHT_THEME = "localOperatorLight"


def main() -> None:
    port, outdir, label = int(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    viewports = VIEWPORTS
    if "--viewport" in sys.argv:
        raw = sys.argv[sys.argv.index("--viewport") + 1]
        width, height = (int(part) for part in raw.split("x"))
        viewports = [(width, height)]
    outdir.mkdir(parents=True, exist_ok=True)
    base = f"http://127.0.0.1:{port}"
    chrome = Chrome()
    report: dict[str, Any] = {}
    try:
        page = Page(chrome.target_ws())
        for width, height in viewports:
            page.metrics(width, height)
            vp = f"{width}x{height}"
            page.goto(f"{base}/login")
            page.js(
                "(() => { const f = document.querySelector('form');"
                f" f.password.value = {fixture_password()!r}; f.submit(); return true; }})()"
            )
            time.sleep(2.0)
            for session_id, purpose in SESSIONS:
                page.goto(f"{base}/#/s/{session_id}")
                time.sleep(1.5)
                page.shot(outdir / f"{label}-{vp}-{session_id}.png")
                report[f"{vp}-{session_id}"] = json.loads(page.js(SPEND_PROBE))
                report[f"{vp}-{session_id}"]["purpose"] = purpose

            # THE LIGHT PASS, one state: this client ships both and the design
            # asks for light where a surface has both. The reload is what
            # applies the stored theme.
            page.goto(f"{base}/#/s/spend-populated")
            page.js(
                "(() => { localStorage.setItem('lo-mobile-theme', "
                f"{LIGHT_THEME!r}); location.reload(); return true; }})()"
            )
            time.sleep(2.5)
            page.shot(outdir / f"{label}-{vp}-spend-populated-light.png")
            report[f"{vp}-spend-populated-light"] = json.loads(page.js(SPEND_PROBE))
            page.js(
                "(() => { localStorage.removeItem('lo-mobile-theme');"
                " location.reload(); return true; })()"
            )
            time.sleep(2.0)
        page.close()
    finally:
        chrome.close()
    (outdir / f"{label}-spend-geometry.json").write_text(json.dumps(report, indent=2))
    for key, geo in report.items():
        print(
            f"{key:<34} spend={geo.get('spendText')!r} chips={geo.get('chips')} "
            f"untracked={geo.get('untracked')!r} inViewport={geo.get('channelsInViewport')}"
        )


if __name__ == "__main__":
    main()
