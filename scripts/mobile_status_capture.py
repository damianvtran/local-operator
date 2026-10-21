"""Capture the session header/status simplification out of headless Chrome.

Run (with the per-run password the fixture was started with, from the repo
root): ``.venv/bin/python scripts/mobile_status_capture.py <port> <outdir> <label>``

Sibling of ``mobile_spend_capture.py``, and deliberately separate: that harness
walks PR-2's published-spend STATES, while this one photographs the OPERATOR'S
defect and its fix — one status line with the per-channel facts behind a tap.

WHAT EACH FRAME IS FOR:

* ``-collapsed-a`` / ``-collapsed-b`` — two consecutive frames with no input
  between them, so the settled layout is proven stable before a tap moves it.
* ``-expanded`` — the inline panel below the row: the three `image` rows now
  qualified with their own model, `channels not tracked`, `· +N more`.
* ``-collapsed-again`` — the same one-line row after the second tap, with the
  panel absent from the DOM (not hidden).
* ``-menu`` — the ⋮ sheet: Pin session, Wide view and Approvals…, each stating
  its current value.
* ``-operator-spend-long-*`` — the long-tail repro (R1-1/U3/Q1): at the
  bottom of 60 entries, the panel stays open through 0/100/500/1000ms while
  the browser tail-follows, then closes on a real touch drag.
* ``-operator-spend-zero-*`` — a published breakdown with no spend figure:
  one `cost detail` control in the row, the facts only after the tap.
* ``-asks-header`` / ``-stacked-approval-header`` — the header at rest:
  back/title/menu, and `needs you` only where a gate actually blocks.

Geometry is recorded beside every frame (the status row and spend button rects
before/after, the panel's presence and rect, the header controls' labels). The
script ASSERTS that the spend button's top/height (and the status row's) are
identical collapsed and expanded — the tapped row must not move while the
panel pushes only what is below it — and exits non-zero if they differ.

The driver (``Chrome``/``Page``, the per-run password contract and the real
touch ``tap``) is IMPORTED from the existing sibling harnesses, so there is one
headless-Chrome launch recipe, one login path and one finger.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.mobile_asks_open_capture import tap  # noqa: E402
from scripts.mobile_overflow_capture import Chrome, Page, fixture_password  # noqa: E402

VIEWPORTS = [(390, 844), (360, 780)]

SPEND_PROBE = """
(() => {
  const rect = (el) => {
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return {
      top: Math.round(r.top), bottom: Math.round(r.bottom), h: Math.round(r.height),
      left: Math.round(r.left), width: Math.round(r.width),
    };
  };
  const text = (el) => (el ? el.textContent.trim() : null);
  const status = document.querySelector('[data-testid="session-status"]');
  const spend = document.querySelector('[data-testid="session-status-spend"]');
  const panel = document.querySelector('[data-testid="session-status-channels"]');
  return JSON.stringify({
    viewport: [window.innerWidth, window.innerHeight],
    status: rect(status),
    spend: rect(spend),
    spendText: text(spend),
    spendLabel: spend ? spend.getAttribute('aria-label') : null,
    spendExpanded: spend ? spend.getAttribute('aria-expanded') : null,
    panelPresent: Boolean(panel),
    panel: rect(panel),
    chips: [...document.querySelectorAll('[data-testid="session-status-channel"]')]
      .map((chip) => chip.textContent.trim()),
    untracked: text(document.querySelector('[data-testid="session-status-channels-untracked"]')),
    more: panel ? (panel.textContent.match(/\\u00b7 \\+\\d+ more/) || [null])[0] : null,
    costDetail: (() => {
      const detail = document.querySelector('[data-testid="session-status-cost-detail"]');
      return detail
        ? {
            rect: rect(detail),
            text: detail.textContent.trim(),
            label: detail.getAttribute('aria-label'),
            expanded: detail.getAttribute('aria-expanded'),
          }
        : null;
    })(),
    /* The transcript scroller's own position: the long-tail repro is about
       what happens to this number while the panel is inserted. */
    scroll: (() => {
      const scroller = [...document.querySelectorAll('.lo-scroll')]
        .find((el) => el.scrollHeight > el.clientHeight + 1);
      return scroller
        ? {
            top: Math.round(scroller.scrollTop),
            height: Math.round(scroller.scrollHeight),
            client: Math.round(scroller.clientHeight),
          }
        : null;
    })(),
    headerControls: [...document.querySelectorAll('header button')].map((button) => ({
      label: button.getAttribute('aria-label'),
      text: button.textContent.trim(),
    })),
  });
})()
"""

MENU_PROBE = """
(() => {
  const dialog = document.querySelector('[role="dialog"]');
  if (!dialog) return JSON.stringify({ open: false });
  return JSON.stringify({
    open: true,
    title: dialog.textContent.includes('Session'),
    rows: [...dialog.querySelectorAll('button')]
      .filter((button) => button.className.includes('w-full'))
      .map((row) => ({
        text: row.textContent.trim(),
        label: row.getAttribute('aria-label'),
        pressed: row.getAttribute('aria-pressed'),
      })),
  });
})()
"""

SPEND = "document.querySelector('[data-testid=\"session-status-spend\"]')"
MENU_TRIGGER = "document.querySelector('header button[aria-label=\"session menu\"]')"
MENU_CLOSE = 'document.querySelector(\'[role="dialog"] button[aria-label="close sheet"]\')'
ANY_DIALOG = "Boolean(document.querySelector('[role=\"dialog\"]'))"
COST_DETAIL = "document.querySelector('[data-testid=\"session-status-cost-detail\"]')"
SCROLL_TOP = """
(() => {
  const scroller = [...document.querySelectorAll('.lo-scroll')]
    .find((el) => el.scrollHeight > el.clientHeight + 1);
  return scroller ? Math.round(scroller.scrollTop) : -1;
})()
"""


def probe(page: Page) -> dict[str, Any]:
    return json.loads(page.js(SPEND_PROBE))


def same_rect(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
    if a is None or b is None:
        return a is b
    return all(a.get(key) == b.get(key) for key in ("top", "bottom", "h", "left", "width"))


def main() -> None:
    port, outdir, label = int(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    outdir.mkdir(parents=True, exist_ok=True)
    base = f"http://127.0.0.1:{port}"
    password = fixture_password()
    chrome = Chrome()
    report: dict[str, Any] = {}
    failures: list[str] = []
    try:
        page = Page(chrome.target_ws())
        for width, height in VIEWPORTS:
            page.metrics(width, height)
            viewport = f"{width}x{height}"
            page.goto(f"{base}/login")
            page.js(
                "(() => { const f = document.querySelector('form');"
                f" f.password.value = {password!r}; f.submit(); return true; }})()"
            )
            time.sleep(2.0)

            page.goto(f"{base}/#/s/operator-spend")
            time.sleep(1.5)
            page.shot(outdir / f"{label}-{viewport}-operator-spend-collapsed-a.png")
            before_a = probe(page)
            time.sleep(0.3)
            page.shot(outdir / f"{label}-{viewport}-operator-spend-collapsed-b.png")
            before_b = probe(page)
            if not same_rect(before_a["spend"], before_b["spend"]):
                failures.append(f"{viewport}: spend moved between settled frames")

            tap(page, SPEND)
            time.sleep(0.4)
            page.shot(outdir / f"{label}-{viewport}-operator-spend-expanded.png")
            expanded = probe(page)
            if expanded["spendExpanded"] != "true":
                failures.append(
                    f"{viewport}: tap did not expand "
                    f"(aria-expanded={expanded['spendExpanded']})"
                )
            if not expanded["panelPresent"]:
                failures.append(f"{viewport}: expanded panel missing")
            if expanded["panel"] and expanded["status"]:
                if expanded["panel"]["top"] < expanded["status"]["bottom"] - 1:
                    failures.append(f"{viewport}: panel is not below the status row")

            tap(page, SPEND)
            time.sleep(0.4)
            page.shot(outdir / f"{label}-{viewport}-operator-spend-collapsed-again.png")
            collapsed = probe(page)
            if collapsed["spendExpanded"] != "false":
                failures.append(f"{viewport}: second tap did not collapse")
            if collapsed["panelPresent"]:
                failures.append(f"{viewport}: collapsed state still has a panel element")

            # THE DEFECT'S CORE CLAIM: the tapped row does not move.
            if not same_rect(before_b["spend"], expanded["spend"]):
                failures.append(
                    f"{viewport}: spend rect changed on expand "
                    f"{before_b['spend']} -> {expanded['spend']}"
                )
            if not same_rect(before_b["status"], expanded["status"]):
                failures.append(
                    f"{viewport}: status row changed on expand "
                    f"{before_b['status']} -> {expanded['status']}"
                )
            if not same_rect(before_b["spend"], collapsed["spend"]):
                failures.append(f"{viewport}: spend rect changed after collapse")

            tap(page, MENU_TRIGGER)
            time.sleep(0.5)
            page.shot(outdir / f"{label}-{viewport}-session-menu.png")
            menu = json.loads(page.js(MENU_PROBE))
            tap(page, MENU_CLOSE)
            time.sleep(0.3)

            # THE LONG-TAIL REPRO (R1-1/U3/Q1). Reach the bottom with REAL
            # touch drags, then open the panel and sample it while the browser
            # is free to tail-follow/anchor the insertion. A programmatic
            # `scroll` must not close it; only a reader gesture may.
            page.goto(f"{base}/#/s/operator-spend-long")
            time.sleep(1.5)
            for _ in range(12):
                was = page.js(SCROLL_TOP)
                page.swipe(width // 2, height - 150, 500)
                if page.js(SCROLL_TOP) == was:
                    break
            page.shot(outdir / f"{label}-{viewport}-operator-spend-long-collapsed.png")
            long_before = probe(page)
            tap(page, SPEND)
            started = time.time()
            long_samples = []
            for target_ms in (0, 100, 500, 1000):
                delay = target_ms / 1000 - (time.time() - started)
                if delay > 0:
                    time.sleep(delay)
                long_samples.append({"ms": target_ms, "probe": json.loads(page.js(SPEND_PROBE))})
            page.shot(outdir / f"{label}-{viewport}-operator-spend-long-expanded.png")
            for sample in long_samples:
                if not sample["probe"]["panelPresent"]:
                    failures.append(f"{viewport}: long-tail panel vanished at {sample['ms']}ms")
                if sample["probe"]["spendExpanded"] != "true":
                    failures.append(
                        f"{viewport}: long-tail aria-expanded at {sample['ms']}ms "
                        f"is {sample['probe']['spendExpanded']!r}"
                    )
            # A REAL USER GESTURE still dismisses it (touchmove is the intent
            # signal now, where a layout-driven `scroll` is not).
            page.swipe(width // 2, height - 200, 320)
            time.sleep(0.4)
            long_after = probe(page)
            page.shot(outdir / f"{label}-{viewport}-operator-spend-long-after-swipe.png")
            if long_after["panelPresent"]:
                failures.append(f"{viewport}: long-tail panel survived a user swipe")

            # THE NO-FIGURE STATE (R1-2/U1): a zero total with a zero-amount
            # row, and no context reading, so `cost detail` is the row's only
            # content and the facts stay behind it.
            page.goto(f"{base}/#/s/operator-spend-zero")
            time.sleep(1.2)
            page.shot(outdir / f"{label}-{viewport}-operator-spend-zero-collapsed.png")
            zero = probe(page)
            zero_open = zero
            if zero["panelPresent"]:
                failures.append(f"{viewport}: zero-state panel is on by default")
            if not zero["costDetail"]:
                failures.append(f"{viewport}: zero-state has no cost detail control")
            else:
                tap(page, COST_DETAIL)
                time.sleep(0.35)
                page.shot(outdir / f"{label}-{viewport}-operator-spend-zero-expanded.png")
                zero_open = probe(page)
                if not zero_open["panelPresent"]:
                    failures.append(f"{viewport}: cost detail did not open the panel")
                if zero_open["costDetail"] and zero_open["costDetail"]["expanded"] != "true":
                    failures.append(f"{viewport}: cost detail aria-expanded stayed false")

            page.goto(f"{base}/#/s/asks")
            time.sleep(1.2)
            # The asks session opens its sheet by policy; the frame this script
            # exists for is the HEADER, so close it first rather than photograph
            # a dialog covering the controls.
            if page.js(ANY_DIALOG) is not None:
                tap(page, MENU_CLOSE)
                time.sleep(0.4)
            page.shot(outdir / f"{label}-{viewport}-asks-header.png")
            asks = probe(page)

            page.goto(f"{base}/#/s/stacked-approval")
            time.sleep(1.2)
            page.shot(outdir / f"{label}-{viewport}-stacked-approval-header.png")
            stacked = probe(page)

            report[viewport] = {
                "before": before_b,
                "before_settled_pair": [before_a, before_b],
                "expanded": expanded,
                "collapsed_again": collapsed,
                "menu": menu,
                "long": {
                    "before": long_before,
                    "samples": long_samples,
                    "after_swipe": long_after,
                },
                "zero": {"collapsed": zero, "expanded": zero_open},
                "asks_header": asks["headerControls"],
                "stacked_approval_header": stacked["headerControls"],
            }
        page.close()
    finally:
        chrome.close()
    (outdir / f"{label}-status-geometry.json").write_text(json.dumps(report, indent=2))
    for viewport, data in report.items():
        before = data["before"]
        expanded = data["expanded"]
        long_presence = [
            (sample["ms"], sample["probe"]["panelPresent"], sample["probe"]["scroll"])
            for sample in data["long"]["samples"]
        ]
        print(
            f"{viewport} spend={before['spendText']!r} label={before['spendLabel']!r} "
            f"spendRect={before['spend']} expandedRect={expanded['spend']} "
            f"panel={expanded['panel']} chips={expanded['chips']} "
            f"untracked={expanded['untracked']!r} more={expanded['more']!r} "
            f"menuRows={[(r['label'] or r['text'], r['pressed']) for r in data['menu']['rows']]} "
            f"longPresence={long_presence} "
            f"longAfterSwipePanel={data['long']['after_swipe']['panelPresent']} "
            f"zeroPanel={data['zero']['collapsed']['panelPresent']} "
            f"zeroDetail={bool(data['zero']['collapsed']['costDetail'])} "
            f"asksHeader={data['asks_header']} "
            f"blockedHeader={data['stacked_approval_header']}"
        )
    if failures:
        for failure in failures:
            print(f"FAIL {failure}", file=sys.stderr)
        raise SystemExit(1)
    print("PASS: settled rows stable; spend/status rects identical expanded vs collapsed")


if __name__ == "__main__":
    main()
