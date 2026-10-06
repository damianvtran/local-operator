"""Round-1 remediation captures for issue #2016 at 390x844.

Five states, each a real client frame plus the numbers behind it:
  01 control-with-count   the control states the pile size (design D2 / UX U4)
  02 scrolled-sticky      the control stays reachable at the list's max scroll
                          while the marks are visible (design D5 / UX U7)
  03 receipt-near-gesture one real tap: receipt in the band, marks cleared
                          (design D3 / UX U1)
  04 degraded-receipt     the unread read fails for real (the fixture patches
                          the store seam); the control must NOT claim the pile is
                          empty (agent MAJOR-1 / design D1)
  05 failure-line         the write is blocked at the network layer; the line
                          names what happened and the recovery (UX U5)

Run (from the PR-head worktree, so ``scripts`` resolves to it):
    PYTHONPATH=. LOP_MOBILE_FIXTURE_PASSWORD=<same as fixture> \
      .venv/bin/python "$LOCAL_OPERATOR_SCRATCHPAD/mobile-2016/round1/scripts/mobile_2016_capture_r1.py" <port> <outdir>

Reuses the repo's own rig (``scripts/mobile_overflow_capture.py``): headless=new,
unique mktemp profile, ``--use-mock-keychain``, Emulation device metrics
(390x844, dpr 2), teardown asserted by the rig. The tap is a real pointer pair
(CDP Input.dispatchMouseEvent pressed+released at the control's measured centre).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from scripts.mobile_overflow_capture import Chrome, Page, fixture_password

STATE_JS = r"""
(() => {
  const rect = (el) => {
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return { top: Math.round(r.top), bottom: Math.round(r.bottom), left: Math.round(r.left),
             right: Math.round(r.right), w: Math.round(r.width), h: Math.round(r.height) };
  };
  const buttons = [...document.querySelectorAll('button')];
  const control = buttons.find((b) => /^\s*mark all \d+ read\s*$/.test(b.innerText || ''));
  const marking = buttons.find((b) => (b.innerText || '').trim() === 'marking\u2026');
  /* Scoped to <main>: the header carries its own status span when offline, and
     the receipt this rig is photographing lives in the list's sticky band. */
  const receipt = document.querySelector('main [role="alert"], main [role="status"]');
  const marks = [...document.querySelectorAll('[aria-label="new activity"]')];
  const main = document.querySelector('main');
  return JSON.stringify({
    control: control ? { label: control.innerText.trim(), rect: rect(control) } : null,
    marking: marking ? { label: marking.innerText.trim(), rect: rect(marking) } : null,
    receipt: receipt
      ? { role: receipt.getAttribute('role'), text: (receipt.innerText || '').trim(), rect: rect(receipt) }
      : null,
    markCount: marks.length,
    markRects: marks.map(rect),
    scrollTop: main ? Math.round(main.scrollTop) : null,
    scrollHeight: main ? Math.round(main.scrollHeight) : null,
    clientHeight: main ? Math.round(main.clientHeight) : null,
    dismiss: !!buttons.find((b) => (b.getAttribute('aria-label') || '') === 'Dismiss'),
  });
})()
"""


def _state(page: Page) -> dict:
    return json.loads(page.js(STATE_JS))


def _tap(page: Page, x: int, y: int) -> None:
    page.send("Input.dispatchMouseEvent", type="mousePressed", x=x, y=y, button="left", clickCount=1)
    time.sleep(0.05)
    page.send("Input.dispatchMouseEvent", type="mouseReleased", x=x, y=y, button="left", clickCount=1)


def _publish_two(page: Page, base: str) -> None:
    for session_id in ("c0ffee000016", "c0ffee000017"):
        page.js(
            "fetch("
            f"{base!r} + '/fixture/publish', "
            "{method:'POST', headers:{'content-type':'application/json'}, "
            f"body: JSON.stringify({{session_id: {session_id!r}}})}}).then((r) => r.json())"
        )


def _arm(page: Page, base: str, *, settle: float = 2.6) -> None:
    """Republish two completions and reload so the control is armed again."""
    _publish_two(page, base)
    time.sleep(0.3)
    page.send("Page.reload")
    time.sleep(settle)


def _tap_control(page: Page, outdir: Path, name: str, *, wait: float = 3.0) -> dict:
    armed = _state(page)
    if not armed["control"]:
        raise SystemExit(f"{name}: the control is not mounted; cannot tap")
    rect = armed["control"]["rect"]
    _tap(page, (rect["left"] + rect["right"]) // 2, (rect["top"] + rect["bottom"]) // 2)
    deadline = time.time() + wait
    settled = _state(page)
    while time.time() < deadline:
        settled = _state(page)
        if settled["receipt"] and not settled["marking"]:
            break
        time.sleep(0.25)
    time.sleep(0.4)
    page.shot(outdir / name)
    return settled


def main() -> None:
    port = int(sys.argv[1])
    outdir = Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    base = f"http://127.0.0.1:{port}"
    password = fixture_password()

    chrome = Chrome()
    report: dict = {"port": port, "viewport": "390x844", "steps": []}
    try:
        page = Page(chrome.target_ws())
        page.metrics(390, 844)

        page.goto(f"{base}/login")
        page.js(
            "(() => { const f = document.querySelector('form');"
            f" f.password.value = {password!r}; f.submit(); return true; }})()"
        )
        time.sleep(2.0)

        # 01 — the armed control, stating the pile size.
        page.goto(f"{base}/#/")
        time.sleep(2.0)
        page.shot(outdir / "01-control-with-count-390x844.png")
        report["steps"].append({"step": "control-with-count", **_state(page)})

        # 02 — scrolled to the bottom: the marks are visible, and the sticky band
        # keeps the control reachable where they are.
        for _ in range(6):
            page.swipe(195, 620, 400)
        time.sleep(0.8)
        page.shot(outdir / "02-scrolled-sticky-390x844.png")
        report["steps"].append({"step": "scrolled", **_state(page)})

        # Back to the top for the gesture.
        page.js("(() => { const m = document.querySelector('main'); m.scrollTop = 0; return true; })()")
        time.sleep(0.6)
        settled = _tap_control(page, outdir, "03-receipt-near-gesture-390x844.png")
        report["steps"].append({"step": "receipt-near-gesture", **_state(page), "settled": settled})

        # 04 — a DEGRADED unread read, through the store's own failing seam.
        _arm(page, base)
        page.js("fetch('/fixture/degrade?on=1').then((r) => r.json())")
        time.sleep(0.4)
        degraded = _tap_control(page, outdir, "04-degraded-receipt-390x844.png")
        report["steps"].append({"step": "degraded-receipt", **_state(page), "settled": degraded})
        page.js("fetch('/fixture/degrade?on=0').then((r) => r.json())")

        # 05 — the write blocked at the network layer.
        _arm(page, base)
        page.send("Network.enable")
        page.send("Network.setBlockedURLs", urls=["*api/attention/seen*"])
        failed = _tap_control(page, outdir, "05-failure-line-390x844.png")
        report["steps"].append({"step": "failure-line", **_state(page), "settled": failed})
        page.send("Network.setBlockedURLs", urls=[])

        page.close()
    finally:
        chrome.close()

    (outdir / "capture-report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
