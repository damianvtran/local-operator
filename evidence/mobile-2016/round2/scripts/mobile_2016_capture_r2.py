"""Round-2 remediation captures for issue #2016 at 390x844.

Four states, each a real client frame plus the numbers behind it:

  01 band-separator    the pinned band's own edge at the list's max scroll, with
                       the rows it overlaps (design round 2, D6)
  02 focus-ring        keyboard activation with a query that registers no card —
                       the focus fallback must land somewhere VISIBLE (design
                       round 2, D7 / QA Q-2 / agent NIT-3)
  03 degraded-no-count the unread read fails for real; the control drops the
                       count it cannot stand behind and posts NOTHING (UX U11;
                       the per-step POST count is the wire artifact for
                       agent NIT-2)
  04 receipt-window    the receipt after an absence LONGER than its TTL, on
                       return (UX U9 — the window is the reader's viewing time)

Run (from the PR-head worktree, so ``scripts`` resolves to it):
    PYTHONPATH=. LOP_MOBILE_FIXTURE_PASSWORD=<same as fixture> \
      .venv/bin/python "$LOCAL_OPERATOR_SCRATCHPAD/mobile-2016/round2/scripts/mobile_2016_capture_r2.py" <port> <outdir>

Reuses the repo's own rig (``scripts/mobile_overflow_capture.py``): headless=new,
unique mktemp profile, ``--use-mock-keychain``, Emulation device metrics
(390x844, dpr 2), teardown asserted by the rig. Taps are real pointer pairs, and
the keyboard step is real CDP key events (`Input.dispatchKeyEvent`).

The POST counter is page-side, installed with
``Page.addScriptToEvaluateOnNewDocument`` so it survives the reloads the arming
step performs: it wraps ``window.fetch`` and counts requests to
``/api/attention/seen``, which is the cheapest honest form of the
``Network.requestWillBeSent`` count the agent round asked for (the rig's
``Page.send`` consumes only command replies, so CDP events are not collectable
through it).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from scripts.mobile_overflow_capture import Chrome, Page, fixture_password

#: Installed on every new document: counts the bulk route's requests, so a step's
#: "no POST is made" is read off the wire rather than assumed.
COUNTER_JS = r"""
(() => {
  window.__seenPosts = 0;
  const original = window.fetch;
  window.fetch = function (...args) {
    const first = args[0];
    const url = typeof first === "string" ? first : (first && first.url) || "";
    if (String(url).includes("/api/attention/seen")) window.__seenPosts += 1;
    return original.apply(this, args);
  };
})();
"""

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
  const plain = buttons.find((b) => (b.innerText || '').trim() === 'mark all as read');
  const marking = buttons.find((b) => (b.innerText || '').trim() === 'marking\u2026');
  const live = [...document.querySelectorAll('main [role="alert"], main [role="status"]')];
  /* The list keeps a 1x1 role="status" announcer reading "working"; the receipt
     is the widest non-empty live region, so skip the announcer by its text. */
  const receipt = live.find((el) => {
    const text = (el.innerText || '').trim();
    return text !== '' && text !== 'working';
  });
  const marks = [...document.querySelectorAll('[aria-label="new activity"]')];
  const main = document.querySelector('main');
  const band = document.querySelector('div.sticky');
  const bandStyle = band ? getComputedStyle(band) : null;
  const active = document.activeElement;
  const activeStyle = active ? getComputedStyle(active) : null;
  const rows = [...document.querySelectorAll('main button')].filter((b) => rect(b) && rect(b).h > 40);
  return JSON.stringify({
    control: control ? { label: control.innerText.trim(), rect: rect(control) } : null,
    plain: plain ? { label: plain.innerText.trim(), rect: rect(plain) } : null,
    marking: marking ? { label: marking.innerText.trim() } : null,
    receipt: receipt
      ? { role: receipt.getAttribute('role'), text: (receipt.innerText || '').trim(), rect: rect(receipt) }
      : null,
    markCount: marks.length,
    markRects: marks.map(rect),
    scrollTop: main ? Math.round(main.scrollTop) : null,
    scrollHeight: main ? Math.round(main.scrollHeight) : null,
    clientHeight: main ? Math.round(main.clientHeight) : null,
    seenPosts: window.__seenPosts ?? null,
    band: band ? {
      classes: band.className,
      rect: rect(band),
      borderBottomWidth: bandStyle.borderBottomWidth,
      borderBottomColor: bandStyle.borderBottomColor,
      boxShadow: bandStyle.boxShadow,
      zIndex: bandStyle.zIndex,
      position: bandStyle.position,
    } : null,
    rowRects: rows.slice(0, 3).map(rect),
    focus: active ? {
      tag: active.tagName,
      isBand: active === band,
      classes: active.className,
      outlineStyle: activeStyle.outlineStyle,
      outlineWidth: activeStyle.outlineWidth,
      outlineColor: activeStyle.outlineColor,
      outlineOffset: activeStyle.outlineOffset,
    } : null,
  });
})()
"""


def _state(page: Page) -> dict:
    return json.loads(page.js(STATE_JS))


def _tap(page: Page, x: int, y: int) -> None:
    page.send("Input.dispatchMouseEvent", type="mousePressed", x=x, y=y, button="left", clickCount=1)
    time.sleep(0.05)
    page.send("Input.dispatchMouseEvent", type="mouseReleased", x=x, y=y, button="left", clickCount=1)


def _key(page: Page, key: str, code: str, vk: int, *, text: str = "") -> None:
    page.send(
        "Input.dispatchKeyEvent",
        type="keyDown",
        key=key,
        code=code,
        windowsVirtualKeyCode=vk,
        nativeVirtualKeyCode=vk,
        text=text,
    )
    page.send(
        "Input.dispatchKeyEvent",
        type="keyUp",
        key=key,
        code=code,
        windowsVirtualKeyCode=vk,
        nativeVirtualKeyCode=vk,
    )


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


def _tap_control(page: Page, *, wait: float = 3.0) -> dict:
    armed = _state(page)
    target = armed["control"] or armed["plain"]
    if not target:
        raise SystemExit("the control is not mounted; cannot tap")
    rect = target["rect"]
    _tap(page, (rect["left"] + rect["right"]) // 2, (rect["top"] + rect["bottom"]) // 2)
    deadline = time.time() + wait
    settled = _state(page)
    while time.time() < deadline:
        settled = _state(page)
        if settled["receipt"] and not settled["marking"]:
            break
        time.sleep(0.25)
    time.sleep(0.4)
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
        # Survives every reload below, so `seenPosts` is a per-document count.
        page.send("Page.addScriptToEvaluateOnNewDocument", source=COUNTER_JS)

        page.goto(f"{base}/login")
        page.js(
            "(() => { const f = document.querySelector('form');"
            f" f.password.value = {password!r}; f.submit(); return true; }})()"
        )
        time.sleep(2.0)

        # 01 — the pinned band's own edge, at the list's maximum scroll, over the
        # rows it overlaps (design D6). The band's computed border is the number.
        page.goto(f"{base}/#/")
        time.sleep(2.0)
        # Re-arm the pile first: a re-run of this rig against a store that a
        # previous run already cleared must still start from two unread rows.
        _arm(page, base)
        for _ in range(6):
            page.swipe(195, 620, 400)
        time.sleep(0.8)
        page.shot(outdir / "01-band-separator-max-scroll-390x844.png")
        report["steps"].append({"step": "band-separator", **_state(page)})

        # 02 — keyboard activation with a query that registers no card: the focus
        # fallback lands on the band, and the band shows it (design D7 / QA Q-2).
        page.js("(() => { const m = document.querySelector('main'); m.scrollTop = 0; return true; })()")
        time.sleep(0.4)
        _arm(page, base)
        page.js(
            "(() => { const i = document.querySelector('input[placeholder^=\"Search\"]');"
            " i.focus(); return document.activeElement === i; })()"
        )
        time.sleep(0.3)
        page.js(
            "(() => { const i = document.querySelector('input[placeholder^=\"Search\"]');"
            " const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;"
            " setter.call(i, 'zzz-no-such-conversation');"
            " i.dispatchEvent(new Event('input', { bubbles: true })); return true; })()"
        )
        time.sleep(0.6)
        _key(page, "Tab", "Tab", 9)
        time.sleep(0.4)
        tabbed = _state(page)
        _key(page, "Enter", "Enter", 13, text="\r")
        time.sleep(3.0)
        page.shot(outdir / "02-focus-ring-390x844.png")
        report["steps"].append({"step": "focus-ring", "afterTab": tabbed, **_state(page)})

        # 03 — a DEGRADED read, with the POST count for that step read off the
        # wire (agent NIT-2). The TTL'd summaries cache means the degrade must be
        # set > 1 s after the last read, or the tap answers from cache.
        _arm(page, base)
        time.sleep(1.4)
        posts_before = _state(page)["seenPosts"]
        page.js("fetch('/fixture/degrade?on=1').then((r) => r.json())")
        time.sleep(0.4)
        degraded = _tap_control(page)
        report["steps"].append(
            {"step": "degraded-no-count", "postsBeforeStep": posts_before, **_state(page), "settled": degraded}
        )
        page.shot(outdir / "03-degraded-no-count-390x844.png")
        page.js("fetch('/fixture/degrade?on=0').then((r) => r.json())")

        # 04 — the receipt's window across an absence LONGER than its TTL (UX U9).
        _arm(page, base)
        cleared = _tap_control(page)
        started = time.time()
        page.js("(() => { location.hash = '#/s/c0ffee000016'; return true; })()")
        time.sleep(0.6)
        away = _state(page)
        time.sleep(14.0)
        page.js("(() => { location.hash = '#/'; return true; })()")
        time.sleep(1.2)
        returned = _state(page)
        elapsed = round(time.time() - started, 1)
        page.shot(outdir / "04-receipt-window-after-absence-390x844.png")
        report["steps"].append(
            {"step": "receipt-window", "elapsedAwaySeconds": elapsed, "cleared": cleared,
             "whileAway": away, "onReturn": returned, **_state(page)}
        )

        page.close()
    finally:
        chrome.close()

    (outdir / "capture-report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
