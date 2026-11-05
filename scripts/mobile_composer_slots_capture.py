"""Capture the phone composer's action cluster in every state, out of headless Chrome.

Run:  PYTHONPATH=. python scripts/mobile_composer_slots_capture.py \
          PORT OUTDIR LABEL [UNAVAILABLE_PORT]

Needs ``scripts/mobile_composer_slots_fixture.py`` serving on PORT (and, for the
mic-unavailable frame, a second fixture started with ``unavailable`` on the port in
argv[4]). The password comes from ``LOP_MOBILE_FIXTURE_PASSWORD`` and is never
printed. Chrome handling is ``mobile_overflow_capture.Chrome``'s (throwaway
profile, ``--headless=new``, own process group, leak-asserted teardown); one
browser is reused for the whole run.

States: idle-empty, idle-text, working-empty, working-text, stopping, recording,
mic-unavailable. Each is driven through the page's own controls, not by setting
state: recording uses a stubbed ``getUserMedia``/``MediaRecorder`` (headless Chrome
has no microphone) and stopping answers the abort POST with ``ok`` so the
pressed state is the real one.

Viewports are iPhone 15 (390x844, safe-area top 47 / bottom 34), 15 Pro Max
(430x932, 59 / 34) and a 320x640 squeeze check, at DPR 3 with
``navigator.standalone = true``, i.e. the installed home-screen app. Every frame
also records the geometry behind it (the x/width of the mic and of the primary
disc), so "nothing moved" is a number, not a look.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from scripts.mobile_overflow_capture import Chrome, Page, fixture_password

VIEWPORTS = [(390, 844, 47, 34), (430, 932, 59, 34), (320, 640, 47, 34)]

# Injected before the app boots. Headless Chrome has no microphone, so the
# recorder is stubbed at the browser API, which is the seam the composer reads;
# the abort POST is answered ok so the "stopping" state is the real latch.
STUBS = """
(() => {
  Object.defineProperty(navigator, 'standalone', { get: () => true });
  class FakeRecorder {
    constructor(stream, opts) { this.state = 'inactive'; this.mimeType = (opts && opts.mimeType) || 'audio/webm'; }
    static isTypeSupported() { return true; }
    start() { this.state = 'recording'; }
    stop() { this.state = 'inactive'; if (this.onstop) this.onstop(); }
  }
  window.MediaRecorder = FakeRecorder;
  const gum = async () => ({ getTracks: () => [{ stop() {} }] });
  Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: { getUserMedia: gum } });
  const realFetch = window.fetch.bind(window);
  window.fetch = (url, init) => {
    if (String(url).includes('/command') && init && init.method === 'POST') {
      return Promise.resolve(new Response(JSON.stringify({ ok: true, detail: '' }), {
        status: 200, headers: { 'content-type': 'application/json' } }));
    }
    return realFetch(url, init);
  };
})()
"""  # noqa: E501

GEOMETRY = """
(() => {
  const r = (el) => { if (!el) return null; const b = el.getBoundingClientRect();
    return { x: +b.x.toFixed(1), y: +b.y.toFixed(1), w: +b.width.toFixed(1), h: +b.height.toFixed(1) }; };
  const q = (s) => document.querySelector(s);
  const byLabel = (l) => [...document.querySelectorAll('button')].find((b) => b.getAttribute('aria-label') === l);
  const cluster = document.querySelector('[data-testid="composer-actions"]');
  const kids = cluster ? [...cluster.children].map((c) => ({
    label: c.getAttribute('aria-label') || (c.hasAttribute('data-mic-reserve') ? 'reserve' : ''),
    role: c.getAttribute('data-role'), ...r(c) })) : [...document.querySelectorAll('.lo-composer button')]
      .filter((b) => ['send','steer','stop'].includes(b.getAttribute('aria-label')))
      .map((b) => ({ label: b.getAttribute('aria-label'), ...r(b) }));
  const labels = [...document.querySelectorAll('button')].map((b) => b.getAttribute('aria-label') || '');
  return JSON.stringify({
    viewport: [innerWidth, innerHeight],
    cluster: kids,
    mic: r(byLabel('start voice input') || byLabel('stop and transcribe') || byLabel('transcribing')),
    shellBottom: r(q('.lo-composer')),
    workingStop: r(q('[data-working-stop]')),
    workingStopVisible: (() => { const b = q('[data-working-stop]'); if (!b) return null;
      const y = b.getBoundingClientRect(); return y.top >= 0 && y.bottom <= innerHeight; })(),
    transcriptScrollTop: (() => { const s = [...document.querySelectorAll('.lo-scroll')]
      .find((e) => e.scrollHeight > e.clientHeight + 50); return s ? Math.round(s.scrollTop) : null; })(),
    // Any two of these visible at once is the reported defect.
    sendAndStopTogether: labels.includes('stop') && (labels.includes('send') || labels.includes('steer')),
    docScrollW: document.documentElement.scrollWidth,
  });
})()
"""  # noqa: E501


def type_text(page: Page, value: str) -> None:
    page.js(
        "(() => { const t = document.querySelector('textarea');"
        " const set = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set;"
        f" set.call(t, {value!r});"
        " t.dispatchEvent(new Event('input', { bubbles: true })); return true; })()"
    )
    time.sleep(0.4)


def click(page: Page, label: str) -> bool:
    ok = page.js(
        "(() => { const b = [...document.querySelectorAll('button')]"
        f".find((x) => x.getAttribute('aria-label') === {label!r});"
        " if (!b) return false; b.click(); return true; })()"
    )
    time.sleep(0.5)
    return bool(ok)


def login(page: Page, base: str) -> None:
    page.goto(f"{base}/login")
    page.js(
        "(() => { const f = document.querySelector('form');"
        f" f.password.value = {fixture_password()!r}; f.submit(); return true; }})()"
    )
    time.sleep(2.0)


def frames_beyond_the_matrix(page: Page, base: str, vp: str, frame: Any) -> None:
    """The positions the state matrix does not reach (review round 1, m1).

    - scrolled-up: a long transcript read from its top, so the working line's Stop must
      still be on screen (it lives outside the scrolling tail);
    - long-activity: an activity label far wider than the line, so the clock and Stop
      must hold their boxes while the label truncates;
    - keyboard-open: the short layout the soft keyboard produces (the visual viewport is
      emulated at 430 px, under SHORT_COLUMN_PX = 480) with the field focused.
    """
    page.goto(f"{base}/#/s/slots-long")
    time.sleep(1.8)
    frame("8-long-activity", base, "slots-long")
    # Real touch drags toward the top of the transcript, as a thumb would.
    for _ in range(12):
        page.swipe(200, 400, -600)
    time.sleep(0.5)
    page.js(
        "(() => { const s = [...document.querySelectorAll('.lo-scroll')]"
        ".find((e) => e.scrollHeight > e.clientHeight + 50); return s ? s.scrollTop : -1; })()"
    )
    frame("9-scrolled-up", base, "slots-long")
    width = int(vp.split("x")[0])
    page.send(
        "Emulation.setDeviceMetricsOverride",
        width=width,
        height=430,
        deviceScaleFactor=3,
        mobile=True,
    )
    time.sleep(0.8)
    page.js("(() => { document.querySelector('textarea').focus(); return true; })()")
    time.sleep(0.8)
    frame("10-keyboard-open", base, "slots-long")


def main() -> None:
    port, outdir, label = int(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    port_off = int(sys.argv[4]) if len(sys.argv) > 4 else None
    outdir.mkdir(parents=True, exist_ok=True)
    chrome = Chrome()
    report: dict[str, Any] = {}
    try:
        page = Page(chrome.target_ws())
        page.send("Page.addScriptToEvaluateOnNewDocument", source=STUBS)
        for width, height, top, bottom in VIEWPORTS:
            vp = f"{width}x{height}"
            page.send(
                "Emulation.setDeviceMetricsOverride",
                width=width,
                height=height,
                deviceScaleFactor=3,
                mobile=True,
            )
            page.send(
                "Emulation.setSafeAreaInsetsOverride",
                insets={"top": top, "bottom": bottom, "left": 0, "right": 0},
            )

            def frame(state: str, base: str, session: str) -> None:
                name = f"{label}-{vp}-{state}"
                page.shot(outdir / f"{name}.png")
                report[name] = json.loads(page.js(GEOMETRY))

            base = f"http://127.0.0.1:{port}"
            login(page, base)
            page.goto(f"{base}/#/s/slots-idle")
            time.sleep(1.5)
            frame("1-idle-empty", base, "slots-idle")
            type_text(page, "Also make the stop button easier to hit")
            frame("2-idle-text", base, "slots-idle")
            type_text(page, "")

            # Recording (stubbed recorder), on the idle conversation.
            click(page, "start voice input")
            time.sleep(1.2)
            frame("6-recording", base, "slots-idle")
            click(page, "cancel recording")

            page.goto(f"{base}/#/s/slots-working")
            time.sleep(1.5)
            time.sleep(0.7)  # past the 500 ms arrival guard so the press is a decision
            frame("3-working-empty", base, "slots-working")
            type_text(page, "Also check the narrow phone")
            frame("4-working-text", base, "slots-working")
            type_text(page, "")
            # stop: working-line Stop when the build has one, else the composer disc
            stopped = click(page, "stop the agent") or click(page, "stop")
            time.sleep(0.6)
            report[f"{label}-{vp}-5-stopping"] = {"pressed": stopped}
            frame("5-stopping", base, "slots-working")

            if port_off is not None:
                page.goto(f"http://127.0.0.1:{port_off}/login")
                login(page, f"http://127.0.0.1:{port_off}")
                page.goto(f"http://127.0.0.1:{port_off}/#/s/slots-idle")
                time.sleep(1.5)
                frame("7-mic-unavailable", base, "slots-idle")
            frames_beyond_the_matrix(page, base, vp, frame)
        page.close()
    finally:
        chrome.close()
    (outdir / f"{label}-geometry.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {len(report)} records to {outdir}")


if __name__ == "__main__":
    main()
