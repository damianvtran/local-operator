"""Post-fix capture for issue #2016: the session list at 390x844 with two unread
rows AND the new `mark all as read` control, a REAL pointer tap on that control,
and the settled frame after the pile clears — plus the DOM inventories behind
each claim.

Run (from the worktree, so ``scripts`` resolves to it):
    PYTHONPATH=. LOP_MOBILE_FIXTURE_PASSWORD=<same as fixture> \
      .venv/bin/python "$LOCAL_OPERATOR_SCRATCHPAD/mobile-2016/postfix/scripts/mobile_2016_capture.py" <port> <outdir>

Reuses the shared rig's own Chrome/Page (``scripts/mobile_overflow_capture.py``):
headless=new, a unique mktemp profile, ``--use-mock-keychain``, a Chrome-chosen
debugging port read from DevToolsActivePort, an Emulation device-metrics
viewport (390x844, dpr 2), and a teardown that ASSERTS zero leftover processes.

THE TAP IS A REAL POINTER PAIR (CDP ``Input.dispatchMouseEvent`` pressed +
released at the control's measured centre), not a scripted ``el.click()``: the
press must land on the element the layout actually paints, or the capture would
prove something a finger cannot do.

Outputs into <outdir>:
  01-list-unread-with-control-390x844.png   both unread rows + the control
  dom-list-initial.json                     control inventory + bulk scan + marks
  list-innertext-initial.txt                the rendered text before the tap
  02-list-marked-390x844.png                the settled frame after the tap
  dom-list-marked.json                      post-tap inventory (marks gone, receipt)
  list-innertext-marked.txt                 the rendered text after the tap
  03-session-390x844.png                    one session screen (scope: "anywhere")
  dom-session.json                          its inventory + bulk scan
  capture-report.json                       summary (control rect, marks, receipt)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from scripts.mobile_overflow_capture import Chrome, Page, fixture_password

#: One scan of every interactive control plus the unread marks. Returned as a
#: JSON string so `Runtime.evaluate`'s returnByValue carries it whole.
#: `bulkRe` is the claim's own vocabulary (the words a mark-all control would
#: carry); the full `controls` list rides along so a reader audits the sweep
#: itself rather than trusting the regex.
SWEEP_JS = r"""
(() => {
  const bulkRe = /mark\s*all|read\s*all|clear\s*all|dismiss\s*all|mark-?all|seen\s*all/i;
  const controls = [...document.querySelectorAll('button, a[href], [role="button"], input[type="submit"]')].map((el) => ({
    tag: el.tagName.toLowerCase(),
    text: (el.innerText || '').trim().slice(0, 120),
    aria: el.getAttribute('aria-label'),
    title: el.getAttribute('title'),
    testid: el.getAttribute('data-testid'),
    disabled: el.disabled === true,
  }));
  const inViewport = (r) => r.top >= -1 && r.bottom <= innerHeight + 1 && r.left >= -1 && r.right <= innerWidth + 1;
  const newMarks = [...document.querySelectorAll('[aria-label="new activity"]')].map((el) => {
    const row = el.closest('button');
    const rect = el.getBoundingClientRect();
    return {
      row: row ? (row.innerText || '').split('\n').map((s) => s.trim()).filter(Boolean).slice(0, 2).join(' | ') : null,
      rect: { top: Math.round(rect.top), bottom: Math.round(rect.bottom), left: Math.round(rect.left) },
      visible: inViewport(rect),
    };
  });
  const bulk = controls.filter((c) => bulkRe.test([c.text, c.aria, c.title].filter(Boolean).join(' ')));
  const textLines = document.body.innerText.split('\n').map((s) => s.trim()).filter((l) => bulkRe.test(l));
  return JSON.stringify({
    url: location.href,
    title: document.title,
    viewport: [innerWidth, innerHeight],
    controlCount: controls.length,
    controls,
    bulkControlMatches: bulk,
    bulkTextMatches: textLines,
    newMarks,
  });
})()
"""

#: The control's own geometry, measured from the element the finger would hit.
CONTROL_JS = r"""
(() => {
  const el = [...document.querySelectorAll('button')].find(
    (b) => (b.innerText || '').trim().toLowerCase() === 'mark all as read'
  );
  if (!el) return 'null';
  const r = el.getBoundingClientRect();
  return JSON.stringify({
    x: Math.round(r.left + r.width / 2),
    y: Math.round(r.top + r.height / 2),
    top: Math.round(r.top),
    left: Math.round(r.left),
    width: Math.round(r.width),
    height: Math.round(r.height),
  });
})()
"""

#: The settled-state probe the tap loop polls: marks gone AND the receipt shown.
STATE_JS = r"""
(() => {
  const text = document.body.innerText || '';
  const marks = document.querySelectorAll('[aria-label="new activity"]').length;
  const control = [...document.querySelectorAll('button')].some(
    (b) => (b.innerText || '').trim().toLowerCase() === 'mark all as read'
  );
  const receipt = [...document.querySelectorAll('p')].some((p) => /^Marked \d+ read\./.test((p.innerText || '').trim()));
  return JSON.stringify({
    marks,
    control,
    receipt,
    tail: text.split('\n').map((s) => s.trim()).filter(Boolean).slice(-6),
  });
})()
"""


def _tap(page: Page, x: int, y: int) -> None:
    """A real pointer pair at (x, y): what a thumb's tap dispatches."""
    page.send("Input.dispatchMouseEvent", type="mousePressed", x=x, y=y, button="left", clickCount=1)
    time.sleep(0.08)
    page.send("Input.dispatchMouseEvent", type="mouseReleased", x=x, y=y, button="left", clickCount=1)


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

        # Log in first: the cookie is per profile; a re-navigation to /login
        # after an auth cookie exists just redirects (same rule the rig uses).
        page.goto(f"{base}/login")
        page.js(
            "(() => { const f = document.querySelector('form');"
            f" f.password.value = {password!r}; f.submit(); return true; }})()"
        )
        time.sleep(2.0)

        page.goto(f"{base}/#/")
        # goto sleeps 2s; give the SSE its first list frame and any settle one
        # more beat before the shutter (consecutive state == what the user sees).
        time.sleep(1.5)
        page.shot(outdir / "01-list-unread-with-control-390x844.png")
        initial = json.loads(page.js(SWEEP_JS))
        (outdir / "dom-list-initial.json").write_text(json.dumps(initial, indent=2))
        (outdir / "list-innertext-initial.txt").write_text(page.js("document.body.innerText"))
        control = json.loads(page.js(CONTROL_JS))
        if control == "null":
            raise SystemExit(
                "the mark-all control is NOT in the DOM — the fix is absent or the list did not load"
            )
        report["steps"].append(
            {
                "step": "list-unread-with-control",
                "title": initial["title"],
                "control_count": initial["controlCount"],
                "bulk_control_matches": initial["bulkControlMatches"],
                "bulk_text_matches": initial["bulkTextMatches"],
                "new_marks": initial["newMarks"],
                "control_rect": control,
            }
        )

        # THE GESTURE: one real tap on the control's measured centre.
        _tap(page, control["x"], control["y"])
        settled: dict = {}
        deadline = time.time() + 15
        while time.time() < deadline:
            settled = json.loads(page.js(STATE_JS))
            if settled["marks"] == 0 and settled["receipt"]:
                break
            time.sleep(0.5)
        time.sleep(0.4)  # one settle beat so the frame is the resting state
        page.shot(outdir / "02-list-marked-390x844.png")
        marked = json.loads(page.js(SWEEP_JS))
        (outdir / "dom-list-marked.json").write_text(json.dumps(marked, indent=2))
        (outdir / "list-innertext-marked.txt").write_text(page.js("document.body.innerText"))
        report["steps"].append(
            {
                "step": "list-marked",
                "marks_after_tap": settled["marks"],
                "receipt_text": settled["tail"],
                "control_after_tap": settled["control"],
                "bulk_control_matches": marked["bulkControlMatches"],
            }
        )

        # Scope check: one session screen too — "anywhere in the client".
        page.goto(f"{base}/#/s/c0ffee000016")
        time.sleep(1.5)
        page.shot(outdir / "03-session-390x844.png")
        session = json.loads(page.js(SWEEP_JS))
        (outdir / "dom-session.json").write_text(json.dumps(session, indent=2))
        report["steps"].append(
            {
                "step": "session-screen",
                "title": session["title"],
                "control_count": session["controlCount"],
                "bulk_control_matches": session["bulkControlMatches"],
                "bulk_text_matches": session["bulkTextMatches"],
            }
        )

        page.close()
    finally:
        chrome.close()

    (outdir / "capture-report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "steps"}, indent=2))


if __name__ == "__main__":
    main()
