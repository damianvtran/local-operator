"""Capture the session list at 390x844 with two unread rows, and sweep the DOM
for any mark-all control — the evidence for issue #2016's "no one-gesture
clear" claim.

Run (from the worktree, so ``scripts`` resolves to it):
    PYTHONPATH=. LOP_MOBILE_FIXTURE_PASSWORD=<same as fixture> \
      .venv/bin/python "$LOCAL_OPERATOR_SCRATCHPAD/mobile-2016/scripts/mobile_2016_capture.py" <port> <outdir>

Reuses the shared rig's own Chrome/Page (``scripts/mobile_overflow_capture.py``):
headless=new, a unique mktemp profile, ``--use-mock-keychain``, a Chrome-chosen
debugging port read from DevToolsActivePort, an Emulation device-metrics
viewport (390x844, dpr 2), and a teardown that ASSERTS zero leftover processes.
No product edits; this script only reads.

Outputs into <outdir>:
  01-list-390x844.png            the list as it loads (before any scrolling)
  dom-list-initial.json          full control inventory + bulk-control scan
  02-list-unread-visible.png     only if real finger swipes were needed
  dom-list-final.json            post-scroll state (same scan)
  list-innertext.txt             the rendered text of the list screen
  03-session-390x844.png         one session screen (scope: "anywhere in the client")
  dom-session.json               its control inventory + bulk scan
  capture-report.json            summary (row marks, visibility, matches)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from scripts.mobile_overflow_capture import Chrome, Page, fixture_password

#: One scan of every interactive control plus the unread marks. Returned as a
#: JSON string so `Runtime.evaluate`'s returnByValue carries it whole.
#: `bulkRe` is the CLAIM's own vocabulary (the words a mark-all control would
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
        page.shot(outdir / "01-list-390x844.png")
        initial = json.loads(page.js(SWEEP_JS))
        (outdir / "dom-list-initial.json").write_text(json.dumps(initial, indent=2))
        (outdir / "list-innertext.txt").write_text(page.js("document.body.innerText"))
        report["steps"].append(
            {
                "step": "list-initial",
                "title": initial["title"],
                "control_count": initial["controlCount"],
                "bulk_control_matches": initial["bulkControlMatches"],
                "bulk_text_matches": initial["bulkTextMatches"],
                "new_marks": initial["newMarks"],
            }
        )

        # If a seeded row sits below the fold, walk it up with REAL finger drags
        # (never scrollTop assignment — the rig's own rule: reachable by script
        # is not reachable by thumb).
        sweeps = 0
        final = initial
        while sweeps < 6 and final["newMarks"] and not all(m["visible"] for m in final["newMarks"]):
            page.swipe(195, 620, 420)
            time.sleep(0.4)
            final = json.loads(page.js(SWEEP_JS))
            sweeps += 1
        if sweeps:
            page.shot(outdir / "02-list-unread-visible.png")
            report["steps"].append(
                {
                    "step": "list-after-swipes",
                    "swipes": sweeps,
                    "new_marks": final["newMarks"],
                    "bulk_control_matches": final["bulkControlMatches"],
                }
            )
        (outdir / "dom-list-final.json").write_text(json.dumps(final, indent=2))

        # Scope check: one session screen too — "anywhere in the client".
        first_id = final["newMarks"][0]["row"] if final["newMarks"] else "c0ffee000016"
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
        report["first_unread_row"] = first_id

        page.close()
    finally:
        chrome.close()

    (outdir / "capture-report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "steps"}, indent=2))


if __name__ == "__main__":
    main()
