"""Capture the phone's QUEUED-ASK surfaces out of headless Chrome over CDP.

Run:  .venv/bin/python scripts/mobile_asks_capture.py <port> <outdir> <label> [--empty-port N]

Sibling of ``mobile_overflow_capture.py`` and deliberately a separate script
rather than a new branch inside it: that harness measures REACHABILITY (does a
real gesture reach every region of a card), and the ask surfaces are about
STATE (which of the two R7 states is mounted, what the composer does, what the
count says). Folding the two would put a scroll-honesty assertion in front of
every ask frame. The driver — ``Chrome``/``Page`` and the per-run password
contract — is IMPORTED from that file, so there is one headless-Chrome launch
recipe and one login path, not two.

AGENTS.md §6 applies and is followed by the imported driver: a unique profile
under the scratch root, ``--remote-debugging-port=0`` with the real port read
from ``DevToolsActivePort``, ``--headless=new`` (a headful window steals the
operator's focus), the viewport set by ``Emulation.setDeviceMetricsOverride``,
``start_new_session=True`` so the browser is one process GROUP, and an ``atexit``
sweep that ASSERTS zero leftovers. One browser instance for the whole run, as
the rig-hygiene rule requires: a fresh launch per case is what made the fleet
allocate gigabytes of profile copies.

WHAT EACH FRAME IS FOR (design §5.0/§5.3, R7):

* ``-bar``      — MINIMIZED: the chip above the composer, with the count and the
  head ask. The composer beneath it is an ordinary conversation composer.
* ``-sheet``    — EXPANDED: the asks sheet, every state on one scroll.
* ``-filled``   — the multi-question form with answers chosen, so the send
  control's enabled state is visible rather than asserted.
* ``-answered`` — the SAME frame after the runtime accepted the answer: the card
  is a receipt and the count fell, with no reload — the live-update half.
* ``-refusal``  — the queue's own sentence ("already answered by desktop.") in
  the card, from the single-winner race.
* ``-loading``  — the read in flight (the fetch is stalled for the frame, so the
  state is reproducible rather than a timing race).
* ``-empty``    — the aggregate answering empty (second fixture, empty index).
* ``-settled``  — a conversation at ZERO outstanding asks: no bar, while its
  transcript still carries the settling rows.
* ``-card``     — that transcript row expanded: the response card.
* ``-none``     — the negative control: a session that never had asks.

Geometry is recorded beside every frame (dock rect, composer visibility, sheet
extents, card count), because the stills show the symptom and the numbers show
the cause: a bar that is present but not inside the viewport, or a sheet whose
form scrolled past the composer, is invisible in a screenshot and obvious here.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

# THE DOCUMENTED RUN LINE HAS TO WORK (agent review round 1, R2). Without this,
# `.venv/bin/python scripts/mobile_asks_capture.py …` — the line this file's own
# docstring gives — died on `ModuleNotFoundError: No module named 'scripts'`
# before `main()` was reached, so nobody could re-derive the evidence. Same
# two-line bootstrap as `scripts/analytics_collapse_probe.py`; the alternative
# (documenting `PYTHONPATH=.`) would leave the Run line a lie.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.mobile_overflow_capture import Chrome, Page, fixture_password  # noqa: E402

VIEWPORTS = [(390, 844), (360, 780)]

#: The dock and the composer it sits above. `composerVisible` is the R7 claim
#: that matters: with the ask surface MINIMIZED the conversation composer is a
#: live control, and a bar that pushed it off the screen would make the rule
#: vacuous.
BAR_PROBE = """
(() => {
  const dock = document.querySelector('[data-testid="ask-dock"]');
  const card = document.querySelector('[data-testid="pending-card"]');
  const ta = document.querySelector('textarea');
  const rect = (el) => {
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return {top: Math.round(r.top), bottom: Math.round(r.bottom), h: Math.round(r.height)};
  };
  return JSON.stringify({
    viewport: [window.innerWidth, window.innerHeight],
    dock: rect(dock),
    dockText: dock ? dock.textContent.trim().slice(0, 90) : null,
    // A mirrored ask must NOT be drawn as a card beside the queued surface
    // (design §4, client rule N3). The fixture publishes the mirror on purpose.
    mirroredCardDrawn: card ? card.getAttribute('data-testid') : null,
    composer: rect(ta),
    composerVisible: (() => {
      if (!ta) return null;
      const r = ta.getBoundingClientRect();
      return r.bottom <= window.innerHeight + 1 && r.top >= 0;
    })(),
    headerAskEntry: [...document.querySelectorAll('button')]
      .filter((b) => /queued asks in this session/.test(b.getAttribute('aria-label') || ''))
      .length,
  });
})()
"""

SHEET_PROBE = """
(() => {
  const dialog = document.querySelector('[role="dialog"]');
  const cards = [...document.querySelectorAll('[data-testid="ask-card"]')];
  const scroller = dialog ? dialog.querySelector('.lo-scroll') : null;
  return JSON.stringify({
    open: Boolean(dialog),
    loading: dialog ? /reading asks/.test(dialog.textContent) : false,
    empty: dialog ? /nothing waiting/.test(dialog.textContent) : false,
    cards: cards.map((c) => ({
      askId: c.getAttribute('data-ask-id'),
      status: c.getAttribute('data-ask-status'),
      controls: c.querySelectorAll('button').length,
      // The refusal is the QUEUE's sentence (design §2.4), so it is read back
      // verbatim rather than described: the evidence for "refusals keep the
      // queue's own sentences" is the string itself.
      refusal: (() => {
        const line = c.querySelector('p.text-danger');
        return line ? line.textContent.trim() : null;
      })(),
      sendDisabled: (() => {
        const send = [...c.querySelectorAll('button')]
            .find((b) => /send answer/.test(b.textContent));
        return send ? send.disabled : null;
      })(),
    })),
    scroller: scroller
      ? {scrollH: scroller.scrollHeight, clientH: scroller.clientHeight}
      : null,
  });
})()
"""

#: What the head ask's own controls say right now, and whether the sheet is
#: mounted — the two facts the collapse/reopen step turns on.
REOPEN_PROBE = """
(() => {
  const dialog = document.querySelector('[role="dialog"]');
  const card = document.querySelector('[data-testid="ask-card"][data-ask-id="qa-head"]');
  if (!card) return JSON.stringify({ dialog: Boolean(dialog), card: false });
  const buttons = [...card.querySelectorAll('button')];
  const send = buttons.find((b) => /send answer/.test(b.textContent));
  return JSON.stringify({
    dialog: Boolean(dialog),
    card: true,
    pressed: buttons
      .filter((b) => !/skip|decline|dismiss|send/.test(b.textContent))
      .map((b) => b.getAttribute('aria-pressed')),
    sendDisabled: send ? send.disabled : null,
  });
})()
"""

#: Fill the HEAD ask's questions (the oldest open one), one option per question,
#: so the multi-question form's completeness rule can be seen switching the send
#: control on. Scoped to the card by its ask id rather than by position.
FILL_HEAD = """
(() => {
  const card = document.querySelector('[data-testid="ask-card"][data-ask-id="qa-head"]');
  if (!card) return "no head card";
  const buttons = [...card.querySelectorAll('button')];
  const pickers = buttons.filter((b) => !/skip|decline|dismiss|send/.test(b.textContent));
  if (pickers.length < 2) return `only ${pickers.length} pickers`;
  // The head ask has two questions, three options each, in DOM order: take the
  // first option of question 1 and the first of question 2, so the form is
  // complete and the send control turns on.
  pickers[0].click();
  pickers[3].click();
  return `picked ${pickers.length} pickers`;
})()
"""

SUBMIT_HEAD = """
(() => {
  const card = document.querySelector('[data-testid="ask-card"][data-ask-id="qa-head"]');
  const send = [...card.querySelectorAll('button')].find((b) => /send answer/.test(b.textContent));
  if (!send) return "no send control";
  if (send.disabled) return "send disabled — the form is incomplete";
  send.click();
  return "sent";
})()
"""

#: The single-winner race, in TWO steps. They cannot be one ``Runtime.evaluate``:
#: React has not re-rendered after the option click, so the send control is still
#: disabled at that moment and the click lands on nothing — which is what the
#: first version of this step did, and the frame showed an untouched card while
#: the log claimed it had sent.
PICK_DEADLINE = """
(() => {
  const card = document.querySelector('[data-testid="ask-card"][data-ask-id="qa-deadline"]');
  if (!card) return "no deadline card";
  const picker = [...card.querySelectorAll('button')]
    .find((b) => !/skip|decline|dismiss|send/.test(b.textContent));
  if (!picker) return "no picker";
  picker.click();
  return "picked";
})()
"""

SUBMIT_DEADLINE = """
(() => {
  const card = document.querySelector('[data-testid="ask-card"][data-ask-id="qa-deadline"]');
  const send = [...card.querySelectorAll('button')].find((b) => /send answer/.test(b.textContent));
  if (!send) return "no send control";
  if (send.disabled) return "send disabled — the form is incomplete";
  send.click();
  return "sent";
})()
"""

STALL_AGGREGATE = """
(() => {
  const real = window.fetch;
  window.fetch = (input, init) => {
    const url = typeof input === "string" ? input : (input && input.url) || "";
    if (url.includes("/api/asks")) return new Promise(() => {});
    return real(input, init);
  };
  return "stalled";
})()
"""

TAP_ROW = """
(() => {
  const dock = document.querySelector('[data-testid="ask-dock"]');
  if (dock) { dock.click(); return "dock"; }
  const entry = [...document.querySelectorAll('button')]
    .find((b) => /queued asks in this session/.test(b.getAttribute('aria-label') || ''));
  if (entry) { entry.click(); return "header"; }
  return "no entry";
})()
"""

EXPAND_ROW = """
(() => {
  const row = document.querySelector('[data-testid="ask-row"]');
  if (!row) return "no row";
  const button = row.querySelector('button');
  if (button) button.click();
  return "expanded";
})()
"""


#: Bring one ask card into the sheet's own viewport with a REAL finger drag
#: (never `scrollIntoView`, which can move a region a thumb cannot — the rule the
#: overflow rig documents). The returned geometry says what the drag had to
#: cover, so the frame's caption can state it.
SCROLL_TO_CARD_JS = """
((askId) => {
  const panel = document.querySelector('[role="dialog"] .lo-scroll');
  const card = document.querySelector('[data-testid="ask-card"][data-ask-id="' + askId + '"]');
  if (!panel || !card) return null;
  const p = panel.getBoundingClientRect();
  const c = card.getBoundingClientRect();
  return JSON.stringify({
    dy: Math.round(c.top - p.top - 8),
    x: Math.round(p.left + p.width / 2),
    y: Math.round(p.top + Math.min(40, p.height / 3)),
    scrollTop: Math.round(panel.scrollTop),
    max: Math.round(panel.scrollHeight - panel.clientHeight),
  });
})
"""


def scroll_to_card(page: Page, ask_id: str) -> dict[str, Any] | None:
    """Drag the sheet until `ask_id`'s card sits near the top of its scroller.

    Returns the geometry it acted on (and what it could not cover), because a
    capture that claims a sentence is visible must be able to prove the card was
    in view when the shutter opened.
    """
    expression = f"({SCROLL_TO_CARD_JS.strip()})({json.dumps(ask_id)})"
    raw = page.js(expression)
    if raw is None:
        return None
    geo = json.loads(raw)
    for _ in range(6):
        if abs(geo["dy"]) < 12:
            break
        page.swipe(geo["x"], geo["y"], geo["dy"])
        raw = page.js(expression)
        if raw is None:
            return None
        geo = json.loads(raw)
    geo["remaining"] = geo["dy"]
    return geo


def main() -> None:
    port, outdir, label = int(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    empty_port = 0
    viewports = VIEWPORTS
    if "--empty-port" in sys.argv:
        empty_port = int(sys.argv[sys.argv.index("--empty-port") + 1])
    if "--viewport" in sys.argv:
        # ONE VIEWPORT PER RUN, pointed at a FRESH fixture: the ask flow
        # SETTLES an ask, so a second viewport against the same daemon would
        # photograph a queue the first pass had already answered.
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

            # 1. MINIMIZED (R7): the bar, with the conversation composer live.
            page.goto(f"{base}/#/s/asks")
            time.sleep(1.5)
            page.shot(outdir / f"{label}-{vp}-bar.png")
            report[f"{vp}-bar"] = json.loads(page.js(BAR_PROBE))

            # 2. EXPANDED: the sheet, every queued state on one scroll.
            page.js(TAP_ROW)
            time.sleep(1.2)
            page.shot(outdir / f"{label}-{vp}-sheet.png")
            report[f"{vp}-sheet"] = json.loads(page.js(SHEET_PROBE))

            # 2a. The recommendation badge (design round 1, D6): the head ask's
            #     first option carries "· recommended", which no frame showed. The
            #     card is dragged into the panel's own viewport with a real touch
            #     gesture first.
            report[f"{vp}-recommended"] = {"drag": scroll_to_card(page, "qa-head")}
            time.sleep(0.6)
            page.shot(outdir / f"{label}-{vp}-recommended.png")
            report[f"{vp}-recommended"].update(json.loads(page.js(SHEET_PROBE)))

            # 2b. The FOREIGN row (design round 1, D6): the sheet opened from a
            #     conversation that is not the one most of its rows belong to, which
            #     is the case the sheet exists for and the case no frame showed.
            page.goto(f"{base}/#/s/asks-foreign")
            time.sleep(1.5)
            page.js(TAP_ROW)
            time.sleep(1.5)
            page.shot(outdir / f"{label}-{vp}-foreign.png")
            report[f"{vp}-foreign"] = json.loads(page.js(SHEET_PROBE))

            # 2c. The BUSIEST STACK (design round 1, D6): todos, a running roster,
            #     an approval card AND the ask chip sharing one column.
            page.goto(f"{base}/#/s/asks-stacked")
            time.sleep(1.5)
            page.shot(outdir / f"{label}-{vp}-stacked.png")
            report[f"{vp}-stacked"] = json.loads(page.js(BAR_PROBE))

            # 2d. COLLAPSE AND RETURN (QA round 1, Q-1 = UX round 1, U1): pick the
            #     head ask's options, close the sheet on the scrim, reopen it. The
            #     draft must survive — §5.0-R7 keeps BOTH buffers, and the chat
            #     buffer always did. The probe reads `aria-pressed` back, so the
            #     frame's claim is the DOM's, not the script's.
            page.goto(f"{base}/#/s/asks")
            time.sleep(1.5)
            page.js(TAP_ROW)
            time.sleep(1.2)
            picked = page.js(FILL_HEAD)
            time.sleep(0.4)
            page.js(
                "(() => { const scrim = document.querySelector('[role=\"dialog\"]');"
                " const close = scrim && scrim.querySelector('button[aria-label]');"
                " if (close) { close.click(); return 'closed'; } return 'no close'; })()"
            )
            time.sleep(0.8)
            collapsed = json.loads(page.js(REOPEN_PROBE))
            page.js(TAP_ROW)
            time.sleep(1.2)
            report[f"{vp}-reopened"] = {
                "fill": picked,
                "collapsed": collapsed,
                "reopened": json.loads(page.js(REOPEN_PROBE)),
            }
            page.shot(outdir / f"{label}-{vp}-reopened.png")

            # 3. The multi-question form, still filled after the collapse and
            #    return — the picks made in 2d are the frame's subject, so nothing
            #    is clicked here: re-clicking would TOGGLE them off (which is what
            #    a draft that survived a collapse looks like).
            time.sleep(0.4)
            page.shot(outdir / f"{label}-{vp}-filled.png")
            report[f"{vp}-filled"] = {
                "draft": json.loads(page.js(REOPEN_PROBE)),
                **json.loads(page.js(SHEET_PROBE)),
            }

            # 4. Send it, and watch the card become a receipt with no reload.
            sent = page.js(SUBMIT_HEAD)
            time.sleep(2.5)
            page.shot(outdir / f"{label}-{vp}-answered.png")
            report[f"{vp}-answered"] = {"submit": sent, **json.loads(page.js(SHEET_PROBE))}

            # 5. The single-winner refusal: the queue's own sentence, verbatim.
            page.js(PICK_DEADLINE)
            time.sleep(0.5)
            page.js(SUBMIT_DEADLINE)
            time.sleep(2.5)
            # THE FRAME MUST SHOW THE SENTENCE IT EXISTS FOR (QA round 1, Q-3 =
            # design round 1, D4). The refusal used to render below the fold of the
            # very card that produced it, so the PNG showed a pressed option and no
            # error. The card's own error line now sits with its controls, and this
            # drags the card into the panel's viewport before the shutter.
            report[f"{vp}-refusal"] = {"drag": scroll_to_card(page, "qa-deadline")}
            time.sleep(0.6)
            page.shot(outdir / f"{label}-{vp}-refusal.png")
            report[f"{vp}-refusal"].update(json.loads(page.js(SHEET_PROBE)))

            # 6. The read in flight, made reproducible by stalling the fetch.
            # A RELOAD, not a `goto`: the sheet's open state and the previous
            # frame's rows live in the session screen, which a hash-only
            # navigation does not remount — the first version of this step
            # therefore photographed the previous step's sheet and called it
            # "loading".
            page.js("location.reload()")
            time.sleep(2.5)
            page.js(STALL_AGGREGATE)
            page.js(TAP_ROW)
            time.sleep(0.8)
            page.shot(outdir / f"{label}-{vp}-loading.png")
            report[f"{vp}-loading"] = json.loads(page.js(SHEET_PROBE))

            # 7. Zero outstanding asks: no bar, but the settling rows remain.
            page.goto(f"{base}/#/s/asks-settled")
            time.sleep(1.5)
            page.shot(outdir / f"{label}-{vp}-settled.png")
            report[f"{vp}-settled"] = json.loads(page.js(BAR_PROBE))
            expanded = page.js(EXPAND_ROW)
            time.sleep(0.4)
            page.shot(outdir / f"{label}-{vp}-card.png")
            report[f"{vp}-card"] = {
                "expand": expanded,
                "rows": page.js(
                    "(() => document.querySelectorAll('[data-testid=\"ask-row\"]').length)()"
                ),
            }

            # 8. Negative control: a conversation that never had an ask.
            page.goto(f"{base}/#/s/roster")
            time.sleep(1.5)
            page.shot(outdir / f"{label}-{vp}-none.png")
            report[f"{vp}-none"] = json.loads(page.js(BAR_PROBE))

            # 9. The aggregate answering empty (the index has no rows at all).
            if empty_port:
                page.goto(f"http://127.0.0.1:{empty_port}/login")
                page.js(
                    "(() => { const f = document.querySelector('form');"
                    f" f.password.value = {fixture_password()!r}; f.submit(); return true; }})()"
                )
                time.sleep(2.0)
                page.goto(f"http://127.0.0.1:{empty_port}/#/s/asks")
                time.sleep(1.5)
                page.js(TAP_ROW)
                time.sleep(1.5)
                page.shot(outdir / f"{label}-{vp}-empty.png")
                report[f"{vp}-empty"] = json.loads(page.js(SHEET_PROBE))

            # 10. THE LIGHT PASS (QA round 1, Q-4 = design round 1, D6). This client
            #     ships 31 themes and the design asks for light where the surface has
            #     both; the shipped set was dark only. The client's own key is the
            #     switch, and the reload is what applies it.
            page.goto(f"{base}/#/s/asks")
            page.js(
                "(() => { localStorage.setItem('lo-mobile-theme', 'localOperatorLight');"
                " location.reload(); return true; })()"
            )
            # The reload keeps the URL, so the screen is already the asks session.
            time.sleep(2.5)
            page.shot(outdir / f"{label}-{vp}-bar-light.png")
            report[f"{vp}-bar-light"] = json.loads(page.js(BAR_PROBE))
            page.js(TAP_ROW)
            time.sleep(1.5)
            page.shot(outdir / f"{label}-{vp}-sheet-light.png")
            report[f"{vp}-sheet-light"] = json.loads(page.js(SHEET_PROBE))
        page.close()
    finally:
        chrome.close()
    (outdir / f"{label}-asks-geometry.json").write_text(json.dumps(report, indent=2))
    for key, geo in report.items():
        if "cards" in geo:
            print(
                f"{key:<22} open={geo['open']} loading={geo['loading']} empty={geo['empty']} "
                "cards="
                + repr(
                    [
                        (c["askId"], c["status"], c["sendDisabled"], c["refusal"])
                        for c in geo["cards"]
                    ]
                )
            )
        elif "dock" in geo:
            print(
                f"{key:<22} dock={geo['dock'] and geo['dock']['top']} "
                f"composerVisible={geo['composerVisible']} mirroredCard={geo['mirroredCardDrawn']} "
                f"headerEntry={geo['headerAskEntry']}"
            )


if __name__ == "__main__":
    main()
