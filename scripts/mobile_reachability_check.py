"""Assert the phone's REACHABILITY contract with real touch input.

Run:  PYTHONPATH=. .venv/bin/python scripts/mobile_reachability_check.py <port>

This is the layer that proves what the vitest suite structurally cannot. The
web tests run under happy-dom, which does no layout: every box is 0x0, so
"approve is 44px tall on arrival" and "a finger reaches option-10" are not
questions that environment can answer. It can only assert the STRUCTURE that
produces the layout. Round 1 of review on PR #1018 is the cautionary case —
four of seven assertions compared a class name in the test to the same class
name in the source, which passes for a cap that is present but ineffective, and
an approval card shipped with its primary action 0px visible at 360x780.

So the properties below are measured, in a real engine, after real gestures:

  R1  a capped card's cap tracks the COLUMN, not the dynamic viewport, so it
      still binds while the keyboard is open (C1/U1)
  R2  the primary action is fully visible ON ARRIVAL, at or above the 44px iOS
      tap-target floor, before any gesture (U2/Q1)
  R3  the last option of a long list is reachable BY GESTURE (the original
      reported defect)
  R4  the stale-tap error is visible where it renders, not below the fold (U3)
  R5  with panels and a pending card stacked, every control is finger-reachable
      and nothing is clipped (D1)

R1's test is the important one to get right, and it is easy to get wrong. The
obvious way to simulate a keyboard — shrinking the viewport with
``Emulation.setDeviceMetricsOverride`` — moves the layout viewport AND the
visual viewport together, so ``dvh`` shrinks along with the column and a broken
``dvh`` cap looks fine. That artefact hid this defect from a review round's
first attempt. A real iOS keyboard is an OVERLAY: it shrinks
``visualViewport.height`` and leaves the dynamic viewport untouched. CDP cannot
express that divergence directly (``Emulation.setVisibleSize`` is accepted and
does nothing observable on current headless Chrome — measured), so this script
reproduces the geometry the way the app itself does: it pins the column to a
keyboard-reduced height exactly as ``screens/session-view.tsx`` does on a
``visualViewport`` resize, leaving device metrics at full height so ``dvh`` is
unchanged, and then measures. That is the divergence, driven through the app's
own mechanism rather than a browser flag.

Follows AGENTS.md §6 for the browser itself; see mobile_overflow_capture.py,
whose Chrome/Page helpers this reuses rather than duplicating.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from scripts.mobile_overflow_capture import (
    PASSWORD,
    VIEWPORTS,
    Chrome,
    Page,
    gesture_scroll_all,
)

# The tap-target floor the Button primitive documents, and the number U2 failed
# against: 30px of a 44px control reads as tappable while sitting under it.
TAP_FLOOR = 44

# Keyboard heights to drive R1 at. These are the ones the UX round measured the
# failure against, and 300px is a plain iOS portrait keyboard — the cap has to
# hold at all of them, not just the smallest.
KEYBOARD_HEIGHTS = (260, 300, 336)


MEASURE_JS = """
(() => {
  const vis = (el) => {
    if (!el) return null;
    const r = el.getBoundingClientRect();
    const top = Math.max(r.top, 0);
    const bottom = Math.min(r.bottom, window.innerHeight);
    return {
      h: Math.round(r.height),
      visible: Math.max(0, Math.round(bottom - top)),
      fully: r.top >= 0 && r.bottom <= window.innerHeight && r.height > 0,
    };
  };
  const byText = (re) =>
    [...document.querySelectorAll('button')].find((b) => re.test(b.textContent || ''));
  // The test id is how the shipped card identifies itself. The `.border-accent`
  // fallback exists so this script can also measure a PRE-FIX tree, which
  // predates the id — a before/after pair has to be captured by one instrument
  // or the comparison measures the instrument. It is a fallback and not the
  // primary selector because that class is also applied by composer.tsx on
  // drag-over and by new-session.tsx on selection.
  const card =
    document.querySelector('[data-testid="pending-card"]') ||
    document.querySelector('.border-accent');
  const col = document.querySelector('[class*="h-dvh"]');
  // Scoped to the CARD. `.border-l-accent` is also the transcript's message
  // bubble, so an unscoped query measures a scrolled-away chat message and
  // reports the card's first option as off screen — a harness bug that reads
  // exactly like the defect under test.
  const opts = card ? [...card.querySelectorAll('.border-l-accent')] : [];
  const err = [...document.querySelectorAll('p')].find((p) =>
    /moved on|Already answered|didn.t respond|went away/.test(p.textContent || ''),
  );
  return JSON.stringify({
    viewport: [window.innerWidth, window.innerHeight],
    visualViewport: window.visualViewport
      ? Math.round(window.visualViewport.height)
      : null,
    column: col
      ? {
          clientH: col.clientHeight,
          scrollH: col.scrollHeight,
          clipped: col.scrollHeight - col.clientHeight,
          styleH: col.style.height || null,
        }
      : null,
    card: card
      ? {
          top: Math.round(card.getBoundingClientRect().top),
          bottom: Math.round(card.getBoundingClientRect().bottom),
          h: Math.round(card.getBoundingClientRect().height),
          capResolved: getComputedStyle(card).maxHeight,
          withinColumn:
            !!col &&
            card.getBoundingClientRect().bottom <= col.getBoundingClientRect().bottom + 1,
        }
      : null,
    approve: vis(byText(/^approve$/)),
    deny: vis(byText(/^deny$/)),
    send: vis(byText(/^send$/)),
    composer: vis(document.querySelector('textarea')),
    lastOption: vis(opts[opts.length - 1]),
    optionCount: opts.length,
    error: err ? { ...vis(err), text: (err.textContent || '').slice(0, 60) } : null,
    panelsExpanded: [...document.querySelectorAll('[aria-expanded]')].map((b) =>
      b.getAttribute('aria-expanded'),
    ),
  });
})()
"""


# Pin the column exactly as screens/session-view.tsx does on a visualViewport
# resize. This is the divergence case: device metrics stay at full height, so
# `dvh` is UNCHANGED, while the column drops by the keyboard's height — which is
# what an overlay keyboard does on iOS and what a uniform viewport shrink cannot
# reproduce. A cap in column units tightens with this; a `dvh` cap does not.
PIN_COLUMN_JS = """
(() => {
  const col = document.querySelector('[class*="h-dvh"]');
  if (!col) return JSON.stringify({ ok: false });
  const pinned = window.innerHeight - %d;
  col.style.height = pinned + 'px';
  col.style.setProperty('--lo-vvh', pinned + 'px');
  // Same fallback as MEASURE_JS, so a pre-fix tree can be pinned and measured
  // by this instrument too.
  const card =
    document.querySelector('[data-testid="pending-card"]') ||
    document.querySelector('.border-accent');
  return JSON.stringify({
    ok: true,
    pinned,
    dvhUnchanged: window.innerHeight,
    capResolved: card ? getComputedStyle(card).maxHeight : null,
  });
})()
"""


def login(page: Page, base: str) -> None:
    page.goto(f"{base}/login")
    page.js(
        "(() => { const f = document.querySelector('form');"
        f" f.password.value = {PASSWORD!r}; f.submit(); return true; }})()"
    )
    import time

    time.sleep(2.0)


def measure(page: Page) -> dict[str, Any]:
    return json.loads(page.js(MEASURE_JS))


def main() -> None:
    port = int(sys.argv[1])
    base = f"http://127.0.0.1:{port}"
    chrome = Chrome()
    failures: list[str] = []
    results: dict[str, Any] = {}

    def check(name: str, ok: bool, detail: str) -> None:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}: {detail}")
        results[name] = {"ok": ok, "detail": detail}
        if not ok:
            failures.append(f"{name}: {detail}")

    try:
        page = Page(chrome.target_ws())
        for width, height in VIEWPORTS:
            vp = f"{width}x{height}"
            page.metrics(width, height)
            login(page, base)
            print(f"\n=== {vp} ===")

            # R2: the approval's primary action, ON ARRIVAL, no gesture.
            page.goto(f"{base}/#/s/approval")
            m = measure(page)
            for label in ("approve", "deny"):
                got = m[label]
                check(
                    f"R2 {vp} {label} visible on arrival",
                    bool(got) and got["visible"] >= TAP_FLOOR,
                    f"{got and got['visible']}/{TAP_FLOOR}px visible"
                    f" (height {got and got['h']}px)",
                )
            check(
                f"R2 {vp} card within column",
                bool(m["card"]) and m["card"]["withinColumn"],
                f"card {m['card'] and (m['card']['top'], m['card']['bottom'])}"
                f" column clientH={m['column'] and m['column']['clientH']}",
            )

            # R1: the keyboard divergence, driven the way the app drives it.
            for kb in KEYBOARD_HEIGHTS:
                pin = json.loads(page.js(PIN_COLUMN_JS % kb))
                m2 = measure(page)
                col_bottom = pin["pinned"]
                # The card must end inside the pinned column, which is the
                # property `send`/`approve` staying on screen depends on.
                card_bottom = m2["card"]["bottom"]
                check(
                    f"R1 {vp} kb={kb} card inside pinned column",
                    card_bottom <= col_bottom + 1,
                    f"card bottom {card_bottom} vs column {col_bottom}"
                    f" (dvh still {pin['dvhUnchanged']}, cap {pin['capResolved']})",
                )
                for label in ("approve", "deny"):
                    got = m2[label]
                    if got:
                        check(
                            f"R1 {vp} kb={kb} {label} reachable",
                            got["visible"] >= TAP_FLOOR,
                            f"{got['visible']}/{TAP_FLOOR}px visible",
                        )

            # R1 for the variant it actually bites: free-text/secret, where the
            # keyboard is open BECAUSE the user is typing and `send` is the only
            # way to submit.
            page.goto(f"{base}/#/s/ask-free")
            for kb in KEYBOARD_HEIGHTS:
                pin = json.loads(page.js(PIN_COLUMN_JS % kb))
                m2 = measure(page)
                got = m2["send"]
                check(
                    f"R1 {vp} kb={kb} send reachable (secret variant)",
                    bool(got) and got["visible"] >= TAP_FLOOR,
                    f"{got and got['visible']}/{TAP_FLOOR}px visible;"
                    f" column {pin['pinned']}, dvh {pin['dvhUnchanged']},"
                    f" cap {pin['capResolved']}",
                )

            # R3: the last of ten options, by gesture only.
            page.goto(f"{base}/#/s/ask-long")
            before = measure(page)
            stranded = gesture_scroll_all(page)
            after = measure(page)
            check(
                f"R3 {vp} option-{after['optionCount']:02d} reachable by gesture",
                bool(after["lastOption"]) and after["lastOption"]["fully"],
                f"last option fully visible={after['lastOption'] and after['lastOption']['fully']}"
                f" (was {before['lastOption'] and before['lastOption']['fully']} on arrival);"
                f" scrollers still stranded: {len(stranded)}",
            )

            # R5: the stacked state, with finger gestures only.
            for session in ("stacked", "stacked-approval"):
                page.goto(f"{base}/#/s/{session}")
                m3 = measure(page)
                clipped = m3["column"]["clipped"] if m3["column"] else -1
                check(
                    f"R5 {vp} {session} column not clipped",
                    clipped <= 1,
                    f"column clientH={m3['column'] and m3['column']['clientH']}"
                    f" scrollH={m3['column'] and m3['column']['scrollH']}"
                    f" clipped={clipped}px;"
                    f" panels aria-expanded={m3['panelsExpanded']}",
                )
                # What must be on screen without a gesture differs by variant,
                # and conflating the two would assert something false. For the
                # approval the DECISION itself must be visible — that is U2. For
                # the options variant the decision is a list that can legitimately
                # be longer than the card, so the contract is weaker but still
                # real: the card and its first option are on screen on arrival,
                # and the tail is reachable by finger. A list needing one swipe
                # is inherent; a list needing a swipe that does nothing is D1.
                if m3["approve"]:
                    check(
                        f"R5 {vp} {session} decision visible on arrival",
                        m3["approve"]["visible"] >= TAP_FLOOR,
                        f"approve {m3['approve']['visible']}/{TAP_FLOOR}px;"
                        f" card {m3['card'] and (m3['card']['top'], m3['card']['bottom'])}",
                    )
                else:
                    first = json.loads(
                        page.js(
                            "(() => { const c ="
                            " document.querySelector('[data-testid=\"pending-card\"]');"
                            " const o = c && c.querySelector('.border-l-accent');"
                            " if (!o) return 'null'; const r = o.getBoundingClientRect();"
                            " return JSON.stringify({ visible: Math.max(0,"
                            " Math.round(Math.min(r.bottom, window.innerHeight)"
                            " - Math.max(r.top, 0))) }); })()"
                        )
                    )
                    check(
                        f"R5 {vp} {session} first option visible on arrival",
                        bool(first) and first["visible"] >= TAP_FLOOR,
                        f"first option {first and first['visible']}/{TAP_FLOOR}px;"
                        f" card {m3['card'] and (m3['card']['top'], m3['card']['bottom'])}",
                    )
                    stranded_stack = gesture_scroll_all(page)
                    m3b = measure(page)
                    check(
                        f"R5 {vp} {session} option tail reachable by gesture",
                        bool(m3b["lastOption"]) and m3b["lastOption"]["fully"],
                        f"last option fully visible="
                        f"{m3b['lastOption'] and m3b['lastOption']['fully']};"
                        f" scrollers still stranded: {len(stranded_stack)}",
                    )

            # R4: the stale-tap error, where it renders.
            page.goto(f"{base}/#/s/stale")
            gesture_scroll_all(page)
            page.js(
                "(() => { const b = [...document.querySelectorAll('button')]"
                ".filter((x) => /option-/.test(x.textContent));"
                " if (b.length) b[b.length - 1].click(); return true; })()"
            )
            import time

            time.sleep(1.5)
            m4 = measure(page)
            err = m4["error"]
            check(
                f"R4 {vp} stale-tap error visible",
                bool(err) and err["visible"] > 0 and err["fully"],
                f"error {err and err['text']!r} visible={err and err['visible']}px"
                f" fully={err and err['fully']}",
            )

        page.close()
    finally:
        chrome.close()

    print(f"\n{len(results) - len(failures)}/{len(results)} checks passed")
    if failures:
        print("\nFAILURES:")
        for line in failures:
            print(f"  - {line}")
        sys.exit(1)
    print("All reachability properties hold.")


if __name__ == "__main__":
    main()
