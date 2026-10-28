"""Capture the phone ask card's ``Other`` free-text row on BOTH builds it photographs.

Run from the repository root (it starts its own fixture, on a free port)::

    PYTHONPATH=. .venv/bin/python scripts/mobile_ask_other_capture.py OUTDIR LABEL \
        --expect before|after [--repo PATH]

``--repo`` names the tree whose ``local_operator/mobile/web/dist`` the fixture serves and
whose ``scripts/mobile_asks_open_fixture.py`` it starts; it defaults to the tree this file
lives in. The BEFORE set therefore needs this file copied into the detached pre-change
worktree, because the fixture serves the tree it sits in — the same arrangement
``scripts/mobile_asks_open_capture.py`` uses, and its banner line ("Fixture serves tree:")
is recorded beside the frames so the provenance of each set is a printed fact.

THE FEATURE UNDER THE LENS: an ``Other`` row on every non-secret option question of the ask
card (``local_operator/mobile/web/src/components/ask-card.tsx``). On the AFTER build it
opens an empty free-text field, focuses it, keeps the question incomplete while the field
is empty, enables the card's send once text is typed, and exists beside a multi-select's
ticked options. On the BEFORE build neither the row nor the field exists anywhere.

THE FIXTURE'S CONVERSATION "Two questions waiting" carries two open asks: a two-question
ask (a single-select picker and a multi-select) and a single-question yes/no ask. The
frames are taken on the SINGLE-QUESTION ask, because it is the only card whose send can
flip from disabled to enabled with one answer — the two-question card stays incomplete
until both are filled, so it cannot show the send transition the feature claims. The
multi-select frame is taken on the two-question card's second question. The probe records
each card's position in the sheet and its ask id, so those identities are facts beside the
frames rather than assumptions in this docstring.

THE FRAME MATRIX (one frame per row, in this order):

====  ==================================================  =========================
 #    what the run does                                    after / before
====  ==================================================  =========================
 1    scrolls the single-question card into view          row present / no row
 2    taps ``Other`` with a real touch                    field open+focus / nothing
 3    types real text (``Input.insertText``)              send enabled / nothing
 4    ticks a multi-select option, taps its ``Other``,
      types                                               typed text / nothing
 5    light palette, a reload, then 1 again on the same
      browser                                             row present / no row
 6    ...and 2 again                                       field open / nothing
====  ==================================================  =========================

The touches are REAL: ``Input.dispatchTouchEvent`` at the element's centre, and the scrolls
that bring a card into view are finger drags (``Page.swipe``), never ``scrollTop``
assignment — the discipline ``scripts/mobile_overflow_capture.py`` exists for. The only
scripted input is ``Input.insertText``, which is how a keyboard delivers text.

Each frame records the probe as JSON beside the PNG: which questions exist, whether each
has an ``Other`` row and a field, the field's value and focus, each option's pressed state,
each card's send label and disabled state, the sheet's scroll geometry, and the theme.
Every frame's checks are evaluated against what THAT build must show, and a mismatch makes
the run exit non-zero: a BEFORE frame that unexpectedly grew an ``Other`` row, or an AFTER
frame whose send did not enable, is a red run rather than a pretty picture.

Isolation and teardown follow AGENTS.md and the rig it inherits: the fixture re-homes HOME
and the config dir through ``scripts.probe_isolation`` on import, Chrome takes a unique
profile under a temp dir and a port it chooses, and both are reaped by the exact pid /
process group this run created, with zero survivors asserted.
"""

from __future__ import annotations

import json
import os
import secrets
import select
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# THE DOCUMENTED RUN LINE HAS TO WORK without PYTHONPATH (same two-line bootstrap, and for
# the same reason, as the open capture): the tree that CONTAINS this file must win over any
# checkout the interpreter's site-packages points at, or the fixture would serve the wrong
# tree's bundle.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.mobile_overflow_capture import Chrome, Page  # noqa: E402

SELF_ROOT = Path(__file__).resolve().parents[1]
VIEWPORT = (390, 844)

#: The conversation the fixture serves, by the NAME the list shows.
CONVERSATION = "Two questions waiting"
#: The single-question yes/no ask: the card the send-transition frames are taken on.
SINGLE_QUESTION = "Ship it behind the flag?"
#: The two-question ask's multi-select: the card the ticked-option frame is taken on.
MULTI_QUESTION = "Which surfaces must the verification cover?"
MULTI_OPTION = "phone"
SINGLE_TEXT = "ship it behind a flag"
MULTI_TEXT = "also cover the terminal"
LIGHT_THEME = "localOperatorLight"

#: How long a freshly opened sheet is watched before its frame is taken.
WATCH_S = 3.0
#: How long a page is given to stop moving before a frame is called unsettled.
SETTLE_S = 20.0
#: Consecutive identical samples (100 ms apart) that count as "stopped moving".
STILL_SAMPLES = 3

RUNNING_ANIMATIONS_JS = """document.getAnimations().filter((a) => {
      if (a.playState !== 'running') return false;
      const t = a.effect && a.effect.getComputedTiming ? a.effect.getComputedTiming() : null;
      return !t || Number.isFinite(t.endTime);
    }).length"""

PROBE = r"""
(() => {
  const box = (el) => {
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return {top: Math.round(r.top), bottom: Math.round(r.bottom),
            left: Math.round(r.left), right: Math.round(r.right),
            w: Math.round(r.width), h: Math.round(r.height)};
  };
  const dialog = document.querySelector('[role="dialog"]');
  const cards = [];
  if (dialog) {
    for (const card of dialog.querySelectorAll('[data-testid="ask-card"]')) {
      const send = card.querySelector('button.flex-1');
      const questions = [];
      const seen = new Set();
      for (const span of card.querySelectorAll('span.min-w-0')) {
        const block = span.parentElement ? span.parentElement.parentElement : null;
        if (!block || seen.has(block)) continue;
        if (!block.querySelector('button[aria-pressed], input')) continue;
        seen.add(block);
        const field = block.querySelector('input');
        const other = block.querySelector('button[data-ask-option-other]');
        questions.push({
          text: span.textContent.trim(),
          box: box(block),
          options: [...block.querySelectorAll('button[aria-pressed]')].map((b) => {
            const span = b.querySelector('span');
            const first = span && span.firstChild ? span.firstChild.textContent : '';
            return {
              label: (first || b.textContent).trim().slice(0, 40),
              pressed: b.getAttribute('aria-pressed') === 'true',
              other: b.hasAttribute('data-ask-option-other'),
            };
          }),
          otherRow: other ? {
            pressed: other.getAttribute('aria-pressed') === 'true',
            text: other.textContent.trim(),
            box: box(other),
          } : null,
          field: field ? {
            value: field.value,
            focused: document.activeElement === field,
            placeholder: field.getAttribute('placeholder'),
            box: box(field),
          } : null,
        });
      }
      cards.push({
        ask_id: card.getAttribute('data-ask-id'),
        status: card.getAttribute('data-ask-status'),
        box: box(card),
        send: send ? {label: send.textContent.trim(), disabled: send.disabled} : null,
        questions,
      });
    }
  }
  const scroller = dialog ? dialog.querySelector('.lo-scroll') : null;
  const s = scroller ? scroller.getBoundingClientRect() : null;
  const active = document.activeElement;
  return JSON.stringify({
    theme: document.documentElement.dataset.theme || null,
    viewport: [window.innerWidth, window.innerHeight],
    dialog: Boolean(dialog),
    panel: dialog ? box(dialog.querySelector('.lo-sheet-panel')) : null,
    cardCount: cards.length,
    cards,
    otherCount: dialog ? dialog.querySelectorAll('[data-ask-option-other]').length : 0,
    scroller: scroller ? {top: Math.round(s.top), bottom: Math.round(s.bottom),
                          left: Math.round(s.left), right: Math.round(s.right),
                          clientH: scroller.clientHeight, scrollH: scroller.scrollHeight,
                          scrollTop: Math.round(scroller.scrollTop)} : null,
    active: active ? {tag: active.tagName.toLowerCase(),
                      label: active.getAttribute('aria-label'),
                      value: typeof active.value === 'string' ? active.value : null} : null,
    animating: __RUNNING__,
  });
})()
"""


PROBE = PROBE.replace("__RUNNING__", RUNNING_ANIMATIONS_JS)

STILL_PROBE = r"""
(() => {
  const running = __RUNNING__;
  const dialog = document.querySelector('[role="dialog"]');
  const panel = dialog ? dialog.querySelector('.lo-sheet-panel') : null;
  const r = panel ? panel.getBoundingClientRect() : null;
  return JSON.stringify({
    running,
    reading: dialog ? /reading asks/.test(dialog.textContent) : false,
    panel: r ? [Math.round(r.top), Math.round(r.bottom), Math.round(r.height)] : null,
  });
})()
"""

STILL_PROBE = STILL_PROBE.replace("__RUNNING__", RUNNING_ANIMATIONS_JS)


def _js_literal(value: str) -> str:
    return json.dumps(value)


def question_block_js(text: str) -> str:
    """JS returning the question block (header + control + skip) for a question's text.

    The DOM gives the block no id of its own; its identity is the question text it paints,
    which is exactly how a reader finds it. The block is the header span's parent.
    """
    lit = _js_literal(text)
    return (
        "(() => {"
        " const cards = [...document.querySelectorAll('[data-testid=\"ask-card\"]')];"
        " for (const card of cards) {"
        f"  const span = [...card.querySelectorAll('span.min-w-0')]"
        f"    .find((s) => s.textContent.trim() === {lit});"
        "  if (span && span.parentElement && span.parentElement.parentElement)"
        "    return span.parentElement.parentElement;"
        " }"
        " return null;"
        "})()"
    )


def other_row_js(text: str) -> str:
    return (
        "(() => {"
        f" const block = ({question_block_js(text)});"
        " return block ? block.querySelector('button[data-ask-option-other]') : null;"
        "})()"
    )


def field_js(text: str) -> str:
    return (
        "(() => {"
        f" const block = ({question_block_js(text)});"
        " return block ? block.querySelector('input') : null;"
        "})()"
    )


def option_js(text: str, label: str) -> str:
    lit = _js_literal(label)
    return (
        "(() => {"
        f" const block = ({question_block_js(text)});"
        " if (!block) return null;"
        " return [...block.querySelectorAll('button[aria-pressed]')]"
        f"  .find((b) => (b.querySelector('span') || b).textContent.trim().startsWith({lit}))"
        " || null;"
        "})()"
    )


def _rect_of(selector_js: str) -> str:
    """JS that finds an element, scrolls it into view and returns the centre of its box."""
    return (
        "(() => {"
        f" const el = ({selector_js});"
        " if (!el) return null;"
        " el.scrollIntoView({block: 'center'});"
        " const r = el.getBoundingClientRect();"
        " return JSON.stringify({x: Math.round(r.left + r.width / 2),"
        " y: Math.round(r.top + r.height / 2), w: Math.round(r.width), h: Math.round(r.height)});"
        "})()"
    )


def row_selector(label: str) -> str:
    return (
        "[...document.querySelectorAll('button')]"
        f".find((b) => (b.textContent || '').includes({label!r}))"
    )


BACK_TO_LIST = "document.querySelector('button[aria-label=\"back to sessions\"]')"


def still_centre(page: Page, selector_js: str, seconds: float = 5.0) -> dict[str, Any] | None:
    """The element's centre once it has stopped moving, or ``None`` when there is no element."""
    deadline = time.time() + seconds
    last: dict[str, Any] | None = None
    while True:
        raw = page.js(_rect_of(selector_js))
        if raw is None:
            return None
        where = json.loads(raw)
        if where == last or time.time() >= deadline:
            return where
        last = where
        time.sleep(0.12)


def tap(page: Page, selector_js: str) -> dict[str, Any] | None:
    """One REAL finger tap on the element, or ``None`` when there is nothing to tap."""
    where = still_centre(page, selector_js)
    if where is None:
        return None
    page.send(
        "Input.dispatchTouchEvent",
        type="touchStart",
        touchPoints=[{"x": where["x"], "y": where["y"]}],
    )
    time.sleep(0.07)
    page.send("Input.dispatchTouchEvent", type="touchEnd", touchPoints=[])
    return where


def probe(page: Page) -> dict[str, Any]:
    return json.loads(page.js(PROBE))


def watch(page: Page, seconds: float = WATCH_S) -> int | None:
    """Poll for a dialog for ``seconds``; the ms until it appeared, or ``None``."""
    start = time.time()
    while time.time() - start < seconds:
        if page.js("document.querySelector('[role=\"dialog\"]') !== null"):
            return round((time.time() - start) * 1000)
        time.sleep(0.05)
    return None


def wait_for(page: Page, selector_js: str, seconds: float = 4.0) -> int | None:
    """Poll for an element; the ms until it existed, or ``None``."""
    start = time.time()
    while time.time() - start < seconds:
        if page.js(f"Boolean({selector_js})"):
            return round((time.time() - start) * 1000)
        time.sleep(0.05)
    return None


def quiesce(page: Page, seconds: float = SETTLE_S) -> int | None:
    """Wait until the page has STOPPED MOVING; the ms it took, or ``None`` if it never did."""
    start = time.time()
    last: dict[str, Any] | None = None
    still = 0
    while time.time() - start < seconds:
        sample = json.loads(page.js(STILL_PROBE))
        moving = sample["running"] > 0 or sample["reading"]
        still = still + 1 if (not moving and sample == last) else 0
        last = sample
        if still >= STILL_SAMPLES - 1:
            return round((time.time() - start) * 1000)
        time.sleep(0.1)
    return None


def free_port() -> int:
    with socket.socket() as probe_socket:
        probe_socket.bind(("127.0.0.1", 0))
        return int(probe_socket.getsockname()[1])


class Fixture:
    """The fixture process for the tree under test, with its banner read back.

    ``scripts/mobile_asks_open_fixture.py`` serves the tree IT SITS IN (its own two-line
    sys.path bootstrap), so this process is started from ``--repo`` and its "Fixture serves
    tree:" line is captured as provenance. stderr goes to a file beside the frames, where a
    startup failure is readable rather than lost.
    """

    def __init__(self, repo: Path, port: int, password: str, log_path: Path) -> None:
        env = {**os.environ, "LOP_MOBILE_FIXTURE_PASSWORD": password}
        self._log = log_path.open("wb")
        self.proc = subprocess.Popen(
            [sys.executable, str(repo / "scripts/mobile_asks_open_fixture.py"), str(port)],
            cwd=repo,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log,
            start_new_session=True,
        )
        self._buffer = b""
        self.banner: list[str] = []
        deadline = time.time() + 60
        while time.time() < deadline:
            line = self._read_line(deadline - time.time())
            if line is None:
                break
            self.banner.append(line)
            if line.startswith("Fixture mobile:"):
                return
        self.close()
        raise RuntimeError(f"the fixture never announced itself; see {log_path}")

    def _read_line(self, timeout: float) -> str | None:
        assert self.proc.stdout is not None
        fd = self.proc.stdout.fileno()
        end = time.time() + max(timeout, 0)
        while b"\n" not in self._buffer:
            remaining = end - time.time()
            if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
                return None
            chunk = os.read(fd, 4096)
            if not chunk:
                return None
            self._buffer += chunk
        line, self._buffer = self._buffer.split(b"\n", 1)
        return line.decode("utf-8", "replace").strip()

    def close(self) -> None:
        """Reap the fixture's whole process group by its own pid, and prove it is gone."""
        try:
            if self.proc.stdin is not None:
                self.proc.stdin.close()
        except OSError:
            pass
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait(timeout=10)
        self._log.close()


def find_question(data: dict[str, Any], text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """(card, question) whose question text is ``text``; ``({}, {})`` where absent.

    Empty dicts rather than ``None`` so an expectation about a missing card reads as a
    mismatch instead of a crash that hides which frame was being checked.
    """
    for card in data.get("cards") or []:
        for question in card.get("questions") or []:
            if question.get("text") == text:
                return card, question
    return {}, {}


def scroll_into_view(
    page: Page, selector_js: str, max_gestures: int = 10
) -> tuple[list[int], dict[str, Any] | None]:
    """Bring an element to the top of the sheet's scroller with REAL finger drags.

    A script may set ``scrollTop`` on anything, including a clipped container; a thumb may
    not. This measures the element against the sheet's own scroller and swipes the content
    by the shortfall, then re-measures — returning the gestures it made and the last
    geometry, so the report states how the frame was reached.
    """
    gestures: list[int] = []
    last: dict[str, Any] | None = None
    for _ in range(max_gestures):
        raw = page.js(
            "(() => {"
            f" const el = ({selector_js});"
            " const d = document.querySelector('[role=\"dialog\"]');"
            " const s = d && d.querySelector('.lo-scroll');"
            " if (!el || !s) return null;"
            " const r = el.getBoundingClientRect();"
            " const b = s.getBoundingClientRect();"
            " return JSON.stringify({top: Math.round(r.top), bottom: Math.round(r.bottom),"
            " h: Math.round(r.height), viewTop: Math.round(b.top),"
            " viewBottom: Math.round(b.bottom),"
            " x: Math.round(b.left + b.width / 2), y: Math.round(b.top + b.height / 2)});"
            "})()"
        )
        if raw is None:
            return gestures, None
        # A local with the declared type: pyright keeps `last` at `dict | None` across
        # the assignment, and json.loads' Any would otherwise only narrow by accident.
        parsed: dict[str, Any] = json.loads(raw)
        last = parsed
        want = parsed["viewTop"] + 12
        delta = parsed["top"] - want
        if abs(delta) <= 4:
            return gestures, parsed
        step = max(-420, min(420, delta))
        page.swipe(parsed["x"], parsed["y"], step)
        gestures.append(step)
    return gestures, last


def check(
    failures: list[str],
    frame_name: str,
    name: str,
    ok: bool,
    detail: Any = "",
) -> bool:
    verdict = "PASS" if ok else "FAIL"
    suffix = "" if ok else f"  ({detail})"
    print(f"  {verdict}  {frame_name}: {name}{suffix}", flush=True)
    if not ok:
        failures.append(f"{frame_name}: {name}{suffix}")
    return ok


def expectations(
    failures: list[str],
    frame_name: str,
    data: dict[str, Any],
    build: str,
) -> list[tuple[str, bool, Any]]:
    """Evaluate one frame against what THIS build must show; print a line per expectation."""
    after = build == "after"
    card, single = find_question(data, SINGLE_QUESTION)
    _, multi = find_question(data, MULTI_QUESTION)
    send_disabled = (card.get("send") or {}).get("disabled")
    other_count = data.get("otherCount") or 0
    rows: list[tuple[str, bool, Any]] = []

    def want(name: str, ok: bool, detail: Any = "") -> None:
        rows.append((name, ok, detail))
        check(failures, frame_name, name, ok, detail)

    want("the sheet is open", data.get("dialog") is True, data.get("dialog"))
    if frame_name.startswith("light-"):
        want(
            f"the light palette is applied ({LIGHT_THEME})",
            data.get("theme") == LIGHT_THEME,
            data.get("theme"),
        )
    want(
        "the single-question card is on screen",
        bool(single),
        sorted(q.get("text") for c in data.get("cards") or [] for q in c.get("questions") or []),
    )
    if frame_name.endswith("other-closed") or frame_name.endswith("light-other-closed"):
        if after:
            want(
                "an Other row exists on the single-select question",
                single.get("otherRow") is not None,
                single.get("otherRow"),
            )
            want("an Other row exists somewhere in the sheet", other_count >= 1, other_count)
            want(
                "the Other field is closed before any press",
                single.get("field") is None,
                single.get("field"),
            )
            want("send is disabled with nothing answered", send_disabled is True, send_disabled)
        else:
            want("NO Other row exists anywhere in the sheet", other_count == 0, other_count)
            want(
                "the single-select question has no Other row",
                single.get("otherRow") is None,
                single.get("otherRow"),
            )
            want("no Other field is rendered", single.get("field") is None, single.get("field"))
            want("send is disabled with nothing answered", send_disabled is True, send_disabled)
    elif frame_name.endswith("other-open") or frame_name.endswith("light-other-open"):
        if after:
            want(
                "the Other row reads as pressed",
                (single.get("otherRow") or {}).get("pressed") is True,
                single.get("otherRow"),
            )
            want(
                "the free-text field is visible",
                single.get("field") is not None,
                single.get("field"),
            )
            want(
                "the field is empty",
                (single.get("field") or {}).get("value") == "",
                (single.get("field") or {}).get("value"),
            )
            want(
                "the field took focus",
                (single.get("field") or {}).get("focused") is True,
                data.get("active"),
            )
            want(
                "send stays disabled while the field is empty", send_disabled is True, send_disabled
            )
            height = ((single.get("field") or {}).get("box") or {}).get("h") or 0
            want("the field is a 44 px touch target", height >= 44, height)
        else:
            want("NO Other row exists anywhere in the sheet", other_count == 0, other_count)
            want(
                "the tap found no Other row", single.get("otherRow") is None, single.get("otherRow")
            )
            want("no Other field is rendered", single.get("field") is None, single.get("field"))
            want("send stays disabled", send_disabled is True, send_disabled)
    elif frame_name.endswith("other-typed"):
        if after:
            want(
                "the field holds the typed text",
                (single.get("field") or {}).get("value") == SINGLE_TEXT,
                (single.get("field") or {}).get("value"),
            )
            want("send is enabled once the answer has text", send_disabled is False, send_disabled)
        else:
            want("NO Other field is rendered", single.get("field") is None, single.get("field"))
            want("send stays disabled", send_disabled is True, send_disabled)
    elif frame_name.endswith("other-multi"):
        ticks = {o.get("label"): o.get("pressed") for o in multi.get("options") or []}
        want(
            f"the multi-select option {MULTI_OPTION!r} is ticked",
            ticks.get(MULTI_OPTION) is True,
            ticks,
        )
        if after:
            want(
                "the multi-select question has its own Other row",
                multi.get("otherRow") is not None,
                multi.get("otherRow"),
            )
            want(
                "that Other row reads as pressed",
                (multi.get("otherRow") or {}).get("pressed") is True,
                multi.get("otherRow"),
            )
            want(
                "the field holds the typed multi text",
                (multi.get("field") or {}).get("value") == MULTI_TEXT,
                (multi.get("field") or {}).get("value"),
            )
        else:
            want("NO Other row exists anywhere in the sheet", other_count == 0, other_count)
            want(
                "the multi-select question has no Other row",
                multi.get("otherRow") is None,
                multi.get("otherRow"),
            )
            want("no Other field is rendered", multi.get("field") is None, multi.get("field"))
    return rows


def main() -> None:
    argv = sys.argv[1:]
    if len(argv) < 2 or "--expect" not in argv:
        raise SystemExit(__doc__)
    outdir, label = Path(argv[0]), argv[1]
    build = argv[argv.index("--expect") + 1]
    if build not in ("before", "after"):
        raise SystemExit("--expect must be 'before' or 'after'")
    repo = Path(argv[argv.index("--repo") + 1]).resolve() if "--repo" in argv else SELF_ROOT
    if not (repo / "local_operator/mobile/web/dist/index.html").exists():
        raise SystemExit(f"{repo} has no built mobile bundle; run pnpm build there first")
    outdir.mkdir(parents=True, exist_ok=True)

    # A THROWAWAY PASSWORD, GENERATED PER RUN AND NEVER PRINTED: handed to the fixture
    # through the environment (not argv, so it is not in `ps`).
    password = secrets.token_urlsafe(16)
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    fixture = Fixture(repo, port, password, outdir / f"{label}-fixture.log")
    chrome = Chrome()
    report: dict[str, Any] = {
        "build": build,
        "repo": str(repo),
        "fixture": fixture.banner,
        "viewport": VIEWPORT,
        "frames": {},
        "checks": {},
    }
    failures: list[str] = []

    def frame(name: str, **extra: Any) -> dict[str, Any]:
        settled_ms = quiesce(page)
        data = {**probe(page), "settledAfterMs": settled_ms, **extra}
        page.shot(outdir / f"{label}-{name}.png")
        if settled_ms is None:
            check(failures, name, "the page settled before the shutter", False, "unsettled")
        rows = expectations(failures, name, data, build)
        report["checks"][name] = [
            {"check": text, "pass": ok, "detail": detail} for text, ok, detail in rows
        ]
        report["frames"][name] = data
        card, single = find_question(data, SINGLE_QUESTION)
        print(
            f"{name:<24} cards={data['cardCount']} otherRows={data['otherCount']} "
            f"singleSend={((card.get('send') or {}).get('label'))!r}/"
            f"{(card.get('send') or {}).get('disabled')} "
            f"field={((single.get('field') or {}).get('value'))!r}",
            flush=True,
        )
        return data

    def open_from_list(name: str) -> int | None:
        tapped = tap(page, row_selector(name))
        if tapped is None:
            raise RuntimeError(f"no list row reads {name!r}; the fixture is not what this expects")
        appeared = watch(page)
        if appeared is not None:
            report["settled-after-open-ms"] = quiesce(page)
        return appeared

    page: Page = Page(chrome.target_ws())
    try:
        width, height = VIEWPORT
        page.metrics(width, height)
        page.goto(f"{base}/login")
        page.js(
            "(() => { const f = document.querySelector('form');"
            f" f.password.value = {password!r}; f.submit(); return true; }})()"
        )
        time.sleep(2.0)
        page.goto(f"{base}/#/")
        time.sleep(1.5)

        report["opened-after-ms"] = open_from_list(CONVERSATION)
        report["card-order-on-open"] = [
            {
                "ask_id": c.get("ask_id"),
                "questions": [q.get("text") for q in c.get("questions") or []],
            }
            for c in (probe(page).get("cards") or [])
        ]

        # 1 — the row that should be there, before anything is pressed.
        gestures, geometry = scroll_into_view(page, question_block_js(SINGLE_QUESTION))
        report["scroll-to-single"] = {"gestures": gestures, "geometry": geometry}
        frame("other-closed")

        # 2 — a real finger on Other: an empty field, focused, send still disabled.
        tapped = tap(page, other_row_js(SINGLE_QUESTION))
        opened_ms = wait_for(page, field_js(SINGLE_QUESTION))
        time.sleep(0.4)
        report["other-tapped"] = tapped
        report["other-field-appeared-ms"] = opened_ms
        if build == "after":
            check(
                failures,
                "other-open",
                "the tap opened the Other field",
                opened_ms is not None,
                opened_ms,
            )
        frame("other-open")

        # 3 — real keyboard text: send enables.
        page.send("Input.insertText", text=SINGLE_TEXT)
        time.sleep(0.4)
        frame("other-typed")

        # 4 — the multi-select's own Other: tick one option, tap its row, type.
        gestures, geometry = scroll_into_view(page, question_block_js(MULTI_QUESTION))
        report["scroll-to-multi"] = {"gestures": gestures, "geometry": geometry}
        ticked = tap(page, option_js(MULTI_QUESTION, MULTI_OPTION))
        report["multi-ticked"] = ticked
        time.sleep(0.3)
        tapped = tap(page, other_row_js(MULTI_QUESTION))
        opened_ms = wait_for(page, field_js(MULTI_QUESTION))
        time.sleep(0.3)
        report["multi-other-tapped"] = tapped
        report["multi-field-appeared-ms"] = opened_ms
        if build == "after":
            check(
                failures,
                "other-multi",
                "the tap opened the multi Other field",
                opened_ms is not None,
                opened_ms,
            )
        page.send("Input.insertText", text=MULTI_TEXT)
        time.sleep(0.4)
        frame("other-multi")

        # 5/6 — the light palette, on the SAME browser: write the app's own theme item and
        # reload (the palette is a boot-time read), then repeat the closed/open pair.
        page.js(f"localStorage.setItem('lo-mobile-theme', {_js_literal(LIGHT_THEME)}); true")
        # THE ANSWER DRAFTS SURVIVE A RELOAD (store.ts keeps them in localStorage,
        # ``lo-mobile-draft:ask:<id>``), so without this the light pass would reopen the
        # card with the previous frames' text still in it. Clear only the answer drafts —
        # the same clean slate a fresh session gives — never the theme item just written.
        page.js(
            "(() => { for (const key of Object.keys(localStorage))"
            " { if (key.startsWith('lo-mobile-draft:ask:')) localStorage.removeItem(key); }"
            " return true; })()"
        )
        # A real reload: navigating to the SAME url is a fragment navigation (no new
        # document), so the palette — a boot-time read — would never be re-applied and
        # the in-memory draft would survive, which is not the state under test.
        page.send("Page.reload", ignoreCache=True)
        time.sleep(2.0)
        # The reload restored the SESSION route the tap had pushed; go back to the list on
        # this fresh document before opening the conversation again, or the "open from
        # list" tap would land on the session header of an already-open screen.
        page.goto(f"{base}/#/")
        wait_for(page, row_selector(CONVERSATION), 6.0)
        report["light-opened-after-ms"] = open_from_list(CONVERSATION)
        gestures, geometry = scroll_into_view(page, question_block_js(SINGLE_QUESTION))
        report["scroll-to-single-light"] = {"gestures": gestures, "geometry": geometry}
        frame("light-other-closed")
        tapped = tap(page, other_row_js(SINGLE_QUESTION))
        opened_ms = wait_for(page, field_js(SINGLE_QUESTION))
        time.sleep(0.4)
        report["light-other-tapped"] = tapped
        report["light-other-field-appeared-ms"] = opened_ms
        if build == "after":
            check(
                failures,
                "light-other-open",
                "the tap opened the Other field",
                opened_ms is not None,
                opened_ms,
            )
        frame("light-other-open")
        page.close()
    finally:
        try:
            chrome.close()
        finally:
            fixture.close()

    (outdir / f"{label}-ask-other-probes.json").write_text(json.dumps(report, indent=2))
    print(f"\nbuild={build} repo={repo} frames={len(report['frames'])}", flush=True)
    print(f"failures={failures or 'none'}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
