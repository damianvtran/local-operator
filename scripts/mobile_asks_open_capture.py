"""Capture the phone's OPEN-BY-DEFAULT asks sheet out of headless Chrome over CDP.

Run from the repository root (it starts its own fixture, on a free port)::

    PYTHONPATH=. .venv/bin/python scripts/mobile_asks_open_capture.py OUTDIR LABEL \
        --expect before|after [--theme localOperatorLight]

``--theme`` (round-1 design D10) pins a palette for the whole run by writing the
relay's own theme item (``lo-mobile-theme``, ``theme.ts``'s key) before the first
document script runs; the sheet reads theme tokens, so a light-palette set is the
single-axis coverage the default dark frames cannot give. Absent, the default
palette is what every other frame here shows.

THE SAME SCRIPT RUNS ON BOTH BUILDS IT PHOTOGRAPHS, which is why it never imports the
policy module: the BEFORE set comes from a detached worktree of the pre-change build (whose
``local_operator/mobile/web/dist`` is built from that tree's sources), the AFTER set from the
branch. ``--expect`` is the self-check that makes the pair trustworthy: every frame's probe
is compared with what THAT build is supposed to do, and a mismatch fails the run, so a
before-set that unexpectedly opened a sheet, or an after-set that did not, is a red exit and
not a pretty picture. The fixture it starts prints the tree it serves, and that line is
recorded beside the frames.

THE STATE MATRIX (one frame per row, in this order; ``before`` = nothing opens by itself):

====  ======================================  ==========================  ==================
 #    what the user does                       after (this change)          before
====  ======================================  ==========================  ==================
 01   opens a conversation with no asks        closed                       closed
 03   opens one whose asks are all answered    closed                       closed
 02   opens one with asks waiting              OPEN (once)                  dock only
 04a  closes it (real tap on the close mark)   closed, dock stays           (nothing to close)
 04b  ...the queue re-publishes (a re-render)  still closed                 dock only
 04c  ...a NEW ask arrives                     still closed, dock counts    dock only
 04d  ...leaves for the list and comes back    still closed                 dock only
 05   types, then the asks arrive              closed, caret + text kept    dock only
====  ======================================  ==========================  ==================

Frame 02 is deliberately AFTER 03 in the run but numbered by the clause it shows (clause
2); the run order is chosen so each conversation is opened from the list by a real tap and
the page is never reloaded, which is what clause 4's "same page lifetime" is about.

THE GESTURES ARE REAL TOUCHES (``Input.dispatchTouchEvent`` tap at the element's centre), not
``element.click()``: the sheet carries a guard that swallows the release click of a press that
was still held when it mounted, and a scripted click never goes through it. A tap that is
swallowed, or a close control that does not answer a finger, would be invisible to a script
that clicks. The one exception is the ``Input.insertText`` that types into the composer, which
is how a keyboard delivers text.

Geometry is recorded beside every frame (dialog and panel boxes, the dock, the composer's
value and focus, which siblings of the dialog went inert, and the history entry the sheet
claimed), because the stills show the symptom and the numbers show the cause.

AGENTS.md section 6 applies and is followed by the imported driver: a unique profile, a port
Chrome picks, ``--headless=new``, ``--use-mock-keychain``, one browser for the whole run, and
a teardown that asserts no helper survived.
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

# THE DOCUMENTED RUN LINE HAS TO WORK (the same bootstrap, for the same reason, as
# ``scripts/mobile_asks_capture.py``: without it the run line dies on
# ``ModuleNotFoundError: No module named 'scripts'`` before ``main()`` is reached).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.mobile_overflow_capture import Chrome, Page  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
VIEWPORT = (390, 844)

#: The conversations the fixture serves, by the NAME the list shows (a row is tapped by its
#: text, exactly as a thumb finds it).
NONE = "Nothing waiting"
SETTLED = "Everything answered"
PENDING = "Two questions waiting"
TYPING = "Typing when it lands"

#: How long a freshly opened conversation is watched before its frame is taken. The first
#: projection arrives within a second or two on this rig; the policy's own window is 45 s, so
#: this is the "settled" bound for a frame, not the policy's.
WATCH_S = 3.0

#: How long a page is given to stop moving before a frame is called unsettled. Generous on
#: purpose: the cost of waiting is seconds, the cost of a thin or half-risen frame is the evidence.
SETTLE_S = 20.0

#: Consecutive identical samples (100 ms apart) that count as "stopped moving".
STILL_SAMPLES = 3

#: The number of FINITE animations still running, as a JS expression: the browser's own answer
#: to "is anything still moving". The sheet's rise is a 240 ms CSS animation; the composer
#: caret's blink is infinite, always running, and not motion, so it is excluded by its end time.
#: One definition, spliced into both probes below, so the frame's recorded number and the
#: stillness wait cannot disagree about what counts.
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
            left: Math.round(r.left), right: Math.round(r.right), h: Math.round(r.height)};
  };
  const dialog = document.querySelector('[role="dialog"]');
  const dock = document.querySelector('[data-testid="ask-dock"]');
  const ta = document.querySelector('textarea');
  const active = document.activeElement;
  const title = dialog ? dialog.querySelector('[id]') : null;
  const panel = dialog ? dialog.querySelector('.lo-sheet-panel') : null;
  const siblings = dialog ? [...dialog.parentElement.children].filter((n) => n !== dialog) : [];
  return JSON.stringify({
    hash: location.hash,
    viewport: [window.innerWidth, window.innerHeight],
    dialog: Boolean(dialog),
    title: title ? title.textContent.trim() : null,
    panel: box(panel),
    cards: dialog ? dialog.querySelectorAll('[data-testid="ask-card"]').length : 0,
    loading: dialog ? /reading asks/.test(dialog.textContent) : false,
    animating: __RUNNING__,
    inertSiblings: siblings.filter((n) => n.hasAttribute('inert')).length,
    siblings: siblings.length,
    dock: Boolean(dock),
    dockText: dock ? dock.textContent.trim().slice(0, 90) : null,
    dockBox: box(dock),
    composer: ta ? {value: ta.value, focused: active === ta, box: box(ta)} : null,
    active: active ? {tag: active.tagName.toLowerCase(),
                      label: active.getAttribute('aria-label')} : null,
    history: {length: history.length, askSheet: Boolean(history.state && history.state.askSheet)},
  });
})()
"""


PROBE = PROBE.replace("__RUNNING__", RUNNING_ANIMATIONS_JS)


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


CLOSE_MARK = 'document.querySelector(\'[role="dialog"] button[aria-label="close sheet"]\')'
BACK_TO_LIST = "document.querySelector('button[aria-label=\"back to sessions\"]')"
COMPOSER = "document.querySelector('textarea')"


def still_centre(page: Page, selector_js: str, seconds: float = 5.0) -> dict[str, Any] | None:
    """The element's centre once it has stopped moving, or ``None`` when there is no element.

    A finger lands where the control IS when it lands, not where it was when someone last
    looked, so a tap is aimed only after two samples agree. The sheet's close mark sits in a
    panel that rises for 240 ms; a rect read mid-rise and tapped 70 ms later is a tap on
    whatever has moved under that point. If the element never holds still within ``seconds``
    the last position is returned and the post-tap probe says what the tap did.
    """
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
    """One REAL finger tap on the element, or ``None`` when there is nothing to tap.

    ``touchStart`` then ``touchEnd`` at the element's centre: Chrome turns that into the
    pointer events and the click a thumb produces, so the sheet's release-click guard and
    the list row's long-press timer both see what they see in use.
    """
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
    """Poll for a dialog for ``seconds``; the ms until it appeared, or ``None``.

    Every frame gets the same observation window on both builds, so "it never opened" means
    "nothing opened in three seconds of looking", not "the shutter beat it".
    """
    start = time.time()
    while time.time() - start < seconds:
        if page.js("document.querySelector('[role=\"dialog\"]') !== null"):
            return round((time.time() - start) * 1000)
        time.sleep(0.05)
    return None


#: One sample of "is the page still moving": finite animations still running (the sheet's 240 ms
#: rise is a CSS animation; the caret's blink is infinite and is not motion), the sheet panel's
#: box, and whether the sheet is still reading. Read through the Web Animations API, which is the
#: browser's own answer, rather than a sleep sized to what the animation was last measured at.
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


def quiesce(page: Page, seconds: float = SETTLE_S) -> int | None:
    """Wait until the page has STOPPED MOVING; the ms it took, or ``None`` if it never did.

    "Stopped" means all three at once, for ``STILL_SAMPLES`` consecutive samples: no finite
    animation running, the sheet (if any) is not still reading the aggregate, and the sheet's
    panel is where it was the sample before.

    THE FIRST FRAME IS NOT THE SETTLED FRAME (AGENTS.md section 5), and on this rig it was
    wrong twice, both only under host load (measured: load 100+, where the compositor lags):

    * the sheet mounts the moment the policy decides, but its rows are a separate read that
      lands afterwards, so "pending on open" photographed "reading asks..." under a probe that
      still said ``dialog: true``;
    * the panel rises over 240 ms, and a frame (and a TAP) taken mid-rise lands on a panel that
      is not where it was measured: the close tap aimed at the x's position of one sample hit
      content that had moved under it, the sheet stayed up, and the "closed by the user" frame
      showed an open sheet. The capture's own ``--expect`` gate caught it and went red, which
      is the reason that gate exists.

    Waiting for the browser's own animation state, not for a sleep sized to the last
    measurement, is what makes the same script right on a quiet host and a loaded one.
    """
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
    """The fixture process, with its one-line-per-command stdin channel.

    stdout carries only ``Fixture ...`` banners and ``ack``/``error`` lines, read here with
    ``select`` on the raw descriptor so a command that never answers is a timeout and not a
    hung capture; stderr goes to a file next to the frames, where a startup failure is
    readable rather than lost.
    """

    def __init__(self, port: int, password: str, log_path: Path) -> None:
        env = {**os.environ, "LOP_MOBILE_FIXTURE_PASSWORD": password}
        self._log = log_path.open("wb")
        self.proc = subprocess.Popen(
            [sys.executable, str(REPO_ROOT / "scripts/mobile_asks_open_fixture.py"), str(port)],
            cwd=REPO_ROOT,
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

    def command(self, text: str, timeout: float = 20.0) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write((text + "\n").encode())
        self.proc.stdin.flush()
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self._read_line(deadline - time.time())
            if line is None:
                break
            if line == f"ack {text}":
                return
            if line.startswith("error "):
                raise RuntimeError(line)
        raise RuntimeError(f"the fixture did not acknowledge {text!r} within {timeout:.0f}s")

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


def expectations(build: str) -> dict[str, dict[str, Any]]:
    """What each frame's probe must say on THIS build (``before`` or ``after``).

    ``after`` is the contract: the sheet opens once for a conversation with asks waiting,
    stays shut for none/all-addressed, and a close the user made is respected. ``before`` is
    the base build: nothing ever opens by itself, so every frame is the dock alone.
    """
    after = build == "after"
    still = {"animating": 0, "loading": False}
    table = {
        "01-no-asks": {"dialog": False, "dock": False},
        "03-all-addressed": {"dialog": False, "dock": False},
        "02-pending-on-open": {"dialog": after, "dock": True},
        "04a-closed-by-the-user": {"dialog": False, "dock": True},
        "04b-after-a-re-render": {"dialog": False, "dock": True},
        "04c-after-a-new-ask": {"dialog": False, "dock": True},
        "04d-after-away-and-back": {"dialog": False, "dock": True},
        "05-typing-then-asks-arrive": {"dialog": False, "dock": True, "composer.focused": True},
    }
    # EVERY frame is a still one: nothing animating and no read in flight. A frame photographed
    # mid-motion is a frame of the transition, not of the state its name claims.
    return {name: {**still, **want} for name, want in table.items()}


def _dig(data: dict[str, Any], dotted: str) -> Any:
    """``a.b.c`` into nested probe dicts; ``None`` where any step is absent or not a dict.

    ``None`` is a real answer here (the composer is ``null`` on a screen with no textarea),
    and an expectation about a field that is not there must come out as a MISMATCH, never as
    a crash that hides which frame was being checked.
    """
    current: Any = data
    for part in dotted.split("."):
        current = current.get(part) if isinstance(current, dict) else None
    return current


def main() -> None:
    if len(sys.argv) < 3 or "--expect" not in sys.argv:
        raise SystemExit(__doc__)
    outdir, label = Path(sys.argv[1]), sys.argv[2]
    build = sys.argv[sys.argv.index("--expect") + 1]
    if build not in ("before", "after"):
        raise SystemExit("--expect must be 'before' or 'after'")
    theme = sys.argv[sys.argv.index("--theme") + 1] if "--theme" in sys.argv else ""
    outdir.mkdir(parents=True, exist_ok=True)
    expected = expectations(build)

    # A THROWAWAY PASSWORD, GENERATED PER RUN AND NEVER PRINTED: handed to the fixture through
    # the environment (not argv, so it is not in `ps`) and typed into the login form below.
    password = secrets.token_urlsafe(16)
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    fixture = Fixture(port, password, outdir / f"{label}-fixture.log")
    chrome = Chrome()
    report: dict[str, Any] = {
        "build": build,
        "fixture": fixture.banner,
        "viewport": VIEWPORT,
        "theme": theme or "default",
    }
    failures: list[str] = []

    def frame(name: str, **extra: Any) -> dict[str, Any]:
        """Photograph the current state and check it against what this build must show."""
        settled_ms = quiesce(page)
        data = {**probe(page), "settledAfterMs": settled_ms, **extra}
        page.shot(outdir / f"{label}-{name}.png")
        wrong = {
            key: {"want": want, "got": _dig(data, key)}
            for key, want in expected[name].items()
            if _dig(data, key) != want
        }
        if settled_ms is None:
            wrong["settledAfterMs"] = {"want": "a settled page", "got": None}
        data["expected"] = expected[name]
        data["mismatch"] = wrong
        report[name] = data
        verdict = "PASS" if not wrong else f"FAIL {wrong}"
        print(
            f"{name:<28} dialog={data['dialog']!s:<5} dock={data['dock']!s:<5} "
            f"title={data['title']!r} dockText={data['dockText']!r} {verdict}",
            flush=True,
        )
        if wrong:
            failures.append(name)
        return data

    def open_from_list(name: str) -> int | None:
        """Tap the conversation's row and watch for a dialog; the ms until one appeared."""
        tapped = tap(page, row_selector(name))
        if tapped is None:
            raise RuntimeError(f"no list row reads {name!r}; the fixture is not what this expects")
        appeared = watch(page)
        if appeared is not None:
            report[f"{name}-settled-after-ms"] = quiesce(page)
        return appeared

    def back_to_list() -> None:
        # A dialog (after) has made the header inert, so a tap on it would land on the
        # scrim; the user closes the sheet first, and that close is itself a dismissal. The
        # steps that call this either have no dialog up or have already closed it.
        if tap(page, BACK_TO_LIST) is None:
            raise RuntimeError("no back control on the conversation screen")
        time.sleep(1.2)

    try:
        fixture_page = Page(chrome.target_ws())
        page = fixture_page
        width, height = VIEWPORT
        page.metrics(width, height)
        if theme:
            # The palette is a boot-time read (``theme.ts`` initTheme, before the first
            # paint), so the item has to exist before the first document script runs; an
            # on-new-document script is the only hook that early, and it survives the
            # run's reloads. The key is the app's own (lo-mobile-theme).
            page.send(
                "Page.addScriptToEvaluateOnNewDocument",
                source=(
                    "try { localStorage.setItem('lo-mobile-theme', "
                    + json.dumps(theme)
                    + "); } catch (error) {}"
                ),
            )
        page.goto(f"{base}/login")
        page.js(
            "(() => { const f = document.querySelector('form');"
            f" f.password.value = {password!r}; f.submit(); return true; }})()"
        )
        time.sleep(2.0)
        page.goto(f"{base}/#/")
        time.sleep(1.5)

        # 01 — nothing waiting.
        report["01-opened-after-ms"] = open_from_list(NONE)
        frame("01-no-asks")
        back_to_list()

        # 03 — every ask already answered or declined.
        report["03-opened-after-ms"] = open_from_list(SETTLED)
        frame("03-all-addressed")
        back_to_list()

        # 02 — asks waiting: this is the frame that changes.
        before_entries = probe(page)["history"]["length"]
        report["02-opened-after-ms"] = open_from_list(PENDING)
        two = frame("02-pending-on-open", historyBeforeOpen=before_entries)

        # 04a — the user closes it. A real tap on the close mark; on the before build there is
        # nothing to close, which is recorded rather than faked.
        closed = tap(page, CLOSE_MARK)
        time.sleep(1.0)
        frame("04a-closed-by-the-user", closedByTap=closed is not None, openBefore=two["dialog"])

        # 04b — the queue re-publishes unchanged: a re-render with no queue change.
        fixture.command("pending-refresh")
        time.sleep(1.5)
        frame("04b-after-a-re-render")

        # 04c — a genuinely NEW ask arrives while the phone watches. The dock counts it; the
        # sheet does not come back.
        fixture.command("pending-new-ask")
        time.sleep(1.5)
        frame("04c-after-a-new-ask")

        # 04d — away and back within the same page lifetime: the screen remounts (the app
        # keys it by session), so only the module-level record can carry the refusal.
        back_to_list()
        report["04d-opened-after-ms"] = open_from_list(PENDING)
        frame("04d-after-away-and-back")
        back_to_list()

        # 05 — the user is typing when the asks land. Nothing may be taken from them.
        open_from_list(TYPING)
        focused = tap(page, COMPOSER)
        page.send("Input.insertText", text="dry run on staging")
        time.sleep(0.4)
        fixture.command("typing-arrive")
        time.sleep(2.0)
        frame("05-typing-then-asks-arrive", composerTapped=focused is not None)
        page.close()
    finally:
        try:
            chrome.close()
        finally:
            fixture.close()
    (outdir / f"{label}-open-by-default-geometry.json").write_text(json.dumps(report, indent=2))
    print(f"\nbuild={build} frames={len(expected)} failures={failures or 'none'}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
