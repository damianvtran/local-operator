"""Capture the phone's Projects sheet out of headless Chrome over CDP.

Run:  PYTHONPATH=. .venv/bin/python scripts/mobile_projects_shot.py OUTDIR

Starts ``scripts/mobile_projects_fixture.py`` (the REAL bundle over a seeded
project store written through the real ``ProjectRegistry``), then photographs
the sheet's five states the design's evidence plan names — list, compact board,
detail, create, delete — at one phone viewport. ``Chrome``/``Page`` are reused
from ``scripts/mobile_overflow_capture.py``: one throwaway ``--headless=new``
browser per run, its own profile, ``--use-mock-keychain``, and a teardown that
asserts no helper survived.

WHAT THE RUN ALSO PROVES, not just photographs: the milestone toggle is driven
through the real POST and the frame after it must show the server's answer (the
glyph and the date flip), and the create/delete receipts come back through the
real routes. The geometry dump beside the frames records the sheet panel's box,
the scroller's extent and every button's height, so "a 44px tap target" is a
measured statement rather than a claim about a PNG.
"""

from __future__ import annotations

import json
import secrets
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from scripts.mobile_overflow_capture import Chrome, Page

VIEWPORT = (390, 844)

GEOMETRY_JS = """
(() => {
  const panel = document.querySelector('.lo-sheet-panel');
  const scroller = panel ? panel.querySelector('.lo-scroll') : null;
  const buttons = panel
    ? [...panel.querySelectorAll('button')].map((b) => ({
        text: (b.textContent || '').trim().slice(0, 70),
        height: Math.round(b.getBoundingClientRect().height),
        disabled: b.disabled,
      }))
    : [];
  const headings = panel
    ? [...panel.querySelectorAll('h3')].map((h) => (h.textContent || '').trim())
    : [];
  return JSON.stringify({
    viewport: [window.innerWidth, window.innerHeight],
    panel: panel
      ? {
          top: Math.round(panel.getBoundingClientRect().top),
          height: Math.round(panel.getBoundingClientRect().height),
        }
      : null,
    scroller: scroller
      ? { scrollH: scroller.scrollHeight, clientH: scroller.clientHeight }
      : null,
    headings,
    buttons,
  });
})()
"""


def must_tap(page: Page, label: str, *, exact: bool = False) -> dict[str, Any]:
    """``_tap`` with the refusal made loud: the choices seen ride in the error."""
    outcome = json.loads(_tap(page, label, exact=exact))
    if not outcome.get("tapped"):
        raise RuntimeError(f"could not tap {label!r} (exact={exact}): {outcome}")
    return outcome


def must_type(page: Page, placeholder: str, text: str) -> dict[str, Any]:
    """``_type`` with the same loud refusal."""
    outcome = json.loads(_type(page, placeholder, text))
    if not outcome.get("typed"):
        raise RuntimeError(f"could not type into {placeholder!r}: {outcome}")
    return outcome


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _tap(page: Page, label: str, *, exact: bool = False) -> Any:
    """Tap a button inside the sheet whose text the label names.

    ``element.click()`` rather than a CDP touch event: this is a TAP on a
    React button, not a scroll, and the cheat this harness refuses is assigning
    ``scrollTop`` past a gesture (see ``mobile_overflow_capture``'s module
    docstring) — a programmatic click on a real button fires the real handler
    with no such shortcut available. Exact matching for the chrome controls
    ("list"/"board"/"new project"), substring matching for rows.
    """
    match = (
        f"(n.textContent || '').trim() === {label!r}"
        if exact
        else f"(n.textContent || '').includes({label!r})"
    )
    return page.js(
        "(() => {"
        # The sheet's panel when one is open, the whole screen otherwise (the
        # footer entry that opens it lives outside the panel).
        " const scope = document.querySelector('.lo-sheet-panel') || document;"
        " const nodes = [...scope.querySelectorAll('button')];"
        f" const hit = nodes.find((n) => {match});"
        " if (!hit) return JSON.stringify({ tapped: false,"
        f" choices: nodes.map((n) => (n.textContent || '').trim().slice(0, 40)) }});"
        " hit.click();"
        " return JSON.stringify({ tapped: true });"
        "})()"
    )


def _type(page: Page, placeholder: str, text: str) -> Any:
    """Type into an input the way a keyboard would: set the value through the
    NATIVE setter (React tracks the value property, so a plain assignment is
    swallowed) and dispatch a bubbling ``input`` event."""
    selector = f'input[placeholder="{placeholder}"]'
    return page.js(
        "(() => {"
        " const panel = document.querySelector('.lo-sheet-panel');"
        f" const input = panel && panel.querySelector({json.dumps(selector)});"
        " if (!input) return JSON.stringify({ typed: false });"
        " const proto = Object.getOwnPropertyDescriptor("
        "window.HTMLInputElement.prototype, 'value');"
        f" proto.set.call(input, {json.dumps(text)});"
        " input.dispatchEvent(new Event('input', { bubbles: true }));"
        " return JSON.stringify({ typed: true });"
        "})()"
    )


def main() -> None:
    outdir = Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    # A THROWAWAY PASSWORD, GENERATED PER RUN AND NEVER PRINTED — the same
    # contract as the sibling capture scripts; the fixture receives it as an
    # argument, this module types it into the login form, and neither prints it.
    password = secrets.token_urlsafe(16)
    fixture = subprocess.Popen(
        [sys.executable, "scripts/mobile_projects_fixture.py", str(port), password],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    chrome = Chrome()
    report: dict[str, Any] = {}
    try:
        deadline = time.time() + 45
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"{base}/login", timeout=2):
                    break
            except Exception:  # noqa: BLE001  -- not up yet
                time.sleep(0.4)
        else:
            raise RuntimeError("fixture never served /login")

        page = Page(chrome.target_ws())
        page.metrics(*VIEWPORT)
        page.goto(f"{base}/login")
        page.js(
            "(() => { const f = document.querySelector('form');"
            f" f.password.value = {password!r}; f.submit(); return true; }})()"
        )
        # Let the POST land and the 303 settle before navigating: an immediate
        # `goto` races the form submission and aborts it, which reads as a
        # rejected password (the sibling capture scripts carry the same sleep).
        time.sleep(2.0)
        page.goto(f"{base}/#/")
        time.sleep(1.5)

        def snap(name: str, note: str = "") -> None:
            time.sleep(1.0)
            report[name] = {"note": note, **json.loads(page.js(GEOMETRY_JS))}
            page.shot(outdir / f"{name}.png")
            print(f"{name}: {note}", flush=True)

        # 0. The sessions screen carrying the new footer entry.
        snap("00-entry", "sessions screen: the projects entry in the footer")

        # 1. The entry point and the list.
        must_tap(page, "projects", exact=True)
        time.sleep(1.2)
        snap("01-list", "footer entry opened the sheet; list rows")

        # 2. The compact board: same rows under status headings.
        must_tap(page, "board", exact=True)
        snap("02-board", "status-grouped sections")

        # 3. Detail: progress + age, milestones, linked sessions.
        must_tap(page, "list", exact=True)
        time.sleep(0.5)
        must_tap(page, "payments-migration")
        time.sleep(1.2)
        snap("03-detail", "progress, milestones, sessions with state")

        # 4. Milestone toggle: the server's answer must repaint the row.
        must_tap(page, "audit")
        snap("04-milestone-toggled", "after tapping the overdue milestone")

        # 5. Create: form, then the receipt + refreshed list.
        must_tap(page, "projects")  # back from detail
        time.sleep(0.5)
        must_tap(page, "new project", exact=True)
        time.sleep(0.5)
        must_type(page, "e.g. payments-migration", "capture-demo")
        snap("05-create", "create form with a typed name")
        must_tap(page, "create", exact=True)
        snap("06-list-after-create", "receipt + the new row")

        # 6. Delete: open the new row, confirm, and show the receipt.
        time.sleep(0.5)
        must_tap(page, "capture-demo")
        time.sleep(1.0)
        must_tap(page, "delete project", exact=True)
        snap("07-delete-confirm", "confirm view")
        must_tap(page, "delete", exact=True)
        snap("08-list-after-delete", "receipt + the row gone")

        (outdir / "projects-mobile-geometry.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print("shots:", ", ".join(f"{key}.png" for key in report))
    finally:
        chrome.close()
        fixture.terminate()
        try:
            fixture.wait(timeout=10)
        except subprocess.TimeoutExpired:
            fixture.kill()


if __name__ == "__main__":
    main()
