"""Fifth probe: the Esc exit, and the reachability of the two edited strings.

Esc is advertised in the ONLY guidance the masked state shows
("... Esc cancels"). This measures what the operator is left holding after
pressing it, and what the very next keystroke does with that.

Also settles, by exhaustive route, whether ``CREDENTIAL_ARMED_NOTICE`` and
``CREDENTIAL_PLACEHOLDER`` — both of which this PR rewrote — paint anywhere.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

for _key in tuple(os.environ):
    if _key.startswith("CMUX_"):
        os.environ.pop(_key)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio  # noqa: E402

from textual import events  # noqa: E402

import scripts.probe_isolation  # noqa: E402, F401
from local_operator.tui.app import (  # noqa: E402
    COMPOSER_CREDENTIAL_CLASS,
    CREDENTIAL_ARMED_NOTICE,
    CREDENTIAL_CHEVRON,
    CREDENTIAL_PLACEHOLDER,
    CREDENTIAL_TYPING_NOTICE,
    PROMPT_CHEVRON,
    OperatorApp,
)
from scripts.visual_capture import save_capture  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402
from tests.unit.tui.test_slash_echo import _boot  # noqa: E402

SECRET = "hunter2-typed-test"
OUT = Path("/tmp/design891-frames/probe5")


def rows(app):
    return [strip.text for strip in app.screen._compositor.render_strips()]


def marker(app):
    """The glyph actually painted in the composer's prompt cell."""
    for line in rows(app):
        stripped = line.strip()
        if stripped.startswith((PROMPT_CHEVRON, CREDENTIAL_CHEVRON, "$")) and (
            "Message Local Operator" in line
            or "/credential" in line
            or SECRET in line
            or "Credential #" in line
        ):
            return stripped[0]
    return None


def dock_class(app):
    try:
        return COMPOSER_CREDENTIAL_CLASS in app.query_one("#input-dock").classes
    except Exception:
        return "n/a"


async def type_text(pilot, text):
    for ch in text:
        await pilot.press(ch)
    for _ in range(3):
        await pilot.pause()


async def esc_exit(label: str, prefix: str) -> None:
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 36)) as pilot:
        await _boot(pilot, app)
        editor = app._editor()
        editor.focus()
        await type_text(pilot, f"{prefix}/credential ")
        await type_text(pilot, SECRET)
        print(f"\n  == {label} ==")
        print(f"     masked        : buffer={editor.text!r}")
        print(f"     marker glyph  : {marker(app)!r}   armed-class={dock_class(app)}")

        await pilot.press("escape")
        for _ in range(10):
            await pilot.pause()
        save_capture(app, OUT / f"esc-{label}-1-after-esc.svg")
        frame = "\n".join(rows(app))
        print(f"     after Esc     : buffer={editor.text!r}")
        print(f"     secret PAINTED IN COMPOSER: {SECRET in frame}")
        print(f"     marker glyph  : {marker(app)!r}   armed-class={dock_class(app)}")
        print(f"     TYPING notice : {CREDENTIAL_TYPING_NOTICE in frame}")
        warn = [
            ln.strip()
            for ln in rows(app)
            if any(w in ln.lower() for w in ("cancel", "plaintext", "visible", "no longer"))
        ]
        print(f"     any warning   : {warn}")

        await pilot.press("enter")
        for _ in range(30):
            await pilot.pause()
        save_capture(app, OUT / f"esc-{label}-2-after-enter.svg")
        frame = "\n".join(rows(app))
        print(f"     -> next Enter : buffer={editor.text!r}")
        print(f"        secret painted anywhere: {SECRET in frame}")
        print(f"        secret UPPERCASED as a key: {SECRET.upper().replace('-', '_') in frame}")
        print(f"        store: {session.variables.credential_names()}")
        for line in rows(app):
            if line.strip() and (SECRET in line or "Paste the value" in line or "▌" in line):
                print(f"          row: {line.strip()!r}")


async def reachability() -> None:
    """Exhaustive: does either edited string paint on ANY route?"""
    print("\n=== REACHABILITY of the two OTHER strings this PR rewrote ===")
    print(f"  ARMED_NOTICE = {CREDENTIAL_ARMED_NOTICE!r}")
    print(f"  PLACEHOLDER  = {CREDENTIAL_PLACEHOLDER!r}")

    async def route(label, drive):
        session = FakeSession()
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(120, 36)) as pilot:
            await _boot(pilot, app)
            editor = app._editor()
            editor.focus()
            await drive(pilot, app, editor, session)
            for _ in range(12):
                await pilot.pause()
            frame = "\n".join(rows(app))
            print(
                f"  {label:38s} ARMED={CREDENTIAL_ARMED_NOTICE in frame:d} "
                f"PLACEHOLDER={CREDENTIAL_PLACEHOLDER in frame:d} "
                f"TYPING={CREDENTIAL_TYPING_NOTICE in frame:d}"
            )

    async def space(pilot, app, editor, session):
        await type_text(pilot, "/credential ")

    async def no_space_then_paste(pilot, app, editor, session):
        await type_text(pilot, "/credential")
        app.post_message(events.Paste(SECRET))

    async def space_then_paste(pilot, app, editor, session):
        await type_text(pilot, "/credential ")
        app.post_message(events.Paste(SECRET))

    async def chip_then_reopen(pilot, app, editor, session):
        await type_text(pilot, "/credential ")
        await type_text(pilot, SECRET)
        await pilot.press("enter")
        for _ in range(10):
            await pilot.pause()
        await type_text(pilot, " and /credential ")

    async def empty_composer(pilot, app, editor, session):
        return

    async def alias_space(pilot, app, editor, session):
        await type_text(pilot, "/cred ")

    for label, drive in (
        ("empty composer (placeholder shown?)", empty_composer),
        ("/credential + space", space),
        ("/cred + space", alias_space),
        ("/credential (no space) then paste", no_space_then_paste),
        ("/credential + space then paste", space_then_paste),
        ("chip minted, then a second arm", chip_then_reopen),
    ):
        await route(label, drive)


async def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== WHAT 'Esc cancels' LEAVES THE OPERATOR HOLDING ===")
    await esc_exit("bare", "")
    await esc_exit("prose", "please store ")
    await reachability()


if __name__ == "__main__":
    asyncio.run(main())
