"""Real-execution evidence for the ``/settings`` page and the #369 grammar.

Not a test. This drives the REAL app against a scratch config directory and
prints the actual ``config.yml`` diff each interaction produced, because a
green unit suite proves the code does what its author expected and not that
the feature works. Run:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python scripts/settings_evidence.py

Every case prints the before/after unified diff of the real file, including
the NEGATIVE cases — an invalid value writing nothing, and a bare
``/model default`` leaving the file byte-identical.
"""

from __future__ import annotations

import asyncio
import difflib
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_SCRATCH = Path(tempfile.mkdtemp(prefix="lo-settings-evidence-"))
os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = str(_SCRATCH)

from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.settings_view import SettingsView  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

CONFIG = _SCRATCH / "config.yml"


def read() -> str:
    return CONFIG.read_text() if CONFIG.exists() else ""


def report(title: str, before: str, after: str, *, expect_change: bool) -> bool:
    """Print the real diff and say whether it matched what was claimed."""
    diff = [
        line.rstrip()
        for line in difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            "config.yml (before)",
            "config.yml (after)",
            lineterm="",
        )
        # The mtime stamp changes on every write and is noise here.
        if "last_modified" not in line
    ]
    body = [
        line for line in diff if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    ]
    changed = bool(body)
    ok = changed == expect_change
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")
    if diff:
        print("\n".join(diff))
    else:
        print("(config.yml byte-identical)")
    verdict = "as expected" if ok else "UNEXPECTED"
    print(f"-> wrote={changed}, expected={expect_change} :: {verdict}")
    return ok


def select(view: SettingsView, key: str) -> None:
    for index, row in enumerate(view._rows):
        if row.kind == "setting" and row.setting is not None and row.setting.key == key:
            view._selected = index
            view._repaint()
            return
    raise SystemExit(f"no row for {key}")


async def main() -> None:
    results: list[bool] = []
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 34)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()

        # -- BOOL: Enter toggles --------------------------------------------
        before = read()
        select(view, "display.shimmer")
        view.action_activate()
        await pilot.pause()
        results.append(
            report("BOOL — Enter on display.shimmer", before, read(), expect_change=True)
        )

        # -- ENUM: expand, then commit a choice ------------------------------
        before = read()
        select(view, "tool_approval_mode")
        view.action_activate()
        await pilot.pause()
        for index, row in enumerate(view._rows):
            if row.kind == "choice" and row.choice is not None and row.choice.value == "auto":
                view._selected = index
                break
        view.action_activate()
        await pilot.pause()
        results.append(
            report("ENUM — choose tool_approval_mode=auto", before, read(), expect_change=True)
        )

        # -- INT: type a value and save --------------------------------------
        before = read()
        select(view, "retry.maxRetries")
        view.action_activate()
        view._buffer = "6"
        view._commit_edit()
        await pilot.pause()
        results.append(
            report("INT — retry.maxRetries=6 (siblings)", before, read(), expect_change=True)
        )

        # -- NEGATIVE: an invalid value writes NOTHING ------------------------
        before = read()
        select(view, "retry.maxRetries")
        view.action_activate()
        view._buffer = "9999"
        view._commit_edit()
        await pilot.pause()
        held_open = view.editing_key == "retry.maxRetries"
        results.append(
            report("NEGATIVE — retry.maxRetries=9999 rejected", before, read(), expect_change=False)
        )
        print(f"-> editor still open: {held_open}; error on screen: {view.error_text!r}")
        results.append(held_open and "at most 100" in view.error_text)
        view._cancel_edit()

        # -- TEXT: free text -------------------------------------------------
        before = read()
        select(view, "web_search.searxng_endpoint")
        view.action_activate()
        view._buffer = "https://searx.example.org"
        view._commit_edit()
        await pilot.pause()
        results.append(
            report("TEXT — web_search.searxng_endpoint", before, read(), expect_change=True)
        )

        # -- LIST: ordered members -------------------------------------------
        before = read()
        select(view, "web_search.providers")
        view.action_activate()
        view._buffer = "brave, exa"
        view._commit_edit()
        await pilot.pause()
        results.append(
            report("LIST — web_search.providers (order)", before, read(), expect_change=True)
        )

        # -- NEGATIVE: an unknown list member --------------------------------
        before = read()
        select(view, "web_search.providers")
        view.action_activate()
        view._buffer = "bing"
        view._commit_edit()
        await pilot.pause()
        results.append(
            report("NEGATIVE — providers=bing rejected", before, read(), expect_change=False)
        )
        print(f"-> error on screen: {view.error_text!r}")
        view._cancel_edit()

        # -- CASCADE: add a hop ----------------------------------------------
        before = read()
        for index, row in enumerate(view._rows):
            if row.kind == "chain_add":
                view._selected = index
                break
        view.action_activate()
        view._buffer = "default anthropic/claude-opus-5"
        view._commit_edit()
        await pilot.pause()
        results.append(
            report("CASCADE — create a chain + first hop", before, read(), expect_change=True)
        )

        # -- RESET: back to the shipped default -------------------------------
        before = read()
        select(view, "display.shimmer")
        view.action_reset()
        await pilot.pause()
        results.append(
            report("RESET — r deletes display.shimmer", before, read(), expect_change=True)
        )

        # -- FLAT-KEY PROOF: the reader sees it, and it is not nested ---------
        from local_operator.tui.settings import settings_get

        select(view, "display.terminal_title")
        view.action_activate()
        await pilot.pause()
        import yaml

        values = yaml.safe_load(read())["values"]
        flat = values.get("display.terminal_title")
        nested = values.get("display")
        print(f"\n{'=' * 72}\nFLAT-KEY TRAP — display.terminal_title\n{'=' * 72}")
        print(f"values['display.terminal_title'] = {flat!r}  (the key tui/settings.py reads)")
        print(f"values['display'] = {nested!r}  (must be None: nothing reads it)")
        seen = settings_get("display.terminal_title")
        print(f"settings_get('display.terminal_title') = {seen!r}")
        results.append(flat is False and nested is None and seen is False)

        app._close_settings_view()
        await pilot.pause()

    # -- #369: the grammar table, each row against the real file -------------
    print(f"\n\n{'#' * 72}\n# issue 369 — the grammar table\n{'#' * 72}")

    session_app = OperatorApp(lambda: _factory(FakeSession()))
    from tests.unit.tui.test_app_pilot import _AccessController, _SwitchableSession

    session = _SwitchableSession()
    ctrl = _AccessController(stored=("openrouter", "anthropic"))
    app = OperatorApp(lambda: _factory(session), provider_controller=ctrl)
    del session_app
    async with app.run_test(size=(110, 30)) as pilot:
        await pilot.pause()

        # /model <p>/<id> — switches the session only.
        before = read()
        app._run_slash_command("/model anthropic/claude-opus-5")
        await pilot.pause()
        results.append(
            report("/model <p>/<id> — session only", before, read(), expect_change=False)
        )
        print(f"-> session model is now {session.model_label!r}")

        # Bare /model default — THE FIX. Confirms; writes nothing.
        before = read()
        app._run_slash_command("/model default")
        await pilot.pause()
        results.append(
            report("/model default (bare) — CONFIRMS (#369)", before, read(), expect_change=False)
        )
        transcript = _transcript(app)
        print(f"-> notice: {transcript.strip().splitlines()[-1].strip()[:110]}")

        # /model default <p>/<id> — unchanged, still writes.
        before = read()
        app._run_slash_command("/model default anthropic/claude-opus-5")
        await pilot.pause()
        results.append(
            report("/model default <p>/<id> — writes", before, read(), expect_change=True)
        )

        # /model saved — switches BACK to the configured default.
        app._run_slash_command("/model openrouter/deepseek/deepseek-chat")
        await pilot.pause()
        moved_to = session.model_label
        before = read()
        app._run_slash_command("/model saved")
        await pilot.pause()
        results.append(report("/model saved — switches back", before, read(), expect_change=False))
        print(f"-> was {moved_to!r}, now {session.model_label!r}")
        results.append(session.model_label == "anthropic/claude-opus-5")

    print(f"\n{'=' * 72}")
    print(f"{sum(results)}/{len(results)} checks matched what was claimed")
    print(f"scratch config: {CONFIG}")
    if not all(results):
        raise SystemExit(1)


def _transcript(app) -> str:
    from tests.unit.tui.test_app_pilot import _transcript_text

    return _transcript_text(app)


asyncio.run(main())
