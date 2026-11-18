"""The ``/move --to`` working-set card (docs/design/move-working-set.md §11.3).

The operator's requirement (2026-11-18): the remote move must warn that moving
mid-conversation involves the working files and require the user's explicit approval
in the dialog. Pinned on the REAL ``OperatorApp`` (so ``local_operator.tcss`` applies):

* the card names the destination device and renders the CLI's summary — worktree,
  counts, the withheld secret, what is never copied;
* Move is disabled until the checkbox is ticked, and focus starts on the checkbox;
* Esc / Cancel dismiss ``False``; a ticked Move dismisses ``True``;
* an ``unknown`` manifest offers no consent at all (only Cancel).
"""

from __future__ import annotations

from typing import Any

import pytest
from textual.widgets import Button, Checkbox, Static

from local_operator.i18n.text import set_active_locale
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.move_workset_confirm import MoveWorkingSetConfirmScreen
from tests.unit.tui.test_app_pilot import FakeSession, _factory

SESSION = "9f3ac1e0b7d2"

BLOCK: dict[str, Any] = {
    "version": 1,
    "scope_digest": "sha256:" + "a" * 64,
    "state": "preflight",
    "carries_files": False,
    "destination": {"device_id": "d_b", "name": "lop-remote-node-2"},
    # The OWNER, named: on a recall this card runs on the destination, so the files
    # staying on "this device" would name the wrong machine.
    "source": {"device_id": "d_a", "name": "laptop"},
    "totals": {"worktrees": 1, "unpushed_commits": 1, "dirty_files": 6, "bytes": 3_200_000},
    "worktrees": [
        {
            "path": "~/proj/.worktrees/feat",
            "repo": "~/proj",
            "head": "0" * 40,
            "state": "planned",
            "branch": "feat",
            "unpushed_commits": 1,
            "staged": 2,
            "unstaged": 2,
            "untracked": 2,
        }
    ],
    "excluded": [{"path": "~/proj/.worktrees/feat/.env.local", "why": "secret_path"}],
    "left_behind": [
        {
            "path": "~/proj/.worktrees/feat/node_modules",
            "kind": "dependency_tree",
            "recipe": {"kind": "pnpm", "run": "pnpm install --frozen-lockfile", "cwd": "~"},
        }
    ],
}


@pytest.fixture(autouse=True)
def _english() -> Any:
    set_active_locale("en")
    yield
    set_active_locale("en")


async def _open(app: OperatorApp, block: dict[str, Any], results: list[bool]) -> Any:
    screen = MoveWorkingSetConfirmScreen(block, session_id=SESSION, device="peer")
    app.push_screen(screen, callback=results.append)
    return screen


@pytest.mark.asyncio
async def test_the_card_renders_the_summary_and_names_the_destination() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    results: list[bool] = []
    async with app.run_test(size=(100, 32)) as pilot:
        await pilot.pause()
        screen = await _open(app, BLOCK, results)
        await pilot.pause()
        title = str(screen.query_one("#move-workset-title", Static).render())
        summary = str(screen.query_one("#move-workset-summary", Static).render())
        check = screen.query_one("#move-workset-check", Checkbox)
    assert title == "Move without the working files?"
    assert f"Moving {SESSION} to lop-remote-node-2 mid-conversation" in summary
    assert "~/proj/.worktrees/feat  feat (1 unpushed, 2 staged, 2 unstaged, 2 untracked)" in summary
    assert "withheld (looks like a secret): ~/proj/.worktrees/feat/.env.local" in summary
    assert "pnpm install --frozen-lockfile" in summary
    assert "They stay on laptop:" in summary
    assert str(check.label) == "I understand these working files stay on laptop"


@pytest.mark.asyncio
async def test_move_is_disabled_until_the_checkbox_is_ticked() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    results: list[bool] = []
    async with app.run_test(size=(100, 32)) as pilot:
        await pilot.pause()
        screen = await _open(app, BLOCK, results)
        await pilot.pause()
        confirm = screen.query_one("#move-workset-confirm", Button)
        check = screen.query_one("#move-workset-check", Checkbox)
        assert confirm.disabled is True
        assert app.focused is check  # the safe default: nothing that moves has focus
        await pilot.press("space")  # tick the focused checkbox
        await pilot.pause()
        assert check.value is True
        assert confirm.disabled is False
        await pilot.click("#move-workset-confirm")
        await pilot.pause()
    assert results == [True]


@pytest.mark.asyncio
async def test_escape_cancels_and_nothing_is_confirmed() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    results: list[bool] = []
    async with app.run_test(size=(100, 32)) as pilot:
        await pilot.pause()
        await _open(app, BLOCK, results)
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
    assert results == [False]


@pytest.mark.asyncio
async def test_a_sends_manifest_asks_for_confidence_in_the_destination() -> None:
    sends = {**BLOCK, "carries_files": True}
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 32)) as pilot:
        await pilot.pause()
        screen = await _open(app, sends, [])
        await pilot.pause()
        title = str(screen.query_one("#move-workset-title", Static).render())
        label = str(screen.query_one("#move-workset-check", Checkbox).label)
    assert title == "Send working files to lop-remote-node-2?"
    assert label == "I'm confident lop-remote-node-2 should receive these files"


@pytest.mark.asyncio
async def test_an_unknown_working_set_offers_only_cancel() -> None:
    unknown = {
        "version": 1,
        "state": "unknown",
        "destination": {"device_id": "d_b", "name": "lop-remote-node-2"},
        "failure": {"step": "status", "repo": "~/proj/.worktrees/feat"},
    }
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 32)) as pilot:
        await pilot.pause()
        screen = await _open(app, unknown, [])
        await pilot.pause()
        summary = str(screen.query_one("#move-workset-summary", Static).render())
        assert not screen.query("#move-workset-check")
        assert not screen.query("#move-workset-confirm")
    assert "step: status" in summary and "--without-working-set" in summary
