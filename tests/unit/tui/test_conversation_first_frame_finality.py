"""A conversation's first painted frame is its settled frame when nothing changed.

The conversation-open finality audit (``TUI-FINALITY-DESIGN``, findings D1-D6)
measured three things moving between a reopen's first paint and its settled
frame while the conversation's data stayed exactly the same:

* **D1** — the first frame mounted MORE history than the settled one and a later
  commit trimmed it (38 -> 18 blocks, virtual height 129 -> 51 on the bench's
  real fixture at 160x48). The text in view did not move, but the scrollbar
  thumb jumped 76 -> 193 px. Cause, measured with an ordered trace: the
  pre-reveal burst read a 39-row viewport from ONE paint-suppressed layout
  while the dock had just grown todo/subagent panels the layout had not yet
  sized, so it filled for 39 rows where the settled frame has 24.
* **D3** — the composer placeholder read ``Draft a message…`` until the owner
  connected and ``Message Local Operator…`` after, with the composer accepting
  a draft in both states.

Each test below reproduces its finding through the REAL ``OperatorApp`` with
production CSS and the real prepare/commit pair (the same rig
``test_sidebar_switch_smoothness`` uses), and asserts the reader-side fact, not
a production predicate agreeing with itself.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from local_operator.session.frontend_state import (
    FrontendModelSpec,
    FrontendSessionState,
    TodoPhaseState,
)
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.editor import DEFAULT_PLACEHOLDER, Editor
from tests.unit.tui.test_app_pilot import _factory
from tests.unit.tui.test_band_panels import _fake_jobs, _Job
from tests.unit.tui.test_sidebar_swap_reset import SidebarRemote, _message


@pytest.fixture(autouse=True)
def isolated_app(tmp_path, monkeypatch):
    # A headless TUI test must never inherit the operator's multiplexer
    # identity: an inherited CMUX_WORKSPACE_ID has previously let a headless run
    # rename his real cmux workspaces.
    for key in tuple(os.environ):
        if key.startswith("CMUX_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("LOCAL_OPERATOR_NO_NOTIFICATIONS", "1")
    monkeypatch.setenv("LOCAL_OPERATOR_NO_TERMINAL_TITLE", "1")
    monkeypatch.setattr(OperatorApp, "_check_for_update", lambda _self: None)
    monkeypatch.setattr(OperatorApp, "_start_terminal_title", lambda _self: None)
    monkeypatch.setattr(OperatorApp, "_start_multiplexer_broadcast", lambda _self: None)
    monkeypatch.setattr(OperatorApp, "_start_herdr_reporter", lambda _self: None)
    # The fake owner carries no build stamp, so every adopt posts the
    # "running an older version" tail notice. Its three rows land under the
    # history after the first commit and are absent from a fresh replay, which
    # moves the reserve goal by exactly the margin this fixture sits on and
    # would make the re-prepare mount a page for a reason that has nothing to do
    # with the viewport. The notice has its own tests; it is silenced here so
    # the history window is the only variable.
    monkeypatch.setattr(OperatorApp, "_check_build_skew", lambda _self, **_kw: None)


def _planned_conversation(
    session_id: str, turns: int, todos: int, *, children: int = 0
) -> SidebarRemote:
    """A long conversation whose session state carries a todo plan.

    The plan is the load-bearing part: it is what makes the todo panel appear in
    the dock on the adopt, i.e. the dock grows in the SAME tick the pre-reveal
    burst measures the transcript's viewport. A conversation without one keeps
    the dock at the composer's height and cannot reproduce D1 at all.
    """
    history = []
    for turn in range(turns):
        user = _message("user", f"question {turn} about the reopen path")
        user.id = f"{session_id}-u{turn}"
        assistant = _message("assistant", f"answer {turn}, with some detail to render")
        assistant.id = f"{session_id}-a{turn}"
        history.extend((user, assistant))
    remote = SidebarRemote(session_id, history=history)
    remote._store.replace(
        FrontendSessionState(
            session_id=session_id,
            epoch=f"epoch-{session_id}",
            selected_model=FrontendModelSpec(provider="test", model_id="model"),
            todos=[
                TodoPhaseState.model_validate(
                    {
                        "name": "Todos",
                        "items": [
                            {"text": f"step {index} of the plan", "status": "done"}
                            for index in range(todos)
                        ],
                    }
                )
            ],
        )
    )
    # The roster the subagent panel reads (``OperatorApp._subagent_roster`` falls
    # back to ``session.jobs.list()`` when there is no comms graph). A running
    # child is never retention-expired, so the panel is shown on the adopt.
    remote.jobs = _fake_jobs(  # type: ignore[attr-defined]
        *(_Job(f"child-{index}", f"child task {index}") for index in range(children))
    )
    return remote


@pytest.mark.asyncio
async def test_the_pre_reveal_measurement_sees_panels_the_adopt_just_showed() -> None:
    """D1's mechanism: the burst's viewport reading must include the grown dock.

    The commit adopts the incoming session, `_refresh_band` shows its todo and
    subagent panels, and the pre-reveal burst then lays out and reads the
    transcript's viewport — all in one tick. Textual only propagates a child's
    size change to its ``height: auto`` ancestors from the idle hook, so a
    synchronous layout at that point re-used ``#input-dock``'s pre-panel
    arrangement and the transcript kept its old height. On the bench's real
    fixture that was a 39-row reading for a 24-row settled view, an extra page
    mounted into the first frame, and a trim after paint (thumb 76 -> 193 px).

    Driven here in exactly that order, on the real app with production CSS, and
    compared against the reading after the app has gone idle — the viewport
    every later frame is laid out in. The end-to-end bench A/B is in the PR;
    the full sidebar switch in the unit rig happens to land on a pass that
    already carries the invalidation, so it cannot tell the two trees apart and
    is not used as the regression guard.
    """
    home = SidebarRemote("home-session")
    target = _planned_conversation("planned-session", 60, todos=8, children=1)
    app = OperatorApp(lambda: _factory(home))
    with patch("local_operator.session.attached.AttachedSession", SidebarRemote):
        async with app.run_test(size=(160, 48)) as pilot:
            for _ in range(20):
                await pilot.pause()
            view = app._transcript_view()
            before = view.container_size.height
            # What the adopt does to the band, without yielding to the loop.
            app._session = target
            app._refresh_band()
            assert app._todo_panel is not None and app._todo_panel.display
            measured = app._settle_reveal_measurement(view)
            for _ in range(10):
                await pilot.pause()
            settled = view.container_size.height

    assert settled < before, "the fixture did not grow the dock, so it cannot reproduce D1"
    assert measured == settled, (
        f"the pre-reveal reading was {measured} rows against a settled {settled} "
        f"(the pre-adopt view had {before}): the burst would fill for a viewport "
        "the first frame does not have"
    )


@pytest.mark.asyncio
async def test_the_composer_placeholder_does_not_change_when_the_owner_connects() -> None:
    """D3: one resting placeholder across the connection states.

    The band row beneath the composer already states the connection; the
    composer accepts a draft in both states, so a second copy for "connecting"
    was a swap carrying no information, painted after first paint on every
    reopen.
    """
    app = OperatorApp(lambda: _factory(SidebarRemote("home-session")))
    async with app.run_test(size=(120, 30)) as pilot:
        for _ in range(10):
            await pilot.pause()
        editor = app.query_one(Editor)
        connecting = app._composer_placeholder_for(editor, connection="Connecting…")
        bound = app._composer_placeholder_for(editor)
    assert connecting == bound == DEFAULT_PLACEHOLDER
