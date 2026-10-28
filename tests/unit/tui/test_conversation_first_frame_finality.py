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
from pathlib import Path
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
from tests.unit.tui.test_sidebar_switch_smoothness import _switch_bound


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


# -- D2: the band paints stored facts at first frame, as last-known ---------------------


def _band(width: int):  # type: ignore[no-untyped-def]
    from typing import cast as _cast

    from textual.widgets import Static

    from local_operator.tui.widgets.status_line import StatusLine
    from tests.unit.tui.test_status_line import FakeDock

    status = StatusLine(_cast(Static, FakeDock(width)))
    status.update(
        model_label="openrouter/deepseek/deepseek-v4.1-flash",
        effort="max",
        cwd="/Users/tester/work",
        context_tokens=179_619,
        context_window=1_048_576,
        cost="≥$53.72",
        conversation_name="Agent hub: agents/teams auto-update + public/private skills repos",
    )
    return status


def _segments(row: str) -> list[tuple[int, str]]:
    """Every non-space run in the row with its starting column."""
    import re

    return [(m.start(), m.group()) for m in re.finditer(r"\S+", row)]


def _frames(width: int) -> tuple[str, str, str]:
    """``(last_known, bound, starting)`` rows of one band at ``width``."""
    status = _band(width)
    status.update(connection="Saved excerpt · Connecting…")
    status.set_connecting(True)
    status.set_last_known(True)
    first = status.render_text(width).plain
    status.update(connection="")
    status.set_connecting(False)
    status.set_last_known(False)
    bound = status.render_text(width).plain
    status.set_starting(True)
    starting = status.render_text(width).plain
    return first, bound, starting


@pytest.mark.parametrize("width", [60, 80, 90, 100, 120, 156, 200])
def test_the_last_known_slot_costs_exactly_what_a_bring_up_costs(width: int, monkeypatch) -> None:
    """D2 constraint (2): no worse than the indicator an ordinary bring-up shows.

    The bar the operator set for the connecting slot: whatever the bind moves,
    it must be no more than what the band already moves when ``starting…``
    clears today. Pinned as the strongest form of that: at EVERY width the
    last-known row is the ``starting…`` row with only the caption swapped — the
    same ladder rungs shed, the same title cut, every column identical — so the
    bind reproduces the existing bring-up transition and nothing beyond it.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_NO_SHIMMER", "1")
    first, _bound, starting = _frames(width)
    assert "Saved excerpt" not in first, "the stored facts were withheld"
    # Shimmer off: `starting…` draws its glyph (it always spells out), the
    # last-known caption draws none, so the slot is `⣾ starting…` (11 cells)
    # against `connecting…` (11 cells). Swapping one for the other must give
    # the identical row.
    glyph_starting = next(part for part in starting.split(" › ") if "starting…" in part)
    slot_starting = glyph_starting.split("    ")[0]
    assert first.replace("connecting…", slot_starting, 1) == starting, (first, starting)


@pytest.mark.parametrize("width", [156, 160, 200])
def test_where_the_slot_fits_the_bind_moves_only_the_slot_and_the_title_cut(width: int) -> None:
    """D2 constraint (3): at widths where nothing is shed for the slot.

    Every left-group segment and every right-group figure sits at the same
    column with the same text in the last-known and the bound frame; the only
    differences are the slot itself and how many title cells the right group
    gets. (Below these widths the slot sheds a rung exactly as ``starting…``
    does — the test above.) 160 is the capture rig's terminal.
    """
    first, bound, _starting = _frames(width)
    slot_at = first.index("connecting…")
    assert [s for s in _segments(first) if s[0] < slot_at - 2] == [
        s for s in _segments(bound) if s[0] < slot_at - 2
    ], (first, bound)

    def right(row: str) -> list[str]:
        return [part.strip() for part in row.split("    ")[-1].split("‹")[:-1]]

    assert right(first) == right(bound), (first, bound)


def test_last_known_never_shows_live_authority() -> None:
    """Running/streaming is live-only: the last-known band never spins.

    The app strips every live field from the stored row before painting it
    (`_paint_last_known_band`); this pins the renderer half: even a band whose
    streaming flag is up does not draw the working indicator while it is in
    last-known mode — the slot says it is connecting, which is the truth.
    """
    status = _band(200)
    status.update(connection="Saved excerpt · Connecting…", streaming=True)
    status.set_last_known(True)
    row = status.render_text(200).plain
    assert "connecting" in row
    assert "working" not in row


def _journal_with_checkpoint(directory: Path, state_session_id: str, *, title: str) -> None:
    import json

    directory.mkdir(parents=True)
    state = FrontendSessionState(
        session_id=state_session_id,
        epoch="old-epoch",
        selected_model=FrontendModelSpec(provider="test", model_id="model"),
        conversation_title=title,
        streaming=True,
        cumulative_parent_cost=1.25,
    ).model_dump(mode="json")
    row = {
        "id": "row-1",
        "ts": 1.0,
        "type": "custom",
        "payload": {
            "kind": "custom",
            "custom_type": "frontend_state_checkpoint_v1",
            "details": {"checkpoint_id": "c1", "state": state},
        },
    }
    # Compact separators: the format `TranscriptEntry.to_json` writes, which
    # is what the byte scan is exact for.
    (directory / "transcript.jsonl").write_text(json.dumps(row, separators=(",", ":")) + "\n")


def test_the_stored_state_is_read_from_the_journal_checkpoint(tmp_path) -> None:
    """The cold source of D2's last-known facts: the newest checkpoint row."""
    from local_operator.tui.app import _read_stored_frontend_state

    directory = tmp_path / "sessions" / "own-session"
    _journal_with_checkpoint(directory, "own-session", title="Stored title")
    state = _read_stored_frontend_state(directory, "own-session")
    assert state is not None
    assert state.conversation_title == "Stored title"
    assert state.cumulative_parent_cost == 1.25


def test_a_forked_journal_keeps_facts_but_not_the_parents_identity(tmp_path) -> None:
    """A fork's journal carries the PARENT's rows; apply the runtime's own rule.

    The runtime restores the parent's durable facts into the fork and
    re-stamps identity (``_inherited_identity_fixups``), so the bound frame
    paints the parent's title/cost — and the last-known frame must paint the
    same, or the bind reflows for unchanged data. The parent's id never leaks.
    """
    from local_operator.tui.app import _read_stored_frontend_state

    directory = tmp_path / "sessions" / "fork-session"
    _journal_with_checkpoint(directory, "parent-session", title="Parent title")
    state = _read_stored_frontend_state(directory, "fork-session")
    assert state is not None
    assert state.session_id == "fork-session"
    assert state.conversation_title == "Parent title"
    assert state.jobs == []


def test_a_journal_without_a_checkpoint_seeds_nothing(tmp_path) -> None:
    """Nothing honest to paint: the band keeps its connecting sentence."""
    from local_operator.tui.app import _read_stored_frontend_state

    directory = tmp_path / "sessions" / "legacy"
    directory.mkdir(parents=True)
    (directory / "transcript.jsonl").write_text(
        '{"id":"m1","ts":1.0,"type":"message","payload":{"kind":"message","role":"user",'
        '"content":[{"text":"hi"}]}}\n'
    )
    assert _read_stored_frontend_state(directory, "legacy") is None


# -- D5: condensed/grouped tool calls are final at first frame -----------------------


def _grouped_conversation(session_id: str) -> SidebarRemote:
    """A long conversation with real quiet groups: runs of persisted peer receipts.

    Three receipts every sixth turn, the shape the TUI folds under one
    ``QuietGroupBlock`` (a maximal run of >= 2 receipt rows). Long enough that
    the bounded first window holds several groups and the paging buffer holds
    more, so a fold derived from a different window than the settled one would
    show up as a moved span.
    """
    from types import SimpleNamespace

    from local_operator.harness.message_types import PEER_MESSAGE_MESSAGE_TYPE

    history: list[object] = []
    for turn in range(60):
        user = _message("user", f"question {turn}")
        user.id = f"{session_id}-u{turn}"
        assistant = _message("assistant", f"answer {turn}")
        assistant.id = f"{session_id}-a{turn}"
        history.extend((user, assistant))
        if turn % 6 == 5:
            history.extend(
                SimpleNamespace(
                    role=None,
                    custom_type=PEER_MESSAGE_MESSAGE_TYPE,
                    id=f"{session_id}-p{turn}-{index}",
                    text="",
                    tool_calls=None,
                    content=[],
                    details={
                        "body": f"ping {turn}.{index}",
                        "sender": {"pid": 9, "conversation_name": "other"},
                    },
                )
                for index in range(3)
            )
    return SidebarRemote(session_id, history=history)


def _group_signature(view):  # type: ignore[no-untyped-def]
    from local_operator.tui.widgets.transcript import QuietGroupBlock

    blocks = view.blocks()
    return {
        "rows": tuple(
            (type(b).__name__, getattr(b, "navigation_anchor_id", "") or "", bool(b.display))
            for b in blocks
            if type(b).__name__ != "NoticeBlock"
        ),
        "groups": tuple(
            (b.group.key, b.group.count, b.expanded, len(b._members))
            for b in blocks
            if isinstance(b, QuietGroupBlock)
        ),
        "hidden": sum(1 for b in blocks if not b.display),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("expand", [False, True])
async def test_grouped_history_is_final_at_first_frame(expand: bool) -> None:
    """D5: group spans, expansion and hidden counts at first frame == settled.

    A fresh open (every group folded, members hidden), and a reopen after the
    reader EXPANDED the newest group and left, are driven through the real
    switch and then through the connect's ``refresh=True`` re-prepare of the
    same unchanged history — the post-paint commit that trimmed D1. The groups
    must be genuinely non-empty, and the first frame's rows, group spans,
    expansion and hidden count must be the settled frame's.

    WHAT THE EXPANDED CASE DOES AND DOES NOT SHOW. A group bar is not
    retainable (`SessionPresentation.retainable`: `QuietGroupBlock` declines
    `retained_payloads`), so leaving the conversation drops the parked view and
    the reopen rebuilds it with every group folded: the reader's expansion is
    not carried across a switch. That is pre-existing behaviour outside this
    lane and is asserted as such below, so the case pins first == settled for
    the frame the reopen actually paints, not an expansion it never had.
    """
    from local_operator.tui.widgets.transcript import QuietGroupBlock

    home = SidebarRemote("home-session")
    target = _grouped_conversation("grouped-session")
    app = OperatorApp(lambda: _factory(home))
    at_first: dict[str, object] = {}
    with (
        patch("local_operator.session.attached.AttachedSession", SidebarRemote),
        patch("local_operator.tui.app.RESUME_PAGE_MESSAGES", 8),
    ):
        async with app.run_test(size=(160, 48)) as pilot:
            for _ in range(20):
                await pilot.pause()
            await _switch_bound(app, pilot, target)
            for _ in range(30):
                await pilot.pause()
            if expand:
                bars = [
                    b for b in app._transcript_view().blocks() if isinstance(b, QuietGroupBlock)
                ]
                bars[-1].toggle_expanded()
                for _ in range(10):
                    await pilot.pause()
                await _switch_bound(app, pilot, home)
                for _ in range(20):
                    await pilot.pause()
            real_commit = app._commit_sidebar_session

            def commit(session_id, prepared, generation):  # type: ignore[no-untyped-def]
                out = real_commit(session_id, prepared, generation)
                if not at_first and session_id == target.session_id:
                    at_first.update(_group_signature(app._transcript_view()))
                return out

            app._commit_sidebar_session = commit  # type: ignore[method-assign]
            if expand:
                await _switch_bound(app, pilot, target)
            else:
                prepared = await app._prepare_sidebar_session(target.session_id, refresh=True)
                commit(target.session_id, prepared, 0)
            for _ in range(30):
                await pilot.pause()
            prepared = await app._prepare_sidebar_session(target.session_id, refresh=True)
            real_commit(target.session_id, prepared, 0)
            for _ in range(30):
                await pilot.pause()
            settled = _group_signature(app._transcript_view())

    groups = settled["groups"]
    assert len(groups) >= 2 and all(count >= 2 for _key, count, _open, _n in groups), (
        "the fixture produced no real groups, so it cannot test D5",
        groups,
    )
    assert settled["hidden"] > 0, "no member rows are folded away"
    assert not any(
        is_open for _k, _c, is_open, _n in groups
    ), "a group reopened expanded — the switch now carries expansion; update this test"
    assert at_first == settled, (at_first, settled)


@pytest.mark.asyncio
async def test_the_connect_refresh_keeps_a_group_the_reader_opened() -> None:
    """D5 defect found here: the bind's re-prepare must not re-fold an open group.

    The reader opens a group while the conversation is on screen, then the
    connect's ``refresh=True`` re-prepare commits a rebuilt view of the SAME
    history. Measured before the fix: the open group came back folded (hidden
    rows 8 -> 11). The group must stay open, and the spans must be unchanged.
    """
    from local_operator.tui.widgets.transcript import QuietGroupBlock

    target = _grouped_conversation("opened-session")
    app = OperatorApp(lambda: _factory(SidebarRemote("home-session")))
    with (
        patch("local_operator.session.attached.AttachedSession", SidebarRemote),
        patch("local_operator.tui.app.RESUME_PAGE_MESSAGES", 8),
    ):
        async with app.run_test(size=(160, 48)) as pilot:
            for _ in range(20):
                await pilot.pause()
            await _switch_bound(app, pilot, target)
            for _ in range(30):
                await pilot.pause()
            bars = [b for b in app._transcript_view().blocks() if isinstance(b, QuietGroupBlock)]
            bars[-1].toggle_expanded()
            for _ in range(10):
                await pilot.pause()
            before = _group_signature(app._transcript_view())
            prepared = await app._prepare_sidebar_session(target.session_id, refresh=True)
            app._commit_sidebar_session(target.session_id, prepared, 0)
            for _ in range(30):
                await pilot.pause()
            after = _group_signature(app._transcript_view())

    assert any(is_open for _k, _c, is_open, _n in before["groups"]), "the fixture opened nothing"
    # Every group the refreshed view still forms carries the SAME span and the
    # same open/closed state it had on screen. Compared per key rather than as
    # a whole tuple because the refresh's bounded window is measured against
    # the reader's EXPANDED view, which reaches its reserve one page sooner and
    # so does not re-mount the oldest offscreen group — a residual recorded in
    # the PR (it is caused by the reader's own expansion, not by unchanged
    # state, and is outside this lane's bounded-window contract).
    shown_before = {key: (count, is_open, n) for key, count, is_open, n in before["groups"]}
    shown_after = {key: (count, is_open, n) for key, count, is_open, n in after["groups"]}
    assert shown_after, "the refresh formed no groups"
    for key, facts in shown_after.items():
        assert shown_before.get(key) == facts, (key, shown_before.get(key), facts)
