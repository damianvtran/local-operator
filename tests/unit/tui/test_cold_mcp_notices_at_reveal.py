"""A cold open carries its DURABLE MCP failure rows in the frame it paints.

The checkpoint every runtime writes at turn end holds the last MCP startup
OUTCOME (``FrontendStateStore.checkpoint`` → ``session.mcp_startup``), and the
transcript's failure rows are rendered from it. A cold open restored every other
part of that snapshot — servers, subagents, todos, the goal — but not this one,
so the first frame painted without the rows and the runtime's first engagement
replayed the very same outcome into a transcript the reader had already started
reading.

What is pinned here is the invariant, stated as the operator meets it:

* the first painted frame that carries content IS the settled frame — same
  notice texts, same row heights, same blank-line placement, same transcript
  height. QA round 1 (Q1) measured the opposite: the rows were authored at the
  80-column fallback, so five of nine were two rows tall with a blank line above
  every one of them, and the next frame took all of that back.
* a checkpoint's rows are worded as HISTORY ("failed last session"), so a live
  round that finds the servers healthy no longer contradicts them; a live round
  that still reports one restates its row IN PLACE (same block, no second row).
  QA round 1 (Q2).
* a session with NO durable outcome is not lied about: nothing is projected, and
  the outcome that arrives later lands once.

The last one is the legacy shape — a journal written before checkpoints existed
carries no outcome at all — and it is written down because "the reveal always
knows" would be the wrong conclusion to draw from the first two.
"""

from __future__ import annotations

import pytest

from local_operator.harness.types import Message, TextContent
from local_operator.session.attached import AttachedSession
from local_operator.session.frontend_state import (
    FRONTEND_CHECKPOINT_CUSTOM_TYPE,
    FrontendSessionState,
)
from local_operator.session.mcp_status import McpStartupOutcome
from local_operator.session.transcript import Transcript
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.transcript import GAP_CLASS, NoticeBlock
from tests.e2e.harness import seed_transcript
from tests.unit.session.test_remote import _never_take_over

#: A terminal wide enough that the notice's signpost fits whole, so the text
#: being compared is the composed sentence and not a width decision.
FRAME = (120, 34)

#: The failures the durable record carries. Two, and deliberately different in
#: kind: one naming a host (the transport family) and one a command, because the
#: signpost rule treats them differently and both must land.
#: One LONG line and one short: the long one is what makes Q1 visible, because
#: at the 80-column fallback it wraps to two rows while the column it is really
#: painted in fits it on one. A pair of short sentences would look identical in
#: both frames and hide exactly the reflow QA measured.
FAILURES = {
    "openaiDeveloperDocs": "network: cannot reach developers.openai.com",
    "slack": "command not found: slack-mcp",
}

ROWS = 6


def _prose(index: int) -> str:
    return " ".join(f"row{index:02} alpha beta gamma delta epsilon zeta" for _ in range(8))


async def _journal(
    tmp_path,
    name: str,
    *,
    failures: dict[str, str] | None,
    goal: str = "",
) -> str:
    """A session directory with ``ROWS`` messages, and optionally a checkpoint.

    ``failures=None`` is the legacy shape: a journal from before the durable
    snapshot existed, which carries no outcome for a cold frame to project.
    """
    config = tmp_path / "config"
    config.mkdir(exist_ok=True)
    (config / "config.yml").write_text(
        "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n"
    )
    directory = config / "sessions" / f"synthetic-{name}"
    await seed_transcript(
        directory,
        [
            Message(
                id=f"cold-row-{index:04}",
                role="assistant",
                content=[TextContent(text=_prose(index))],
            )
            for index in range(ROWS)
        ],
    )
    if failures is not None:
        transcript = Transcript(directory)
        await transcript.append_custom(
            FRONTEND_CHECKPOINT_CUSTOM_TYPE,
            {
                "checkpoint_id": "c1",
                "state": FrontendSessionState(
                    session_id=directory.name,
                    goal=goal,
                    epoch="previous-owner",
                    mcp_startup={
                        "configured": sorted(failures),
                        "connected": [],
                        "failures": dict(failures),
                        "tool_count": 0,
                        "settling": False,
                    },
                ).model_dump(mode="json"),
            },
        )
        transcript.flush()
        transcript.close()
    return directory.name


async def _cold(tmp_path, name: str, *, failures: dict[str, str] | None, goal: str = ""):
    session_id = await _journal(tmp_path, name, failures=failures, goal=goal)
    return await AttachedSession.cold(
        session_id,
        config_dir=tmp_path / "config",
        cwd=str(tmp_path),
        takeover_factory=_never_take_over,
    )


class _Frames:
    """Per-DISPLAY samples of the transcript's rows, notices included.

    ``post_display_hook`` is the only place a frame is a fact rather than an
    inference — the instrument the reveal tests use — so "is the first frame the
    settled one" is answered from painted displays, with the row HEIGHTS and the
    blank-line classes in the sample, not only the texts.
    """

    def __init__(self, app: OperatorApp, session: object) -> None:
        self.samples: list[dict[str, object]] = []
        real = app.post_display_hook
        samples = self.samples

        def hook() -> None:
            try:
                view = app._transcript_view()
                blocks = view.blocks()
                samples.append(
                    {
                        "target": app._session is session,
                        "parked": bool(view.disabled),
                        "blocks": len(blocks),
                        "virtual_h": int(view.virtual_size.height),
                        "max_scroll_y": float(view.max_scroll_y),
                        "region_w": int(view.region.width),
                        "rows": [
                            {
                                # ``_text`` is the composed sentence; asked
                                # defensively because a frame can be sampled
                                # while a block is still being authored.
                                "text": str(getattr(block, "_text", "") or ""),
                                "history": bool(getattr(block, "mcp_failure_history", False)),
                                "h": int(block.outer_size.height),
                                "gap": bool(block.has_class(GAP_CLASS)),
                                "notice": isinstance(block, NoticeBlock),
                            }
                            for block in blocks
                        ],
                        "notice_positions": [
                            index
                            for index, block in enumerate(blocks)
                            if isinstance(block, NoticeBlock)
                        ],
                        "last_message_index": max(
                            (
                                index
                                for index, block in enumerate(blocks)
                                if not isinstance(block, NoticeBlock)
                            ),
                            default=-1,
                        ),
                    }
                )
            except Exception:  # noqa: BLE001 — a frame before the transcript exists is not a state
                pass
            real()

        app.post_display_hook = hook  # type: ignore[method-assign]

    def content(self) -> list[dict]:
        """Frames painted with the session in front of the reader, carrying rows."""
        return [
            sample
            for sample in self.samples
            if sample["target"] and not sample["parked"] and sample["blocks"] > 0
        ]

    @staticmethod
    def texts(frame: dict) -> list[str]:
        return [row["text"] for row in frame["rows"]]

    @staticmethod
    def notices(frame: dict) -> list[dict]:
        return [row for row in frame["rows"] if row["notice"]]

    @classmethod
    def geometry(cls, frame: dict) -> dict:
        """What a painted frame LOOKS like, heights and spacing included.

        The claim QA round 1 could not find in the texts: a row that is authored
        narrow is two rows tall and carries a blank line above it, so an equal
        notice list can still be a different frame.
        """
        return {
            "virtual_h": frame["virtual_h"],
            "notices": [(row["h"], row["gap"], row["history"]) for row in cls.notices(frame)],
        }


async def _settle(pilot, frames: _Frames, *, quiet: int = 6) -> None:
    """Pump until the frame signature stops moving."""
    seen = -1
    stable = 0
    while stable < quiet:
        await pilot.pause()
        if len(frames.samples) == seen:
            stable += 1
        else:
            stable = 0
            seen = len(frames.samples)


def _mcp_rows(frame: dict, failures: dict[str, str]) -> list[dict]:
    return [
        row
        for row in _Frames.notices(frame)
        if any(f"MCP {name} failed" in row["text"] for name in failures)
    ]


def _healthy_live(app: OperatorApp, viewer: AttachedSession) -> None:
    """Push the owner's snapshot the way the bind does: through the frontend state.

    ``_install_frontend(..., publish=True)`` is the real hop a bound viewer takes
    when its owner reports a round (``_apply_frontend_facades`` →
    ``_fire_mcp_startup_sink`` → this app), so the sink wiring is exercised
    rather than bypassed.
    """
    state = viewer.frontend_state.model_copy(
        update={
            "mcp_startup": {
                "configured": sorted(FAILURES),
                "connected": sorted(FAILURES),
                "failures": {},
                "tool_count": 300,
                "settling": False,
            }
        }
    )
    viewer._install_frontend(state, publish=True)


@pytest.mark.asyncio
async def test_a_cold_open_paints_its_durable_mcp_failures_before_the_first_frame(
    tmp_path, monkeypatch
) -> None:
    """RED on a tree that restores every durable field except ``mcp_startup``.

    The first contentful frame carries no MCP rows there, and the replay that
    the bind performs appends both of them — so the settled frame differs from
    the first in count, in text, and in height.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    viewer = await _cold(tmp_path, "durable", failures=FAILURES)
    try:

        async def factory():
            return viewer

        app = OperatorApp(factory)
        frames = _Frames(app, viewer)
        async with app.run_test(size=FRAME) as pilot:
            await _settle(pilot, frames)
            content = frames.content()
            assert content, f"no painted frame carried the conversation: {frames.samples[-4:]}"
            first = content[0]
            settled = content[-1]
            assert len(_mcp_rows(first, FAILURES)) == len(FAILURES), (
                "the first contentful frame did not carry the durable failure rows: "
                f"{frames.texts(first)}"
            )
            assert all(
                position > first["last_message_index"] for position in first["notice_positions"]
            ), (
                "a durable row was painted ABOVE the replayed conversation, where the "
                "reader has to scroll back for it: "
                f"notices at {first['notice_positions']}, conversation ends at "
                f"{first['last_message_index']}"
            )
            # Q1: the first frame must already be the SETTLED frame, heights and
            # spacing included — not just the same sentences.
            assert _Frames.geometry(first) == _Frames.geometry(settled), (
                "the first contentful frame reflowed after it was painted: "
                f"{_Frames.geometry(first)} -> {_Frames.geometry(settled)}"
            )

            # What the post-paint bind does with the SAME outcome the checkpoint
            # already carried. It must recognise the rows as painted.
            app._report_mcp_startup(viewer)
            await _settle(pilot, frames)
            assert frames.texts(frames.content()[-1]) == frames.texts(settled), (
                "replaying the durable outcome appended rows the first frame already had: "
                f"{frames.texts(settled)} -> {frames.texts(frames.content()[-1])}"
            )
    finally:
        await viewer.dispose()


@pytest.mark.asyncio
async def test_a_checkpoint_row_is_history_and_a_healthy_live_round_never_contradicts_it(
    tmp_path, monkeypatch
) -> None:
    """Q2: the row says LAST SESSION, so the healthy toast does not argue with it.

    The reported shape: a cold open on a checkpoint with nine failures, then the
    owner's snapshot arrives with every server connected. On the pre-round-1 head
    the frame ended up with a green "MCP ready: 12 servers" above nine red
    present-tense "failed" rows that nothing could ever retract.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    viewer = await _cold(tmp_path, "stale", failures=FAILURES)
    try:

        async def factory():
            return viewer

        app = OperatorApp(factory)
        frames = _Frames(app, viewer)
        async with app.run_test(size=FRAME) as pilot:
            await _settle(pilot, frames)
            first = frames.content()[0]
            rows = _mcp_rows(first, FAILURES)
            assert rows, f"the durable rows are missing: {frames.texts(first)}"
            assert all("failed last session" in row["text"] for row in rows), (
                "a checkpoint's failure was stated as present-tense news: "
                f"{[row['text'] for row in rows]}"
            )
            assert all(row["history"] for row in rows)

            # NIT-1: through the REAL hop — `_install_frontend(publish=True)`
            # is what a bound viewer does when its owner reports a round, and
            # the sink it fires is the app's `_report_mcp_startup`. Counted so
            # the cell cannot pass by the snapshot merely landing on the facade.
            calls: list[object] = []
            real_report = OperatorApp._report_mcp_startup

            def spy(self, session):  # noqa: ANN001
                calls.append(session)
                return real_report(self, session)

            OperatorApp._report_mcp_startup = spy  # type: ignore[method-assign]
            try:
                _healthy_live(app, viewer)
                await _settle(pilot, frames)
            finally:
                OperatorApp._report_mcp_startup = real_report  # type: ignore[method-assign]
            assert calls, (
                "the owner's snapshot never reached the app's startup report: the sink "
                "wiring was bypassed, so this cell proved nothing about the live path"
            )
            after = frames.content()[-1]
            assert frames.texts(after) == frames.texts(first), (
                "a healthy live round changed the history rows: "
                f"{frames.texts(first)} -> {frames.texts(after)}"
            )
            assert [row["text"] for row in _mcp_rows(after, FAILURES)] == [
                row["text"] for row in rows
            ]
            assert _Frames.geometry(after) == _Frames.geometry(first), (
                "a healthy live round reflowed the history rows: "
                f"{_Frames.geometry(first)} -> {_Frames.geometry(after)}"
            )
    finally:
        await viewer.dispose()


@pytest.mark.asyncio
async def test_a_live_round_that_confirms_a_history_row_restates_it_in_place(
    tmp_path, monkeypatch
) -> None:
    """Q2's other half: a failure that is STILL failing becomes present tense.

    Restated on the block that is already on screen — one row, one statement,
    and nothing appended beside it — so the confirmation costs no line and
    leaves no stale claim above its own correction. The live session is a double
    because "a runtime is attached and synced" is exactly what a synthetic cold
    facade is not; the ledger key it shares with the facade is what makes the
    row findable.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    viewer = await _cold(tmp_path, "confirm", failures=FAILURES)
    try:

        async def factory():
            return viewer

        app = OperatorApp(factory)
        frames = _Frames(app, viewer)
        async with app.run_test(size=FRAME) as pilot:
            await _settle(pilot, frames)
            first = frames.content()[0]
            before = _mcp_rows(first, FAILURES)
            assert before and all("last session" in row["text"] for row in before)

            class _Live:
                session_id = viewer.session_id
                is_cold = False
                mcp_startup = McpStartupOutcome(
                    configured=tuple(sorted(FAILURES)),
                    connected=(),
                    failures=dict(FAILURES),
                    tool_count=0,
                )

            app._report_mcp_startup(_Live())
            await _settle(pilot, frames)
            after = frames.content()[-1]
            confirmed = _mcp_rows(after, FAILURES)
            assert len(confirmed) == len(FAILURES), (
                "the confirmed failure was appended instead of restated: "
                f"{frames.texts(first)} -> {frames.texts(after)}"
            )
            assert not any("last session" in row["text"] for row in confirmed), (
                f"a failure the live round still reports stayed in the past tense: "
                f"{[row['text'] for row in confirmed]}"
            )
            assert not any(row["history"] for row in confirmed)
    finally:
        await viewer.dispose()


@pytest.mark.asyncio
async def test_a_cold_open_without_a_durable_outcome_projects_nothing_and_appends_once(
    tmp_path, monkeypatch
) -> None:
    """The legacy shape: no checkpoint, so the outcome is genuinely NEW later.

    Projecting rows from a record that does not exist would be inventing them,
    so the assertion is the honest one: the first frame is clean, and the
    outcome that arrives later lands exactly once rather than twice.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    viewer = await _cold(tmp_path, "legacy", failures=None)
    try:
        assert viewer.mcp_startup is None, "a journal with no checkpoint has no outcome"

        async def factory():
            return viewer

        app = OperatorApp(factory)
        frames = _Frames(app, viewer)
        async with app.run_test(size=FRAME) as pilot:
            await _settle(pilot, frames)
            first = frames.content()[0]
            assert _mcp_rows(first, FAILURES) == []

            class _Live:
                session_id = viewer.session_id
                is_cold = False
                mcp_startup = McpStartupOutcome(
                    configured=tuple(sorted(FAILURES)),
                    connected=(),
                    failures=dict(FAILURES),
                    tool_count=0,
                )

            app._report_mcp_startup(_Live())
            await _settle(pilot, frames)
            settled = frames.content()[-1]
            rows = _mcp_rows(settled, FAILURES)
            assert len(rows) == len(
                FAILURES
            ), f"the first known outcome must land once: {frames.texts(settled)}"
            assert not any(row["history"] for row in rows), (
                "a failure the live runtime reports was worded as history: "
                f"{[row['text'] for row in rows]}"
            )
            # And the replay stays quiet, the same as on the durable path.
            app._report_mcp_startup(_Live())
            await _settle(pilot, frames)
            assert frames.texts(frames.content()[-1]) == frames.texts(settled), (
                "a second replay of an unchanged outcome appended its rows again: "
                f"{frames.texts(settled)} -> {frames.texts(frames.content()[-1])}"
            )
    finally:
        await viewer.dispose()


@pytest.mark.asyncio
async def test_the_other_checkpoint_backed_notice_kind_is_in_the_first_frame_too(
    tmp_path, monkeypatch
) -> None:
    """The enumeration, pinned: ``goal`` is the only OTHER durable row.

    Of the fields ``_restore_cold_details`` restores, exactly two render as
    transcript rows: ``mcp_startup`` (the failures above) and ``goal``, whose
    "goal restored" receipt is raised by ``_report_attachment_restore``. Every
    other restored field — roster, todos, title, identity, spend, servers —
    renders into a panel, the band or the tab, never a ``NoticeBlock``. So the
    goal receipt is the one that would be missed if reporting moved back before
    the replay or out to the bind, and this cell is what notices if it does.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    viewer = await _cold(tmp_path, "goal-only", failures={}, goal="ship the rollout")
    try:

        async def factory():
            return viewer

        app = OperatorApp(factory)
        frames = _Frames(app, viewer)
        async with app.run_test(size=FRAME) as pilot:
            await _settle(pilot, frames)
            first = frames.content()[0]
            assert any(
                str(row["text"]).startswith("goal restored:") for row in _Frames.notices(first)
            ), (
                "the durable goal's receipt was not in the first contentful frame: "
                f"{frames.texts(first)}"
            )
            assert all(
                position > first["last_message_index"] for position in first["notice_positions"]
            ), "a durable row was painted above the replayed conversation"
            settled = frames.content()[-1]
            assert frames.texts(settled) == frames.texts(first), (
                f"the settled frame differs from the first: {frames.texts(first)} -> "
                f"{frames.texts(settled)}"
            )
    finally:
        await viewer.dispose()


@pytest.mark.asyncio
async def test_a_restored_outcome_records_without_interrupting(tmp_path, monkeypatch) -> None:
    """Review round 1, MINOR-1: a checkpoint's failures are not breaking news.

    The toast is one-shot, it takes the app's single courtesy slot, and on a cold
    open it announced a round that had already finished — sometimes for a session
    whose runtime would never engage at all. The row is the record and it is
    already on screen; the card is the interruption, and it now waits for a round
    that actually just happened.
    """
    from local_operator.tui.widgets.toast import Toast

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    viewer = await _cold(tmp_path, "quiet", failures=FAILURES)
    try:

        async def factory():
            return viewer

        app = OperatorApp(factory)
        frames = _Frames(app, viewer)
        async with app.run_test(size=FRAME) as pilot:
            await _settle(pilot, frames)
            assert _mcp_rows(frames.content()[0], FAILURES), "the rows are missing"
            assert app._announced_mcp_startup == {}, (
                "a restored outcome took the announce slot: " f"{app._announced_mcp_startup}"
            )

            class _Live:
                session_id = viewer.session_id
                is_cold = False
                mcp_startup = McpStartupOutcome(
                    configured=tuple(sorted(FAILURES)),
                    connected=(),
                    failures=dict(FAILURES),
                    tool_count=0,
                )

            app._report_mcp_startup(_Live())
            await _settle(pilot, frames)
            assert (
                app._announced_mcp_startup
            ), "a live round did not announce, so the restored one had spent the slot"
            assert app.query_one(Toast).display
    finally:
        await viewer.dispose()
