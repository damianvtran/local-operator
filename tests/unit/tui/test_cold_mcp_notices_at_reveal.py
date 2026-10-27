"""A cold open carries its DURABLE MCP failure rows, so the bind adds none.

The checkpoint every runtime writes at turn end holds the last MCP startup
OUTCOME (``FrontendStateStore.checkpoint`` → ``session.mcp_startup``), and the
transcript's failure rows are rendered from it. A cold open restored every other
part of that snapshot — servers, subagents, todos, the goal — but not this one,
so the first frame painted without the rows and the runtime's first engagement
replayed the very same outcome into a transcript the reader had already started
reading. The rows appended under them: the tail moved, and the "durable record"
that ``_report_mcp_startup`` promises arrived at a moment the user had no reason
to look.

What is pinned here is the invariant, stated as the operator meets it:

* the first painted frame that carries content already carries one row per
  durable failure;
* replaying that same outcome (what the bind does, and what a re-attach does)
  appends nothing, so the count and the text are unchanged in the settled frame;
* a session with NO durable outcome is not lied about: nothing is projected, and
  the outcome that arrives later lands once.

The last one is the legacy shape — a journal written before checkpoints existed
carries no outcome at all — and it is written down because "the reveal always
knows" would be the wrong conclusion to draw from the first two.

The size is small on purpose: what is being measured is which frame a block
appears in, not how a long conversation pages.
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
from local_operator.tui.widgets.transcript import NoticeBlock
from tests.e2e.harness import seed_transcript
from tests.unit.session.test_remote import _never_take_over

#: A terminal wide enough that the notice's signpost fits whole, so the text
#: being compared is the composed sentence and not a width decision.
FRAME = (120, 34)

#: The failures the durable record carries. Two, and deliberately different in
#: kind: one naming a host (the transport family) and one a command, because the
#: signpost rule treats them differently and both must land.
FAILURES = {
    "linear": "cannot reach linear.example",
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
    """Per-DISPLAY samples of the transcript's notice rows.

    ``post_display_hook`` is the only place a frame is a fact rather than an
    inference — the same instrument the reveal tests use — so "which frame did
    this row appear in" is answered from painted displays, not from awaits.
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
                        "notices": [
                            # ``NoticeBlock._text`` is the composed sentence the
                            # row was built with (the same attribute every
                            # removal by text reads); asked defensively because a
                            # frame can be sampled while a block is still being
                            # authored.
                            str(getattr(block, "_text", "") or "")
                            for block in blocks
                            if isinstance(block, NoticeBlock)
                        ],
                    }
                )
            except Exception:  # noqa: BLE001 — a frame before the transcript exists is not a state
                pass
            real()

        app.post_display_hook = hook  # type: ignore[method-assign]

    def content(self) -> list[dict[str, object]]:
        """Frames painted with the session in front of the reader, carrying rows."""
        return [
            sample
            for sample in self.samples
            if sample["target"] and not sample["parked"] and sample["blocks"] > 0
        ]


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


def _lines_mentioning(lines: list[str], failures: dict[str, str]) -> list[str]:
    return [line for line in lines if any(f"MCP {name} failed" in line for name in failures)]


@pytest.mark.asyncio
async def test_a_cold_open_paints_its_durable_mcp_failures_before_the_first_frame(
    tmp_path, monkeypatch
) -> None:
    """RED on a tree that restores every durable field except ``mcp_startup``.

    The first contentful frame carries no MCP rows there, and the replay that
    the bind performs appends both of them — so the settled frame differs from
    the first in count and in text, which is exactly the shift reported.
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
            assert len(_lines_mentioning(first["notices"], FAILURES)) == len(FAILURES), (
                "the first contentful frame did not carry the durable failure rows: "
                f"{first['notices']}"
            )
            assert all(
                position > first["last_message_index"] for position in first["notice_positions"]
            ), (
                "a durable row was painted ABOVE the replayed conversation, where the "
                "reader has to scroll back for it: "
                f"notices at {first['notice_positions']}, conversation ends at "
                f"{first['last_message_index']}"
            )

            # What the post-paint bind does with the SAME outcome the checkpoint
            # already carried. It must recognise the rows as painted.
            app._report_mcp_startup(viewer)
            await _settle(pilot, frames)
            settled = frames.content()[-1]
            assert settled["notices"] == first["notices"], (
                "replaying the durable outcome appended rows the first frame already had: "
                f"{first['notices']} -> {settled['notices']}"
            )
    finally:
        await viewer.dispose()


@pytest.mark.asyncio
async def test_a_cold_open_without_a_durable_outcome_projects_nothing_and_appends_once(
    tmp_path, monkeypatch
) -> None:
    """The legacy shape: no checkpoint, so the outcome is genuinely NEW at bind.

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
            assert _lines_mentioning(first["notices"], FAILURES) == []

            viewer.mcp_startup = McpStartupOutcome(
                configured=tuple(sorted(FAILURES)),
                connected=(),
                failures=dict(FAILURES),
                tool_count=0,
            )
            app._report_mcp_startup(viewer)
            await _settle(pilot, frames)
            settled = frames.content()[-1]
            assert len(_lines_mentioning(settled["notices"], FAILURES)) == len(FAILURES), (
                "the first known outcome must land once: " f"{settled['notices']}"
            )
            # And the replay stays quiet, the same as on the durable path.
            app._report_mcp_startup(viewer)
            await _settle(pilot, frames)
            assert frames.content()[-1]["notices"] == settled["notices"], (
                "a second replay of an unchanged outcome appended its rows again: "
                f"{settled['notices']} -> {frames.content()[-1]['notices']}"
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
            assert [row for row in first["notices"] if row.startswith("goal restored:")], (
                "the durable goal's receipt was not in the first contentful frame: "
                f"{first['notices']}"
            )
            assert all(
                position > first["last_message_index"] for position in first["notice_positions"]
            ), "a durable row was painted above the replayed conversation"
            settled = frames.content()[-1]
            assert settled["notices"] == first["notices"], (
                f"the settled frame differs from the first: {first['notices']} -> "
                f"{settled['notices']}"
            )
    finally:
        await viewer.dispose()
