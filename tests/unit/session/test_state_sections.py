"""Section-granular ``[session-state]`` deltas.

A state delta used to re-ship a WHOLE block whenever anything inside it moved:
a one-character change in block 3 re-injected all 7.5-22.8k chars of the tail,
and most of those re-sent bytes were sections that had not moved at all.
Records now carry only the changed sections, and the resume fold replays them,
so the two contracts this file pins are:

- a change inside one section ships only that section, a section that goes away
  ships as an explicit empty section, and a block-1 change does not drag the
  unchanged capability notes along; and
- folding a section-granular transcript lands on the SAME bytes as the
  whole-block path did, across a real change sequence — the test that catches a
  split whose boundaries do not match the renderer's composition, or a removal
  the delta forgot to state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import (
    AbortSignal,
    ChatRequest,
    CustomMessage,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
)
from local_operator.prompts_api import (
    CHANNEL_NONE,
    TOOL_INVENTORY_HEADING,
    build_system_blocks,
)
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript

MODEL = ModelSpec(provider="test", model_id="m", context_window=100_000)

SKILLS = "## Guides\n\n- demo: A demo guide."
ENV = "cwd: /tmp/project\nOS: Darwin"
DATE = "2026-09-29"


class RecordingStream:
    """Answers every turn with one line, keeping the requests it was given."""

    def __init__(self) -> None:
        self.requests: list[ChatRequest] = []

    def __call__(self, request: ChatRequest, signal: AbortSignal | None):
        self.requests.append(request)

        async def gen():
            yield StreamTextDelta(delta="ok")
            yield StreamEndEvent(stop_reason="stop")

        return gen()


def _state_provider(state: dict[str, Any], tools: list[Any]):
    """A session-blocks provider over a live state dict, deterministic per state.

    The host answers are PINNED (not probed) so the run is identical on every
    machine, and published on the provider attributes because the session's own
    block-1 re-render reads them — the two renderers must agree byte for byte
    or every prompt would journal a phantom inventory delta.
    """

    def provider(model_label: str = "") -> list[str]:
        knowledge = "\n\n".join(piece for piece in (SKILLS, state.get("recs", "")) if piece)
        return build_system_blocks(
            tools,
            knowledge,
            ENV,
            DATE,
            goal=state["goal"],
            team_brief=state["team"],
            agent_brief=state["agent"],
            interactive=(
                state["interactive_from"].interactivity()
                if "interactive_from" in state
                else state["interactive"]
            ),
            channel=CHANNEL_NONE,
            credentials=state["credentials"],
            host_has_browser=False,
            host_has_console=False,
        )

    setattr(provider, "append_only_state", True)
    setattr(provider, "host_has_browser", False)
    setattr(provider, "host_has_console", False)
    return provider


def _initial_state() -> dict[str, Any]:
    return {
        "goal": "ship it",
        "team": "",
        "agent": "",
        "interactive": True,
        "credentials": [],
        "recs": "",
    }


def _make_session(
    directory: Path, stream: RecordingStream, state: dict[str, Any], tools: list[Any]
) -> Session:
    return Session(
        model=MODEL,
        stream_fn=stream,
        tools=tools,
        transcript=Transcript(directory),
        system_blocks_provider=_state_provider(state, tools),
        has_ui=True,
    )


def _state_records(session: Session) -> list[CustomMessage]:
    return [
        message
        for message in session._context.messages
        if isinstance(message, CustomMessage) and message.custom_type == "session_state"
    ]


async def _answer_nothing(questions: list[Any]) -> Any:
    """A host hook that exists: the tool gates on its presence, not its answer."""
    return None


@pytest.mark.asyncio
async def test_a_change_inside_one_tail_section_ships_only_that_section(tmp_path) -> None:
    state = _initial_state()
    tools: list[Any] = []
    stream = RecordingStream()
    session = _make_session(tmp_path / "sess", stream, state, tools)
    await session.prompt("freeze the prefix")

    state["team"] = "collaborate on the release"
    await session.prompt("attach the team")

    records = _state_records(session)
    assert len(records) == 1
    assert records[-1].details["blocks"] == {
        "3": {"team": "<team>\ncollaborate on the release\n</team>"}
    }
    text = str(records[-1].details["text"])
    assert text.startswith("[session-state]\n## Team\n")
    # The sections that did not move did not ride: neither the knowledge piece
    # nor the (unchanged) goal, which the whole-block protocol re-shipped.
    assert SKILLS not in text
    assert "<goal>" not in text
    await session.dispose()


@pytest.mark.asyncio
async def test_a_cleared_section_ships_as_an_explicit_empty_section(tmp_path) -> None:
    state = _initial_state()
    state["team"] = "start together"
    tools: list[Any] = []
    stream = RecordingStream()
    session = _make_session(tmp_path / "sess", stream, state, tools)
    await session.prompt("freeze the prefix")

    state["team"] = ""
    await session.prompt("detach the team")

    records = _state_records(session)
    assert len(records) == 1
    assert records[-1].details["blocks"] == {"3": {"team": ""}}
    # Silence would read as "unchanged", not as "gone"; the record says empty.
    assert "## Team\n(empty)" in str(records[-1].details["text"])
    await session.dispose()


@pytest.mark.asyncio
async def test_a_late_recommendation_ships_only_the_recommendation(tmp_path) -> None:
    """The measured driver of the residual re-sends: a late classification
    answer re-renders the knowledge block, and under whole-block deltas that
    re-sent the whole guides/skills listing beside it. The recommendation block
    is its own section now, so only it moves."""
    state = _initial_state()
    tools: list[Any] = []
    stream = RecordingStream()
    session = _make_session(tmp_path / "sess", stream, state, tools)
    await session.prompt("freeze the prefix")

    recs = "<resource_recommendations>\nRecommend `skill://a`.\n</resource_recommendations>"
    state["recs"] = recs
    await session.prompt("a late answer arrives")

    records = _state_records(session)
    assert len(records) == 1
    assert records[-1].details["blocks"] == {"3": {"recs": recs}}
    # The listing beside it was not re-sent.
    assert SKILLS not in str(records[-1].details["text"])
    await session.dispose()


@pytest.mark.asyncio
async def test_an_inventory_change_does_not_reship_the_capability_notes(tmp_path) -> None:
    state = _initial_state()
    tools: list[Any] = []
    stream = RecordingStream()
    session = _make_session(tmp_path / "sess", stream, state, tools)
    await session.prompt("before the front end wires itself up")

    session.set_ask_handler(_answer_nothing)
    await session.prompt("and now?")

    records = _state_records(session)
    assert len(records) == 1
    blocks = records[-1].details["blocks"]
    assert set(blocks) == {"1"}
    assert set(blocks["1"]) == {"tools"}, "the unchanged capability notes were re-shipped"
    assert any(line == "- ask" for line in blocks["1"]["tools"].splitlines())
    # The notes existed in the frozen block — they were deliberately not re-sent.
    frozen = session._frozen_system_blocks or []
    assert "NO browser tool" in frozen[1]
    assert "NO browser tool" not in str(records[-1].details["text"])
    await session.dispose()


@pytest.mark.asyncio
async def test_folding_a_change_sequence_lands_on_the_full_block_path_bytes(tmp_path) -> None:
    """The accuracy test: byte equality across a real sequence of changes.

    Three views of the same run must agree — the live session's state after the
    last delta, a fresh session frozen in the final state, and a session resumed
    from the section-granular records — and so must the whole-block path that a
    pre-sections transcript carries (last writer wins per index, which is what
    ``main``'s fold did). Any boundary mismatch, missed change, or unstated
    removal shows up here as a byte difference.
    """
    state = _initial_state()
    tools: list[Any] = []
    stream = RecordingStream()
    session = _make_session(tmp_path / "sess", stream, state, tools)
    await session.prompt("freeze the prefix")
    renders: list[list[str]] = [list(session._last_system_blocks or [])]

    changes: list[dict[str, Any]] = [
        {"team": "collaborate on the release"},
        {"goal": "land the section deltas"},
        {"team": ""},
        {"interactive": False},
        {"credentials": ["DEPLOY_TOKEN"]},
        {
            "recs": (
                "<resource_recommendations>\nRecommend `skill://a`.\n" "</resource_recommendations>"
            )
        },
        {"agent": "reviewer"},
        {"interactive": True},
    ]
    for step, change in enumerate(changes, start=1):
        state.update(change)
        await session.prompt(f"step {step}")
        renders.append(list(session._last_system_blocks or []))
    final = renders[-1]

    # (a) a fresh session frozen in the final state renders the same bytes.
    probe_stream = RecordingStream()
    probe = _make_session(tmp_path / "probe", probe_stream, state, tools)
    await probe._prepare_system_blocks()
    assert probe._frozen_system_blocks == final

    # (b) resume folds the section-granular records to the same bytes, and a
    # resumed session with nothing new writes nothing (the fold is stable).
    await session.dispose()
    resumed = _make_session(tmp_path / "sess", stream, state, tools)
    assert resumed._last_system_blocks == final
    entry_ids = [entry.id for entry in resumed._transcript.entries()]
    await resumed._prepare_system_blocks()
    assert [entry.id for entry in resumed._transcript.entries()] == entry_ids
    await resumed.dispose()

    # (c) the whole-block path carries the same bytes: a transcript with one
    # record per changed index (the shape every pre-sections run wrote).
    legacy = Transcript(tmp_path / "legacy")
    await legacy.append_custom("system_prefix", {"blocks": renders[0]})
    for previous, blocks in zip(renders, renders[1:]):
        changed = {
            str(index): block
            for index, block in enumerate(blocks)
            if index >= len(previous) or block != previous[index]
        }
        if changed:
            await legacy.append_custom("session_state", {"blocks": changed})
    view = _make_session(tmp_path / "legacy", stream, state, tools)
    assert view._last_system_blocks == final
    await view.dispose()
    await probe.dispose()


@pytest.mark.asyncio
async def test_a_section_record_applies_on_top_of_a_legacy_whole_block_record(tmp_path) -> None:
    """Mixed histories: the section protocol must patch a block a legacy record
    set, not only blocks the fold itself reconstructed from the frozen prefix."""
    state = _initial_state()
    tools: list[Any] = []
    stream = RecordingStream()
    tail = build_system_blocks([], SKILLS, ENV, DATE, team_brief="first team")[3]
    updated = "<team>\nsecond team\n</team>"

    legacy = Transcript(tmp_path / "mix")
    await legacy.append_custom("system_prefix", {"blocks": ["i", "t", "e", "old"]})
    await legacy.append_custom("session_state", {"blocks": {"3": tail}})
    await legacy.append_custom("session_state", {"blocks": {"3": {"team": updated}}})

    session = _make_session(tmp_path / "mix", stream, state, tools)
    assert session._last_system_blocks is not None
    assert session._last_system_blocks[3] == tail.replace("<team>\nfirst team\n</team>", updated)
    await session.dispose()


def test_the_whole_block_render_is_unchanged() -> None:
    """Backward compatibility at the renderer: a legacy whole-block value is
    labelled exactly as it always was, heading asymmetry included."""
    text = str(
        Session._system_state_message(
            {
                "1": f"{TOOL_INVENTORY_HEADING}\n\n- bash",
                "2": "Today is 2026-09-07.",
                "3": "goal: ship it",
            }
        ).details["text"]
    )
    assert text == (
        "[session-state]\n"
        f"{TOOL_INVENTORY_HEADING}\n\n- bash\n\n"
        "## Environment\nToday is 2026-09-07.\n\n"
        "## Knowledge and session state\ngoal: ship it"
    )


@pytest.mark.asyncio
async def test_a_multi_change_sequence_ships_far_fewer_bytes_than_whole_blocks(tmp_path) -> None:
    """The saving, measured on the same sequence the fold test replays.

    Method: sum the ``details.blocks`` payload size of each record — the bytes
    a reader processes and the session re-sends — under the section protocol,
    against the whole-block protocol's re-send for the same sequence (the full
    text of every block that changed at that step). Fix the direction, not a
    headline: the frozen composition here is small, and the fleet-scale replay
    belongs on the PR.
    """
    state = _initial_state()
    tools: list[Any] = []
    stream = RecordingStream()
    session = _make_session(tmp_path / "sess", stream, state, tools)
    await session.prompt("freeze the prefix")
    previous = list(session._last_system_blocks or [])
    sent_whole = 0
    sent_sections = 0

    changes: list[dict[str, Any]] = [
        {"team": "collaborate on the release"},
        {"goal": "land the section deltas"},
        {"team": "a renamed collaboration"},
        {"interactive": False},
        {"credentials": ["DEPLOY_TOKEN"]},
        {
            "recs": (
                "<resource_recommendations>\nRecommend `skill://a`.\n" "</resource_recommendations>"
            )
        },
        {"agent": "reviewer"},
        {"interactive": True},
        {"team": ""},
    ]
    for step, change in enumerate(changes, start=1):
        state.update(change)
        await session.prompt(f"step {step}")
        desired = list(session._last_system_blocks or [])
        sent_whole += sum(
            len(block)
            for index, block in enumerate(desired)
            if index > 0 and block != previous[index]
        )
        previous = desired

    sent_sections = sum(
        len(text)
        for record in _state_records(session)
        for block in record.details["blocks"].values()
        for text in block.values()
    )

    assert sent_whole > 0 and sent_sections > 0
    assert sent_sections < sent_whole * 0.5, (sent_sections, sent_whole)
    await session.dispose()


# --- the compaction re-anchor and the interactivity latch ---------------------
#
# Measured 2026-10-08 over 1,142 transcripts: every byte-identical re-send of a
# state section followed a compaction marker, because the re-anchor shipped
# EVERY section whenever the last state record was compacted away — although the
# frozen prefix never leaves the request and later records survive in context.
# Separately, the interactivity section flipped three times inside one manager
# turn as the control socket blinked.


async def _compact_after(session: Session, keep: CustomMessage | None = None) -> None:
    """Compact so that only ``keep`` (and what follows it) survives the cut."""
    from local_operator.harness.types import Message

    kept = Message.user("retained task")
    await session._transcript.append_message(kept)
    first_kept = keep.id if keep is not None else kept.id
    await session._transcript.append_compaction("summary", first_kept, 100)
    session._context.messages = session._transcript.build_llm_history()


@pytest.mark.asyncio
async def test_a_reanchor_reships_only_sections_the_model_cannot_see(tmp_path) -> None:
    state = _initial_state()
    state["team"] = "the roster"
    stream = RecordingStream()
    session = _make_session(tmp_path / "sess", stream, state, [])
    await session.prompt("freeze the prefix")  # team rides the frozen prefix

    state["goal"] = "a changed goal"
    await session.prompt("move the goal")
    assert len(_state_records(session)) == 1

    # The record carrying the new goal is compacted away; the prefix is not.
    await _compact_after(session)
    await session._prepare_system_blocks()

    records = _state_records(session)
    reanchor = records[-1].details["blocks"]
    # The goal the model can no longer see is re-shipped...
    assert "goal" in reanchor["3"] and "a changed goal" in reanchor["3"]["goal"]
    # ...and nothing it can still see in the frozen prefix rides along.
    assert set(reanchor) == {"3"}
    assert set(reanchor["3"]) == {"goal"}
    await session.dispose()


@pytest.mark.asyncio
async def test_a_reanchor_skips_sections_a_surviving_record_still_carries(tmp_path) -> None:
    state = _initial_state()
    stream = RecordingStream()
    session = _make_session(tmp_path / "sess", stream, state, [])
    await session.prompt("freeze the prefix")
    state["team"] = "the roster"
    await session.prompt("attach the team")
    team_record = _state_records(session)[-1]

    # The cut lands ON the team record, so it survives in context, but the
    # compaction id still moves past the session's last-published marker.
    await _compact_after(session, keep=team_record)
    before = len(_state_records(session))
    await session._prepare_system_blocks()
    # Nothing changed that the model cannot see: no record at all.
    assert len(_state_records(session)) == before
    await session.dispose()


@pytest.mark.asyncio
async def test_interactivity_moves_only_at_a_turn_boundary(tmp_path) -> None:
    state = _initial_state()
    attached = {"value": True}
    stream = RecordingStream()
    session = _make_session(tmp_path / "sess", stream, state, [])

    # The provider reads the session's holder, as the real factory's closure
    # does (``session_factory``: ``goal_state.interactivity()`` per render).
    state["interactive_from"] = session._goal_state
    session._goal_state.interactive_probe = lambda: attached["value"]

    await session.prompt("freeze the prefix")
    # A blink INSIDE a turn: every provider call in it must read the latched
    # answer, so no record is journalled however many calls the turn makes.
    attached["value"] = False
    for _ in range(3):
        await session._prepare_system_blocks()
    attached["value"] = True
    await session._prepare_system_blocks()
    assert _state_records(session) == []
    # Decisions that act NOW still see the present.
    attached["value"] = False
    assert session._goal_state.is_interactive() is False

    # A real detach reaches the model at the next turn — exactly once.
    await session.prompt("next turn")
    records = _state_records(session)
    assert len(records) == 1
    assert "No interface is attached" in records[0].details["blocks"]["3"]["interactivity"]
    await session.prompt("and another")
    assert len(_state_records(session)) == 1
    await session.dispose()
