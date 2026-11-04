"""Per-session rules ride the outbound system tail on every call.

Why this file exists, in one measurement: a rule delivered only as a
``[session-state]`` delta did not beat the same text in custom instructions
(merge case: 3/5 replays broken against 1/5). The prefix freeze is what made it
so -- after the first commit the system blocks a session sends are
``_frozen_system_blocks``, and a changed tail section was reduced to a delta
message, so the system prompt never contained the rule again. A standing rule
must be system text on every request, so a live section is spliced into a copy
of the frozen tail and the delta path skips it.

Everything here drives a real ``Session`` over the real factory provider,
imported from ``test_session_rules_acceptance`` rather than copied: the sidecar
is written by the operator's own module, the render is the production one, and
the assertions read both the blocks the reconcile path returns and the records
the transcript receives.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import (
    AbortSignal,
    ChatRequest,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
)
from local_operator.prompts_api import (
    _TAIL_SECTION_MARKERS,
    ADVISORY_TAIL_SECTIONS,
    LIVE_TAIL_SECTIONS,
    assemble_system_block_sections,
    build_system_blocks,
    split_system_block_sections,
    with_live_tail_sections,
)
from local_operator.session.rules import add_rule, edit_rule, remove_rule
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from tests.unit.test_session_rules_acceptance import _bump_mtime, _make_provider

MODEL = ModelSpec(provider="test", model_id="m", context_window=100_000)

RULES_ENV = "LOP_SESSION_RULES"
SIDECAR = "rules.json"
RULE_TEXT = "End every reply with ALPHA."
EDITED_TEXT = "End every reply with BRAVO."

SKILLS = "## Guides\n\n- demo: A demo guide."
ENV = "cwd: /tmp/project\nOS: Darwin"
DATE = "2026-11-04"


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


def _make_session(directory: Path, stream: RecordingStream) -> Session:
    """A real session over the production per-turn blocks closure."""
    transcript = Transcript(directory)
    provider = _make_provider(transcript, directory)
    return Session(
        model=MODEL,
        stream_fn=stream,
        tools=[],
        transcript=transcript,
        system_blocks_provider=provider,
        has_ui=True,
    )


def _frozen(session: Session) -> list[str]:
    """The session's frozen prefix, asserted to exist (it always has after a turn)."""
    blocks = session._frozen_system_blocks
    assert blocks is not None, "the prefix was never frozen"
    return blocks


def _state_rows(session: Session) -> list[dict[str, Any]]:
    """Every ``[session-state]`` record this session's journal holds."""
    return [
        entry.payload.get("details", {}) or {}
        for entry in session._transcript.entries()
        if entry.payload.get("custom_type") == "session_state"
    ]


def _rules_rows(session: Session) -> list[str]:
    """The state texts that claim anything about the rules section.

    Searched by the rendered heading rather than by ``details.blocks`` so a
    whole-block record (the pre-sections shape) is caught as well.
    """
    return [str(row.get("text", "")) for row in _state_rows(session) if "## Rules" in str(row)]


async def _answer_nothing(questions: list[Any]) -> Any:
    """A host hook that exists: the tool gates on its presence, not its answer."""
    return None


def _wire_turn_blocks(stream: RecordingStream) -> list[list[str]]:
    """The multi-block requests: turn-level calls, not one-shot helpers."""
    return [request.system_blocks for request in stream.requests if len(request.system_blocks) >= 4]


@pytest.mark.asyncio
async def test_a_rule_added_after_the_freeze_is_in_the_outbound_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(RULES_ENV, "1")
    session_dir = tmp_path / "session"
    stream = RecordingStream()
    session = _make_session(session_dir, stream)
    try:
        await session.prompt("freeze the prefix")
        frozen = list(_frozen(session))
        assert "<rules>" not in frozen[3]

        add_rule(session_dir, RULE_TEXT)
        _bump_mtime(session_dir / SIDECAR)

        blocks = await session._prepare_system_blocks()

        assert "<rules>" in blocks[3], "the rule never reached the system blocks"
        assert RULE_TEXT in blocks[3]
        assert "<rules>" not in _frozen(session)[3], "the frozen prefix was rewritten"
        assert _frozen(session) == frozen, "the frozen prefix moved"

        # The same blocks a real turn sends, not a private view.
        await session.prompt("follow the rule now")
        turns = _wire_turn_blocks(stream)
        assert turns, "no turn-level request was recorded"
        assert "<rules>" in turns[-1][3]
        assert RULE_TEXT in turns[-1][3]
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_adding_a_rule_mid_session_changes_the_blocks_without_a_state_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(RULES_ENV, "1")
    session_dir = tmp_path / "session"
    stream = RecordingStream()
    session = _make_session(session_dir, stream)
    try:
        await session.prompt("freeze the prefix")
        before = await session._prepare_system_blocks()

        add_rule(session_dir, RULE_TEXT)
        _bump_mtime(session_dir / SIDECAR)
        after = await session._prepare_system_blocks()

        assert after[3] != before[3], "a rule added after the freeze was invisible"
        assert after[:3] == before[:3], "a tail rule moved a cached head block"
        assert _rules_rows(session) == [], "a rule change was shipped twice, as a delta as well"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_removing_a_rule_removes_the_section_from_the_outbound_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(RULES_ENV, "1")
    session_dir = tmp_path / "session"
    rule = add_rule(session_dir, RULE_TEXT)
    stream = RecordingStream()
    session = _make_session(session_dir, stream)
    try:
        await session.prompt("freeze the prefix")
        assert "<rules>" in _frozen(session)[3], "the rule was not frozen in"

        remove_rule(session_dir, rule.id)
        _bump_mtime(session_dir / SIDECAR)
        blocks = await session._prepare_system_blocks()

        assert "<rules>" not in blocks[3], "a removed rule still rides the system blocks"
        assert RULE_TEXT not in blocks[3]
        assert "<rules>" in _frozen(session)[3], "the frozen prefix was rewritten"
        assert _rules_rows(session) == [], "a removal was shipped twice, as a delta as well"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_an_edit_moves_only_the_rules_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(RULES_ENV, "1")
    session_dir = tmp_path / "session"
    rule = add_rule(session_dir, RULE_TEXT)
    stream = RecordingStream()
    session = _make_session(session_dir, stream)
    try:
        await session.prompt("freeze the prefix")
        frozen = list(_frozen(session))
        frozen_sections = split_system_block_sections(3, frozen[3])

        edit_rule(session_dir, rule.id, EDITED_TEXT)
        _bump_mtime(session_dir / SIDECAR)
        blocks = await session._prepare_system_blocks()
        sections = split_system_block_sections(3, blocks[3])

        assert sections["rules"] != frozen_sections["rules"]
        assert EDITED_TEXT in sections["rules"]
        assert set(sections) == set(frozen_sections)
        for name, text in frozen_sections.items():
            if name not in LIVE_TAIL_SECTIONS:
                assert sections[name] == text, f"{name} moved with a rules edit"
        assert blocks[:3] == frozen[:3]
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_compaction_reanchor_does_not_ship_rules_as_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ``lost_state`` branch is the same walk, so it must skip live sections too.

    A compaction that outlives the last state record re-anchors every section
    against what the model can see. A rule added after the freeze is NOT in
    what it can see under the old protocol -- that is exactly why the re-anchor
    used to ship it as a ``## Rules`` record -- and it still must not, because
    the splice puts it in the system blocks of the same request.
    """
    monkeypatch.setenv(RULES_ENV, "1")
    session_dir = tmp_path / "session"
    stream = RecordingStream()
    session = _make_session(session_dir, stream)
    try:
        await session.prompt("freeze the prefix")
        # A real state record first: ``lost_state`` only fires past the id it set.
        session.set_ask_handler(_answer_nothing)
        await session.prompt("install the ask door")
        assert session._system_state_message_id is not None

        add_rule(session_dir, RULE_TEXT)
        _bump_mtime(session_dir / SIDECAR)
        entries = session._transcript.entries()
        await session._transcript.append_compaction("summary", entries[0].id, 0)

        blocks = await session._prepare_system_blocks()

        assert "<rules>" in blocks[3]
        assert _rules_rows(session) == []
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_with_the_flag_off_the_blocks_are_the_frozen_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(RULES_ENV, raising=False)
    session_dir = tmp_path / "session"
    # A sidecar on disk that the flag must make inert.
    add_rule(session_dir, RULE_TEXT)
    stream = RecordingStream()
    session = _make_session(session_dir, stream)
    try:
        await session.prompt("freeze the prefix")
        frozen = list(_frozen(session))

        blocks = await session._prepare_system_blocks()

        assert blocks == frozen
        assert "<rules>" not in "\n".join(blocks)
        assert _state_rows(session) == []
    finally:
        await session.dispose()


def test_live_sections_are_tail_sections_and_never_advisory() -> None:
    assert LIVE_TAIL_SECTIONS <= {name for name, _ in _TAIL_SECTION_MARKERS}
    assert not (LIVE_TAIL_SECTIONS & ADVISORY_TAIL_SECTIONS)


def test_the_splice_with_no_live_section_keeps_the_bytes() -> None:
    frozen = build_system_blocks(
        [], SKILLS, ENV, DATE, goal="ship it", host_has_browser=False, host_has_console=False
    )

    assert with_live_tail_sections(frozen, list(frozen)) == frozen


def test_the_splice_adds_the_section_in_its_composition_position() -> None:
    frozen = build_system_blocks(
        [], SKILLS, ENV, DATE, goal="ship it", host_has_browser=False, host_has_console=False
    )
    rendered = "<rules>\n1. [r1] Alpha\n</rules>"
    desired = build_system_blocks(
        [],
        SKILLS,
        ENV,
        DATE,
        goal="ship it",
        session_rules=rendered,
        host_has_browser=False,
        host_has_console=False,
    )

    blocks = with_live_tail_sections(frozen, desired)

    assert blocks[3].index("<goal>") < blocks[3].index("<rules>")
    assert blocks[3] == assemble_system_block_sections(3, split_system_block_sections(3, blocks[3]))
    assert "<rules>" not in frozen[3], "the input list was mutated"
