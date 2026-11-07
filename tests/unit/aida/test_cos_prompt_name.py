"""The chief of staff is TOLD her configured name — and a rename reaches her.

The incident (2026-11): an operator renamed his chief of staff "Jian-Yang";
Team Bridge labelled her "Jian-Yang (Victor Tay)" while the body she wrote
opened "Aida here". The label path read ``aida.name``; nothing put that name in
front of the model. These tests drive the REAL composition root
(``session_factory.create_session``) and read the blocks and the journalled
``[session-state]`` rows a turn would carry, because a builder-level test alone
passes with the wiring deleted.

The cache contract is the other half: the name rides the head block ONCE
(frozen per session), a no-rename session's prefix never moves, and a mid-
session rename costs exactly one tail ``<identity>`` section — never a new
persisted prefix epoch.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest

from local_operator.prompts_api import build_system_blocks, split_tail_sections

HEAD_MARKER = "## Your name"


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        hosting="test", model="test", agent_name=None, agent_id=None, yolo=True, train=False
    )


async def _session(root: Path) -> Any:
    from local_operator.agents import AgentRegistry
    from local_operator.config import ConfigManager
    from local_operator.session_factory import create_session

    return await create_session(_args(), ConfigManager(root), AgentRegistry(root))


def _state_rows(session: Any) -> list[Any]:
    return [
        entry
        for entry in session._transcript.entries()
        if str(entry.payload.get("custom_type", "")) == "session_state"
    ]


def _prefix_rows(session: Any) -> list[Any]:
    return [
        entry
        for entry in session._transcript.entries()
        if str(entry.payload.get("custom_type", "")) == "system_prefix"
    ]


@pytest.fixture()
def as_cos(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every session constructed in the test HER session.

    Patched at the predicate rather than writing ``state.json`` because the
    session id is minted inside ``create_session``; the predicate itself is
    covered by ``test_aida_state``.
    """
    from local_operator.aida import state as aida_state

    monkeypatch.setattr(aida_state, "is_aida_session", lambda _root, sid: bool(sid))


@pytest.mark.asyncio
async def test_a_renamed_cos_reads_her_name_in_the_head_block(
    isolated_root: Path, as_cos: None
) -> None:
    from local_operator.aida import naming

    naming.set_name(isolated_root, "Jian-Yang")
    session = await _session(isolated_root)
    try:
        blocks = await session._prepare_system_blocks()
    finally:
        await session.dispose()
    assert "Your name is Jian-Yang." in blocks[0]
    assert "introduce yourself, sign messages and refer to yourself as Jian-Yang" in blocks[0]
    # After the profile body, so a stale "You are Aida" in it is the earlier,
    # overridden statement.
    assert blocks[0].rindex(HEAD_MARKER) > blocks[0].find("<user_instructions>")
    assert not any("Jian-Yang" in block for block in blocks[1:])


@pytest.mark.asyncio
async def test_an_unrenamed_cos_is_aida(isolated_root: Path, as_cos: None) -> None:
    session = await _session(isolated_root)
    try:
        blocks = await session._prepare_system_blocks()
    finally:
        await session.dispose()
    assert "Your name is Aida." in blocks[0]


@pytest.mark.asyncio
async def test_an_ordinary_session_is_never_given_a_name(isolated_root: Path) -> None:
    from local_operator.aida import naming

    naming.set_name(isolated_root, "Jian-Yang")
    session = await _session(isolated_root)
    try:
        blocks = await session._prepare_system_blocks()
        naming.set_name(isolated_root, "Someone Else")
        await session._prepare_system_blocks()
        assert _state_rows(session) == []
    finally:
        await session.dispose()
    assert HEAD_MARKER not in blocks[0]
    assert not any("<identity>" in block or "Jian-Yang" in block for block in blocks)


@pytest.mark.asyncio
async def test_no_rename_keeps_the_prefix_byte_stable(isolated_root: Path, as_cos: None) -> None:
    session = await _session(isolated_root)
    try:
        first = await session._prepare_system_blocks()
        for _ in range(20):
            assert await session._prepare_system_blocks() == first
        assert _state_rows(session) == []
        assert len(_prefix_rows(session)) == 1
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_mid_session_rename_is_one_delta_and_the_prefix_stays(
    isolated_root: Path, as_cos: None
) -> None:
    from local_operator.aida import naming

    session = await _session(isolated_root)
    try:
        first = await session._prepare_system_blocks()
        naming.set_name(isolated_root, "Jian-Yang")
        # The NEXT turn: the persisted prefix is returned unchanged...
        assert await session._prepare_system_blocks() == first
        for _ in range(10):
            await session._prepare_system_blocks()
        # ...and exactly one state row carries the new name, as one section.
        rows = _state_rows(session)
        assert len(rows) == 1
        details = rows[0].payload["details"]
        assert details["blocks"] == {
            "3": {
                "identity": (
                    "<identity>\nRenamed this session: your name is now Jian-Yang "
                    "(supersedes the name above).\n</identity>"
                )
            }
        }
        assert "## Your name" in details["text"]
        assert len(_prefix_rows(session)) == 1
        assert "Your name is Aida." in first[0]

        # Writing the SAME name again is not a change: nothing journals.
        naming.set_name(isolated_root, "Jian-Yang")
        await session._prepare_system_blocks()
        assert len(_state_rows(session)) == 1
    finally:
        await session.dispose()

    # A fresh session (and every later prefix rebuild) carries the new name in
    # the head itself, with no tail section.
    fresh = await _session(isolated_root)
    try:
        blocks = await fresh._prepare_system_blocks()
    finally:
        await fresh.dispose()
    assert "Your name is Jian-Yang." in blocks[0]
    assert "<identity>" not in blocks[3]


@pytest.mark.asyncio
async def test_renaming_back_ships_a_removal(isolated_root: Path, as_cos: None) -> None:
    from local_operator.aida import naming

    session = await _session(isolated_root)
    try:
        await session._prepare_system_blocks()
        naming.set_name(isolated_root, "Jian-Yang")
        await session._prepare_system_blocks()
        naming.set_name(isolated_root, "Aida")
        await session._prepare_system_blocks()
        rows = _state_rows(session)
        assert len(rows) == 2
        assert rows[1].payload["details"]["blocks"] == {"3": {"identity": ""}}
    finally:
        await session.dispose()


def test_builder_is_byte_identical_without_a_name() -> None:
    """Every non-CoS caller passes nothing; its bytes must not move."""
    base = build_system_blocks([], "<skills/>", "", "2026-11-07")
    explicit = build_system_blocks([], "<skills/>", "", "2026-11-07", identity_name="", renamed_to="")
    assert base == explicit
    assert HEAD_MARKER not in base[0]


def test_identity_section_round_trips_through_the_splitter() -> None:
    blocks = build_system_blocks(
        [],
        "<skills/>",
        "",
        "2026-11-07",
        goal="ship it",
        agent_brief="be brief",
        interactive=True,
        identity_name="Aida",
        renamed_to="Jian-Yang",
    )
    sections = split_tail_sections(blocks[3])
    assert list(sections) == ["knowledge", "goal", "agent", "identity", "interactivity"]
    assert "Jian-Yang" in sections["identity"]
    assert "\n\n".join(sections.values()) == blocks[3]
