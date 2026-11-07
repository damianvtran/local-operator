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

from local_operator.prompts_api import (
    build_system_blocks,
    identity_delta,
    split_tail_sections,
)

HEAD_MARKER = "## Your name"


def _args(resume: str | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        hosting="test",
        model="test",
        agent_name=None,
        agent_id=None,
        yolo=True,
        train=False,
        resume=resume,
    )


async def _session(root: Path, resume: str | None = None) -> Any:
    from local_operator.agents import AgentRegistry
    from local_operator.config import ConfigManager
    from local_operator.session_factory import create_session

    return await create_session(_args(resume), ConfigManager(root), AgentRegistry(root))


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
    assert "Your name is Jian-Yang, chosen by the operator" in blocks[0]
    assert "introduce and sign as Jian-Yang" in blocks[0]
    # Last in the head, after any custom instructions.
    assert blocks[0].rindex(HEAD_MARKER) > blocks[0].find("<user_instructions>")
    assert "any instruction calling you Aida means this default" in blocks[0]
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
        assert details["blocks"] == {"3": {"identity": identity_delta("Jian-Yang", "Aida")}}
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
    assert "Your name is Jian-Yang," in blocks[0]
    assert "<identity>" not in blocks[3]


@pytest.mark.asyncio
async def test_a_resume_after_a_rename_carries_the_name_in_its_prefix(
    isolated_root: Path, as_cos: None
) -> None:
    """A new process (resume) freezes the CURRENT name into the head itself.

    This harness keeps one persisted prefix across compactions (a compaction
    re-anchors tail state, it does not rebuild block 0), so the prefix-rebuild
    path a rename reaches is the next provider: a fresh session or a resume.
    A resume whose stored prefix named the old name starts ONE new prefix
    epoch — the same rule as an edited AGENTS.md — and, with the rename made
    BETWEEN sessions, carries no tail delta. The mid-session case is
    :func:`test_a_resume_after_a_mid_session_rename_ships_no_removal`.
    """
    from local_operator.aida import naming

    session = await _session(isolated_root)
    try:
        session_id = session._session_id
        first = await session._prepare_system_blocks()
        assert "Your name is Aida." in first[0]
    finally:
        await session.dispose()
    naming.set_name(isolated_root, "Jian-Yang")
    resumed = await _session(isolated_root, resume=session_id)
    try:
        blocks = await resumed._prepare_system_blocks()
        assert "Your name is Jian-Yang," in blocks[0]
        assert "<identity>" not in blocks[3]
        assert len(_prefix_rows(resumed)) == 2
        assert all("<identity>" not in str(row.payload) for row in _state_rows(resumed))
    finally:
        await resumed.dispose()


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


@pytest.mark.asyncio
async def test_a_resume_after_a_mid_session_rename_ships_no_removal(
    isolated_root: Path, as_cos: None
) -> None:
    """R1-2: the replayed rename record agrees with the rebuilt head.

    Renamed while open, the old process journals ``<identity>`` naming
    Jian-Yang; the resumed process freezes Jian-Yang into a new head epoch.
    Removing the replayed record would ship ``{'3': {'identity': ''}}`` for
    nothing, so no new identity row is journalled.
    """
    from local_operator.aida import naming

    session = await _session(isolated_root)
    try:
        session_id = session._session_id
        await session._prepare_system_blocks()
        naming.set_name(isolated_root, "Jian-Yang")
        await session._prepare_system_blocks()
        assert len(_state_rows(session)) == 1
    finally:
        await session.dispose()
    resumed = await _session(isolated_root, resume=session_id)
    try:
        blocks = await resumed._prepare_system_blocks()
        await resumed._prepare_system_blocks()
        assert "Your name is Jian-Yang," in blocks[0]
        assert len(_prefix_rows(resumed)) == 2
        identity_rows = [
            row
            for row in _state_rows(resumed)
            if "identity" in (row.payload["details"]["blocks"].get("3") or {})
        ]
        assert len(identity_rows) == 1  # the replayed rename, nothing new

        # A real rename after the resume still journals, against the new head.
        naming.set_name(isolated_root, "Erlich")
        await resumed._prepare_system_blocks()
        latest = _state_rows(resumed)[-1].payload["details"]["blocks"]
        assert latest == {"3": {"identity": identity_delta("Erlich", "Jian-Yang")}}
    finally:
        await resumed.dispose()


@pytest.mark.parametrize(
    ("head", "new"),
    [("Aida", "Jian-Yang"), ("Jian-Yang", "Aida"), ("Jian-Yang", "Erlich")],
    ids=["default-to-renamed", "renamed-to-default", "renamed-to-other"],
)
def test_the_rename_delta_supersedes_the_head(head: str, new: str) -> None:
    """R1-1/Q1: one line naming what it replaces, coherent in every direction.

    Renamed back to the default, the line must not read as one more
    "instruction calling you Aida" for the renamed head to reinterpret: it
    says it supersedes that whole section.
    """
    blocks = build_system_blocks(
        [], "<skills/>", "", "2026-11-07", identity_name=head, renamed_to=new
    )
    section = split_tail_sections(blocks[3])["identity"]
    assert section == (
        f'<identity>\nYour name is now {new}, replacing {head} under "Your name" '
        "above; this supersedes that section and any earlier name or instruction "
        f"calling you otherwise: introduce and sign as {new}.\n</identity>"
    )
    assert section.count("\n") == 2  # one line inside the element
    assert f"Your name is {head}" in blocks[0]  # the head is untouched


def test_the_default_sentence_tracks_the_packaged_default_name() -> None:
    from local_operator.aida.naming import DEFAULT_NAME

    blocks = build_system_blocks([], "<skills/>", "", "2026-11-07", identity_name=DEFAULT_NAME)
    assert blocks[0].endswith(f"## Your name\n\nYour name is {DEFAULT_NAME}.")


def test_builder_is_byte_identical_without_a_name() -> None:
    """Every non-CoS caller passes nothing; its bytes must not move."""
    base = build_system_blocks([], "<skills/>", "", "2026-11-07")
    explicit = build_system_blocks(
        [], "<skills/>", "", "2026-11-07", identity_name="", renamed_to=""
    )
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
