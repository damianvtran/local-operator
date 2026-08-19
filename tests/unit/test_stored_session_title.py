"""The picker labels a session with the name the user last SAW.

Before this, every row was labelled with the conversation's opening message,
so a session renamed to something memorable was still listed under whatever
happened to be typed first — and a user who could not recall that opening line
could not find the session at all.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from local_operator.harness.types import Message, TextContent
from local_operator.resume import (
    _TITLE_CUSTOM_TYPE,
    TITLE_SCAN_BYTES,
    session_name,
    stored_session_title,
)
from local_operator.session.naming import CONVERSATION_NAME_CUSTOM_TYPE
from local_operator.session.transcript import Transcript


def _session(tmp_path: Path, opener: str, *titles: tuple[str, bool]) -> Path:
    session = tmp_path / "sessions" / "abcd1234"

    async def build() -> None:
        transcript = Transcript(session)
        await transcript.append_message(Message(role="user", content=[TextContent(text=opener)]))
        for text, user_set in titles:
            await transcript.append_custom(
                CONVERSATION_NAME_CUSTOM_TYPE, {"text": text, "user_set": user_set}
            )

    asyncio.run(build())
    return session


def test_the_journalled_title_type_matches_the_writer():
    """``resume`` may not import the engine, so it spells the entry type again.

    This is the pin that keeps the two spellings together: rename one and this
    fails, instead of every session silently reverting to its opening message.
    """
    assert _TITLE_CUSTOM_TYPE == CONVERSATION_NAME_CUSTOM_TYPE


def test_a_stored_title_is_the_name_the_picker_shows(tmp_path: Path):
    session = _session(tmp_path, "some forgettable opening line", ("Retention Fix", False))
    assert stored_session_title(session) == "Retention Fix"
    assert session_name(session) == "Retention Fix"


def test_the_newest_title_wins(tmp_path: Path):
    """Each rename appends a snapshot, so the last row is the one in force."""
    session = _session(
        tmp_path,
        "opening",
        ("First Generated Name", False),
        ("What The User Renamed It To", True),
    )
    assert stored_session_title(session) == "What The User Renamed It To"


def test_a_session_with_no_stored_title_falls_back_to_its_opener(tmp_path: Path):
    """Every transcript written before titles were journalled is in this
    state, and must still get a recognisable row."""
    session = _session(tmp_path, "the thing I typed first")
    assert stored_session_title(session) == ""
    assert session_name(session) == "the thing I typed first"


def test_a_title_with_json_escapes_reads_back_as_the_user_saw_it(tmp_path: Path):
    """Decoded through the JSON decoder, not a hand-rolled unescape."""
    session = _session(tmp_path, "opening", ('A "quoted" name \\ with escapes', True))
    assert stored_session_title(session) == 'A "quoted" name \\ with escapes'


def test_a_missing_transcript_yields_no_title_rather_than_raising(tmp_path: Path):
    """A picker row must never be the thing that takes the picker down."""
    assert stored_session_title(tmp_path / "sessions" / "nope") == ""


def test_a_title_is_found_past_a_megabyte_of_conversation(tmp_path: Path):
    """The scan reads the TAIL, because the title in force is the newest entry
    and on a long conversation that is megabytes past the opener."""
    session = tmp_path / "sessions" / "longone"

    async def build() -> None:
        transcript = Transcript(session)
        await transcript.append_message(
            Message(role="user", content=[TextContent(text="opening line")])
        )
        for index in range(200):
            await transcript.append_message(
                Message(role="assistant", content=[TextContent(text=f"{index} " + "z" * 8_000)])
            )
        await transcript.append_custom(
            CONVERSATION_NAME_CUSTOM_TYPE, {"text": "Named At The End", "user_set": True}
        )

    asyncio.run(build())
    assert session.joinpath("transcript.jsonl").stat().st_size > TITLE_SCAN_BYTES
    assert stored_session_title(session) == "Named At The End"


def test_a_long_title_is_condensed_for_the_row_but_not_for_a_caller(tmp_path: Path):
    """``condense=False`` is used where the full text is wanted; the picker
    row is the only place a name must survive being cut."""
    long_title = "An extremely long conversation title that runs well past the row budget"
    session = _session(tmp_path, "opening", (long_title, True))
    assert session_name(session, condense=False) == long_title
    assert session_name(session).endswith("…")


def test_a_title_written_early_survives_a_long_conversation(tmp_path: Path):
    """The case a tail-only scan missed on 78% of a real store.

    The title is journalled when the session is auto-named -- turn 2, near the
    HEAD -- and every turn after it pushes it further from the tail. Scanning
    only the tail therefore reverted most real sessions to their opening
    message, which is the exact failure this function exists to fix.
    """
    session = tmp_path / "sessions" / "buriedtitle"

    async def build() -> None:
        transcript = Transcript(session)
        await transcript.append_message(
            Message(role="user", content=[TextContent(text="check the flake build please")])
        )
        await transcript.append_custom(
            CONVERSATION_NAME_CUSTOM_TYPE, {"text": "Nix Flake Build Failure", "user_set": False}
        )
        for index in range(120):
            await transcript.append_message(
                Message(role="assistant", content=[TextContent(text=f"{index} " + "z" * 8_000)])
            )

    asyncio.run(build())
    # Comfortably past a tail-only window, and past both windows combined.
    assert session.joinpath("transcript.jsonl").stat().st_size > TITLE_SCAN_BYTES * 2
    assert stored_session_title(session) == "Nix Flake Build Failure"


def test_a_late_rename_still_outranks_the_title_journalled_at_the_head(tmp_path: Path):
    """Reading both ends must not cost the newest-wins rule: the head holds
    the ORIGINAL name, and a rename made hours later is the one in force."""
    session = tmp_path / "sessions" / "renamedlate"

    async def build() -> None:
        transcript = Transcript(session)
        await transcript.append_message(
            Message(role="user", content=[TextContent(text="opening line")])
        )
        await transcript.append_custom(
            CONVERSATION_NAME_CUSTOM_TYPE, {"text": "Generated At The Start", "user_set": False}
        )
        for index in range(120):
            await transcript.append_message(
                Message(role="assistant", content=[TextContent(text=f"{index} " + "z" * 8_000)])
            )
        await transcript.append_custom(
            CONVERSATION_NAME_CUSTOM_TYPE, {"text": "Renamed Much Later", "user_set": True}
        )

    asyncio.run(build())
    assert stored_session_title(session) == "Renamed Much Later"
