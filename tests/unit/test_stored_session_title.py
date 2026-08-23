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
    TITLE_SIDECAR_NAME,
    _read_title_sidecar,
    backfill_session_titles,
    read_title_names,
    session_name,
    stored_session_title,
    write_session_title,
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


def test_a_transcript_between_one_and_two_windows_still_finds_the_newest_title(
    tmp_path: Path,
):
    """The band where the two scan windows would overlap.

    This size class had no coverage while the other tests all built files
    several windows wide, and the branch serving it once read the tail from a
    handle already at EOF -- so it searched the head only and returned the name
    the user had renamed AWAY from. A stale title is worse than a missing one,
    because the picker states it with the same confidence as a correct one.
    """
    session = tmp_path / "sessions" / "bandsized"

    async def build() -> None:
        transcript = Transcript(session)
        await transcript.append_message(
            Message(role="user", content=[TextContent(text="opening line")])
        )
        await transcript.append_custom(
            CONVERSATION_NAME_CUSTOM_TYPE, {"text": "Original Head Title", "user_set": False}
        )
        for index in range(24):
            await transcript.append_message(
                Message(role="assistant", content=[TextContent(text=f"{index} " + "z" * 8_000)])
            )
        await transcript.append_custom(
            CONVERSATION_NAME_CUSTOM_TYPE, {"text": "Renamed Much Later", "user_set": True}
        )

    asyncio.run(build())
    size = session.joinpath("transcript.jsonl").stat().st_size
    assert TITLE_SCAN_BYTES < size <= TITLE_SCAN_BYTES * 2, size
    assert stored_session_title(session) == "Renamed Much Later"


def test_a_band_sized_transcript_whose_title_is_only_at_the_head(tmp_path: Path):
    """The other half of the band: a session auto-named at turn 2 behind a
    very large opening turn, never renamed. The title is only near the head."""
    session = tmp_path / "sessions" / "bigfirstturn"

    async def build() -> None:
        transcript = Transcript(session)
        await transcript.append_message(
            Message(role="user", content=[TextContent(text="x" * 240_000)])
        )
        await transcript.append_custom(
            CONVERSATION_NAME_CUSTOM_TYPE, {"text": "Auto Named At Turn 2", "user_set": False}
        )

    asyncio.run(build())
    size = session.joinpath("transcript.jsonl").stat().st_size
    assert TITLE_SCAN_BYTES < size <= TITLE_SCAN_BYTES * 2, size
    assert stored_session_title(session) == "Auto Named At Turn 2"


# ---------------------------------------------------------------------------
# Title sidecar (title.json): the O(1) fast path that closes the window-scan
# gap TITLE_SCAN_BYTES documents. See resume.write_session_title.
# ---------------------------------------------------------------------------


def test_the_sidecar_round_trips_text_and_names(tmp_path: Path):
    """write_session_title then _read_title_sidecar returns what went in."""
    session = tmp_path / "sessions" / "roundtrip"
    session.mkdir(parents=True)
    write_session_title(session, "Second Name", user_set=True, past_names=["First Name"])
    sidecar = _read_title_sidecar(session)
    assert sidecar is not None
    assert sidecar.text == "Second Name"
    assert sidecar.user_set is True
    # The in-force title is accumulated into names, first-seen order preserved.
    assert sidecar.names == ("First Name", "Second Name")
    assert read_title_names(session) == ["First Name", "Second Name"]


def test_the_sidecar_dedupes_names_keeping_first_seen_order(tmp_path: Path):
    """A re-title back to a former name does not duplicate it in the list."""
    session = tmp_path / "sessions" / "dedup"
    session.mkdir(parents=True)
    write_session_title(session, "Alpha", user_set=False, past_names=["Alpha", "Beta"])
    assert read_title_names(session) == ["Alpha", "Beta"]


def test_the_sidecar_wins_over_a_title_in_an_unscanned_window(tmp_path: Path):
    """THE NAMED REGRESSION, distilled: a title in the middle of a large
    transcript — invisible to both scan windows — is found via the sidecar.

    This is the topic-pivot / ADM shape: > 2x TITLE_SCAN_BYTES with the only
    conversation_name entries buried strictly between the head and tail
    windows. The window scan returns "" for it; the sidecar returns the title.
    """
    session = tmp_path / "sessions" / "middletitle"

    async def build() -> None:
        transcript = Transcript(session)
        await transcript.append_message(
            Message(role="user", content=[TextContent(text="an unrelated opening topic")])
        )
        # Bulk before the title, pushing it past the head window.
        for index in range(20):
            await transcript.append_message(
                Message(role="assistant", content=[TextContent(text=f"pre {index} " + "z" * 8_000)])
            )
        await transcript.append_custom(
            CONVERSATION_NAME_CUSTOM_TYPE,
            {"text": "Buried In The Middle", "user_set": False},
        )
        # Bulk after the title, so it sits strictly between the two windows.
        for index in range(40):
            body = f"post {index} " + "w" * 8_000
            await transcript.append_message(
                Message(role="assistant", content=[TextContent(text=body)])
            )

    asyncio.run(build())
    size = session.joinpath("transcript.jsonl").stat().st_size
    assert size > TITLE_SCAN_BYTES * 2, size
    # Precondition of the regression: the window scan alone cannot find it.
    assert not (session / TITLE_SIDECAR_NAME).exists()
    assert stored_session_title(session) == ""
    # The fix: the backfill writes the sidecar, and now it is found.
    assert backfill_session_titles(tmp_path) == 1
    assert stored_session_title(session) == "Buried In The Middle"


def test_a_corrupt_sidecar_falls_back_to_the_scan_rather_than_raising(tmp_path: Path):
    """A truncated sidecar (mid multi-byte char) must never take the picker
    down, mirroring session_origin's errors='replace' tolerance."""
    session = _session(tmp_path, "opening", ("Scannable Title", False))
    # A byte sequence that is invalid UTF-8 and not JSON.
    (session / TITLE_SIDECAR_NAME).write_bytes(b'{"text": "\xff\xfe truncated')
    # _read returns None (unparseable), so stored_session_title uses the scan.
    assert _read_title_sidecar(session) is None
    assert stored_session_title(session) == "Scannable Title"


def test_a_sidecar_with_no_text_falls_back_to_the_scan(tmp_path: Path):
    """An empty in-force title in the sidecar must not shadow a scannable one:
    stored_session_title only takes the sidecar when it carries real text."""
    session = _session(tmp_path, "opening", ("Scannable Title", False))
    write_session_title(session, "", user_set=False, past_names=[])
    assert stored_session_title(session) == "Scannable Title"


def test_the_sidecar_write_preserves_directory_mtime(tmp_path: Path):
    """Journalling a title is bookkeeping ABOUT a session, never activity IN
    it, so it must not move the mtime retention and listings read."""
    session = tmp_path / "sessions" / "mtime"
    session.mkdir(parents=True)
    before = session.stat().st_mtime
    import os
    import time

    # Backdate so a preserved mtime is distinguishable from a fresh one.
    os.utime(session, (before - 10_000, before - 10_000))
    expected = session.stat().st_mtime
    time.sleep(0.01)
    write_session_title(session, "A Title", user_set=True, past_names=[])
    assert session.stat().st_mtime == expected


def test_backfill_never_restamps_an_existing_sidecar(tmp_path: Path):
    """A second sweep is a no-op: the sidecar is event-sourced from its first
    write on, so re-stamping risks clobbering a newer name with an old scan."""
    session = _session(tmp_path, "opening", ("Only Title", False))
    assert backfill_session_titles(tmp_path) == 1
    # Rename after backfill: the sidecar now leads the transcript.
    write_session_title(session, "Renamed After", user_set=True, past_names=["Only Title"])
    assert backfill_session_titles(tmp_path) == 0
    assert stored_session_title(session) == "Renamed After"


def test_backfill_limit_caps_work_done(tmp_path: Path):
    """limit bounds sidecars written per run, like backfill_session_origins."""
    for i in range(3):
        session = tmp_path / "sessions" / f"sess{i}"

        async def build(s=session) -> None:
            transcript = Transcript(s)
            await transcript.append_message(
                Message(role="user", content=[TextContent(text="opening")])
            )
            await transcript.append_custom(
                CONVERSATION_NAME_CUSTOM_TYPE, {"text": "A Name", "user_set": False}
            )

        asyncio.run(build())
    assert backfill_session_titles(tmp_path, limit=2) == 2
    # The remaining one is stamped on a later run — nothing is skipped forever.
    assert backfill_session_titles(tmp_path, limit=2) == 1
