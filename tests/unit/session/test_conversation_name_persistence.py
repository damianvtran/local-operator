"""A conversation's title has to survive the session that earned it.

``ConversationName`` used to live only in memory, so every ``--resume`` opened
a conversation that had forgotten what it was called: the status band fell back
to the working directory and the terminal tab read ``lo › <dir>`` — which is
precisely the "five sessions, five identical rows" failure the terminal title
exists to fix, reintroduced on the one path where the name was already known.

The title is journalled as a ``conversation_name`` custom entry, replayed at
construction alongside the wake schedules. Three properties are load-bearing
and each has a test below:

* **``user_set`` rides along.** It is precedence, not decoration — a name the
  human typed has to keep outranking generated titles across a resume, or the
  resumed session's first re-title check silently overwrites it.
* **The write survives teardown.** It is a background task, and ``dispose``
  cancels background tasks, so a ``/rename`` moments before ctrl+d would never
  reach disk without an explicit flush.
* **Only real changes are journalled.** A generated title that loses to a
  user-set one stores nothing, and re-appending the standing name every turn
  would grow the transcript for no information.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import StreamEndEvent
from local_operator.session import session as session_mod
from local_operator.session.naming import CONVERSATION_NAME_CUSTOM_TYPE
from local_operator.session.session import _NAME_CHASE_ATTEMPTS, Session
from local_operator.session.transcript import Transcript
from tests.unit.session.test_session import MODEL, ScriptedStream


def _session(tmp_path: Path) -> Session:
    """A session over ``tmp_path/sess`` — reopening one resumes it."""
    return Session(
        model=MODEL,
        stream_fn=ScriptedStream([[StreamEndEvent(stop_reason="stop")]]),
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: [],
    )


def _name_entries(tmp_path: Path) -> list[dict[str, Any]]:
    """Every journalled title, oldest first."""
    path = tmp_path / "sess" / "transcript.jsonl"
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        payload = entry.get("payload", {})
        if entry.get("type") == "custom" and payload.get("custom_type") == (
            CONVERSATION_NAME_CUSTOM_TYPE
        ):
            entries.append(payload["details"])
    return entries


@pytest.mark.asyncio
async def test_a_generated_title_is_restored_by_the_next_resume(tmp_path) -> None:
    """THE bug: a resumed session opened nameless and wore its cwd instead."""
    session = _session(tmp_path)
    await session.async_init()
    session.set_conversation_name("Reduce agent RAM usage", user_set=False)
    await session.dispose()

    resumed = _session(tmp_path)
    assert resumed.conversation_name == "Reduce agent RAM usage"
    # Restored as a GENERATED title, so a later explicit rename still outranks it.
    assert resumed.conversation_name_state.user_set is False
    await resumed.dispose()


@pytest.mark.asyncio
async def test_a_human_rename_still_outranks_a_generated_title_after_a_resume(
    tmp_path,
) -> None:
    """``user_set`` is precedence and has to cross the process boundary.

    Restored as a plain string, the resumed session would have believed its
    name was a generated one and let the first re-title check replace a title
    the user chose by hand — the one thing the flag exists to prevent.
    """
    session = _session(tmp_path)
    await session.async_init()
    session.set_conversation_name("Q3 billing migration", user_set=True)
    await session.dispose()

    resumed = _session(tmp_path)
    assert resumed.conversation_name_state.user_set is True
    # A generated title landing in the resumed session must lose, exactly as it
    # would have lost in the session where the rename happened.
    assert resumed.set_conversation_name("Some model title", user_set=False) == (
        "Q3 billing migration"
    )
    await resumed.dispose()


@pytest.mark.asyncio
async def test_a_title_stored_moments_before_teardown_still_lands(tmp_path) -> None:
    """A `/rename` right before ctrl+d must not be cancelled on the way to disk.

    The write is a background task and ``dispose`` cancels those, so without the
    flush this stored nothing at all: the task had not had one turn of the event
    loop between the store and the teardown.
    """
    session = _session(tmp_path)
    await session.async_init()
    session.set_conversation_name("Named as the user quit", user_set=True)
    await session.dispose()  # no await in between: the write never got a tick

    assert _name_entries(tmp_path), "the title never reached the transcript"
    assert _session(tmp_path).conversation_name == "Named as the user quit"


@pytest.mark.asyncio
async def test_a_store_that_changes_nothing_journals_nothing(tmp_path) -> None:
    """The transcript must not grow by a row per turn for a stable title.

    Both no-op shapes are covered: re-storing the identical name, and a
    generated title losing to a user-set one.
    """
    session = _session(tmp_path)
    await session.async_init()
    session.set_conversation_name("Fix the importer", user_set=True)
    session.set_conversation_name("Fix the importer", user_set=True)  # identical
    session.set_conversation_name("A model title", user_set=False)  # loses
    await session.dispose()

    assert _name_entries(tmp_path) == [{"text": "Fix the importer", "user_set": True}]


@pytest.mark.asyncio
async def test_the_newest_journalled_title_is_the_one_restored(tmp_path) -> None:
    """Replay reads the newest entry, so a rename supersedes what came before."""
    session = _session(tmp_path)
    await session.async_init()
    session.set_conversation_name("First subject", user_set=False)
    session.set_conversation_name("Second subject", user_set=True)
    await session.dispose()

    resumed = _session(tmp_path)
    assert resumed.conversation_name == "Second subject"
    assert resumed.conversation_name_state.user_set is True
    await resumed.dispose()


@pytest.mark.asyncio
async def test_a_restored_title_has_already_spent_its_naming_attempt(tmp_path) -> None:
    """A resumed conversation must not buy a title it is already wearing.

    ``claim_request`` is the once-per-conversation latch; a restored session
    that left it unspent would pay for a provider call to re-derive the name
    sitting on its own band.
    """
    session = _session(tmp_path)
    await session.async_init()
    session.set_conversation_name("Already named", user_set=False)
    await session.dispose()

    resumed = _session(tmp_path)
    assert resumed.conversation_name_state.claim_request() is False
    await resumed.dispose()


@pytest.mark.asyncio
async def test_an_unreadable_title_entry_never_refuses_the_resume(tmp_path) -> None:
    """Decoration must not be able to take a conversation down with it.

    Same tolerance ``_load_wake_schedules`` has: a malformed entry yields a
    nameless session, not a resume that fails.
    """
    session = _session(tmp_path)
    await session.async_init()
    session.set_conversation_name("A good title", user_set=False)
    await session.dispose()

    # Rows are written with compact separators (`{"text":"…"}`), so the
    # substitution is spelled the way the file actually reads.
    path = tmp_path / "sess" / "transcript.jsonl"
    corrupted = path.read_text(encoding="utf-8").replace('"text":"A good title"', '"text":42')
    assert '"text":42' in corrupted, "the entry was not corrupted — the test proves nothing"
    path.write_text(corrupted, encoding="utf-8")

    resumed = _session(tmp_path)
    assert resumed.conversation_name == ""
    await resumed.dispose()


@pytest.mark.asyncio
async def test_a_rename_landing_mid_write_is_not_lost(tmp_path) -> None:
    """A title stored WHILE the previous one is being appended must still win.

    The append holds a lock and touches the filesystem, so the window between
    "payload read" and "row on disk" is real. Marking the write clean
    afterwards regardless of what the holder now says reported the NEWER title
    as saved and left the OLDER one on disk — the next resume then restored a
    name the user had already replaced. Caught with the slowed append below,
    which widens the window to something a test can hit deterministically.
    """

    class SlowTranscript(Transcript):
        async def append_custom(self, custom_type: str, details: dict[str, Any]):
            await asyncio.sleep(0.15)  # the payload was read BEFORE this
            return await super().append_custom(custom_type, details)

    session = Session(
        model=MODEL,
        stream_fn=ScriptedStream([[StreamEndEvent(stop_reason="stop")]]),
        tools=[],
        transcript=SlowTranscript(tmp_path / "sess"),
        system_blocks_provider=lambda: [],
    )
    await session.async_init()
    session.set_conversation_name("First title", user_set=False)
    await asyncio.sleep(0.05)  # the write is now inside append_custom
    session.set_conversation_name("Second title", user_set=True)
    await asyncio.sleep(0.5)  # the chase lands without any dispose

    # On disk BEFORE teardown: the correction must not depend on quitting.
    assert _session(tmp_path).conversation_name == "Second title"
    await session.dispose()
    assert _session(tmp_path).conversation_name == "Second title"


@pytest.mark.asyncio
async def test_a_failing_journal_write_is_logged_not_raised(tmp_path, caplog) -> None:
    """A full or read-only volume must not print a traceback for decoration.

    The write is deliberately NOT routed through ``_spawn_background`` (dispose
    has to await it, not cancel it), so it needs that helper's guard of its own.
    Without it the exception was never retrieved and asyncio wrote
    ``Task exception was never retrieved`` into the user's terminal — for a
    title (review round 1, F2).
    """

    class FailingTranscript(Transcript):
        async def append_custom(self, custom_type: str, details: dict[str, Any]):
            raise OSError("disk full")

    session = Session(
        model=MODEL,
        stream_fn=ScriptedStream([[StreamEndEvent(stop_reason="stop")]]),
        tools=[],
        transcript=FailingTranscript(tmp_path / "sess"),
        system_blocks_provider=lambda: [],
    )
    await session.async_init()
    with caplog.at_level("WARNING"):
        session.set_conversation_name("A title nobody can store", user_set=True)
        # The BACKGROUND task fails here, before any dispose runs. Asserting
        # after dispose instead is what made the first version of this test
        # vacuous: the flush's own error path logs a near-identical line, so
        # the assertion passed against the unguarded code while asyncio still
        # printed the traceback (review round 2, F11).
        await asyncio.sleep(0.05)
        background_logs = [r for r in caplog.records if "conversation name" in r.message]
        assert background_logs, "the background write's failure was not logged"
        # And the task really did finish, so nothing is left holding an
        # unretrieved exception for the GC to complain about.
        task = session._conversation_name_task
        assert task is not None and task.done()
        assert task.exception() is None, "the failure escaped the guard"

        # Teardown still succeeds — a decoration failure may not take it down.
        await session.dispose()

    assert session.conversation_name == "A title nobody can store"


@pytest.mark.asyncio
async def test_a_title_that_keeps_moving_cannot_spin_the_journal(tmp_path) -> None:
    """The chase is CAPPED, not merely "bounded by user behaviour".

    A writer that renames during every append never converges, and as recursion
    that was 2 987 frames, 2 986 rows and 522 KB before ``RecursionError``. The
    loop stops after ``_NAME_CHASE_ATTEMPTS`` and leaves the rest to the dispose
    flush (review round 1, F3).
    """
    renames = iter(range(1, 10_000))

    class MovingTranscript(Transcript):
        async def append_custom(self, custom_type: str, details: dict[str, Any]):
            result = await super().append_custom(custom_type, details)
            # Move the holder under the write, every single time.
            session._conversation_name.text = f"Moving title {next(renames)}"
            return result

    session = Session(
        model=MODEL,
        stream_fn=ScriptedStream([[StreamEndEvent(stop_reason="stop")]]),
        tools=[],
        transcript=MovingTranscript(tmp_path / "sess"),
        system_blocks_provider=lambda: [],
    )
    await session.async_init()
    session.set_conversation_name("First", user_set=True)
    await asyncio.sleep(0.1)
    await session.dispose()

    # Capped rather than unbounded: a handful of rows, not thousands.
    rows = _name_entries(tmp_path)
    assert len(rows) <= _NAME_CHASE_ATTEMPTS * 2 + 2, f"the chase spun: {len(rows)} rows"


@pytest.mark.asyncio
async def test_a_slow_write_is_not_duplicated_by_the_dispose_flush(tmp_path, monkeypatch) -> None:
    """Dispose must not re-append a row a slow write already put on disk.

    The flush waits on the in-flight write and then retried whenever the dirty
    flag was still set — which a slow append holds for its whole duration, so
    the identical payload was written twice (review round 1, F5).

    The append must outlast ``_NAME_FLUSH_TIMEOUT_S``, or the flush's wait
    simply completes and the ordinary "not dirty" return does the work — the
    branch under test is never entered and the test passes against the broken
    code. The first version of this test slept 0.3 s against a 2 s timeout and
    did exactly that (review round 2, F10), so the delay is derived from the
    timeout rather than hard-coded, and the timeout itself is monkeypatched
    down so the test does not pay for it in wall clock.
    """
    monkeypatch.setattr(session_mod, "_NAME_FLUSH_TIMEOUT_S", 0.2)

    class SlowTranscript(Transcript):
        async def append_custom(self, custom_type: str, details: dict[str, Any]):
            # Comfortably PAST the flush's patience, so the shield-wait times
            # out with the write still in flight — the state F9 was about.
            await asyncio.sleep(0.5)
            return await super().append_custom(custom_type, details)

    session = Session(
        model=MODEL,
        stream_fn=ScriptedStream([[StreamEndEvent(stop_reason="stop")]]),
        tools=[],
        transcript=SlowTranscript(tmp_path / "sess"),
        system_blocks_provider=lambda: [],
    )
    await session.async_init()
    session.set_conversation_name("Slow write", user_set=True)
    await session.dispose()
    # The shielded write outlives dispose by design; give it a moment to land.
    await asyncio.sleep(0.5)

    assert _name_entries(tmp_path) == [{"text": "Slow write", "user_set": True}]


@pytest.mark.asyncio
async def test_a_wedged_write_cannot_hang_dispose(tmp_path, monkeypatch) -> None:
    """ctrl+d must return even when the volume never answers.

    The flush's timeout used to bound only the WAIT for an in-flight write; the
    retry it fell through to was an unbounded await on the same wedged append,
    and ``dispose`` awaits the flush directly — so a hung filesystem hung the
    app on exit, for decoration (review round 2, F9).
    """
    monkeypatch.setattr(session_mod, "_NAME_FLUSH_TIMEOUT_S", 0.2)

    class WedgedTranscript(Transcript):
        async def append_custom(self, custom_type: str, details: dict[str, Any]):
            await asyncio.sleep(3600)  # never answers
            raise AssertionError("unreachable: the sleep outlives the test")

    session = Session(
        model=MODEL,
        stream_fn=ScriptedStream([[StreamEndEvent(stop_reason="stop")]]),
        tools=[],
        transcript=WedgedTranscript(tmp_path / "sess"),
        system_blocks_provider=lambda: [],
    )
    await session.async_init()
    session.set_conversation_name("Never lands", user_set=True)
    # The bound applies to the wait AND the retry, so dispose returns in about
    # two budgets rather than never. Generous ceiling: the assertion is
    # "bounded", not a stopwatch on the scheduler.
    await asyncio.wait_for(session.dispose(), timeout=5.0)


@pytest.mark.asyncio
async def test_a_transient_write_failure_still_lands_the_title(tmp_path) -> None:
    """A hiccup must cost a retry, not the name.

    ``_conversation_name_journalled`` is claimed BEFORE the append so the flush
    can recognise a write in flight (F9). Left standing when that append
    RAISES, the claim is false — the title is neither in flight nor on disk —
    and the flush skipped the very retry that exists to rescue it, losing the
    name outright.

    The always-failing test above cannot catch this: with every append doomed,
    the retry fails too and "no row on disk" holds either way. It takes a
    TRANSIENT failure — first append fails, the next would succeed — which is
    an ENOSPC/EIO/lock blip rather than a broken disk (review round 3, F12).
    """
    calls = {"n": 0}

    class FlakyTranscript(Transcript):
        async def append_custom(self, custom_type: str, details: dict[str, Any]):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("transient hiccup")
            return await super().append_custom(custom_type, details)

    session = Session(
        model=MODEL,
        stream_fn=ScriptedStream([[StreamEndEvent(stop_reason="stop")]]),
        tools=[],
        transcript=FlakyTranscript(tmp_path / "sess"),
        system_blocks_provider=lambda: [],
    )
    await session.async_init()
    session.set_conversation_name("Survives a hiccup", user_set=True)
    await asyncio.sleep(0.05)
    await session.dispose()

    assert calls["n"] >= 2, "the failed write was never retried"
    assert _session(tmp_path).conversation_name == "Survives a hiccup"


@pytest.mark.asyncio
async def test_teardown_costs_one_budget_not_two(tmp_path, monkeypatch, caplog) -> None:
    """The flush's whole cost is ``_NAME_FLUSH_TIMEOUT_S``, once.

    The wait and the retry used to charge a full budget each and ran in
    sequence, so a wedged volume plus a title that moved under the write made
    teardown twice what the constant advertises (measured at 10 s). They now
    share one deadline (review round 3, F14).

    The warning is asserted too, so that bounding the wait stays visible in a
    log rather than becoming a silent drop.
    """
    monkeypatch.setattr(session_mod, "_NAME_FLUSH_TIMEOUT_S", 0.4)

    class WedgedTranscript(Transcript):
        async def append_custom(self, custom_type: str, details: dict[str, Any]):
            await asyncio.sleep(3600)
            raise AssertionError("unreachable: the sleep outlives the test")

    session = Session(
        model=MODEL,
        stream_fn=ScriptedStream([[StreamEndEvent(stop_reason="stop")]]),
        tools=[],
        transcript=WedgedTranscript(tmp_path / "sess"),
        system_blocks_provider=lambda: [],
    )
    await session.async_init()
    session.set_conversation_name("Title A", user_set=False)
    await asyncio.sleep(0.05)
    # Moves the title under the in-flight write, which is what sends the flush
    # through BOTH halves — the case that used to cost two budgets.
    session.set_conversation_name("Title B", user_set=True)

    with caplog.at_level("WARNING"):
        started = time.monotonic()
        await session.dispose()
        elapsed = time.monotonic() - started

    # One budget plus scheduling slack, nowhere near two.
    assert elapsed < 0.4 * 1.8, f"teardown took {elapsed:.2f}s, more than one budget"
    assert any("stopped waiting for the conversation name" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_the_timeout_warning_claims_only_what_it_knows(tmp_path, monkeypatch, caplog) -> None:
    """The log line must not report a loss that has not happened.

    A timed-out retry says nothing about the title's fate: the shielded write it
    gave up waiting for survives dispose, and its own chase loop re-appends a
    title that moved — so the row frequently lands moments later. "Gave up
    journalling the conversation name" reported those as dropped names, and a
    warning that cries wolf is worth less than none (review round 3, F13).

    Whether a slow write will finish or is wedged forever is not knowable at
    this point, so the message states the fact it has — teardown stopped
    waiting — and names the consequence conditionally. This pins that wording
    against a case where the title DOES land.
    """
    monkeypatch.setattr(session_mod, "_NAME_FLUSH_TIMEOUT_S", 0.4)

    class SlowTranscript(Transcript):
        async def append_custom(self, custom_type: str, details: dict[str, Any]):
            await asyncio.sleep(1.0)  # outlives the flush, but does finish
            return await super().append_custom(custom_type, details)

    session = Session(
        model=MODEL,
        stream_fn=ScriptedStream([[StreamEndEvent(stop_reason="stop")]]),
        tools=[],
        transcript=SlowTranscript(tmp_path / "sess"),
        system_blocks_provider=lambda: [],
    )
    await session.async_init()
    session.set_conversation_name("Title A", user_set=False)
    await asyncio.sleep(0.05)
    session.set_conversation_name("Title B", user_set=True)  # moves under the write

    with caplog.at_level("WARNING"):
        await session.dispose()
        await asyncio.sleep(2.5)  # let the shielded write and its chase land

    # The title reached disk despite the flush giving up on it...
    assert _session(tmp_path).conversation_name == "Title B"
    # ...so nothing may claim it was lost or dropped.
    for record in caplog.records:
        assert "gave up journalling" not in record.message
        assert "stopped waiting" not in record.message or "if the write does not finish" in (
            record.message
        )


@pytest.mark.asyncio
async def test_a_fresh_session_is_nameless_and_journals_nothing(tmp_path) -> None:
    """No title, no entry: the ordinary new conversation is unaffected."""
    session = _session(tmp_path)
    await session.async_init()
    assert session.conversation_name == ""
    await session.dispose()
    assert _name_entries(tmp_path) == []
