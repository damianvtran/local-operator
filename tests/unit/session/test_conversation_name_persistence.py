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
import gc
import json
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import StreamEndEvent
from local_operator.session import session as session_mod
from local_operator.session.naming import CONVERSATION_NAME_CUSTOM_TYPE
from local_operator.session.session import Session
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
    assert any("conversation name" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_a_slow_but_real_write_keeps_its_title(tmp_path, monkeypatch) -> None:
    """The budget must not drop a write that was going to finish.

    The wait and the retry used to charge a full budget EACH and run in
    sequence, so teardown could take twice the advertised bound — and because
    the bound is what decides whether a slow-but-real write is lost rather than
    merely un-awaited, two tight budgets also threw away titles: an append just
    under the sum was cut off by the first half. One shared deadline keeps any
    append inside the budget and still bounds teardown.
    """
    # NOT monkeypatched. The defect was a hard-coded per-half budget, so a test
    # that patches the shared constant proves nothing about it: the patched name
    # simply is not read by the broken code, and the test passes either way
    # (review round 1, F4). The delay is therefore expressed against the REAL
    # `_NAME_FLUSH_TIMEOUT_S` and placed where the two shapes disagree — past
    # what a single half used to allow, comfortably inside the shared budget.
    delay = session_mod._NAME_FLUSH_TIMEOUT_S * 0.6
    assert delay > 2.0, "the delay must exceed the old per-half budget to discriminate"

    class SlowTranscript(Transcript):
        async def append_custom(self, custom_type: str, details: dict[str, Any]):
            await asyncio.sleep(delay)
            return await super().append_custom(custom_type, details)

    session = Session(
        model=MODEL,
        stream_fn=ScriptedStream([[StreamEndEvent(stop_reason="stop")]]),
        tools=[],
        transcript=SlowTranscript(tmp_path / "sess"),
        system_blocks_provider=lambda: [],
    )
    await session.async_init()
    session.set_conversation_name("Slow but real", user_set=True)
    await session.dispose()

    assert _session(tmp_path).conversation_name == "Slow but real"


@pytest.mark.asyncio
async def test_a_fresh_session_is_nameless_and_journals_nothing(tmp_path) -> None:
    """No title, no entry: the ordinary new conversation is unaffected."""
    session = _session(tmp_path)
    await session.async_init()
    assert session.conversation_name == ""
    await session.dispose()
    assert _name_entries(tmp_path) == []


@pytest.mark.asyncio
async def test_the_title_sidecar_is_written_on_the_same_event_as_the_transcript(
    tmp_path,
) -> None:
    """The picker's O(1) title fast path: journalling a title also writes
    title.json beside the transcript, so a title buried in a large transcript
    is still found without a full read on the picker's synchronous path.
    """
    from local_operator.resume import stored_session_title

    session = _session(tmp_path)
    await session.async_init()
    session.set_conversation_name("Auto Generated Name", user_set=False)
    session.set_conversation_name("What The User Renamed It To", user_set=True)
    await session.dispose()

    session_dir = tmp_path / "sess"
    assert (session_dir / "title.json").is_file()
    # The in-force title is the newest, read straight from the sidecar.
    assert stored_session_title(session_dir) == "What The User Renamed It To"


@pytest.mark.asyncio
async def test_the_sidecar_retains_every_name_the_session_has_borne(tmp_path) -> None:
    """A search must reach a session by a name it was renamed AWAY from, so the
    sidecar accumulates every distinct title across renames, first-seen order
    preserved.

    Each rename is disposed before the next so it is a distinct persist event:
    two renames within one persist window coalesce into a single journalled row
    (only the final name is written), which is the writer's existing contract —
    accumulation is over names actually journalled, not every transient value.
    """
    from local_operator.resume import read_title_names

    session = _session(tmp_path)
    await session.async_init()
    session.set_conversation_name("First Subject", user_set=False)
    await session.dispose()

    resumed = _session(tmp_path)
    await resumed.async_init()
    resumed.set_conversation_name("Second Subject", user_set=True)
    await resumed.dispose()

    names = read_title_names(tmp_path / "sess")
    assert names == ["First Subject", "Second Subject"]


def test_written_names_are_whitespace_normalized_like_the_title(tmp_path) -> None:
    """``names`` must be normalised on write the same way ``text`` is, so the
    digest folds a name in exactly as the reader tokenises it and two names that
    differ only in internal whitespace collapse to one. Otherwise a hand-edited
    or foreign sidecar could fold ``Old   Name`` into the digest unnormalised."""
    from local_operator.resume import read_title_names, write_session_title

    session = tmp_path / "sess"
    session.mkdir()
    write_session_title(
        session,
        "Final  Title",
        user_set=False,
        past_names=["Old   Name", "Old Name", "  Second\tName  "],
    )

    # Internal runs collapse; the duplicate that only differed by whitespace is
    # deduped after normalisation; first-seen order and the in-force title last.
    assert read_title_names(session) == ["Old Name", "Second Name", "Final Title"]


@pytest.mark.asyncio
async def test_a_resumed_session_findable_by_either_name_after_reindex(tmp_path) -> None:
    """End to end: create, auto-name, dispose, resume, rename, dispose; then a
    fresh recent_session_rows + build_index makes it findable by BOTH names."""
    from local_operator.resume import recent_session_rows
    from local_operator.session.search_index import build_index, search_digests

    # Build under config_dir/sessions/<id>, where recent_session_rows scans.
    session_id = "abcd1234beef"
    session_dir = tmp_path / "sessions" / session_id

    def _at(directory: Path) -> Session:
        return Session(
            model=MODEL,
            stream_fn=ScriptedStream([[StreamEndEvent(stop_reason="stop")]]),
            tools=[],
            transcript=Transcript(directory),
            system_blocks_provider=lambda: [],
        )

    session = _at(session_dir)
    await session.async_init()
    session.set_conversation_name("Load Test Review", user_set=False)
    await session.dispose()

    resumed = _at(session_dir)
    await resumed.async_init()
    resumed.set_conversation_name("Improve ADM Classifier Throughput", user_set=True)
    await resumed.dispose()

    # The row shows the CURRENT title.
    rows = recent_session_rows(tmp_path, limit=10)
    assert rows and rows[0].name == "Improve ADM Classifier Throughput"

    # The index folds both names in, so either finds it.
    digests = build_index(tmp_path, [rows[0].id])
    assert search_digests(digests, "classifier") == {rows[0].id}
    assert search_digests(digests, "load test review") == {rows[0].id}


@pytest.mark.asyncio
async def test_a_landing_title_renames_an_open_browser_tab_group(tmp_path, monkeypatch) -> None:
    """The title has to reach browser chrome that was created before it existed.

    A conversation names itself a second or two INTO its first turn, which is
    normally after an opening "look at this page" already created the tab group
    — so the group latched the label the session had at open time (the bare
    fallback) and, for an open→screenshot→close session, no later command ever
    came along to reconcile it. Three concurrent sessions read ``LO · Session``,
    ``LO · Session (2)`` and ``LO · Session (3)`` for exactly that reason.
    """
    pushed: list[tuple[str, str]] = []

    async def fake_retitle(state: Any, context: Any) -> None:
        pushed.append((state.surface_id, context.session_name))

    monkeypatch.setattr("local_operator.tools.builtin.retitle_browser_surface", fake_retitle)

    session = _session(tmp_path)
    await session.async_init()
    session._browser.surface_id = "bridge:5:n0nce"

    session.set_conversation_name("Fix the tab groups", user_set=False)
    # Fire-and-forget through the session's task group: the rename must not sit
    # in front of the paint path that stored the title.
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert pushed == [("bridge:5:n0nce", "Fix the tab groups")]

    # A no-op store (same name again) journals nothing and pushes nothing —
    # otherwise every turn's re-title check would re-push an unchanged label.
    pushed.clear()
    session.set_conversation_name("Fix the tab groups", user_set=False)
    await asyncio.sleep(0)
    assert pushed == []
    await session.dispose()


@pytest.mark.asyncio
async def test_a_session_with_no_open_tab_pushes_no_rename(tmp_path, monkeypatch) -> None:
    """The overwhelmingly common case must cost nothing — not even the import."""
    called = False

    async def fake_retitle(state: Any, context: Any) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr("local_operator.tools.builtin.retitle_browser_surface", fake_retitle)
    session = _session(tmp_path)
    await session.async_init()
    session.set_conversation_name("No browsing here", user_set=False)
    await asyncio.sleep(0)
    assert called is False
    await session.dispose()


@pytest.mark.asyncio
async def test_a_failing_rename_never_costs_the_title(tmp_path, monkeypatch) -> None:
    """Tab chrome is presentation; a push that blows up must not lose the name."""

    async def boom(state: Any, context: Any) -> None:
        raise RuntimeError("extension went away")

    monkeypatch.setattr("local_operator.tools.builtin.retitle_browser_surface", boom)
    session = _session(tmp_path)
    await session.async_init()
    session._browser.surface_id = "bridge:5:n0nce"
    stored = session.set_conversation_name("Survives the failure", user_set=True)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert stored == "Survives the failure"
    assert session.conversation_name == "Survives the failure"
    await session.dispose()


@pytest.mark.asyncio
async def test_the_tool_context_reads_the_live_title_not_a_stale_snapshot(tmp_path) -> None:
    """``_build_tool_context`` is a per-TURN snapshot, so the browser label has
    to reach through it to the live holder — that is the latch race itself."""
    session = _session(tmp_path)
    await session.async_init()
    context = session._build_tool_context()
    assert context.session_name == ""
    assert context.session_name_provider is not None
    assert context.session_name_provider() == ""

    # Named after the context was built, exactly as the naming errand does.
    session.set_conversation_name("Named mid-turn", user_set=False)
    assert context.session_name == "", "the snapshot itself is expected to be stale"
    assert context.session_name_provider() == "Named mid-turn"
    await session.dispose()


def test_naming_a_browsing_session_outside_a_loop_does_not_raise(tmp_path) -> None:
    """A rename must never take down its caller for want of an event loop.

    Deliberately NOT ``async``: the whole point is the loop-less case.
    ``set_conversation_name`` is called from the TUI's SYNCHRONOUS paint path
    (``_store_title``, ``_cmd_rename``), and a session can be constructed and
    named outside a running loop — which is exactly the case
    ``_spawn_conversation_name_write`` has always guarded. The browser rename
    reaches ``_spawn_background`` -> ``asyncio.ensure_future``, which raises
    ``RuntimeError`` there, and the swallow inside the pushed coroutine cannot
    help because the raise happens at SCHEDULING time, before it ever runs.

    Regression guard: on the first draft of this feature the call below raised
    ``RuntimeError: There is no current event loop`` where ``main`` returned the
    title, for any session that had a browser tab open.
    """
    session = _session(tmp_path)
    # A session that IS browsing: the guard only matters once a surface exists,
    # because the no-tab path returns before it ever tries to schedule.
    session._browser.surface_id = "bridge:5:n0nce"

    stored = session.set_conversation_name("Named with no loop", user_set=True)

    assert stored == "Named with no loop"
    assert session.conversation_name == "Named with no loop"


def test_a_loopless_spawn_leaves_no_unawaited_coroutine(tmp_path, recwarn) -> None:
    """The declined rename must not be reported by asyncio at GC time.

    ``_spawn_background`` builds its ``_guarded`` wrapper BEFORE calling
    ``ensure_future``, so a raise there stranded two coroutines — the wrapper
    and the caller's — and Python blames the session for work it deliberately
    declined to schedule. Both are closed on that path now; this pins it,
    because an un-awaited-coroutine warning surfaces at an unrelated GC point
    and is miserable to trace back.
    """
    session = _session(tmp_path)
    session._browser.surface_id = "bridge:5:n0nce"

    session.set_conversation_name("Named with no loop", user_set=True)
    gc.collect()

    unawaited = [
        w
        for w in recwarn.list
        if issubclass(w.category, RuntimeWarning) and "never awaited" in str(w.message)
        # The title's own journal write has its own long-standing guard with the
        # same shape; this test is about the browser rename's coroutines.
        and "_persist_conversation_name" not in str(w.message)
    ]
    assert unawaited == [], [str(w.message) for w in unawaited]
