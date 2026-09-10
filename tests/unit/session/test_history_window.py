"""Canonical display pages never substitute a truncated context for history."""

import asyncio
import base64
import json
from pathlib import Path

import pytest

from local_operator.harness.types import (
    CustomMessage,
    Message,
    TextContent,
    ToolCall,
    ToolResult,
)
from local_operator.session.history_window import (
    _AUDIT_TAIL_CURSOR,
    DisplayHistoryWindow,
    _audit_entry_cursor,
    _sign,
    display_window,
    wire_payload,
)
from local_operator.session.remote import RemoteSession
from local_operator.session.runtime.owned import OwnedSessionHandle
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.transcript import (
    ENTRY_COMPACTION,
    Transcript,
    context_cut_index,
    replay_entries,
)
from tests.e2e.harness import ScriptedStream, build_session, seed_transcript, text_turn
from tests.unit.session.test_remote import _never_take_over


def window(transcript: Transcript, **kwargs):  # noqa: ANN003, ANN201
    return display_window(
        transcript,
        conversation_id="window-test",
        owner_epoch="synthetic-epoch",
        through_id=transcript.entries()[-1].id if transcript.entries() else None,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_pages_match_canonical_cut_across_appends_and_reset_on_prune(tmp_path: Path) -> None:
    transcript = Transcript(tmp_path)
    messages = [Message.user(f"row {index}") for index in range(270)]
    await transcript.append_messages(messages)
    first = window(transcript)
    assert first.total_message_count == 270
    assert len(first.messages) == 120
    assert first.start == 150
    assert first.before_token and first.snapshot_token
    await transcript.append_message(Message.assistant("after captured cut"))
    rows = list(first.messages)
    token = first.before_token
    while token:
        page = window(transcript, before=token)
        assert page.through_id == first.through_id
        rows[:0] = page.messages
        token = page.before_token
    assert [m.model_dump() for m in rows] == [m.model_dump() for m in messages]
    await transcript.append_prune(messages[-1].id, "pruned")
    assert window(transcript, before=first.before_token).status == "reset"
    last = transcript.build_llm_history(through_id=first.through_id)[-1]
    assert isinstance(last, Message) and last.text == "row 269"


@pytest.mark.asyncio
async def test_compaction_roles_and_delayed_tool_results_share_canonical_replay(
    tmp_path: Path,
) -> None:
    transcript = Transcript(tmp_path)
    opener = Message.user("preserved verbatim")
    call = Message.assistant("running a tool")
    call.tool_calls = [ToolCall(id="tool-one", name="bash", arguments={"command": "echo hello"})]
    custom = CustomMessage(custom_type="aside", attribution="user", details={"text": "context"})
    result = Message.tool_result(
        ToolResult(tool_call_id="tool-one", tool_name="bash", content=[TextContent(text="hello")])
    )
    await transcript.append_messages([opener, call, custom, result])
    compact = await transcript.append_compaction(
        "summary", call.id, 500, preserved_user_turns=[{"id": opener.id, "text": opener.text}]
    )
    canonical = transcript.build_llm_history()
    assert canonical[0].id == compact.id
    assert canonical[1].id == opener.id
    first = window(transcript, max_messages=3)
    assert [m.id for m in first.messages] == [call.id, custom.id, result.id]
    assert first.before_token
    previous = window(transcript, before=first.before_token, max_messages=3)
    assert [m.model_dump() for m in previous.messages + first.messages] == [
        m.model_dump() for m in canonical
    ]
    anchored = window(transcript, before=first.snapshot_token, anchor="tool:tool-one")
    assert any(m.id == call.id for m in anchored.messages)


@pytest.mark.asyncio
async def test_wire_budget_never_returns_oversized_or_truncated_required_prose(
    tmp_path: Path,
) -> None:
    transcript = Transcript(tmp_path)
    message = Message.assistant("required prose " * 100_000)
    await transcript.append_message(message)
    page = window(transcript)
    assert page.status == "full_required"
    assert page.messages == []
    assert len(page.model_dump_json().encode()) < 1024 * 1024
    replayed = transcript.build_llm_history()[0]
    assert isinstance(replayed, Message) and replayed.text == message.text


@pytest.mark.asyncio
async def test_signed_page_scope_and_anchor_validation(tmp_path: Path) -> None:
    transcript = Transcript(tmp_path)
    await transcript.append_messages([Message.user(str(i)) for i in range(130)])
    first = window(transcript)
    assert first.before_token
    with pytest.raises(ValueError, match="invalid history token"):
        window(transcript, before=first.before_token + "bad")
    with pytest.raises(ValueError, match="another conversation"):
        display_window(
            transcript,
            conversation_id="wrong",
            owner_epoch="synthetic-epoch",
            through_id=first.through_id,
            before=first.before_token,
        )
    assert (
        display_window(
            transcript,
            conversation_id="window-test",
            owner_epoch="new-epoch",
            through_id=first.through_id,
            before=first.before_token,
        ).status
        == "reset"
    )
    assert window(transcript, before=first.snapshot_token, anchor="missing-id").status == "reset"


@pytest.mark.asyncio
async def test_an_older_runtime_page_still_validates_and_reads_by_epoch(
    tmp_path: Path,
) -> None:
    """The ``owner_epoch`` → ``runtime_epoch`` rename must not strand a runtime.

    The DTO is ``extra="forbid"`` and crosses the socket, so a viewer that
    stopped accepting the old key would fail ``model_validate`` on every page
    an older runtime mints — sidebar history unavailable, a degrade not an
    error. Reads accept both keys; the wire emits only the new one.
    """
    transcript = Transcript(tmp_path)
    await transcript.append_message(Message.user("row"))
    page = window(transcript)
    legacy = wire_payload(page, audit_capable=True)
    legacy["owner_epoch"] = legacy.pop("runtime_epoch")
    restored = DisplayHistoryWindow.model_validate(legacy)
    assert restored.runtime_epoch == "synthetic-epoch"
    # ``remote.py`` reconciles pages through this attribute and is owned by
    # the viewer-identifier rename in flight, so the old name must keep
    # reading one release past the field rename.
    assert restored.owner_epoch == "synthetic-epoch"
    fresh = wire_payload(page, audit_capable=True)
    assert "owner_epoch" not in fresh
    assert fresh["runtime_epoch"] == "synthetic-epoch"


@pytest.mark.asyncio
async def test_a_before_token_minted_with_the_old_claims_key_still_verifies(
    tmp_path: Path,
) -> None:
    """HMAC page tokens outlive the claims-key rename.

    A token whose claims carry ``owner_epoch`` must keep verifying as ``ok``:
    a mismatch here reads as ``status="reset"`` — a silent window reset plus
    a full-replay refetch, never an error.
    """
    transcript = Transcript(tmp_path)
    await transcript.append_messages([Message.user(str(i)) for i in range(130)])
    first = window(transcript)
    assert first.before_token
    header = first.before_token.split(".")[0]
    claims = json.loads(base64.urlsafe_b64decode(header + "=" * (-len(header) % 4)))
    assert claims["runtime_epoch"] == "synthetic-epoch"
    legacy_claims = {**claims, "owner_epoch": claims.pop("runtime_epoch")}
    legacy_token = _sign(legacy_claims, transcript._history_page_key)
    assert window(transcript, before=legacy_token).status == "ok"
    # The dual read must not swallow a genuine mismatch: a stale epoch still
    # resets, so this compat read cannot mask a cross-epoch replay.
    stale = _sign({**legacy_claims, "owner_epoch": "gone"}, transcript._history_page_key)
    assert window(transcript, before=stale).status == "reset"


@pytest.mark.asyncio
async def test_real_attach_pages_without_viewer_journal_parse_and_records_shell_once(
    tmp_path: Path, monkeypatch
) -> None:  # noqa: ANN001
    config = tmp_path / "config"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    directory = config / "sessions" / "window-test"
    messages = [Message.user(f"canonical {index}") for index in range(250)]
    await seed_transcript(directory, messages)
    session = build_session(directory, ScriptedStream([text_turn("owner reply")]), cwd=tmp_path)
    handle = OwnedSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    server = RuntimeServer(handle, kind="daemon")
    remote = None
    await server.start_in_process()
    try:

        async def forbidden(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            raise AssertionError("window attach must not parse the viewer journal")

        monkeypatch.setattr(RemoteSession, "_read_transcript", forbidden)
        remote = await RemoteSession.connect(
            server._record,
            "window-test",
            config_dir=config,
            takeover_factory=_never_take_over,
            display_window=True,
        )
        assert not remote.is_cold
        assert remote.history_message_count == 250
        assert len(remote.display_history_window()) == 120
        with pytest.raises(RuntimeError, match="not hydrated"):
            remote.history()
        await remote.seed_history([Message.user("must not seed a nonempty window")])
        assert remote.history_message_count == 250
        old_token = remote.history_before_token
        assert old_token is not None
        await remote.ensure_display_anchor(messages[10].id)
        assert any(m.id == messages[10].id for m in remote.display_history_window())
        assert [m.id for m in await remote.materialize_history()] == [m.id for m in messages]
        result = ToolResult(
            tool_call_id="synthetic-shell",
            tool_name="bash",
            content=[TextContent(text="exit code: 0\nhello")],
        )
        await remote.record_shell("echo hello", result)
        await remote.record_shell("echo hello", result)
        durable = session._transcript.build_llm_history()
        assert sum(m.id == "shell:synthetic-shell:user" for m in durable) == 1
        assert sum(m.id == "shell:synthetic-shell:result" for m in durable) == 1
        assert (
            sum(m.id == "shell:synthetic-shell:user" for m in remote.display_history_window()) == 1
        )
        assert json.loads(remote.frontend_state.model_dump_json())["session_id"] == "window-test"
        received = []
        remote.subscribe(received.append)
        await session._transcript.append_compaction("new canonical cut", messages[-1].id, 500)
        stale = await remote.history_page(old_token)
        assert stale.status == "reset"
        await remote._refresh_display_history()
        assert not remote.is_cold
        assert any(getattr(event, "reset", False) for event in received)
        marker = remote.display_history_window()[0]
        assert isinstance(marker, CustomMessage) and marker.custom_type == "compaction_summary"
        assert not await remote.ensure_display_anchor(messages[0].id)
    finally:
        if remote is not None:
            await remote.dispose()
        server.close()
        await handle.dispose()


@pytest.mark.asyncio
async def test_live_replay_mutations_refresh_without_replacing_the_connection(
    tmp_path, monkeypatch
):
    from local_operator.harness.types import CompactionEndEvent, NoticeEvent

    config = tmp_path / "config"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    directory = config / "sessions" / "window-live"
    messages = [Message.user(f"canonical {index}") for index in range(250)]
    await seed_transcript(directory, messages)
    session = build_session(directory, ScriptedStream([text_turn("unused")]), cwd=tmp_path)
    handle = OwnedSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    server = RuntimeServer(handle, kind="daemon")
    await server.start_in_process()
    assert server._record is not None
    remote = None
    try:
        remote = await RemoteSession.connect(
            server._record,
            "window-live",
            config_dir=config,
            takeover_factory=_never_take_over,
            display_window=True,
        )
        connection = remote._client
        received = asyncio.Event()
        remote.subscribe(
            lambda event: received.set() if isinstance(event, CompactionEndEvent) else None
        )
        old_token = remote.history_before_token
        await session._transcript.append_compaction("canonical summary", messages[-1].id, 1000)
        await session._emit(CompactionEndEvent(reason="manual", success=True))
        await asyncio.wait_for(received.wait(), 3)
        await remote.ensure_display_current()
        assert remote._client is connection
        assert remote.display_history_current
        assert remote.history_message_count == 2
        assert remote._display_history is not None
        assert remote._display_history.history_generation == session._transcript._history_generation
        assert any(
            isinstance(row, CustomMessage) and row.custom_type == "compaction_summary"
            for row in remote.display_history_window()
        )
        assert old_token is not None and (await remote.history_page(old_token)).status == "reset"
        # Non-compaction mutations publish the same generation fence before
        # ordinary events; callers need not page an obsolete token to find out.
        await session._transcript.append_prune(messages[-1].id, "pruned")
        await session._emit(NoticeEvent(text="pruned", kind="info"))
        for _ in range(100):
            if remote.frontend_state.history_generation == session._transcript._history_generation:
                break
            await asyncio.sleep(0.01)
        await remote.ensure_display_current()
        assert remote._client is connection
        assert [row.model_dump() for row in remote.display_history_window()] == [
            row.model_dump() for row in session._transcript.build_llm_history()
        ]
        assert remote._display_history.history_generation == session._transcript._history_generation
        assert len(server._clients) == 1
    finally:
        if remote is not None:
            await remote.dispose()
        server.close()
        await handle.dispose()


@pytest.mark.asyncio
async def test_prompt_and_wait_does_not_complete_on_admission(tmp_path, monkeypatch):
    config = tmp_path / "config"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    directory = config / "sessions" / "window-loop"
    await seed_transcript(directory, [Message.user("initial")])
    started, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def stream(request, signal=None):
        # The session also issues a tool-less naming errand off the first
        # prompt. Only a real agent turn carries the tool schema, and counting
        # the errand would make "how many turns ran" unreadable here.
        if not getattr(request, "tools", None):
            for event in text_turn("named"):
                yield event
            return
        calls.append(len(calls) + 1)
        started.set()
        await release.wait()
        for event in text_turn("completed"):
            yield event

    session = build_session(directory, stream, cwd=tmp_path)
    handle = OwnedSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    server = RuntimeServer(handle, kind="daemon")
    await server.start_in_process()
    assert server._record is not None
    remote = None
    task = None
    try:
        remote = await RemoteSession.connect(
            server._record,
            "window-loop",
            config_dir=config,
            takeover_factory=_never_take_over,
            display_window=True,
        )
        task = asyncio.create_task(
            remote.prompt_and_wait("one", message_id="11111111-1111-4111-8111-111111111111")
        )

        async def provider_started():
            while not started.is_set():
                if task.done():
                    await task
                    raise AssertionError("turn completed before the provider started")
                await asyncio.sleep(0.01)

        await asyncio.wait_for(provider_started(), 3)
        await asyncio.sleep(0.05)
        assert not task.done(), "durable admission is not terminal completion"
        assert calls == [1]
        # A refresh can pause UI event delivery. The scheduler still observes
        # the authenticated terminal outcome, not whether a card was painted.
        remote._ready_for_events = False
        release.set()
        await asyncio.wait_for(task, 3)
        remote._ready_for_events = True
        await asyncio.wait_for(
            remote.prompt_and_wait("two", message_id="22222222-2222-4222-8222-222222222222"), 3
        )
        assert calls == [1, 2]
        assert not remote._prompt_completion_waiters
    finally:
        release.set()
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if remote is not None:
            await remote.dispose()
        server.close()
        await handle.dispose()


# --- Audit paging ------------------------------------------------------------
#
# The chain must reach the journal's FIRST message row. Before this, it stopped
# at the compaction cut and told the reader that was the start of the
# conversation — measured at 97% of rows unreachable on a real 17,345-row
# session.


async def _compacted_journal(directory: Path, *, compactions: int, rows_each: int):
    transcript = Transcript(directory)
    written = []
    for cut in range(compactions):
        batch = [Message.user(f"cut {cut} row {index}") for index in range(rows_each)]
        await transcript.append_messages(batch)
        written.extend(batch)
        await transcript.append_compaction(
            f"summary {cut}",
            batch[-1].id,
            500,
            preserved_user_turns=[{"id": batch[0].id, "text": batch[0].text}],
        )
    tail = [Message.assistant(f"tail {index}") for index in range(rows_each)]
    await transcript.append_messages(tail)
    written.extend(tail)
    return transcript, written


def _walk(transcript: Transcript, **kwargs):
    """Follow the whole backward chain, returning (rows, pages, audit_pages)."""
    page = window(transcript, **kwargs)
    rows = list(page.messages)
    pages = [page]
    while page.before_token:
        page = window(transcript, before=page.before_token, **kwargs)
        assert page.status == "ok", page.status
        rows[0:0] = list(page.messages)
        pages.append(page)
    return rows, pages


@pytest.mark.asyncio
async def test_chain_terminates_at_the_first_message_row_not_the_compaction_cut(tmp_path):
    """The defect, stated as a test.

    ``before_token`` may become ``None`` only when the journal's first message
    row has been delivered. Terminating at the compaction cut is what made the
    TUI print "start of conversation" above thousands of real rows.
    """
    transcript, written = await _compacted_journal(tmp_path / "s", compactions=3, rows_each=40)
    rows, pages = _walk(transcript, max_messages=30)
    message_rows = [r for r in rows if getattr(r, "custom_type", None) != "compaction_summary"]
    assert message_rows[0].id == written[0].id
    assert pages[-1].before_token is None
    assert any(p.audit for p in pages), "the chain never entered the audit phase"


@pytest.mark.asyncio
async def test_every_row_appears_exactly_once_across_the_chain(tmp_path):
    """Audit completeness: the chain tiles the journal with no gap and no repeat.

    This is the assertion that proves the two phases meet exactly at the
    compaction cut — a page-seam off-by-one shows up here as a missing row or a
    duplicated one rather than as a subtly short transcript nobody notices.
    """
    transcript, written = await _compacted_journal(tmp_path / "s", compactions=3, rows_each=25)
    rows, pages = _walk(transcript, max_messages=20)
    delivered = [r.id for r in rows if getattr(r, "custom_type", None) != "compaction_summary"]
    # Completeness and exactly-once are the contract.
    assert set(delivered) == {m.id for m in written}
    assert len(delivered) == len(set(delivered)) == len(written)

    # Global order is NOT asserted, and the reason is a pre-existing property
    # of the context phase rather than a concession: the latest compaction's
    # ``preserved_user_turns`` are re-emitted at the HEAD of the model's replay
    # under their original ids, so that one row is displayed where compaction
    # put it rather than at its journal position. The audit phase suppresses
    # its own in-place copy so the row is delivered once instead of twice (the
    # duplicate-id hazard). Within the audit phase itself, order is journal
    # order, and that IS asserted.
    # ``_walk`` collects pages newest-first, so reverse to read the audit phase
    # in the order a reader scrolling upward actually assembles it.
    audit_rows = [
        r.id
        for page in reversed(pages)
        if page.audit
        for r in page.messages
        if getattr(r, "custom_type", None) != "compaction_summary"
    ]
    order = {m.id: index for index, m in enumerate(written)}
    positions = [order[row_id] for row_id in audit_rows]
    assert positions == sorted(positions)


@pytest.mark.asyncio
async def test_audit_paging_does_not_move_the_context_derived_counts(tmp_path):
    """``total_message_count``/``theme_turn_count`` stay CONTEXT-scoped.

    The TUI reads the first as a monotonic context-growth signal for its
    presentation cache and the second as the retitle growth gate. Inflating
    either by the audit depth would invalidate every cached presentation and
    re-fire retitling across every session on the machine.
    """
    transcript, _ = await _compacted_journal(tmp_path / "s", compactions=2, rows_each=20)
    first = window(transcript, max_messages=15)
    _rows, pages = _walk(transcript, max_messages=15)
    audit_pages = [p for p in pages if p.audit]
    assert audit_pages
    for page in audit_pages:
        assert page.total_message_count == 0
        assert page.theme_turn_count == 0
        assert page.opener_text == ""
    # And the context page's own counts are untouched by the audit phase.
    assert first.total_message_count == len(transcript.build_llm_history())


@pytest.mark.asyncio
async def test_an_audit_token_resets_after_a_compaction_bumps_the_generation(tmp_path):
    """An outstanding audit cursor must not survive the journal changing.

    The cursor names an entry id rather than an offset precisely because
    ``compact_file`` rewrites the file; a generation bump invalidates it into
    ``reset`` so the viewer re-syncs instead of paging a stale coordinate space.
    """
    transcript, written = await _compacted_journal(tmp_path / "s", compactions=2, rows_each=20)
    page = window(transcript, max_messages=15)
    while page.before_token and not page.audit:
        page = window(transcript, before=page.before_token, max_messages=15)
    assert page.audit and page.before_token
    await transcript.append_compaction("later cut", written[-1].id, 500)
    assert window(transcript, before=page.before_token, max_messages=15).status == "reset"


@pytest.mark.asyncio
async def test_a_tool_call_and_its_result_never_split_across_an_audit_page(tmp_path):
    """A result is drawn on its call's card; a page may not open on one."""
    directory = tmp_path / "s"
    transcript = Transcript(directory)
    rows = []
    for index in range(12):
        call = Message.assistant(
            f"calling {index}",
            tool_calls=[ToolCall(id=f"call-{index}", name="probe", arguments={})],
        )
        result = Message.tool_result(
            ToolResult(
                tool_call_id=f"call-{index}",
                tool_name="probe",
                content=[TextContent(text=f"done {index}")],
            )
        )
        rows.extend([Message.user(f"ask {index}"), call, result])
    await transcript.append_messages(rows)
    await transcript.append_compaction("cut", rows[-1].id, 500)
    await transcript.append_messages([Message.user("after")])

    _all_rows, pages = _walk(transcript, max_messages=4)
    for page in pages:
        if not page.messages:
            continue
        first = page.messages[0]
        assert getattr(first, "role", None) != "tool", f"page opens on a tool result: {first.id}"


@pytest.mark.asyncio
async def test_a_journal_without_compaction_still_terminates_where_it_always_did(tmp_path):
    """No compaction means no audit phase: unchanged behaviour, unchanged copy."""
    transcript = Transcript(tmp_path / "s")
    messages = [Message.user(f"row {index}") for index in range(50)]
    await transcript.append_messages(messages)
    rows, pages = _walk(transcript, max_messages=20)
    assert [r.id for r in rows] == [m.id for m in messages]
    assert not any(p.audit for p in pages)
    assert not any(p.audit_available for p in pages)


@pytest.mark.asyncio
async def test_the_audit_chain_advances_even_when_a_page_delivers_no_rows(tmp_path):
    """Forward progress is a property of the WINDOW, not of the rows kept.

    An audit page can legitimately deliver nothing: every row in its window was
    a preserved user turn the context phase already re-emitted at its head, so
    the audit copy is suppressed to avoid a duplicate id. If the next cursor
    were minted from the surviving rows, such a page would re-mint the cursor
    it was fetched with and the chain would spin on one window forever.

    Found on a real 231 MB journal, where it presented as a hang rather than as
    a wrong answer — which is why the assertion is a bounded loop.
    """
    directory = tmp_path / "s"
    transcript = Transcript(directory)
    # A run of user turns, ALL of which the compaction preserves. Their audit
    # copies are therefore all suppressed, so some audit window is empty.
    preserved = [Message.user(f"preserved {index}") for index in range(12)]
    await transcript.append_messages(preserved)
    later = [Message.assistant(f"later {index}") for index in range(30)]
    await transcript.append_messages(later)
    await transcript.append_compaction(
        "summary",
        later[-1].id,
        500,
        preserved_user_turns=[{"id": m.id, "text": m.text} for m in preserved],
    )
    await transcript.append_messages([Message.user("after the cut")])

    page = window(transcript, max_messages=4)
    seen_cursors = set()
    for _ in range(200):  # bounded: a stalled chain must fail, not hang
        token = page.before_token
        if not token:
            break
        assert token not in seen_cursors, "the audit chain re-issued a cursor"
        seen_cursors.add(token)
        page = window(transcript, before=token, max_messages=4)
        assert page.status == "ok"
    else:
        raise AssertionError("the audit chain did not terminate")
    assert page.before_token is None


@pytest.mark.asyncio
async def test_a_superseded_compaction_row_above_the_cut_renders_no_marker(tmp_path):
    """Repeated compaction leaves compaction rows INSIDE the kept suffix.

    Review round 1 (R2) read the omission of these as unreachable markers. It is
    the opposite: each names a boundary the latest compaction already moved, so
    the seam it describes no longer exists in the rendered transcript. Rendering
    them would place ``COMPACTION_MARKER_NOTICE`` — which claims the history
    above it is outside the model's context — above rows the model can still
    see, reintroducing the inversion D1 fixed.

    Pinned because the tempting "fix" (emit them from the context phase) also
    breaks the byte-identical guard below and pushes each marker's
    ``preserve_data`` into the model's own context.
    """
    directory = tmp_path / "s"
    transcript = Transcript(directory)
    rows = [Message.user(f"row {index}") for index in range(40)]
    await transcript.append_messages(rows)
    # Two compactions whose kept windows OVERLAP, so the earlier compaction row
    # ends up at or above the later one's cut — the shape the real journal
    # reaches by compacting repeatedly against a short kept suffix.
    superseded = await transcript.append_compaction("older summary", rows[20].id, 500)
    latest = await transcript.append_compaction("latest summary", rows[30].id, 500)

    entries = transcript.entries()
    cut = context_cut_index(entries, quiet=True)
    positions = {entry.id: index for index, entry in enumerate(entries)}
    # THE STARTING STATE MUST BE REACHABLE, not merely arranged: assert the
    # superseded row really does sit at/above the cut, or this test passes
    # while exercising nothing.
    assert positions[superseded.id] >= cut, "fixture did not reproduce a superseded marker"
    assert positions[latest.id] >= cut

    _, pages = _walk(transcript, max_messages=15)
    marker_ids = [
        row.id
        for page in pages
        for row in page.messages
        if getattr(row, "custom_type", None) == "compaction_summary"
    ]
    assert latest.id in marker_ids, "the live seam must still be rendered"
    assert superseded.id not in marker_ids, (
        "a superseded compaction row was rendered as a seam; its notice would "
        "claim the rows above it are outside the model's context, but they are "
        "still in the kept window"
    )


@pytest.mark.asyncio
async def test_emitting_superseded_markers_would_break_the_context_replay_guard(tmp_path):
    """Why R2's suggested route is rejected, asserted rather than argued.

    The PR's central regression guard is that ``mode="context"`` stays
    byte-for-byte what ``build_llm_history`` always produced. A context phase
    that also emitted the superseded markers would violate exactly that.
    """
    directory = tmp_path / "s"
    transcript = Transcript(directory)
    rows = [Message.user(f"row {index}") for index in range(40)]
    await transcript.append_messages(rows)
    superseded = await transcript.append_compaction("older summary", rows[20].id, 500)
    await transcript.append_compaction("latest summary", rows[30].id, 500)

    entries = transcript.entries()
    assert {e.id for e in entries} >= {superseded.id}
    replayed = replay_entries(entries, None, mode="context")
    assert [m.model_dump() for m in replayed] == [
        m.model_dump() for m in transcript.build_llm_history()
    ]
    # Exactly ONE marker at the head: the live seam.
    markers = [m for m in replayed if getattr(m, "custom_type", None) == "compaction_summary"]
    assert len(markers) == 1 and markers[0].id != superseded.id


@pytest.mark.asyncio
async def test_full_required_hands_back_an_audit_cursor_and_the_reader_adopts_it(tmp_path):
    """The DOMINANT real path into the audit phase, which had no coverage.

    All eight reference journals reach audit through this escalation rather
    than through a plain context drain: a group larger than the frame budget
    returns ``full_required``, the reader replays the model's history locally,
    and that replay stops at the context cut. The audit cursor returned
    alongside the escalation is the only way back to the pre-compaction rows,
    so a regression here silently restores the original defect for every
    compacted session while every other test stays green (review round 1, R3).
    """
    directory = tmp_path / "s"
    transcript = Transcript(directory)
    early = [Message.user(f"early {index}") for index in range(6)]
    await transcript.append_messages(early)
    cut_row = Message.user("cut row")
    await transcript.append_message(cut_row)
    await transcript.append_compaction("summary", cut_row.id, 500)
    # One group larger than the whole frame budget, which is what forces the
    # escalation rather than an ordinary page.
    await transcript.append_message(Message.assistant("oversized prose " * 100_000))

    page = window(transcript)
    # Reachability of the starting state, again asserted rather than assumed:
    # this must be the REAL escalation, carrying a REAL audit cursor.
    assert page.status == "full_required", page.status
    assert page.audit_available is True
    assert page.before_token, "the escalation dropped the audit cursor"

    # The reader's half: adopting that cursor must leave the chain live.
    remote = RemoteSession.__new__(RemoteSession)
    remote._display_history = page
    # The state a reader is REALLY in at this moment, not a convenient one:
    # ``materialize_history`` has just replayed the model's history in full, so
    # the context half is hydrated and only the audit half is outstanding.
    # ``history_before_token`` returns None unless BOTH are set, so getting
    # either wrong would hide the regression this test exists to catch.
    remote._history_hydrated = True
    remote._audit_exhausted = True
    assert remote.history_before_token is None, (
        "fixture is not in the post-escalation state: the chain must look "
        "exhausted BEFORE the cursor is adopted, or adopting it proves nothing"
    )
    remote._adopt_audit_cursor(page.before_token)
    assert remote.history_before_token == page.before_token
    assert remote._audit_exhausted is False
    assert remote._display_history.has_more is True
    assert remote._display_history.audit_available is True

    # And the adopted cursor actually fetches pre-compaction rows.
    resumed = window(transcript, before=page.before_token)
    assert resumed.status == "ok"
    assert resumed.audit is True
    assert {row.id for row in resumed.messages} & {row.id for row in early}


@pytest.mark.asyncio
async def test_an_empty_kept_suffix_pages_the_audit_tail_and_terminates(tmp_path):
    """``_AUDIT_TAIL_CURSOR``: the cut lands PAST the journal's last row.

    ``context_cut_index`` returns ``len(entries)`` when the newest compaction
    row carries no ``first_kept_entry_id``, so there is no entry id at the cut
    for the first audit cursor to name and the sentinel stands in for one. A
    regression here is a HUNG chain rather than a wrong answer (review round 1,
    R5).

    REACHABILITY, stated honestly because it decides what this test is worth:
    the CURRENT writer cannot produce this row. ``append_compaction`` takes
    ``first_kept_entry_id: str`` as a required argument and always records it,
    and a scan of 401 real session journals found 280 compaction rows with the
    field and none without. So the sentinel is defensive — it covers a journal
    written by a build predating the field, which is exactly the mixed-build
    situation this PR's capability negotiation exists for. The row is therefore
    appended through ``_append``, the same primitive ``append_compaction`` uses,
    rather than by passing a bogus id to the public writer: an unresolvable id
    is a DIFFERENT branch (it returns 0 and replays everything), and using it
    here would silently test the wrong path.
    """
    directory = tmp_path / "s"
    transcript = Transcript(directory)
    rows = [Message.user(f"row {index}") for index in range(8)]
    await transcript.append_messages(rows)
    await transcript._append(ENTRY_COMPACTION, {"summary": "summary", "tokens_before": 500})

    entries = transcript.entries()
    cut = context_cut_index(entries, quiet=True)
    assert cut == len(entries), "fixture did not reproduce an empty kept suffix"
    # And the sentinel is genuinely the cursor in play, not merely the state.
    assert _audit_entry_cursor(transcript) == _AUDIT_TAIL_CURSOR

    walked, pages = _walk(transcript, max_messages=30)
    assert any(p.audit for p in pages), "the tail cursor never entered the audit phase"
    assert pages[-1].before_token is None, "the chain did not terminate"
    delivered = {row.id for row in walked}
    assert delivered >= {row.id for row in rows}, "audit tail paging lost rows"
