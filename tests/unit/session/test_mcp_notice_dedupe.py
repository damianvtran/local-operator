"""The MCP-unavailable notice is written ONCE per state change.

The measured defect these tests pin: ``session_mcp_unavailable`` rows are
byte-identical for the same (server, reason), and every process boot/resume
that re-attempted a dead server appended the card again — 96 identical
``minerva-qa`` rows over ~29 h on session ``1375449bf925``, including a
four-card cluster inside seven minutes (the operator's "four identical cards
stacked", 2026-09-30). The rule (``session/notice_guard.py`` plus
``Session.journal_mcp_unavailable``): suppress an identical card while it is
outstanding in the replay — the in-process guard, else a durable transcript
scan for a fresh boot — and re-emit on a changed card, on a re-failure after
a LIVE recovery, or after the 24 h staleness reminder. Both halves are
bounded at the latest compaction cut, so a card no surface still shows
re-arms the next identical failure rather than suppressing it silently
(review round 1, M1).

The session-level tests drive the REAL sink the MCP manager is wired to
(``Session._on_mcp_incident`` / ``_on_mcp_recovery``) over real session
directories and read back the rows a surface would, rather than asserting on
the guard in isolation — that is the first class below.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.message_types import (
    SESSION_MCP_RECOVERY_MESSAGE_TYPE,
    SESSION_MCP_UNAVAILABLE_MESSAGE_TYPE,
)
from local_operator.harness.types import (
    ChatRequest,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamEvent,
)
from local_operator.incidents import format_mcp_unavailable_message
from local_operator.session.notice_guard import (
    MCP_UNAVAILABLE_REMIND_S,
    NoticeGuard,
    fingerprint_text,
)
from local_operator.session.session import Session, _default_convert_to_llm
from local_operator.session.transcript import Transcript, TranscriptEntry

MODEL = ModelSpec(provider="test", model_id="m", context_window=100_000)

#: The exact card from the operator's screenshot session (the four-card stack
#: at 08:42-08:49Z on 2026-09-30).
SCREENSHOT_SERVER = "minerva-qa"
SCREENSHOT_REASON = "/mcp reauth minerva-qa — sign-in expired"


class ScriptedStream:
    """Replays per-call event scripts; records requests."""

    def __init__(self, turns: list[list[StreamEvent]] | None = None) -> None:
        self.turns = turns or []
        self.requests: list[ChatRequest] = []

    def __call__(self, request: ChatRequest, signal: Any):
        self.requests.append(request)
        turn = self.turns.pop(0)

        async def gen():
            for event in turn:
                yield event

        return gen()


def _make_session(session_dir: Path, stream: ScriptedStream | None = None) -> Session:
    """A real Session over ``session_dir`` — the unit every cycle re-creates.

    Kept module-local, like ``test_session.py``'s own helper rather than a
    shared one: each file owning its fakes is what stops them drifting into a
    de-facto library the suite depends on by accident.
    """
    return Session(
        model=MODEL,
        stream_fn=stream if stream is not None else ScriptedStream(),
        tools=[],
        transcript=Transcript(session_dir),
        system_blocks_provider=lambda: ["stable", "env"],
    )


async def _drain(session: Session) -> None:
    """Run every background journal task to completion, oldest first."""
    for _ in range(10):
        pending = [task for task in list(session._background_tasks) if not task.done()]
        if not pending:
            return
        await asyncio.gather(*pending, return_exceptions=True)
    raise AssertionError("background journal tasks never settled")


async def _fire(
    session: Session,
    *,
    server: str = SCREENSHOT_SERVER,
    reason: str = SCREENSHOT_REASON,
) -> None:
    """One REAL emission: the sink ``McpManager.on_incident`` is wired to."""
    session._on_mcp_incident(server, reason)
    await _drain(session)


def _unavailable_rows(session_dir: Path) -> list[TranscriptEntry]:
    """The persisted warning rows, read off disk by a FRESH Transcript.

    Freshly constructed on purpose: it proves the row is durable — readable by
    a process that was never the writer — not merely in the writer's memory.
    """
    return [
        entry
        for entry in Transcript(session_dir).entries()
        if str(entry.payload.get("custom_type", "")) == SESSION_MCP_UNAVAILABLE_MESSAGE_TYPE
    ]


def _backdate_unavailable_row(session_dir: Path, *, by: float) -> None:
    """Age the newest warning row's timestamp on disk by ``by`` seconds."""
    path = session_dir / "transcript.jsonl"
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    changed = False
    for row in reversed(rows):
        is_warning = (
            str(row.get("payload", {}).get("custom_type", ""))
            == SESSION_MCP_UNAVAILABLE_MESSAGE_TYPE
        )
        if not changed and is_warning:
            row["ts"] = float(row["ts"]) - by
            changed = True
    assert changed, "no warning row on disk to backdate"
    path.write_text(
        "\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n",
        encoding="utf-8",
    )


def _model_warnings(replayed: list[Any]) -> list[Any]:
    """The model-visible ``[session warning]`` injections in a replay."""
    rendered = _default_convert_to_llm(replayed)
    return [
        message
        for message in rendered
        if "[session warning]"
        in " ".join(getattr(part, "text", "") for part in getattr(message, "content", []) or [])
    ]


async def _tui_warnings(replayed: list[Any]) -> list[Any]:
    """The TUI fold of ``replayed``: one real-app boot, no manual fold.

    The real ``OperatorApp`` (production stylesheet) over a ``FakeSession``
    carrying the replay, read back as the painted warning blocks — the
    surface the operator sees rather than a fold helper's output. Filtered
    to the ``[session warning]`` card by TEXT: compaction markers are also
    ``NoticeBlock`` subclasses, and this helper must count the warning, not
    the fold's bookkeeping.

    The wait before the read is the ADOPTION, not the clock. The replay is
    applied by ``_adopt_session`` INSIDE the boot worker the app spawns in
    ``on_mount`` (``run_worker(..., group="session")``), and that method is
    synchronous end to end: it commits ``self._session`` and the painted
    fold in one uninterrupted stretch. A single ``pilot.pause()`` is one
    message-drain hop against that worker, so under load it reads the app
    BEFORE the adoption lands — measured: the parity test read "the TUI
    painted 0 notices" twice in one CI cell (run 36802746654, attempts 1+2,
    shard ``test (3.12, 4)``) and again on an unrelated branch (run
    36811665066 job 110233263809, shard 2), each time while the replay
    itself carried the warning. So poll for ``_session`` to land, bounded
    at 200 turns (~4 s at Textual's 20 ms pause floor). On exhaustion the
    helper asserts the adoption happened — a never-adopting boot must fail
    loudly, not return an empty fold that the ``[]``-expecting call sites
    would accept (review round 1, MINOR-1) — then one settle pause for the
    mount before reading the blocks.
    """
    from local_operator.tui.app import OperatorApp
    from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    shell = FakeSession()
    shell._history = list(replayed)
    app = OperatorApp(lambda: _factory(shell))
    async with app.run_test(size=(100, 30)) as pilot:
        # No manual fold: the app's own boot replays `session.history()`.
        # Wait on the adoption (the docstring's measured race), kept bounded
        # so a broken boot fails loudly instead of hanging.
        for _ in range(200):
            await pilot.pause()
            if getattr(app, "_session", None) is not None:
                break
        # Exhaustion must be loud (MINOR-1): a never-adopting boot must not
        # hand back an empty fold the empty-expectation call sites accept.
        assert (
            getattr(app, "_session", None) is not None
        ), "the app never adopted the session within the bound"
        # One settle turn for the mount that follows the replay's paint.
        await pilot.pause()
        return [
            block
            for block in app.query_one(TranscriptView).blocks()
            if isinstance(block, NoticeBlock)
            and str(getattr(block, "_text", "")).startswith("[session warning]")
        ]


class TestNoticeGuard:
    """The guard's own algebra, without a Session."""

    def test_an_identical_card_is_suppressed_and_a_changed_one_emits(self) -> None:
        guard = NoticeGuard(remind_after_s=100.0)
        card = fingerprint_text("card A")
        assert guard.should_emit("server", card, now=0.0) is True
        guard.note_emitted("server", card, now=0.0)
        assert guard.should_emit("server", card, now=99.9) is False
        assert guard.should_emit("server", fingerprint_text("card B"), now=1.0) is True
        # The changed card is now the outstanding one...
        other = fingerprint_text("card B")
        guard.note_emitted("server", other, now=1.0)
        # ...so card A again IS a change (the newest state moved).
        assert guard.should_emit("server", card, now=2.0) is True

    def test_clear_re_arms_and_the_reminder_window_re_arms(self) -> None:
        guard = NoticeGuard(remind_after_s=100.0)
        card = fingerprint_text("card")
        guard.note_emitted("s", card, now=0.0)
        assert guard.should_emit("s", card, now=99.999) is False
        assert (
            guard.should_emit("s", card, now=100.0) is True
        ), "the boundary itself re-reminds — the window is the longest silence"
        guard.note_emitted("s", card, now=100.0)
        assert guard.should_emit("s", card, now=100.001) is False
        guard.clear("s")
        assert guard.should_emit("s", card, now=100.002) is True

    def test_no_reminder_window_suppresses_until_re_armed(self) -> None:
        guard = NoticeGuard(remind_after_s=None)
        card = fingerprint_text("card")
        guard.note_emitted("s", card, now=0.0)
        assert guard.should_emit("s", card, now=10**9) is False
        guard.note_recovered("s")
        assert guard.should_emit("s", card, now=10**9) is True

    def test_a_recovered_marker_re_arms_and_skips_the_durable_lookup(self) -> None:
        guard = NoticeGuard(remind_after_s=100.0)
        card = fingerprint_text("card")
        calls: list[tuple[str, str]] = []

        def find_previous(subject: str, fingerprint: str) -> float:
            calls.append((subject, fingerprint))
            return 1.0  # a FRESH durable row that would suppress

        assert guard.should_emit("s", card, find_previous=find_previous, now=2.0) is False
        assert calls == [("s", card)]
        guard.note_recovered("s")
        calls.clear()
        assert guard.should_emit("s", card, find_previous=find_previous, now=2.0) is True
        assert calls == [], (
            "a live recovery must skip the durable lookup: the store cannot see "
            "the recovery (it is never persisted), so its row is superseded"
        )
        guard.note_emitted("s", card, now=2.0)
        assert guard.should_emit("s", card, find_previous=find_previous, now=3.0) is False

    def test_a_stale_or_absent_durable_row_emits_and_a_fresh_one_suppresses(self) -> None:
        guard = NoticeGuard(remind_after_s=100.0)
        card = fingerprint_text("card")
        assert guard.should_emit("s", card, find_previous=lambda *_: 0.0, now=99.0) is False
        assert guard.should_emit("s", card, find_previous=lambda *_: 0.0, now=100.0) is True
        assert guard.should_emit("s", card, find_previous=lambda *_: None, now=1.0) is True

    def test_the_fingerprint_is_over_the_rendered_card(self) -> None:
        prefix = "x" * 200
        first = format_mcp_unavailable_message("files", prefix + "-one")
        second = format_mcp_unavailable_message("files", prefix + "-two")
        assert first == second
        assert fingerprint_text(first) == fingerprint_text(second), (
            "reasons differing only past the formatter's 200-char clip render "
            "ONE card, so they dedupe as one card"
        )
        assert fingerprint_text(format_mcp_unavailable_message("other", prefix)) != (
            fingerprint_text(first)
        )

    def test_subjects_do_not_leak_into_each_other(self) -> None:
        guard = NoticeGuard(remind_after_s=100.0)
        card = fingerprint_text("card")
        guard.note_emitted("a", card, now=0.0)
        guard.note_emitted("b", card, now=0.0)
        guard.note_recovered("a")
        assert guard.should_emit("a", card, now=1.0) is True
        assert guard.should_emit("b", card, now=1.0) is False

    def test_a_void_record_is_dropped_and_a_shown_one_still_suppresses(self) -> None:
        """``record_visible`` re-validates the live record (review round 1, M1).

        A caller whose store can RETIRE an emission it once showed (the MCP
        replay drops rows below a compaction cut) must be able to void the
        record — otherwise the guard keeps suppressing a card no surface
        shows. A record the caller still shows keeps its suppression, and the
        callback is not consulted at all for a changed card.
        """
        guard = NoticeGuard(remind_after_s=100.0)
        card = fingerprint_text("card")
        guard.note_emitted("s", card, now=0.0)

        # A record the caller still SHOWS keeps suppressing...
        calls: list[tuple[str, str]] = []

        def shown(subject: str, fingerprint: str) -> bool:
            calls.append((subject, fingerprint))
            return True

        assert guard.should_emit("s", card, record_visible=shown, now=1.0) is False
        assert calls == [("s", card)]

        # ...but a record it no longer shows is VOID: dropped, and it must emit.
        seen: list[tuple[str, str]] = []

        def gone(subject: str, fingerprint: str) -> bool:
            seen.append((subject, fingerprint))
            return False

        assert (
            guard.should_emit("s", card, record_visible=gone, now=2.0) is True
        ), "a record the caller no longer shows must not suppress"
        assert seen == [("s", card)]
        assert guard._outstanding.get("s") is None, "the void record must be dropped"

        # A changed card emits without consulting the callback at all.
        calls.clear()
        assert (
            guard.should_emit("s", fingerprint_text("other"), record_visible=shown, now=3.0) is True
        )
        assert calls == []

    def test_a_void_record_hands_the_decision_to_the_durable_lookup(self) -> None:
        """With the record dropped, ``find_previous`` decides as on a fresh boot."""
        guard = NoticeGuard(remind_after_s=100.0)
        card = fingerprint_text("card")
        guard.note_emitted("s", card, now=0.0)
        assert (
            guard.should_emit(
                "s",
                card,
                find_previous=lambda *_: 5.0,
                record_visible=lambda *_: False,
                now=6.0,
            )
            is False
        ), "the durable row is fresh; the void record falls through to it"


class TestSessionJournalDedupe:
    """The rule at the REAL write, over real session directories."""

    @pytest.mark.asyncio
    async def test_a_repeated_identical_notice_writes_one_row(self, tmp_path: Path) -> None:
        session_dir = tmp_path / "sess"
        session = _make_session(session_dir)
        try:
            await _fire(session)
            await _fire(session)
            assert len(_unavailable_rows(session_dir)) == 1, "a byte-identical duplicate re-flagged"

            # A changed reason is a new state...
            await _fire(session, reason="MCP authorization failed")
            assert len(_unavailable_rows(session_dir)) == 2

            # ...but a changed reason that still RENDERS the same card is not.
            await _fire(session, server="files", reason="y" * 300 + "-first")
            await _fire(session, server="files", reason="y" * 300 + "-second")
            assert (
                len(_unavailable_rows(session_dir)) == 3
            ), "reasons differing only past the 200-char clip are the same card"
        finally:
            await session.dispose()

    @pytest.mark.asyncio
    async def test_recovery_re_arms_and_is_not_persisted(self, tmp_path: Path) -> None:
        session_dir = tmp_path / "sess"
        session = _make_session(session_dir)
        try:
            await _fire(session)
            assert len(_unavailable_rows(session_dir)) == 1

            session._on_mcp_recovery(SCREENSHOT_SERVER, 41)
            await _drain(session)

            # The recovery is live-only: it must not reach the transcript...
            dumped = "\n".join(
                json.dumps(entry.payload, default=str)
                for entry in Transcript(session_dir).entries()
            )
            assert SESSION_MCP_RECOVERY_MESSAGE_TYPE not in dumped, (
                "the recovery was persisted; a resumed session would replay a "
                "claim about a connection it cannot re-verify"
            )

            # ...but it re-arms the warning: the SAME card after a live
            # recovery is a state change and must re-flag.
            await _fire(session)
            assert (
                len(_unavailable_rows(session_dir)) == 2
            ), "recovered -> fails again must emit a new notice"
        finally:
            await session.dispose()

    @pytest.mark.asyncio
    async def test_a_resumed_process_does_not_re_flag_the_same_card(self, tmp_path: Path) -> None:
        session_dir = tmp_path / "sess"
        first = _make_session(session_dir)
        try:
            await _fire(first)
        finally:
            await first.dispose()

        second = _make_session(session_dir)
        try:
            # A new process cannot know the guard — the DURABLE scan is what
            # suppresses here, and this is the shape of every boot/resume.
            await _fire(second)
            assert len(_unavailable_rows(session_dir)) == 1

            # A changed reason crosses the process boundary as a new card.
            await _fire(second, reason="/mcp reauth minerva-qa — refresh unconfirmed")
            assert len(_unavailable_rows(session_dir)) == 2
        finally:
            await second.dispose()

    @pytest.mark.asyncio
    async def test_a_row_older_than_the_reminder_window_re_reminds(self, tmp_path: Path) -> None:
        session_dir = tmp_path / "sess"
        first = _make_session(session_dir)
        try:
            await _fire(first)
        finally:
            await first.dispose()

        _backdate_unavailable_row(session_dir, by=MCP_UNAVAILABLE_REMIND_S + 60.0)

        second = _make_session(session_dir)
        try:
            await _fire(second)
            assert len(_unavailable_rows(session_dir)) == 2, (
                "a notice older than the reminder window must re-surface "
                "instead of being suppressed forever"
            )
        finally:
            await second.dispose()

    @pytest.mark.asyncio
    async def test_a_compaction_cut_re_emits_on_resume(self, tmp_path: Path) -> None:
        """A row the replay no longer shows must not suppress a fresh process.

        Review round 1, M1: the durable scan used to reach rows below the
        latest compaction cut — rows both the model replay and the TUI fold
        have dropped — so a resumed session kept the card suppressed with
        NOTHING visible on any surface. Bounded at the cut: the re-fire emits.
        """
        session_dir = tmp_path / "sess"
        first = _make_session(session_dir)
        try:
            await _fire(first)
            filler = await first._transcript.append_message(Message.user("filler after fire"))
            await first._transcript.append_compaction("cut above the notice", filler.id, 0)
        finally:
            await first.dispose()

        # The premise the finding measured: no surface shows the card any more.
        assert _model_warnings(Transcript(session_dir).build_llm_history()) == []

        resumed = _make_session(session_dir)
        try:
            await _fire(resumed)
        finally:
            await resumed.dispose()

        assert len(_unavailable_rows(session_dir)) == 2, (
            "the resume re-fire was suppressed although no visible card was "
            "outstanding; suppression must not outlive visibility"
        )

    @pytest.mark.asyncio
    async def test_a_mid_process_compaction_re_arms_the_live_guard(self, tmp_path: Path) -> None:
        """The in-memory record must not suppress on a row that just vanished.

        Review round 1, M1 (in-process arm): fire -> append_compaction cutting
        the row -> re-fire in the SAME process. The guard's own record is
        re-validated against the cut (``record_visible``) instead of trusting
        a row the replay no longer shows.
        """
        session_dir = tmp_path / "sess"
        session = _make_session(session_dir)
        try:
            await _fire(session)
            filler = await session._transcript.append_message(Message.user("filler after fire"))
            await session._transcript.append_compaction("cut above the notice", filler.id, 0)
            await _fire(session)
            assert len(_unavailable_rows(session_dir)) == 2, (
                "a mid-process compaction left the guard suppressing a card that "
                "no longer replays; it must re-emit"
            )
        finally:
            await session.dispose()

    @pytest.mark.asyncio
    async def test_a_row_below_the_cut_never_suppresses_a_re_flag(self, tmp_path: Path) -> None:
        """No scan may reach a row below the cut, matching row or not (M1).

        The store here carries several rows for the one server and the newest
        of them sits below the eventual cut: an unbounded scan finds it (the
        pre-fix defect), the cut-bounded scan must not — "outstanding" is a
        row the replay still shows, and the re-fire emits.
        """
        session_dir = tmp_path / "sess"
        first = _make_session(session_dir)
        try:
            await _fire(first)
            await _fire(first, reason="MCP authorization failed")
            await _fire(first)  # the card again: the newest row for the server
            filler = await first._transcript.append_message(Message.user("filler after fires"))
            await first._transcript.append_compaction("cut above every notice", filler.id, 0)

            card = fingerprint_text(
                format_mcp_unavailable_message(SCREENSHOT_SERVER, SCREENSHOT_REASON)
            )
            assert (
                first._mcp_unavailable_previous_ts(SCREENSHOT_SERVER, card) is None
            ), "the scan reached a row below the cut"
        finally:
            await first.dispose()

        resumed = _make_session(session_dir)
        try:
            await _fire(resumed)
        finally:
            await resumed.dispose()

        assert (
            len(_unavailable_rows(session_dir)) == 4
        ), "a row below the compaction cut suppressed the re-flag"

    @pytest.mark.asyncio
    async def test_the_notice_dedupes_across_a_turn_boundary(self, tmp_path: Path) -> None:
        """Cross-turn persistence: a repeat after a completed turn is the same card."""
        session_dir = tmp_path / "sess"
        stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")]])
        session = _make_session(session_dir, stream)
        try:
            await _fire(session)
            await session.prompt("keep going")
            await _fire(session)
            assert len(_unavailable_rows(session_dir)) == 1
        finally:
            await session.dispose()

    @pytest.mark.asyncio
    async def test_the_screenshot_case_writes_exactly_one_row(self, tmp_path: Path) -> None:
        """The operator's four-card stack: repeated cycles, in-process + boots."""
        session_dir = tmp_path / "sess"

        # Cycle 1 — one process fires the identical card twice (a boot that
        # settles discovery and then re-attempts on its continuation).
        first = _make_session(session_dir)
        try:
            await _fire(first)
            await _fire(first)
        finally:
            await first.dispose()

        # Cycles 2-4 — fresh processes over the same directory (the resume path).
        for _ in range(3):
            session = _make_session(session_dir)
            try:
                await _fire(session)
            finally:
                await session.dispose()

        rows = _unavailable_rows(session_dir)
        assert len(rows) == 1, f"the screenshot case re-flagged: {len(rows)} rows"
        text = rows[0].payload["details"]["text"]
        assert text == format_mcp_unavailable_message(SCREENSHOT_SERVER, SCREENSHOT_REASON)

    @pytest.mark.asyncio
    async def test_every_surface_shows_exactly_one_notice(self, tmp_path: Path) -> None:
        """Parity: one transcript row -> one model warning -> one TUI block.

        Extended for review round 1 (M1): when a compaction drops the card
        from every replay surface at once, the next identical failure must
        RE-EMIT — a suppressed silence with no visible card is the defect the
        guard may not produce.
        """
        session_dir = tmp_path / "sess"
        for _ in range(4):
            session = _make_session(session_dir)
            try:
                await _fire(session)
            finally:
                await session.dispose()

        # (1) The transcript every surface reads.
        assert len(_unavailable_rows(session_dir)) == 1

        # (2) The model's render pass, over a fresh replay of that transcript.
        replayed = Transcript(session_dir).build_llm_history()
        injected = _model_warnings(replayed)
        assert len(injected) == 1, f"the model is injected {len(injected)} warnings"

        # (3) The TUI fold, through the real app: one warning NoticeBlock.
        notices = await _tui_warnings(replayed)
        assert len(notices) == 1, f"the TUI painted {len(notices)} notices"
        assert notices[0]._token == "warning"
        assert notices[0]._text.startswith("[session warning] MCP server 'minerva-qa'")

        # (4) The cut case (M1): a compaction whose cut keeps only newer rows
        # drops the card from the replay every surface is fed from...
        cut_session = _make_session(session_dir)
        try:
            filler = await cut_session._transcript.append_message(Message.user("filler after fire"))
            await cut_session._transcript.append_compaction("cut above the notice", filler.id, 0)
        finally:
            await cut_session.dispose()

        replayed = Transcript(session_dir).build_llm_history()
        assert _model_warnings(replayed) == [], "a pre-cut card must leave the model replay"
        assert await _tui_warnings(replayed) == [], "a pre-cut card must leave the TUI fold"

        # ...so the SAME card's next failure must re-emit on every surface.
        resumed = _make_session(session_dir)
        try:
            await _fire(resumed)
        finally:
            await resumed.dispose()

        rows = _unavailable_rows(session_dir)
        assert len(rows) == 2, "append-only history keeps the old row; the re-fire adds a new one"
        replayed = Transcript(session_dir).build_llm_history()
        injected = _model_warnings(replayed)
        assert len(injected) == 1, f"the model is injected {len(injected)} warnings"
        notices = await _tui_warnings(replayed)
        assert len(notices) == 1, f"the TUI painted {len(notices)} notices"
        assert notices[0]._text.startswith("[session warning] MCP server 'minerva-qa'")
