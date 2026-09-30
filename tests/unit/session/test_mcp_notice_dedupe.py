"""The MCP-unavailable notice is written ONCE per state change.

The measured defect these tests pin: ``session_mcp_unavailable`` rows are
byte-identical for the same (server, reason), and every process boot/resume
that re-attempted a dead server appended the card again — 96 identical
``minerva-qa`` rows over ~29 h on session ``1375449bf925``, including a
four-card cluster inside seven minutes (the operator's "four identical cards
stacked", 2026-09-30). The rule (``session/notice_guard.py`` plus
``Session.journal_mcp_unavailable``): suppress an identical card while it is
outstanding — the in-process guard, else a durable transcript scan for a fresh
boot — and re-emit only on a changed card, on a re-failure after a LIVE
recovery, or after the 24 h staleness reminder.

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
        """Parity: one transcript row -> one model warning -> one TUI block."""
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
        rendered = _default_convert_to_llm(replayed)
        injected = [
            message
            for message in rendered
            if "[session warning]"
            in " ".join(getattr(part, "text", "") for part in getattr(message, "content", []) or [])
        ]
        assert len(injected) == 1, f"the model is injected {len(injected)} warnings"

        # (3) The TUI fold, through the real app: one warning NoticeBlock.
        from local_operator.tui.app import OperatorApp
        from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView
        from tests.unit.tui.test_app_pilot import FakeSession, _factory

        shell = FakeSession()
        shell._history = list(replayed)
        app = OperatorApp(lambda: _factory(shell))
        async with app.run_test(size=(100, 30)) as pilot:
            # No manual fold: the app's own boot replays `session.history()`.
            await pilot.pause()
            notices = [
                block
                for block in app.query_one(TranscriptView).blocks()
                if isinstance(block, NoticeBlock)
            ]

        assert len(notices) == 1, f"the TUI painted {len(notices)} notices"
        assert notices[0]._token == "warning"
        assert notices[0]._text.startswith("[session warning] MCP server 'minerva-qa'")
