"""The runtime's ``/highlights`` command: target selection, refusals, receipts.

WHY THIS FILE EXISTS. The command shipped with zero references in ``tests/`` -- its
only evidence was a manual drive pasted in a PR comment -- and the post-merge review
round on PR #2189 (comment 6103781990, F-2) found that gap only after the race it
should have caught beside it (F-1, ``SupplementRunner.newest_actionable``). This file
pins the command the way its siblings are pinned (``test_serving_monitor_command``,
``test_serving_wake_command``): a REAL session, handle, runner and journal -- the
generator's two outside edges stubbed -- driven through ``_highlights_slash``, because
reaching the ladder branch IS the subject.

The receipt words asserted here are byte-for-byte the TUI's own map
(``OperatorApp._run_highlights_op``'s ``words``): one gesture, one sentence, whichever
surface runs it.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import ModelSpec, StreamEndEvent
from local_operator.session.frontend_state import SlashResult
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.session.runtime.supplements import SupplementRunner
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.supplements import generator, policy
from local_operator.supplements.contract import SUPPLEMENT_CUSTOM_TYPE
from local_operator.supplements.decision import Decision
from local_operator.supplements.evidence import Dataset
from local_operator.supplements.persistence import append_row, build_details
from tests.unit.session.test_session import ScriptedStream, wait_for

pytestmark = pytest.mark.asyncio

MODEL = ModelSpec(provider="test", model_id="m", context_window=100_000)

DATASETS = (
    Dataset(
        title="Latency by region",
        source="bench.csv",
        columns=("region", "ms"),
        rows=(("us-east", "120"), ("us-west", "98.5")),
        n_rows=2,
        numeric_columns=("ms",),
    ),
)


class _Usage:
    input_tokens = 10
    output_tokens = 10


def _settings() -> policy.SupplementSettings:
    """The shipped defaults: the runner reads its snapshot at build."""
    return policy.SupplementSettings.from_values({})


def _decision() -> Decision:
    return Decision(vendor="heuristic", files_p={}, graphics_p=0.9, featured=(), more=())


def _make_session(directory: Path, cwd: Path) -> Session:
    return Session(
        model=MODEL,
        stream_fn=ScriptedStream([[StreamEndEvent(stop_reason="stop")]]),
        tools=[],
        transcript=Transcript(directory),
        system_blocks_provider=lambda: ["stable"],
        cwd=str(cwd),
    )


def _rows(transcript: Transcript) -> list[dict[str, Any]]:
    return [
        entry.payload["details"]
        for entry in transcript.entries()
        if entry.type == "custom" and entry.payload.get("custom_type") == SUPPLEMENT_CUSTOM_TYPE
    ]


async def _handle(session: Session, cwd: Path) -> ServingSessionHandle:
    return ServingSessionHandle(
        session, asyncio.get_running_loop(), install_gates=False, cwd=str(cwd)
    )


async def _run(handle: ServingSessionHandle, arg: str) -> SlashResult:
    return await handle._highlights_slash(arg, SlashResult)


def _remember(runner: SupplementRunner, anchor: str, job: str) -> None:
    """Seed one remembered job the way ``_render``/``_start`` do.

    The selection under test reads THIS structure (``_inputs``), so the real private
    dataclass goes in -- a double of it would assert the reading into existence.
    """
    from local_operator.session.runtime.supplements import _JobInputs

    runner._remember(
        _JobInputs(
            anchor=anchor,
            job=job,
            decision=_decision(),
            datasets=(),
            user_text="u",
            answer_text="a",
        )
    )


def _stub(runner: SupplementRunner, monkeypatch: pytest.MonkeyPatch, *, answers: list[str]) -> None:
    """The runner's two outside edges, replaced (the ``test_runner_generator`` shape)."""
    spec = generator.DesignModel(MODEL, "session", "test/m")

    def resolve(settings: Any) -> generator.DesignModel:
        return spec

    async def complete(request: Any) -> tuple[str, Any]:
        return answers.pop(0) if answers else "NONE", _Usage()

    monkeypatch.setattr(runner, "_resolve_model", resolve)
    monkeypatch.setattr(runner, "_complete", complete)


async def test_cancel_takes_the_running_job_and_says_the_words_the_tui_shows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``cancel`` addresses the job RUNNING now -- never a newest journal row.

    The running job is parked mid-generator; the command must cut THAT job, write its
    ``cancelled`` row and answer with the TUI's word for the op. ``CANCEL`` arrives in
    caps: the TUI lowercases the typed verb, and the runtime must agree.
    """
    session = _make_session(tmp_path / "sessions" / "hl-cancel", tmp_path)
    handle = await _handle(session, tmp_path)
    started = asyncio.Event()
    release = asyncio.Event()
    spec = generator.DesignModel(MODEL, "session", "test/m")

    def resolve(settings: Any) -> generator.DesignModel:
        return spec

    async def complete(request: Any) -> tuple[str, Any]:
        started.set()
        await release.wait()
        return "NONE", _Usage()

    monkeypatch.setattr(handle._supplements, "_resolve_model", resolve)
    monkeypatch.setattr(handle._supplements, "_complete", complete)
    try:
        handle._supplements._render("anchor-hl", _decision(), DATASETS, "u", "a", _settings())
        await asyncio.wait_for(started.wait(), timeout=10)
        await wait_for(
            lambda: bool(_rows(session.transcript))
            and _rows(session.transcript)[0]["state"] == "decided",
            timeout=10,
        )
        result = await _run(handle, "CANCEL")
        assert (result.kind, result.text, result.style) == (
            "notice",
            "Highlights cancelled",
            "info",
        )
        rows = _rows(session.transcript)
        assert [row["state"] for row in rows] == ["decided", "cancelled"], rows
        # The runner is idle now: the second cancel answers the honest idle line.
        idle = await _run(handle, "cancel")
        assert (idle.kind, idle.text, idle.style) == (
            "notice",
            "nothing is being prepared right now",
            "warning",
        )
    finally:
        release.set()
        await handle.dispose()
        await session.dispose()


async def test_retry_and_adjust_pick_the_newest_actionable_remembered_row(
    tmp_path: Path,
) -> None:
    """The selection rules, on remembered anchors and a seeded journal.

    ``retry`` takes the newest remembered job whose newest row is failed/cancelled --
    a NEWER ``done`` row is skipped, it has nothing to retry; ``adjust`` takes the
    newest row in any non-excluded state; superseded/dismissed/skipped rows are never
    actionable; and a verb the caller refuses answers ``None`` rather than guessing.
    """
    session = _make_session(tmp_path / "sessions" / "hl-select", tmp_path)
    handle = await _handle(session, tmp_path)
    runner = handle._supplements
    await append_row(
        session.transcript,
        dict(
            build_details(
                anchor="anchor-a",
                job="j-a",
                version=2,
                state="cancelled",
                decision=_decision(),
                at=100.0,
            )
        ),
    )
    await append_row(
        session.transcript,
        dict(
            build_details(
                anchor="anchor-b",
                job="j-b",
                version=2,
                state="done",
                decision=_decision(),
                at=200.0,
            )
        ),
    )
    dismissed = dict(
        build_details(
            anchor="anchor-c",
            job="j-c",
            version=2,
            state="skipped",
            decision=_decision(),
            at=300.0,
        )
    )
    dismissed["dismissed"] = True
    await append_row(session.transcript, dismissed)
    for anchor, job in (("anchor-a", "j-a"), ("anchor-b", "j-b"), ("anchor-c", "j-c")):
        _remember(runner, anchor, job)
    try:
        # retry: the NEWER done row is not retryable; the older cancelled row wins,
        # and the even-newer dismissed row never competes.
        assert await runner.newest_actionable("retry") == ("anchor-a", "j-a")
        # adjust: the newest row in any non-excluded state -- b's done row, not c's.
        assert await runner.newest_actionable("adjust") == ("anchor-b", "j-b")
        # A newer actionable row moves the pick: newest wins between live candidates.
        await append_row(
            session.transcript,
            dict(
                build_details(
                    anchor="anchor-b",
                    job="j-b",
                    version=3,
                    state="cancelled",
                    decision=_decision(),
                    at=400.0,
                )
            ),
        )
        assert await runner.newest_actionable("retry") == ("anchor-b", "j-b")
        # A verb the caller refuses answers None rather than guessing a target.
        assert await runner.newest_actionable("dance") is None
    finally:
        await handle.dispose()
        await session.dispose()


async def test_retry_and_adjust_receipts_are_the_tui_words_and_start_a_new_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The two op receipts, word for word, and the journal effect behind them.

    A cancelled row is the retry shape (``Preparing highlights again…``); the settled
    do-over after it is the adjust shape (``Updating highlights…``). Each starts a
    fresh version of the CHOSEN job -- receipt and write must name the same work.
    """
    session = _make_session(tmp_path / "sessions" / "hl-words", tmp_path)
    handle = await _handle(session, tmp_path)
    runner = handle._supplements
    _stub(runner, monkeypatch, answers=["NONE", "NONE"])
    await append_row(
        session.transcript,
        dict(
            build_details(
                anchor="anchor-hw",
                job="j-hw",
                version=2,
                state="cancelled",
                decision=_decision(),
                at=100.0,
            )
        ),
    )
    _remember(runner, "anchor-hw", "j-hw")
    try:
        retry = await _run(handle, "retry")
        assert (retry.kind, retry.text, retry.style) == (
            "notice",
            "Preparing highlights again…",
            "info",
        )
        await wait_for(lambda: len(_rows(session.transcript)) == 3, timeout=10)
        adjust = await _run(handle, "adjust now show the tail")
        assert (adjust.kind, adjust.text, adjust.style) == (
            "notice",
            "Updating highlights…",
            "info",
        )
        await wait_for(lambda: len(_rows(session.transcript)) == 5, timeout=10)
        rows = _rows(session.transcript)
        assert [row["state"] for row in rows] == [
            "cancelled",
            "decided",
            "done",
            "decided",
            "done",
        ], rows
        assert {row["job"] for row in rows} == {"j-hw"}
    finally:
        await handle.dispose()
        await session.dispose()


async def test_every_refusal_is_the_tui_sentence_in_warning_ink(tmp_path: Path) -> None:
    """Nothing to act on, or nothing readable: the TUI's own refusal lines.

    Each verb has its own missing-target sentence; the malformed shapes (an unknown
    verb, an empty argument, ``adjust`` without text) are the two usage lines. All of
    them are ``warning`` notices, as the TUI paints them.
    """
    session = _make_session(tmp_path / "sessions" / "hl-refuse", tmp_path)
    handle = await _handle(session, tmp_path)
    try:
        for arg, text in (
            ("cancel", "nothing is being prepared right now"),
            ("retry", "nothing to retry — no highlights were cancelled or failed"),
            ("adjust make it nicer", "no highlights to adjust yet"),
        ):
            result = await _run(handle, arg)
            assert (result.kind, result.text, result.style) == ("notice", text, "warning")
        for arg, text in (
            ("dance", "Usage: /highlights cancel | retry | adjust <text>"),
            ("", "Usage: /highlights cancel | retry | adjust <text>"),
            ("adjust", "Usage: /highlights adjust <text>"),
            ("adjust   ", "Usage: /highlights adjust <text>"),
        ):
            result = await _run(handle, arg)
            assert (result.kind, result.text, result.style) == ("notice", text, "warning")
    finally:
        await handle.dispose()
        await session.dispose()


async def test_a_forgotten_or_rowless_anchor_is_not_a_target(tmp_path: Path) -> None:
    """Selection reads BOTH halves: inputs and journal.

    A journal row can outlive the inputs (judged on a previous process, or evicted by
    the three-job bound): the runtime must not promise a retry it cannot run. And an
    anchor whose newest read says NOTHING is skipped rather than selected blind.
    """
    session = _make_session(tmp_path / "sessions" / "hl-forgot", tmp_path)
    handle = await _handle(session, tmp_path)
    runner = handle._supplements
    await append_row(
        session.transcript,
        dict(
            build_details(
                anchor="anchor-forgotten",
                job="j-f",
                version=2,
                state="cancelled",
                decision=_decision(),
                at=100.0,
            )
        ),
    )
    try:
        # Journal row, no remembered inputs: the same refusal as no job at all.
        retry = await _run(handle, "retry")
        assert retry.text == "nothing to retry — no highlights were cancelled or failed"
        adjust = await _run(handle, "adjust make it nicer")
        assert adjust.text == "no highlights to adjust yet"
        # Remembered inputs with NO journal rows: the read finds nothing to act on.
        _remember(runner, "anchor-rowless", "j-norows")
        assert await runner.newest_actionable("retry") is None
        assert (await _run(handle, "adjust make it nicer")).text == "no highlights to adjust yet"
    finally:
        await handle.dispose()
        await session.dispose()


async def test_cancel_racing_a_settling_job_says_already_finished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The neutral receipt still has its sentence: the settle race's own words.

    A cancel can arrive while the job is mid-terminal-beat -- the row is durable, the
    task is not done -- so the pick succeeds and the op must answer for the settled
    pair without touching it (the round-1 R1 gate, from the command's side). The
    command maps that answer to the TUI's ``those highlights already finished``.
    """
    session = _make_session(tmp_path / "sessions" / "hl-settle", tmp_path)
    handle = await _handle(session, tmp_path)
    runner = handle._supplements
    _stub(runner, monkeypatch, answers=["NONE"])
    entered = asyncio.Event()
    release_beat = asyncio.Event()
    original_progress = runner._progress
    gate = {"first": True}

    async def hold_done_beat(details: Any, state: str, **kwargs: Any) -> None:
        if state == "done" and gate["first"]:
            gate["first"] = False
            entered.set()
            await release_beat.wait()
        await original_progress(details, state, **kwargs)

    monkeypatch.setattr(runner, "_progress", hold_done_beat)
    try:
        runner._render("anchor-settle", _decision(), DATASETS, "u", "a", _settings())
        await asyncio.wait_for(entered.wait(), timeout=10)
        # The terminal row is journaled; the task is parked INSIDE the beat, so the
        # pick still sees a running job and the op must answer from the row.
        assert [row["state"] for row in _rows(session.transcript)] == ["decided", "done"]
        result = await _run(handle, "cancel")
        assert (result.kind, result.text, result.style) == (
            "notice",
            "those highlights already finished",
            "info",
        )
        assert [row["state"] for row in _rows(session.transcript)] == ["decided", "done"]
    finally:
        release_beat.set()
        await handle.dispose()
        await session.dispose()


async def test_retry_survives_an_anchor_remembered_during_its_journal_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F-1: the dictionary-changed-size race, driven deterministically.

    ``newest_actionable`` walks the remembered anchors and awaits each journal read on
    a worker thread; a turn ending in that window remembers a NEW anchor (``_render``
    -> ``_remember`` -- the ordinary path), which used to make the walk raise
    ``RuntimeError: dictionary changed size during iteration`` and fail the whole
    command. The window is widened here by gating the read, but the interleave itself
    is EVENT-driven, not timing: the render provably lands while the read is in
    flight, and the read returns only after it. Pre-fix this test fails with that
    RuntimeError; post-fix the walk iterates its snapshot and the retry answers.
    """
    session = _make_session(tmp_path / "sessions" / "hl-race", tmp_path)
    handle = await _handle(session, tmp_path)
    runner = handle._supplements
    _stub(runner, monkeypatch, answers=["NONE"])
    await append_row(
        session.transcript,
        dict(
            build_details(
                anchor="anchor-race",
                job="j-race",
                version=2,
                state="cancelled",
                decision=_decision(),
                at=100.0,
            )
        ),
    )
    _remember(runner, "anchor-race", "j-race")

    read_started = threading.Event()
    read_release = threading.Event()
    real_read = runner._newest_row

    def gated_read(anchor: str, *, job: str | None = None) -> Any:
        read_started.set()
        assert read_release.wait(10), "the gated read was never released"
        return real_read(anchor, job=job)

    async def render_during_read() -> None:
        await wait_for(read_started.is_set, timeout=10)
        runner._render("anchor-during", _decision(), DATASETS, "u", "a", _settings())
        read_release.set()

    monkeypatch.setattr(runner, "_newest_row", gated_read)
    renderer = asyncio.create_task(render_during_read())
    try:
        result = await _run(handle, "retry")
        # The command completed: the snapshot held only the anchor that was
        # remembered when the walk began, and the retry started it.
        assert (result.kind, result.text, result.style) == (
            "notice",
            "Preparing highlights again…",
            "info",
        )
        assert "anchor-during" in runner._inputs, "the render's remember never landed"
        await wait_for(lambda: not runner.running, timeout=10)
        raced = [row for row in _rows(session.transcript) if row["anchor"] == "anchor-race"]
        assert [(row["version"], row["state"]) for row in raced] == [
            (2, "cancelled"),
            (3, "decided"),
            (4, "done"),
        ], raced
    finally:
        read_release.set()
        renderer.cancel()
        await asyncio.gather(renderer, return_exceptions=True)
        await handle.dispose()
        await session.dispose()
