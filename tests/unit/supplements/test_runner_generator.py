"""The runner's generator tail and its four control ops (memo §2.5-§2.7).

WHAT IS AND IS NOT STUBBED. The session, the transcript, the handle and the RUNNER are real;
the provider call and the model resolution are not — they are the two things this file's
subject does not own. That leaves every persistence decision under test: the ``decided`` row,
the single terminal row at ``version + 1`` (a regeneration REPLACES a version rather than
appending a duplicate), the content-addressed blob the row points at, and the row a cancel or
a dismiss leaves behind.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import ModelSpec, StreamEndEvent
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.supplements import generator, policy
from local_operator.supplements.contract import (
    SUPPLEMENT_CUSTOM_TYPE,
    newest_per_anchor,
    reader_disposition,
)
from local_operator.supplements.decision import Decision
from local_operator.supplements.evidence import Dataset
from local_operator.supplements.persistence import append_row, build_details
from tests.unit.session.test_session import ScriptedStream, wait_for

pytestmark = pytest.mark.asyncio

MODEL = ModelSpec(provider="test", model_id="m", context_window=100_000)

BLOCK = (
    '<component title="Latency" source="bench.csv">\n'
    '<data>{"lat":{"title":"Latency (ms)","columns":["region","ms"],'
    '"rows":[["us-east",120],["us-west",98.5]]}}</data>\n'
    '<html><div id="c"></div><script>LO.bar(document.getElementById("c"),"lat",'
    '{x:"region",y:"ms",unit:"ms"})</script></html>\n'
    "</component>"
)

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


def _settings() -> policy.SupplementSettings:
    """The shipped defaults: the runner reads its snapshot at build, and these tests are
    about the tail, not about which row of the settings page was set."""
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


def _hold_terminal_write(
    monkeypatch: pytest.MonkeyPatch, transcript: Transcript
) -> tuple[threading.Event, threading.Event]:
    """Hold the next supplement TERMINAL write open inside ``_commit``'s worker thread.

    The round-3 review's R3-1 window is the append's suspension (thread dispatch + file
    write): the row reaches disk and is published by ``_commit`` while the writing task is
    suspended before it can return, so only the journal knows the row exists. Returns
    ``(entered, release)``: wait for ``entered`` (the write is inside the thread), position
    the op, then set ``release``. The filter matches terminal ``done`` supplement rows
    only -- turn entries and ``decided`` rows pass straight through.
    """
    entered = threading.Event()
    release = threading.Event()
    original = transcript._write_entries

    def held(entries: list[Any], *, preserve_mtime: bool = False) -> None:
        terminal = any(
            entry.type == "custom"
            and isinstance(entry.payload, dict)
            and entry.payload.get("custom_type") == SUPPLEMENT_CUSTOM_TYPE
            and isinstance(entry.payload.get("details"), dict)
            and entry.payload["details"].get("state") == "done"
            for entry in entries
        )
        if terminal:
            entered.set()
            assert release.wait(10), "the held terminal write was never released"
        original(entries, preserve_mtime=preserve_mtime)

    monkeypatch.setattr(transcript, "_write_entries", held)
    return entered, release


class StubRunner:
    """The real runner with its two outside dependencies replaced."""

    def __init__(self, runner: Any, *, answers: list[str], price: float = 0.0) -> None:
        self.runner = runner
        self.answers = list(answers)
        self.price = price
        self.requests: list[Any] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        spec = generator.DesignModel(MODEL, "session", "test/m")

        def resolve(settings: Any) -> generator.DesignModel:
            return spec

        async def complete(request: Any) -> tuple[str, Any]:
            self.requests.append(request)
            text = self.answers.pop(0) if self.answers else "NONE"
            return text, _Usage()

        monkeypatch.setattr(self.runner, "_resolve_model", resolve)
        monkeypatch.setattr(self.runner, "_complete", complete)


class _Usage:
    input_tokens = 10
    output_tokens = 10


async def _handle(session: Session, cwd: Path) -> ServingSessionHandle:
    handle = ServingSessionHandle(
        session, asyncio.get_running_loop(), install_gates=False, cwd=str(cwd)
    )
    return handle


async def _drain(handle: ServingSessionHandle) -> None:
    """Let the runner's detached task settle (the job is not in ``_background_tasks``)."""
    await wait_for(lambda: not handle._supplements.running, timeout=10)


async def test_a_graphics_job_writes_decided_then_one_terminal_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _make_session(tmp_path / "sessions" / "tail", tmp_path)
    handle = await _handle(session, tmp_path)
    stub = StubRunner(handle._supplements, answers=[BLOCK])
    stub.install(monkeypatch)
    try:
        handle._supplements._render(
            "anchor-1", _decision(), DATASETS, "how did it go?", "120 east.", _settings()
        )
        await _drain(handle)
        rows = _rows(session.transcript)
        assert [row["state"] for row in rows] == ["decided", "done"], rows
        assert [row["version"] for row in rows] == [1, 2]
        decided, done = rows
        assert decided["job"] == done["job"], "one job id across the versions"
        assert done["components"], done
        component = done["components"][0]
        assert component["mime"] == "text/html" and component["height_hint"] == 320
        assert (component["attachment"], component["source"]) == (
            component["attachment"],
            "bench.csv",
        )
        # The blob is on disk, content-addressed, and it is the component the validator saw.
        from local_operator.session.attachments import AttachmentStore

        stored = AttachmentStore().get_bytes(component["attachment"])
        assert stored is not None
        raw, mime = stored
        assert mime == "text/html" and b'id="c"' in raw
        # The reader rule: the newest version wins, and the pair is terminal.
        assert reader_disposition(done, job_live=False) == "block"
        assert handle._supplements.is_live(done["job"]) is False
    finally:
        await handle.dispose()


async def test_a_second_version_replaces_the_first_rather_than_duplicating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _make_session(tmp_path / "sessions" / "versions", tmp_path)
    handle = await _handle(session, tmp_path)
    stub = StubRunner(handle._supplements, answers=[BLOCK])
    stub.install(monkeypatch)
    try:
        handle._supplements._render("anchor-2", _decision(), DATASETS, "u", "a", _settings())
        await _drain(handle)
        before = _rows(session.transcript)
        # A restart is the re-generation path: version + 1 with the same job id.
        receipt = await handle._supplements.restart_job("anchor-2", before[0]["job"])
        assert receipt == "restarting"
        await _drain(handle)
        after = _rows(session.transcript)
        # 1: the first job's decided row, 2: its terminal row, 3: the restart's decided row,
        # 4: the restart's terminal row. Every version appears ONCE -- a regeneration
        # REPLACES a version rather than appending a duplicate of it.
        assert [row["version"] for row in after] == [1, 2, 3, 4], after
        assert len({row["job"] for row in after}) == 1
        assert after[-1]["state"] == "done"
        # Newest-version-wins: exactly one row answers for the anchor.
        from local_operator.supplements.contract import newest_per_anchor

        assert list(newest_per_anchor(after)) == ["anchor-2"]
    finally:
        await handle.dispose()


async def test_cancel_writes_cancelled_and_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _make_session(tmp_path / "sessions" / "cancel", tmp_path)
    handle = await _handle(session, tmp_path)
    started = asyncio.Event()
    release = asyncio.Event()
    spec = generator.DesignModel(MODEL, "session", "test/m")

    def resolve(settings: Any) -> generator.DesignModel:
        return spec

    async def complete(request: Any) -> tuple[str, Any]:
        started.set()
        await release.wait()
        return BLOCK, _Usage()

    monkeypatch.setattr(handle._supplements, "_resolve_model", resolve)
    monkeypatch.setattr(handle._supplements, "_complete", complete)
    try:
        handle._supplements._render("anchor-3", _decision(), DATASETS, "u", "a", _settings())
        await asyncio.wait_for(started.wait(), timeout=10)
        await wait_for(
            lambda: _rows(session.transcript)
            and _rows(session.transcript)[0]["state"] == "decided",
            timeout=10,
        )
        job = _rows(session.transcript)[0]["job"]
        assert await handle._supplements.cancel_job("anchor-3", job) == "cancelled"
        rows = _rows(session.transcript)
        assert rows[-1]["state"] == "cancelled" and rows[-1]["version"] == 2, rows
        # Idempotent: the same op again answers the neutral receipt, image-gen's rule.
        assert await handle._supplements.cancel_job("anchor-3", job) == "already finished"
        # A cancelled pair renders the retry line, never the spinner.
        assert reader_disposition(rows[-1], job_live=False) == "cancelled_retry"
    finally:
        release.set()
        await handle.dispose()


async def test_dismiss_writes_skipped_and_survives_a_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _make_session(tmp_path / "sessions" / "dismiss", tmp_path)
    handle = await _handle(session, tmp_path)
    stub = StubRunner(handle._supplements, answers=[BLOCK])
    stub.install(monkeypatch)
    try:
        handle._supplements._render("anchor-4", _decision(), DATASETS, "u", "a", _settings())
        await _drain(handle)
        # Forget the in-memory inputs: this is the restart/history-page case, where only the
        # journal knows the row.
        handle._supplements._inputs.clear()
        assert await handle._supplements.dismiss("anchor-4") == "dismissed"
        rows = _rows(session.transcript)
        assert rows[-1]["state"] == "skipped" and rows[-1]["dismissed"] is True
        assert reader_disposition(rows[-1], job_live=False) == "nothing"
        newest = handle._supplements._newest_row("anchor-4")
        assert newest is not None and newest["state"] == "skipped"
    finally:
        await handle.dispose()


async def test_the_fork_answering_NONE_writes_one_terminal_row_with_no_components(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A legitimate "no figure helps": the row is terminal, carries no components, and the
    reader shows the files-only block rather than a spinner or a failure line."""
    session = _make_session(tmp_path / "sessions" / "none", tmp_path)
    handle = await _handle(session, tmp_path)
    stub = StubRunner(handle._supplements, answers=["NONE"])
    stub.install(monkeypatch)
    try:
        handle._supplements._render("anchor-5", _decision(), DATASETS, "u", "a", _settings())
        await _drain(handle)
        rows = _rows(session.transcript)
        assert [row["state"] for row in rows] == ["decided", "done"], rows
        assert rows[-1]["components"] == [] and rows[-1].get("error", "") == ""
        assert rows[-1]["model"] == "test/m" and rows[-1]["turns"] == 1
        assert isinstance(rows[-1]["cost_usd"], float)
        assert session.transcript.latest_custom(SUPPLEMENT_CUSTOM_TYPE)["state"] == "done"
    finally:
        await handle.dispose()


LYING_BLOCK = BLOCK.replace('["us-east",120],["us-west",98.5]', '["us-east",120],["us-west",7000]')


async def test_cancel_for_a_settled_job_never_touches_a_newer_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1 REPRO-1: a cancel naming a SETTLED job answers "already finished".

    It must not cut the newer job that is running, and it must not rewrite the settled
    row (the pre-fix behaviour: newer job killed, its own row left non-terminal, and the
    settled row's state done -> cancelled).
    """
    session = _make_session(tmp_path / "sessions" / "settled-cancel", tmp_path)
    handle = await _handle(session, tmp_path)
    stub = StubRunner(handle._supplements, answers=[BLOCK])
    stub.install(monkeypatch)
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocking_complete(request: Any) -> tuple[str, Any]:
        started.set()
        await release.wait()
        return BLOCK, _Usage()

    try:
        handle._supplements._render("anchor-old", _decision(), DATASETS, "u", "a", _settings())
        await _drain(handle)
        settled = _rows(session.transcript)
        assert [row["state"] for row in settled] == ["decided", "done"]
        old_job = settled[0]["job"]

        # A NEWER job starts and blocks mid-flight.
        monkeypatch.setattr(handle._supplements, "_complete", blocking_complete)
        handle._supplements._render("anchor-new", _decision(), DATASETS, "u", "a", _settings())
        await asyncio.wait_for(started.wait(), timeout=10)
        new_job = next(
            row["job"] for row in _rows(session.transcript) if row["anchor"] == "anchor-new"
        )

        assert await handle._supplements.cancel_job("anchor-old", old_job) == "already finished"
        # The newer job is still running...
        assert handle._supplements.running
        assert handle._supplements.is_live(new_job)
        # ...and the settled pair is untouched: still two rows, newest still ``done``.
        old_rows = [row for row in _rows(session.transcript) if row["anchor"] == "anchor-old"]
        assert [row["state"] for row in old_rows] == ["decided", "done"], old_rows

        release.set()
        await _drain(handle)
        new_rows = [row for row in _rows(session.transcript) if row["anchor"] == "anchor-new"]
        assert [row["state"] for row in new_rows] == ["decided", "done"], new_rows
    finally:
        release.set()
        await handle.dispose()


async def test_steer_on_an_older_anchor_queues_behind_the_running_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1 REPRO-3 + memo §2.9: a steer on an older anchor waits, depth 1.

    The newer job must NOT be cut; when it settles, the older anchor's version + 1 runs
    with the instruction (the pre-fix behaviour: newer job cut, the old anchor's next
    version started immediately).
    """
    session = _make_session(tmp_path / "sessions" / "queued-steer", tmp_path)
    handle = await _handle(session, tmp_path)
    stub = StubRunner(handle._supplements, answers=[BLOCK])
    stub.install(monkeypatch)
    started = asyncio.Event()
    release = asyncio.Event()
    calls: list[Any] = []

    async def complete(request: Any) -> tuple[str, Any]:
        calls.append(request)
        if len(calls) == 1:
            started.set()
            await release.wait()
        return BLOCK, _Usage()

    try:
        handle._supplements._render("anchor-old", _decision(), DATASETS, "u", "a", _settings())
        await _drain(handle)
        old_job = _rows(session.transcript)[0]["job"]

        monkeypatch.setattr(handle._supplements, "_complete", complete)
        handle._supplements._render("anchor-new", _decision(), DATASETS, "u", "a", _settings())
        await asyncio.wait_for(started.wait(), timeout=10)

        assert (
            await handle._supplements.steer_job("anchor-old", old_job, "make it a table")
            == "steering"
        )
        # The newer job was not cut and the old anchor gained no row yet: it is QUEUED.
        assert handle._supplements.running
        old_rows = [row for row in _rows(session.transcript) if row["anchor"] == "anchor-old"]
        assert [row["state"] for row in old_rows] == ["decided", "done"], old_rows

        release.set()
        await _drain(handle)
        # Queued behind the newer job: once it settles, version + 1 runs for the older anchor.
        old_rows = [row for row in _rows(session.transcript) if row["anchor"] == "anchor-old"]
        assert [row["version"] for row in old_rows] == [1, 2, 3, 4], old_rows
        assert old_rows[-1]["state"] == "done"
        assert old_rows[-1]["instruction"] == "make it a table"
        new_rows = [row for row in _rows(session.transcript) if row["anchor"] == "anchor-new"]
        assert [row["state"] for row in new_rows] == ["decided", "done"], new_rows
    finally:
        release.set()
        await handle.dispose()


async def test_a_graphics_job_reports_live_while_it_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R2: ``is_live`` -- the runtime's half of the reader rule -- must be True mid-job.

    C1a left the ``_live_jobs`` registry as the C1b seam; the graphics path never added to
    it, so the answer was False for exactly the long-lived jobs the rule exists for.
    """
    session = _make_session(tmp_path / "sessions" / "live", tmp_path)
    handle = await _handle(session, tmp_path)
    started = asyncio.Event()
    release = asyncio.Event()

    async def complete(request: Any) -> tuple[str, Any]:
        started.set()
        await release.wait()
        return BLOCK, _Usage()

    monkeypatch.setattr(handle._supplements, "_complete", complete)
    try:
        handle._supplements._render("anchor-live", _decision(), DATASETS, "u", "a", _settings())
        await asyncio.wait_for(started.wait(), timeout=10)
        job = _rows(session.transcript)[0]["job"]
        assert handle._supplements.is_live(job) is True, "a running job must read live"
        release.set()
        await _drain(handle)
        assert handle._supplements.is_live(job) is False, "a settled job must not read live"
    finally:
        release.set()
        await handle.dispose()


async def test_the_wall_clock_bound_keeps_the_blocks_that_already_passed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R4, the report's repro: timeoutS=1; turn 1 passes one block (and asks to repair
    one), turn 2 hangs, the bound fires -- and the accepted block must SURVIVE on the row.

    Pre-fix the row was ``failed`` / ``bound:time`` with no components: the ``wait_for``
    cancelled the attempt and the accepted blocks lived only in its locals.
    """
    session = _make_session(tmp_path / "sessions" / "bound-time", tmp_path)
    handle = await _handle(session, tmp_path)
    calls: list[Any] = []

    async def complete(request: Any) -> tuple[str, Any]:
        calls.append(request)
        if len(calls) == 1:
            return BLOCK + "\n" + LYING_BLOCK, _Usage()
        # The repair turn never answers: the wall clock ends it. (The raise keeps the
        # declared return type honest; reaching it would mean the bound did not fire.)
        await asyncio.Event().wait()
        raise AssertionError("the wall clock must cut this turn before it answers")

    monkeypatch.setattr(handle._supplements, "_complete", complete)
    settings = policy.SupplementSettings.from_values({"supplements": {"timeoutS": 1}})
    assert settings.timeout_s == 1
    try:
        handle._supplements._render("anchor-bound", _decision(), DATASETS, "u", "a", settings)
        await _drain(handle)
        rows = _rows(session.transcript)
        assert [row["state"] for row in rows] == ["decided", "done"], rows
        done = rows[-1]
        assert done["components"], "the accepted block was lost to the bound"
        assert any("bound:time" in line for line in done.get("detail", [])), done
        assert done.get("error", "") == "", "a kept row must not read as a failure"
        assert reader_disposition(done, job_live=False) == "block"
    finally:
        await handle.dispose()


# --- the settle races and the per-anchor version rule (round-2 R2-3/R2-4; QA Q-1/Q-2) -------


async def test_cancel_that_races_a_settling_job_answers_already_finished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R2-3: a cancel arriving as the job settles must not rewrite the settled row.

    The gate runs before the op's first await; the job can write its terminal row inside
    the ``cancelling`` beat. Pre-fix the cancel then cut the settling task and wrote
    ``cancelled`` over ``done`` -- a cold reader flips the settled block to
    "cancelled - Retry". The receipt is the neutral one and the row stays ``done``.
    """
    session = _make_session(tmp_path / "sessions" / "settle-cancel", tmp_path)
    handle = await _handle(session, tmp_path)
    stub = StubRunner(handle._supplements, answers=[BLOCK])
    stub.install(monkeypatch)
    entered = asyncio.Event()
    release_beat = asyncio.Event()
    original_progress = handle._supplements._progress
    gate = {"first": True}

    async def hold_done_beat(details: Any, state: str, **kwargs: Any) -> None:
        if state == "done" and gate["first"]:
            gate["first"] = False
            entered.set()
            await release_beat.wait()
        await original_progress(details, state, **kwargs)

    monkeypatch.setattr(handle._supplements, "_progress", hold_done_beat)
    try:
        handle._supplements._render("anchor-sc", _decision(), DATASETS, "u", "a", _settings())
        await asyncio.wait_for(entered.wait(), timeout=10)
        # The terminal row is written (and ``inputs.details`` advanced); the task is stuck
        # mid-beat, so ``task.done()`` is still False -- the exact gate-blind spot.
        assert [row["state"] for row in _rows(session.transcript)] == ["decided", "done"]
        job = _rows(session.transcript)[0]["job"]
        receipt = await asyncio.wait_for(handle._supplements.cancel_job("anchor-sc", job), 10)
        assert receipt == "already finished"
        rows = _rows(session.transcript)
        assert [row["state"] for row in rows] == ["decided", "done"], rows
        release_beat.set()
        await _drain(handle)
        rows = _rows(session.transcript)
        assert [row["state"] for row in rows] == ["decided", "done"], rows
        assert reader_disposition(rows[-1], job_live=False) == "block"
    finally:
        release_beat.set()
        await handle.dispose()


async def test_restart_that_races_a_settling_job_skips_the_cut(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R2-3 for the restart shape: a job that settled inside the beat is NOT cut.

    The next version starts straight away; no ``cancelled`` row lands over the settled
    pair (pre-fix: the settling task was cut and ``cancelled`` was written at version 3,
    painting a settled block as cancelled for a moment).
    """
    session = _make_session(tmp_path / "sessions" / "settle-restart", tmp_path)
    handle = await _handle(session, tmp_path)
    stub = StubRunner(handle._supplements, answers=[BLOCK, BLOCK])
    stub.install(monkeypatch)
    entered = asyncio.Event()
    release_beat = asyncio.Event()
    original_progress = handle._supplements._progress
    gate = {"first": True}

    async def hold_done_beat(details: Any, state: str, **kwargs: Any) -> None:
        if state == "done" and gate["first"]:
            gate["first"] = False
            entered.set()
            await release_beat.wait()
        await original_progress(details, state, **kwargs)

    monkeypatch.setattr(handle._supplements, "_progress", hold_done_beat)
    try:
        handle._supplements._render("anchor-sr", _decision(), DATASETS, "u", "a", _settings())
        await asyncio.wait_for(entered.wait(), timeout=10)
        job = _rows(session.transcript)[0]["job"]
        receipt = await asyncio.wait_for(handle._supplements.restart_job("anchor-sr", job), 10)
        assert receipt == "restarting"
        release_beat.set()
        await _drain(handle)
        rows = _rows(session.transcript)
        states = [row["state"] for row in rows]
        assert states == ["decided", "done", "decided", "done"], rows
        assert "cancelled" not in states, rows
        assert [row["version"] for row in rows] == [1, 2, 3, 4], rows
    finally:
        release_beat.set()
        await handle.dispose()


async def test_a_restart_racing_a_paused_cancel_cannot_clobber_the_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R2-4: two overlapping ops must not clear or settle over each other.

    Probe: pause the cancel inside its ``cancelling`` beat, start a restart, then let the
    cancel resume. Pre-fix the restart had run to completion in the window: the resumed
    cancel cleared ``_task`` (None while the restarted job ran), a further cancel for that
    live job answered "already finished", and version 4 ended up with two rows. The ops
    serialise, so the restart waits; the observable end state must be coherent either way.
    """
    session = _make_session(tmp_path / "sessions" / "race", tmp_path)
    handle = await _handle(session, tmp_path)

    def resolve(settings: Any) -> generator.DesignModel:
        return generator.DesignModel(MODEL, "session", "test/m")

    monkeypatch.setattr(handle._supplements, "_resolve_model", resolve)
    started = asyncio.Event()
    release_job = asyncio.Event()

    async def complete(request: Any) -> tuple[str, Any]:
        started.set()
        await release_job.wait()
        return BLOCK, _Usage()

    monkeypatch.setattr(handle._supplements, "_complete", complete)
    entered = asyncio.Event()
    release_beat = asyncio.Event()
    original_progress = handle._supplements._progress
    gate = {"first": True}

    async def hold_cancelling_beat(details: Any, state: str, **kwargs: Any) -> None:
        if state == "cancelling" and gate["first"]:
            gate["first"] = False
            entered.set()
            await release_beat.wait()
        await original_progress(details, state, **kwargs)

    monkeypatch.setattr(handle._supplements, "_progress", hold_cancelling_beat)
    try:
        handle._supplements._render("anchor-race", _decision(), DATASETS, "u", "a", _settings())
        await asyncio.wait_for(started.wait(), timeout=10)
        job = _rows(session.transcript)[0]["job"]
        cancel = asyncio.create_task(handle._supplements.cancel_job("anchor-race", job))
        await asyncio.wait_for(entered.wait(), timeout=10)
        restart = asyncio.create_task(handle._supplements.restart_job("anchor-race", job))
        # Let the restart run to completion if it can: pre-fix it does, resuming the
        # interleave; post-fix it is waiting on the ops lock behind the paused cancel.
        try:
            await asyncio.wait_for(asyncio.shield(restart), timeout=1.0)
        except asyncio.TimeoutError:
            pass
        release_beat.set()
        assert await asyncio.wait_for(cancel, timeout=10) == "cancelled"
        assert await asyncio.wait_for(restart, timeout=10) == "restarting"
        await wait_for(lambda: handle._supplements.is_live(job), timeout=10)
        # The restarted version is live and coherently owned.
        assert handle._supplements._task is not None, "a live job's task was cleared"
        assert handle._supplements.running
        assert handle._supplements._running == ("anchor-race", job)
        # A further cancel for that live job cuts it, rather than answering a stale
        # "already finished" (the probe's exact symptom).
        again = await asyncio.wait_for(handle._supplements.cancel_job("anchor-race", job), 10)
        assert again == "cancelled"
        release_job.set()
        await _drain(handle)
        rows = _rows(session.transcript)
        versions = [row["version"] for row in rows]
        assert len(set(versions)) == len(versions), rows
        assert [row["state"] for row in rows] == [
            "decided",
            "cancelled",
            "decided",
            "cancelled",
        ], rows
    finally:
        release_beat.set()
        release_job.set()
        await handle.dispose()


# --- the in-flight terminal append (round-3 review R3-1) ------------------------------------
#
# The three reproductions round 3 ran on all three op sites, pinned: the op is delivered
# while the job's terminal append sits inside ``_commit``'s worker thread, so the ``done``
# row is durable and published while ``inputs.details`` still reads ``decided``. Pre-fix
# each op cut the settling task and rebuilt the version that row already held. Each test
# asserts the post-fix journal and receipt, and fails on the pre-fix source.


async def test_a_cancel_during_the_terminal_append_answers_already_finished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R3-1 repro A: the cancel lands while the terminal append is in the worker thread.

    Pre-fix: the cancel cut the settling task and settled from the stale pointer --
    [(1,decided),(2,done),(2,cancelled)], receipt ``cancelled``, the delivered block
    flipping to "cancelled - Retry". Post-fix: the append is part of the cancel's own
    write lock, the decision reads the journal, and the sequential answer applies --
    nothing to cut, receipt ``already finished``, rows [(1,decided),(2,done)].
    """
    session = _make_session(tmp_path / "sessions" / "r31-cancel", tmp_path)
    handle = await _handle(session, tmp_path)
    stub = StubRunner(handle._supplements, answers=[BLOCK])
    stub.install(monkeypatch)
    entered, release = _hold_terminal_write(monkeypatch, session.transcript)
    try:
        handle._supplements._render("anchor-r31c", _decision(), DATASETS, "u", "a", _settings())
        assert await asyncio.to_thread(entered.wait, 10), "the terminal append was never held"
        job = _rows(session.transcript)[0]["job"]
        cancel = asyncio.create_task(handle._supplements.cancel_job("anchor-r31c", job))
        for _ in range(20):  # let the op reach its suspension: the write lock, or the reap
            await asyncio.sleep(0)
        assert not cancel.done(), "the op finished without waiting on the held write"
        release.set()
        receipt = await asyncio.wait_for(cancel, 10)
        rows = _rows(session.transcript)
        states = [row["state"] for row in rows]
        versions = [row["version"] for row in rows]
        assert (receipt, states, versions) == (
            "already finished",
            ["decided", "done"],
            [1, 2],
        ), (receipt, states, versions)
        assert reader_disposition(rows[-1], job_live=False) == "block"
        await _drain(handle)
    finally:
        release.set()
        await handle.dispose()


async def test_a_restart_during_the_terminal_append_skips_the_cut(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R3-1 repro B: the restart lands while the terminal append is in the worker thread.

    Pre-fix: [(1,decided),(2,done),(2,cancelled),(3,decided),(4,done)] -- a duplicate
    version 2, then the next version. Post-fix: the restart reads the journal, does not
    cut the settled job and runs version 3 straight away -- [(1,decided),(2,done),
    (3,decided),(4,done)].
    """
    session = _make_session(tmp_path / "sessions" / "r31-restart", tmp_path)
    handle = await _handle(session, tmp_path)
    stub = StubRunner(handle._supplements, answers=[BLOCK, BLOCK])
    stub.install(monkeypatch)
    entered, release = _hold_terminal_write(monkeypatch, session.transcript)
    try:
        handle._supplements._render("anchor-r31r", _decision(), DATASETS, "u", "a", _settings())
        assert await asyncio.to_thread(entered.wait, 10), "the terminal append was never held"
        job = _rows(session.transcript)[0]["job"]
        restart = asyncio.create_task(handle._supplements.restart_job("anchor-r31r", job))
        for _ in range(20):  # let the op reach its suspension: the write lock, or the reap
            await asyncio.sleep(0)
        assert not restart.done(), "the op finished without waiting on the held write"
        release.set()
        receipt = await asyncio.wait_for(restart, 10)
        await _drain(handle)
        rows = _rows(session.transcript)
        states = [row["state"] for row in rows]
        versions = [row["version"] for row in rows]
        assert (receipt, states, versions) == (
            "restarting",
            ["decided", "done", "decided", "done"],
            [1, 2, 3, 4],
        ), (receipt, states, versions)
    finally:
        release.set()
        await handle.dispose()


async def test_a_dismiss_during_the_terminal_append_builds_on_the_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R3-1 repro C: the dismiss lands while the terminal append is in the worker thread.

    Pre-fix: [(1,decided),(2,done),(2,skipped)] -- the stale pointer rebuilt version 2
    over the delivered ``done``. Post-fix: the dismiss reads the journal inside the write
    lock and the skipped row follows it -- [(1,decided),(2,done),(3,skipped)], with both
    readers agreeing on the newest row.
    """
    session = _make_session(tmp_path / "sessions" / "r31-dismiss", tmp_path)
    handle = await _handle(session, tmp_path)
    stub = StubRunner(handle._supplements, answers=[BLOCK])
    stub.install(monkeypatch)
    entered, release = _hold_terminal_write(monkeypatch, session.transcript)
    try:
        handle._supplements._render("anchor-r31d", _decision(), DATASETS, "u", "a", _settings())
        assert await asyncio.to_thread(entered.wait, 10), "the terminal append was never held"
        dismiss = asyncio.create_task(handle._supplements.dismiss("anchor-r31d"))
        for _ in range(20):  # let the op reach its suspension: the write lock, or the reap
            await asyncio.sleep(0)
        assert not dismiss.done(), "the op finished without waiting on the held write"
        release.set()
        receipt = await asyncio.wait_for(dismiss, 10)
        await _drain(handle)
        rows = _rows(session.transcript)
        states = [row["state"] for row in rows]
        versions = [row["version"] for row in rows]
        runtime_newest = handle._supplements._newest_row("anchor-r31d")
        contract_newest = newest_per_anchor(rows)["anchor-r31d"]
        assert (receipt, states, versions) == (
            "dismissed",
            ["decided", "done", "skipped"],
            [1, 2, 3],
        ), (receipt, states, versions)
        assert runtime_newest is not None, rows
        assert runtime_newest["state"] == contract_newest["state"] == "skipped"
    finally:
        release.set()
        await handle.dispose()


async def test_cancel_then_dismiss_keeps_one_version_sequence_per_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q-1 (QA round 1): cancel-then-dismiss must not put two rows at one version.

    ``_settle`` used to leave ``inputs.details`` on the ``decided`` row, so the dismiss
    built ``next_version`` from a version the cancelled row already held -- two rows at
    version 2, and ``_newest_row`` (first-wins) and ``contract.newest_per_anchor``
    (later-wins) then disagreed on "newest".
    """
    session = _make_session(tmp_path / "sessions" / "cancel-dismiss", tmp_path)
    handle = await _handle(session, tmp_path)

    def resolve(settings: Any) -> generator.DesignModel:
        return generator.DesignModel(MODEL, "session", "test/m")

    monkeypatch.setattr(handle._supplements, "_resolve_model", resolve)
    started = asyncio.Event()
    release = asyncio.Event()

    async def complete(request: Any) -> tuple[str, Any]:
        started.set()
        await release.wait()
        return BLOCK, _Usage()

    monkeypatch.setattr(handle._supplements, "_complete", complete)
    try:
        handle._supplements._render("anchor-cd", _decision(), DATASETS, "u", "a", _settings())
        await asyncio.wait_for(started.wait(), timeout=10)
        job = _rows(session.transcript)[0]["job"]
        assert (
            await asyncio.wait_for(handle._supplements.cancel_job("anchor-cd", job), 10)
            == "cancelled"
        )
        assert await asyncio.wait_for(handle._supplements.dismiss("anchor-cd"), 10) == "dismissed"
        rows = _rows(session.transcript)
        assert [row["version"] for row in rows] == [1, 2, 3], rows
        assert [row["state"] for row in rows] == ["decided", "cancelled", "skipped"], rows
        # Both readers name the same newest row -- the dismiss, carrying ``dismissed``.
        runtime_newest = handle._supplements._newest_row("anchor-cd")
        agreed = newest_per_anchor(rows)["anchor-cd"]
        assert runtime_newest is not None
        assert runtime_newest["state"] == agreed["state"] == "skipped", (runtime_newest, agreed)
        assert runtime_newest.get("dismissed") is True and agreed.get("dismissed") is True
    finally:
        release.set()
        await handle.dispose()


async def test_a_second_job_ever_started_for_one_anchor_keeps_versions_monotone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q-2 (QA round 1): one anchor, one version sequence.

    A second job on an anchor (synthetic here: the render path is the only traffic, and a
    turn carries a fresh answer anchor) used to start at version 1 again -- rows [1, 2]
    under two job ids, and the two readers disagreeing on which is newest. The attempt
    re-reads the journal before its first write and sits past the rows the anchor holds.
    """
    session = _make_session(tmp_path / "sessions" / "second-job", tmp_path)
    handle = await _handle(session, tmp_path)
    stub = StubRunner(handle._supplements, answers=[BLOCK, BLOCK])
    stub.install(monkeypatch)
    try:
        handle._supplements._render("anchor-q2", _decision(), DATASETS, "u", "a", _settings())
        await _drain(handle)
        first = _rows(session.transcript)
        assert [row["version"] for row in first] == [1, 2], first
        handle._supplements._render("anchor-q2", _decision(), DATASETS, "u", "a", _settings())
        await _drain(handle)
        rows = _rows(session.transcript)
        assert [row["version"] for row in rows] == [1, 2, 3, 4], rows
        assert len({row["job"] for row in rows}) == 2, rows
        assert [row["state"] for row in rows] == ["decided", "done", "decided", "done"], rows
        runtime_newest = handle._supplements._newest_row("anchor-q2")
        agreed = newest_per_anchor(rows)["anchor-q2"]
        assert runtime_newest is not None
        assert runtime_newest["version"] == agreed["version"] == 4, (runtime_newest, agreed)
        assert runtime_newest["job"] == agreed["job"]
    finally:
        await handle.dispose()


async def test_the_runtime_reader_breaks_version_ties_like_the_contract(
    tmp_path: Path,
) -> None:
    """Q-1 (QA round 1): one tie-break for "newest", wherever it is read.

    The write paths keep versions unique per anchor now, but the reader rules must still
    agree when a journal carries a tie: ``contract.newest_per_anchor`` documents later-row-
    in-journal-order; ``_newest_row`` used ``max``'s first-wins and named the other row.
    """
    session = _make_session(tmp_path / "sessions" / "tie", tmp_path)
    handle = await _handle(session, tmp_path)
    try:
        earlier = dict(
            build_details(
                anchor="anchor-tie", job="j1", version=2, state="cancelled", decision=_decision()
            )
        )
        later = dict(
            build_details(
                anchor="anchor-tie", job="j1", version=2, state="skipped", decision=_decision()
            )
        )
        later["dismissed"] = True
        await append_row(session.transcript, earlier)
        await append_row(session.transcript, later)
        runtime_newest = handle._supplements._newest_row("anchor-tie")
        agreed = newest_per_anchor(_rows(session.transcript))["anchor-tie"]
        assert runtime_newest is not None
        assert runtime_newest["state"] == "skipped", runtime_newest
        assert agreed["state"] == runtime_newest["state"]
        assert agreed.get("dismissed") is True
    finally:
        await handle.dispose()


async def test_a_superseded_row_leaves_the_next_version_to_the_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q-1's supersede site, re-derived on the round-3 invariant: the journal decides.

    ``_close_superseded`` journals ``cancelled``/``superseded`` over the job's leftover
    non-terminal row, and the dismiss a surface sends next reads THAT row as its base, so
    its version follows it (3, not a rebuild of 2). Driven directly -- the full supersede
    wiring (a second turn) is test_runner.py's, and this pins the journal half.
    """
    session = _make_session(tmp_path / "sessions" / "supersede", tmp_path)
    handle = await _handle(session, tmp_path)

    def resolve(settings: Any) -> generator.DesignModel:
        return generator.DesignModel(MODEL, "session", "test/m")

    monkeypatch.setattr(handle._supplements, "_resolve_model", resolve)
    started = asyncio.Event()
    release = asyncio.Event()

    async def complete(request: Any) -> tuple[str, Any]:
        started.set()
        await release.wait()
        return BLOCK, _Usage()

    monkeypatch.setattr(handle._supplements, "_complete", complete)
    try:
        handle._supplements._render("anchor-sup", _decision(), DATASETS, "u", "a", _settings())
        await asyncio.wait_for(started.wait(), timeout=10)
        await handle._supplements._close_superseded()
        assert [row["state"] for row in _rows(session.transcript)] == [
            "decided",
            "cancelled",
        ], _rows(session.transcript)
        receipt = await asyncio.wait_for(handle._supplements.dismiss("anchor-sup"), 10)
        assert receipt == "dismissed"
        rows = _rows(session.transcript)
        assert [row["version"] for row in rows] == [1, 2, 3], rows
        assert [row["state"] for row in rows] == ["decided", "cancelled", "skipped"], rows
    finally:
        release.set()
        await handle.dispose()
