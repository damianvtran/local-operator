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
    reader_disposition,
)
from local_operator.supplements.decision import Decision
from local_operator.supplements.evidence import Dataset
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
