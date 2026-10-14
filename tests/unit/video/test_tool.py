"""The ``generate_video``/``video_status`` tools: gates, params, receipts.

The submission path is exercised with the real parameter surface and source
resolution; the job manager, the journal and the scheduler are stubs (what
the tools DO with them is the contract under test), and the walk never runs —
registration returns a fake id without executing the runner.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.artifacts import RungAvailability
from local_operator.harness.types import TextContent, ToolContext
from local_operator.tools import video_tool
from local_operator.video import VideoRoute
from local_operator.video import cascade as video_cascade
from local_operator.video import records as video_records


class _FakeJobs:
    """Minimal structural stand-in for ``JobManagerProtocol`` (runtime_checkable:
    the protocol members must EXIST for pydantic to accept it) plus the methods
    the video tool actually uses.

    Registered ids read back as LIVE jobs via ``get``/``list`` — the
    watcher-dedupe contract needs a second ``video_status`` call to SEE the
    watcher the first one registered.
    """

    def __init__(self) -> None:
        self.registered: list[dict[str, Any]] = []
        self.outputs: list[tuple[str, str]] = []
        self.live: dict[str, Any] = {}
        self._next = 0

    def get(self, job_id: str, *, registrant_id: str | None = None) -> Any:
        return self.live.get(job_id)

    def list(self, *, registrant_id: str | None = None) -> list[Any]:
        return list(self.live.values())

    async def cancel(self, job_id: str, *, registrant_id: str | None = None) -> bool:
        return False

    def register(self, *, type: str, label: str, run: Any, **_kwargs: Any) -> str:
        self._next += 1
        job_id = f"job-{self._next}"
        self.registered.append({"type": type, "label": label, "run": run, "id": job_id})
        self.live[job_id] = SimpleNamespace(id=job_id, status="running")
        return job_id

    def append_output(self, job_id: str, text: str) -> None:
        self.outputs.append((job_id, text))


class _FakeScheduler:
    def __init__(self) -> None:
        self.schedules: list[Any] = []

    async def update(self, rows: list[Any]) -> None:
        self.schedules = rows


def _resolution(*, radient: bool = True, fal: bool = True) -> video_cascade.VideoRouteResolution:
    rungs = (
        RungAvailability(
            route=str(VideoRoute.RADIENT),
            available=radient,
            reason="Signed in to Radient." if radient else "Not signed in to Radient.",
        ),
        RungAvailability(
            route=str(VideoRoute.FAL),
            available=fal,
            reason="A FAL key is stored." if fal else "No FAL key is stored.",
        ),
    )
    route = VideoRoute.RADIENT if radient else (VideoRoute.FAL if fal else VideoRoute.NONE)
    return video_cascade.VideoRouteResolution(route=route, reason="test", rungs=rungs)


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def test_builders_gate_on_jobs_and_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    # No job manager: neither tool exists.
    assert video_tool.build_generate_video_tool(ToolContext(cwd=".")) is None
    assert video_tool.build_video_status_tool(ToolContext(cwd=".")) is None

    monkeypatch.setattr(video_tool, "video_provider_reachable", lambda *a: False)
    ctx = ToolContext(cwd=".", jobs=_FakeJobs())  # type: ignore[arg-type]
    assert video_tool.build_generate_video_tool(ctx) is None
    assert video_tool.build_video_status_tool(ctx) is None

    monkeypatch.setattr(video_tool, "video_provider_reachable", lambda *a: True)
    tool = video_tool.build_generate_video_tool(ctx)
    assert tool is not None
    assert tool.name == "generate_video"
    assert tool.approval_tier == "write"
    assert tool.interruptible is False, "the call returns instantly; there is no wait to interrupt"
    assert len(tool.description.split(". ", 1)[0] + ".") <= 160
    status = video_tool.build_video_status_tool(ctx)
    assert status is not None and status.name == "video_status"
    assert status.approval_tier == "read"


# ---------------------------------------------------------------------------
# Source resolution (the frozen §2.2 rules)
# ---------------------------------------------------------------------------


def test_resolve_sources_one_source_only() -> None:
    params = video_tool.GenerateVideoParams.model_validate(
        {"prompt": "p", "source_image_path": "a.png", "source_video_path": "https://x/y.mp4"}
    )
    with pytest.raises(Exception, match="exactly one source"):
        video_tool._resolve_sources(params, ".")


def test_resolve_sources_video_rules() -> None:
    params = video_tool.GenerateVideoParams.model_validate(
        {"prompt": "p", "source_video_path": "http://x/y.mp4"}
    )
    with pytest.raises(Exception, match="https URL"):
        video_tool._resolve_sources(params, ".")
    params = video_tool.GenerateVideoParams.model_validate(
        {"prompt": "p", "source_video_attachment": "a" * 32}
    )
    with pytest.raises(Exception, match="not supported yet"):
        video_tool._resolve_sources(params, ".")
    params = video_tool.GenerateVideoParams.model_validate(
        {"prompt": "p", "source_video_path": "https://x/y.mp4"}
    )
    mode, kind, wire, rows = video_tool._resolve_sources(params, ".")
    assert (mode, kind, wire) == ("v2v", "video", "https://x/y.mp4")
    assert rows == [{"kind": "video", "url": "https://x/y.mp4"}]


def test_resolve_sources_image_uses_image_lane_loaders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_operator.tools import image_tool

    monkeypatch.setattr(
        image_tool, "_load_source_image", lambda raw, cwd: "data:image/png;base64,AAAA"
    )
    params = video_tool.GenerateVideoParams.model_validate(
        {"prompt": "p", "source_image_path": "pic.png"}
    )
    mode, kind, wire, rows = video_tool._resolve_sources(params, ".")
    assert (mode, kind) == ("i2v", "image")
    assert wire is not None and wire.startswith("data:image/png;base64,")
    # The record row carries the PATH, never the bytes.
    assert rows == [{"kind": "image", "path": "pic.png"}]


def test_source_image_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    # The byte arithmetic (base64 inflates by ~4/3) is what decides; shrink
    # the cap so the check is exercised without allocating 20+ MiB.
    monkeypatch.setattr(video_tool, "_SOURCE_IMAGE_MAX_BYTES", 8)
    with pytest.raises(Exception, match="source cap"):
        video_tool._check_source_image_cap("data:image/png;base64,AAAAAAAA")
    # Under the cap stays accepted.
    monkeypatch.setattr(video_tool, "_SOURCE_IMAGE_MAX_BYTES", 1024)
    video_tool._check_source_image_cap("data:image/png;base64,AAAA")


# ---------------------------------------------------------------------------
# The submit receipt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_generate_video_submits_and_reports(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    journal: list[dict[str, Any]] = []
    jobs = _FakeJobs()
    scheduler = _FakeScheduler()

    async def fake_resolution(*_a: Any, **_k: Any) -> Any:
        return _resolution()

    async def fake_journal(record: dict[str, Any]) -> None:
        journal.append(record)

    monkeypatch.setattr(video_tool, "resolve_video_route", fake_resolution)
    ctx = ToolContext(
        cwd=str(tmp_path),
        jobs=jobs,  # type: ignore[arg-type]
        journal_video_job=fake_journal,
        wake_scheduler=scheduler,  # type: ignore[arg-type]
    )
    result = await video_tool.execute_generate_video(
        "call-1",
        {"prompt": "paper boat", "duration_s": 8, "resolution": "720p"},
        None,
        None,
        ctx,
    )
    assert not result.is_error
    details = result.details or {}
    assert details["job_id"] == "job-1"
    assert details["stage"] == "queued"
    assert details["provider"] == "radient"
    assert details["cancel_hint"] == "jobs op='cancel', job_id='job-1'"
    assert details["cancel_support"] == "queued_only"
    assert details["estimated_cost_usd"] == pytest.approx(1.6)
    assert jobs.registered and jobs.registered[0]["type"] == "video"
    assert "8s" in jobs.registered[0]["label"] and "Radient" in jobs.registered[0]["label"]
    # The registered record went to the transcript door; the wake was armed.
    assert journal[0]["state"] == "registered"
    assert journal[0]["mode"] == "t2v"
    assert [row.id for row in scheduler.schedules] == ["video-check-job-1"]
    first_block = result.content[0]
    assert isinstance(first_block, TextContent)
    text = first_block.text
    assert "job `job-1`" in text and "cancel with" in text


@pytest.mark.asyncio
async def test_execute_generate_video_refuses_no_capable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def fake_resolution(*_a: Any, **_k: Any) -> Any:
        return _resolution(radient=True, fal=False)

    monkeypatch.setattr(video_tool, "resolve_video_route", fake_resolution)
    # The image loader is the IMAGE lane's own (one "readable image" rule);
    # patch it at its home so this test is about the capability refusal.
    import local_operator.tools.image_tool as image_tool

    monkeypatch.setattr(
        image_tool, "_load_source_image", lambda path, cwd: "data:image/png;base64,AAAA"
    )
    jobs = _FakeJobs()
    ctx = ToolContext(cwd=str(tmp_path), jobs=jobs)  # type: ignore[arg-type]
    result = await video_tool.execute_generate_video(
        "call-2",
        {"prompt": "p", "source_image_path": "pic.png"},
        None,
        None,
        ctx,
    )
    assert result.is_error
    refused = result.content[0]
    assert isinstance(refused, TextContent)
    assert "No available provider can generate a video from an image" in refused.text
    assert jobs.registered == []  # nothing was registered


def test_approval_describer_names_duration_and_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(video_tool, "_preferred_video_route", lambda mode: "radient")
    text = video_tool._describe_generate_video_approval(
        {"prompt": "p", "duration_s": 8, "resolution": "720p"}, "."
    )
    assert "Generate an 8s video" in text
    assert "Radient" in text
    assert "$1.60 at 720p" in text
    assert "paid provider call" in text
    # A v2v request is described as one (and priced from the FAL table).
    monkeypatch.setattr(video_tool, "_preferred_video_route", lambda mode: "fal")
    text = video_tool._describe_generate_video_approval(
        {"prompt": "p", "duration_s": 5, "source_video_path": "https://x/y.mp4"}, "."
    )
    assert "from a source video" in text
    assert "FAL" in text


# ---------------------------------------------------------------------------
# video_status
# ---------------------------------------------------------------------------


def _write_records(directory: Path, rows: list[dict[str, Any]]) -> None:
    """Hand-write the row shape ``Transcript.append_custom`` produces."""
    import json

    from local_operator.session import transcript as transcript_module

    lines = [
        json.dumps(
            {
                "id": f"r{index}",
                "ts": float(index),
                "type": transcript_module.ENTRY_CUSTOM,
                "payload": {
                    "custom_type": video_records.VIDEO_JOB_CUSTOM_TYPE,
                    "details": row,
                },
            },
            separators=(",", ":"),
        )
        for index, row in enumerate(rows)
    ]
    (directory / "transcript.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.mark.asyncio
async def test_video_status_terminal_record(tmp_path: Path) -> None:
    _write_records(
        tmp_path,
        [
            {
                "job": "jjj",
                "v": 1,
                "at": "2026-10-10T10:00:00+00:00",
                "state": "completed",
                "provider": "fal",
                "model": "m",
                "result": {
                    "digest": "d" * 32,
                    "model": "m",
                    "duration_s": 8.0,
                    "cost_usd": 1.6,
                    "cost_source": "reported",
                    "billing_basis": "billed",
                },
            }
        ],
    )
    ctx = ToolContext(
        cwd=str(tmp_path), session_dir=str(tmp_path), jobs=_FakeJobs()  # type: ignore[arg-type]
    )
    result = await video_tool.execute_video_status("call-3", {"job_id": "jjj"}, None, None, ctx)
    assert not result.is_error
    completed_block = result.content[0]
    assert isinstance(completed_block, TextContent)
    text = completed_block.text
    assert "completed" in text and "digest dddd" in text
    assert result.details and result.details["digest"] == "d" * 32


@pytest.mark.asyncio
async def test_video_status_unknown_and_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = ToolContext(
        cwd=str(tmp_path), session_dir=str(tmp_path), jobs=_FakeJobs()  # type: ignore[arg-type]
    )
    result = await video_tool.execute_video_status("call-4", {"job_id": "nope"}, None, None, ctx)
    # An unknown id is an ANSWER, not a fault: the record may have been swept,
    # and the model must not be pushed to retry an id that will never exist.
    assert not result.is_error
    unknown_block = result.content[0]
    assert isinstance(unknown_block, TextContent)
    assert "no video job" in unknown_block.text.lower()

    _write_records(
        tmp_path,
        [
            {
                "job": "ppp",
                "v": 1,
                "at": "2026-10-10T10:00:00+00:00",
                "state": "running",
                "provider": "fal",
                "model": "m",
                "provider_ref": {"request_id": "r"},
            },
        ],
    )

    import local_operator.video.rungs as video_rungs

    async def fake_poll_once(**_kwargs: Any) -> Any:
        # The provider is still working: the tool must hand the job to a
        # watcher rather than answer with a stale record alone.
        return video_rungs.VideoPollOutcome(state="running")

    async def fake_journal(record: dict[str, Any]) -> None:
        return None

    runner_seen: list[Any] = []

    def fake_watcher_runner(**kwargs: Any) -> Any:
        runner_seen.append(kwargs)
        return lambda *a, **k: None

    monkeypatch.setattr(video_tool, "poll_video_once", fake_poll_once)
    monkeypatch.setattr(video_tool, "make_video_watcher_runner", fake_watcher_runner)
    ctx2 = ToolContext(
        cwd=str(tmp_path),
        session_dir=str(tmp_path),
        jobs=_FakeJobs(),  # type: ignore[arg-type]
        journal_video_job=fake_journal,
    )
    result = await video_tool.execute_video_status("call-5", {}, None, None, ctx2)
    assert not result.is_error
    pending_block = result.content[0]
    assert isinstance(pending_block, TextContent)
    text = pending_block.text
    assert "still running" in text
    assert "observation resumed" in text
    assert runner_seen, "the watcher runner must be built for the pending record"


@pytest.mark.asyncio
async def test_video_status_no_id_on_a_record_free_session(tmp_path: Path) -> None:
    """QA round 3: a fresh session with NO records gets the honest absence,
    not `max()`'s ValueError — that traceback would reach the model as the
    tool's answer.

    Repro (one line): ``execute_video_status("call", {}, …)`` on an empty
    session dir used to raise ``ValueError: max() iterable argument is
    empty`` at the newest-record pick.
    """
    ctx = ToolContext(
        cwd=str(tmp_path), session_dir=str(tmp_path), jobs=_FakeJobs()  # type: ignore[arg-type]
    )
    result = await video_tool.execute_video_status("call-8", {}, None, None, ctx)
    assert not result.is_error
    no_job_block = result.content[0]
    assert isinstance(no_job_block, TextContent)
    assert no_job_block.text == "No video job is recorded for this session."
    assert result.details and result.details["job_id"] is None


@pytest.mark.asyncio
async def test_video_status_dedupes_the_watcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second ``video_status`` on the same pending record must NOT register
    a twin watcher (round-1 review V1): each watcher settles through the
    ordinary delivery path, so N registrations deliver N completion rows for
    one generation.

    The dedupe rides the record itself — the watcher's id is journaled on a
    revision — so the journal here APPENDS the row to the same transcript the
    scanner reads, exactly as ``Session.journal_video_job`` does.
    """
    import json

    from local_operator.session import transcript as transcript_module

    _write_records(
        tmp_path,
        [
            {
                "job": "www",
                "v": 1,
                "at": "2026-10-10T10:00:00+00:00",
                "state": "running",
                "provider": "fal",
                "model": "m",
                "mode": "t2v",
                "provider_ref": {"request_id": "r"},
            }
        ],
    )

    import local_operator.video.rungs as video_rungs

    async def fake_poll_once(**_kwargs: Any) -> Any:
        return video_rungs.VideoPollOutcome(state="running")

    stamped: list[dict[str, Any]] = []

    async def fake_journal(record: dict[str, Any]) -> None:
        stamped.append(record)
        path = tmp_path / "transcript.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        line = json.dumps(
            {
                "id": f"r{len(lines)}",
                "ts": float(len(lines)),
                "type": transcript_module.ENTRY_CUSTOM,
                "payload": {
                    "custom_type": video_records.VIDEO_JOB_CUSTOM_TYPE,
                    "details": record,
                },
            },
            separators=(",", ":"),
        )
        path.write_text("\n".join([*lines, line]) + "\n", encoding="utf-8")

    monkeypatch.setattr(video_tool, "poll_video_once", fake_poll_once)
    monkeypatch.setattr(
        video_tool, "make_video_watcher_runner", lambda **kw: (lambda *a, **k: None)
    )
    jobs = _FakeJobs()
    ctx = ToolContext(
        cwd=str(tmp_path),
        session_dir=str(tmp_path),
        jobs=jobs,  # type: ignore[arg-type]
        journal_video_job=fake_journal,
    )

    first = await video_tool.execute_video_status("call-6", {}, None, None, ctx)
    assert not first.is_error
    assert len(jobs.registered) == 1
    assert stamped and stamped[0].get("watcher") == "job-1"
    assert stamped[0].get("provider_ref") == {"request_id": "r"}

    second = await video_tool.execute_video_status("call-7", {}, None, None, ctx)
    assert not second.is_error
    assert len(jobs.registered) == 1, "a second status call registered a twin watcher"
    second_block = second.content[0]
    assert isinstance(second_block, TextContent)
    second_text = second_block.text
    assert "already resumed as job job-1" in second_text
    assert "jobs op='cancel', job_id='job-1'" in second_text


def test_terminal_text_conflict_branch_uses_the_frozen_sentence() -> None:
    """The record's conflict pair renders as the frozen "already finished"
    sentence, and a plain cancel renders the note as its own sentence — no
    ``(cancelled — ….).`` nesting (round-1 design D1/D2)."""
    from local_operator.video import jobs as video_jobs

    conflict = {
        "state": "cancelled",
        "error_type": "media_already_completed",
        "error": video_jobs._CANCEL_CONFLICT_SENTENCE,
        "outcome_note": (
            "Provider job not cancelled — it had already completed; its result was discarded."
        ),
    }
    assert video_tool._terminal_text(conflict, "jjj") == (
        "Video job jjj was cancelled. The generation had already completed when the "
        "cancel arrived; its result was discarded."
    )
    assert (
        video_tool._terminal_text(
            {"state": "cancelled", "outcome_note": "Provider job cancelled."}, "kkk"
        )
        == "Video job kkk was cancelled. Provider job cancelled."
    )
    assert video_tool._terminal_text({"state": "cancelled"}, "mmm") == (
        "Video job mmm was cancelled."
    )


@pytest.mark.asyncio
async def test_write_settle_carries_the_provider_ref_forward(tmp_path: Path) -> None:
    """V2 (the ``video_status`` half): a settle row must keep the record's
    ``provider_ref`` — the §1.5 refetch exists precisely for a terminal row
    that could not register, and it cannot poll without the ref."""
    journal: list[dict[str, Any]] = []

    async def fake_journal(record: dict[str, Any]) -> None:
        journal.append(record)

    ctx = ToolContext(
        cwd=str(tmp_path),
        session_dir=str(tmp_path),
        journal_video_job=fake_journal,
    )
    record = {
        "job": "sss",
        "state": "running",
        "provider": "fal",
        "model": "m",
        "provider_ref": {"request_id": "r"},
    }
    await video_tool._write_settle(ctx, "sss", record, state="failed", outcome_note="boom")
    assert journal and journal[0]["provider_ref"] == {"request_id": "r"}
    assert journal[0]["state"] == "failed"
    assert journal[0]["outcome_note"] == "boom"
