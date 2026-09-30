"""The record sink: an episode's product must survive the disk.

WHAT THESE TESTS PIN, and why each is here. Two measured episodes (2026-09-28)
completed their whole interaction -- one through the completion gate and
scored -- and sealed ``status: failed, steps: 0`` because a single record write
met ENOSPC mid-run; ``record_sink``'s module docstring carries the
measurements. Each test below pins one of the four properties that fix that,
plus the run-level regressions that would have caught it:

* a volume too small for a record refuses BEFORE ``launch`` (no spend);
* a failed write truncates back to the last complete line (a partial record
  stays usable);
* the reserve is spent only when the seal actually needs it (the record stops
  at 100% minus the margin, and the seal lands on top);
* the failure is classified out-of-room vs anything else (legible, so a full
  disk never again reads as "the run misbehaved");
* ``run_session_episode`` keeps the run's real status/steps/tool inventory and
  marks ``record_incomplete`` instead of replacing the outcome with the void.

Failure injection is at the syscall seam (``record_sink._CALLS``) because the
classification, the truncate-back and the reserve arithmetic ARE the units
under test; ``scripts/repro-enospc-record.py`` is the companion that drives a
real bounded APFS image to the same conclusion end to end.
"""

from __future__ import annotations

import errno
import json
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.evaluation import record_sink
from local_operator.evaluation.action_server import ACTION_TOOL_NAME, SERVER_NAME
from local_operator.evaluation.evidence.models import ScoreArtifact
from local_operator.evaluation.record_sink import (
    RECORD_EXPECTED_BYTES,
    SEAL_RESERVE_BYTES,
    SEAL_RESERVE_NAME,
    RecordSink,
    RecordSinkError,
)
from local_operator.evaluation.session_arm import run_session_episode
from local_operator.harness.types import ReasoningDeltaEvent
from local_operator.mcp.tool_bridge import create_mcp_tool_name
from local_operator.session.spec import ApprovalPolicy, SessionRoots, SessionSpec
from tests.unit.evaluation.runner.conftest import (
    FakeAdapter,
    build_config,
    build_spec,
    selector,
)

#: The name ``declare_action_server`` mints, computed the same way it does.
ACTION_TOOL = create_mcp_tool_name(SERVER_NAME, ACTION_TOOL_NAME)


def _free(monkeypatch: pytest.MonkeyPatch, free: int | None) -> None:
    """Pin the free-space reading the pre-flight performs.

    ``None`` makes the measurement raise, the unmeasurable case.
    """

    if free is None:

        def broken(_path: Any) -> Any:
            raise OSError(errno.EIO, "cannot measure")

        monkeypatch.setattr(record_sink.shutil, "disk_usage", broken)
        return
    monkeypatch.setattr(record_sink.shutil, "disk_usage", lambda _path: SimpleNamespace(free=free))


def _parse_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def short_scratch() -> Iterator[Path]:
    """A socket-capable scratch directory short enough for ``sun_path``.

    The action bridge binds a UNIX socket under the episode scratch, and its
    path is bounded (~104 bytes on macOS). Under xdist the ``tmp_path`` layer
    plus a descriptive test name goes over that bound (measured), so the
    run-level tests stage the session scratch here and keep the RECORD under
    ``tmp_path``; the same constraint the repro harness documents.
    """

    path = Path(tempfile.mkdtemp(prefix="lore-t-"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(autouse=True)
def _pin_a_healthy_volume(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the pre-flight's DEFAULT free-space reading to a healthy value.

    The pre-flight reads the REAL volume, and this fleet runs at ~97-100%; a
    test whose subject is a torn write must not go red because the host is
    nearly full (measured 2026-09-28: this file failed wholesale while the
    live campaign's disks were full, every failure a legitimate refusal). The
    tests that ASSERT the refusal set their own reading from the test body,
    which runs after this fixture and therefore wins.
    """

    _free(monkeypatch, 8 * RECORD_EXPECTED_BYTES)


class _TornWrite:
    """A ``_CALLS.write`` that writes half a line and then reports ENOSPC.

    The partial write is the point: a block-device write that runs out of room
    can land some bytes, and the sink's truncate-back is what keeps the file
    parseable through it.
    """

    def __init__(self, real: Any) -> None:
        self._real = real
        self.armed = False

    def __call__(self, fd: int, data: Any) -> int:
        if not self.armed:
            return self._real(fd, data)
        half = max(1, len(data) // 2)
        self._real(fd, bytes(data[:half]))
        raise OSError(errno.ENOSPC, "No space left on device")


class TestPreflight:
    def test_refuses_below_the_record_plus_reserve_budget(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _free(monkeypatch, RECORD_EXPECTED_BYTES + SEAL_RESERVE_BYTES - 1)
        root = tmp_path / "evidence"
        with pytest.raises(RecordSinkError) as raised:
            RecordSink(root / "events.jsonl")
        error = raised.value
        assert error.preflight is True
        assert error.out_of_room is True
        assert "refusing to start" in error.sentence
        assert str(RECORD_EXPECTED_BYTES) in error.sentence
        assert str(SEAL_RESERVE_BYTES) in error.sentence
        # The refusal leaves nothing behind to clean up: no reserve, no file.
        assert not (root / SEAL_RESERVE_NAME).exists()
        assert not (root / "events.jsonl").exists()

    def test_accepts_exactly_the_budget(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _free(monkeypatch, RECORD_EXPECTED_BYTES + SEAL_RESERVE_BYTES)
        sink = RecordSink(tmp_path / "events.jsonl")
        assert (tmp_path / SEAL_RESERVE_NAME).stat().st_size == SEAL_RESERVE_BYTES
        assert sink.failure is None
        sink.close()

    def test_an_unmeasurable_volume_proceeds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A failed stat is not evidence of a full disk; the write path still
        # classifies whatever it meets.
        _free(monkeypatch, None)
        sink = RecordSink(tmp_path / "events.jsonl")
        assert sink.failure is None
        sink.close()

    def test_a_negative_budget_is_refused(self, tmp_path: Path) -> None:
        # A negative expected/reserve would WEAKEN the floor the pre-flight
        # applies, so it is a caller error: a ValueError, not a refusal shape
        # the caller might try to reason about.
        with pytest.raises(ValueError):
            RecordSink(tmp_path / "events.jsonl", reserve_bytes=-1)

    def test_a_reserve_that_cannot_be_created_refuses(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _free(monkeypatch, RECORD_EXPECTED_BYTES + SEAL_RESERVE_BYTES + 1)
        real_open = Path.open

        def refusing_open(self: Path, *args: Any, **kwargs: Any) -> Any:
            if self.name == SEAL_RESERVE_NAME:
                raise OSError(errno.ENOSPC, "No space left on device")
            return real_open(self, *args, **kwargs)

        monkeypatch.setattr(Path, "open", refusing_open)
        with pytest.raises(RecordSinkError, match="refusing to start"):
            RecordSink(tmp_path / "events.jsonl")


class TestReserve:
    def test_preallocated_released_once_then_idempotent(self, tmp_path: Path) -> None:
        sink = RecordSink(tmp_path / "events.jsonl")
        reserve = tmp_path / SEAL_RESERVE_NAME
        assert reserve.stat().st_size == SEAL_RESERVE_BYTES
        assert sink.release_reserve() is True
        assert not reserve.exists()
        assert sink.release_reserve() is False
        sink.close()

    def test_close_keeps_the_reserve_for_the_seal(self, tmp_path: Path) -> None:
        sink = RecordSink(tmp_path / "events.jsonl")
        sink.close()
        assert (tmp_path / SEAL_RESERVE_NAME).exists()

    def test_a_zero_reserve_configures_no_file(self, tmp_path: Path) -> None:
        sink = RecordSink(tmp_path / "events.jsonl", reserve_bytes=0)
        assert not (tmp_path / SEAL_RESERVE_NAME).exists()
        assert sink.release_reserve() is False
        sink.close()


class TestWrites:
    def test_a_torn_write_is_truncated_back_to_the_last_complete_line(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sink = RecordSink(tmp_path / "events.jsonl")
        sink.write("a", {"n": 1})
        sink.write("b", {"n": 2})
        complete = sink.bytes_written

        torn = _TornWrite(record_sink._CALLS.write)
        monkeypatch.setattr(record_sink._CALLS, "write", torn)
        torn.armed = True
        sink.write("c", {"n": 3})

        # The torn line is gone: every byte on disk parses, and the sink's own
        # byte count matches the file exactly.
        lines = _parse_lines(tmp_path / "events.jsonl")
        assert [line["kind"] for line in lines] == ["a", "b"]
        assert (tmp_path / "events.jsonl").stat().st_size == sink.bytes_written == complete
        assert sink.failure is not None
        assert sink.failure.out_of_room is True
        assert "ran out of room" in sink.failure.sentence
        assert "ENOSPC" in sink.failure.sentence

        # The volume coming back (a neighbour deleting files, the reserve being
        # freed later) lets the SAME sink continue on the same line stream.
        torn.armed = False
        sink.write("d", {"n": 4})
        assert [line["kind"] for line in _parse_lines(tmp_path / "events.jsonl")] == [
            "a",
            "b",
            "d",
        ]
        sink.close()

    def test_a_failed_truncate_is_named_in_the_failure_sentence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Review round 1 (R1-N2): the truncate-back is best effort, and a dead
        # volume can refuse the fixup itself. That case is not silent -- the
        # failure sentence says the file ends in a torn fragment, because a
        # reader cannot tell a cut that happened from one that did not.
        sink = RecordSink(tmp_path / "events.jsonl")
        sink.write("a", {"n": 1})
        complete = sink.bytes_written

        torn = _TornWrite(record_sink._CALLS.write)
        monkeypatch.setattr(record_sink._CALLS, "write", torn)

        def denying_truncate(_fd: int, _size: int) -> None:
            raise OSError(errno.EIO, "the fixup met a dead volume")

        monkeypatch.setattr(record_sink._CALLS, "ftruncate", denying_truncate)
        torn.armed = True
        sink.write("b", {"n": 2})

        assert sink.failure is not None
        assert "ran out of room" in sink.failure.sentence
        assert "torn fragment" in sink.failure.sentence
        # The half-written tail is still on disk: the sentence is the only
        # thing that can carry the fact on this volume.
        assert (tmp_path / "events.jsonl").stat().st_size > complete

    def test_a_non_space_failure_is_not_classified_as_out_of_room(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sink = RecordSink(tmp_path / "events.jsonl")

        def denying(fd: int, data: Any) -> int:
            raise OSError(errno.EACCES, "Permission denied")

        monkeypatch.setattr(record_sink._CALLS, "write", denying)
        sink.write("a", {"n": 1})
        assert sink.failure is not None
        assert sink.failure.out_of_room is False
        assert "Permission denied" in sink.failure.sentence
        assert "ran out of room" not in sink.failure.sentence
        sink.close()

    def test_only_the_first_failure_is_kept_and_counted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sink = RecordSink(tmp_path / "events.jsonl")
        failures = iter(
            [
                OSError(errno.ENOSPC, "No space left on device"),
                OSError(errno.EIO, "Input/output error"),
            ]
        )

        def failing(fd: int, data: Any) -> int:
            del fd, data
            raise next(failures)

        monkeypatch.setattr(record_sink._CALLS, "write", failing)
        sink.write("a", {"n": 1})
        sink.write("b", {"n": 2})
        assert sink.failures == 2
        first = sink.failure
        assert first is not None
        assert first.out_of_room is True
        assert "ENOSPC" in first.sentence
        assert sink.bytes_written == 0

    def test_writing_after_close_raises(self, tmp_path: Path) -> None:
        sink = RecordSink(tmp_path / "events.jsonl")
        sink.close()
        with pytest.raises(RecordSinkError, match="closed"):
            sink.write("a", {"n": 1})


class TestSealWrite:
    def test_does_not_spend_the_reserve_when_the_write_lands(self, tmp_path: Path) -> None:
        sink = RecordSink(tmp_path / "events.jsonl")
        sink.seal_write(tmp_path / "score.json", '{"a": 1}\n')
        assert (tmp_path / "score.json").read_text(encoding="utf-8") == '{"a": 1}\n'
        assert (tmp_path / SEAL_RESERVE_NAME).exists()
        sink.release_reserve()
        sink.close()

    def test_spends_the_reserve_when_the_volume_is_full(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sink = RecordSink(tmp_path / "events.jsonl")
        # Everything is full until the reserve is released; the release is what
        # makes the retry land. This is the exact conversion the reserve exists
        # for.
        state = {"full": True}
        real_write = record_sink._CALLS.write
        real_unlink = record_sink._CALLS.unlink

        def write(fd: int, data: Any) -> int:
            if state["full"]:
                raise OSError(errno.ENOSPC, "No space left on device")
            return real_write(fd, data)

        def unlink(path: str) -> None:
            real_unlink(path)
            if path.endswith(SEAL_RESERVE_NAME):
                state["full"] = False

        monkeypatch.setattr(record_sink._CALLS, "write", write)
        monkeypatch.setattr(record_sink._CALLS, "unlink", unlink)
        sink.seal_write(tmp_path / "outcome.json", '{"outcome": true}\n')
        assert (tmp_path / "outcome.json").read_text(encoding="utf-8") == '{"outcome": true}\n'
        assert not (tmp_path / SEAL_RESERVE_NAME).exists()
        sink.close()

    def test_raises_when_the_released_reserve_is_still_not_enough(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sink = RecordSink(tmp_path / "events.jsonl")
        sink.release_reserve()
        real_write = record_sink._CALLS.write

        def write(fd: int, data: Any) -> int:
            del fd, data
            raise OSError(errno.ENOSPC, "No space left on device")

        monkeypatch.setattr(record_sink._CALLS, "write", write)
        with pytest.raises(RecordSinkError) as raised:
            sink.seal_write(tmp_path / "outcome.json", "{}\n")
        assert raised.value.out_of_room is True
        assert "already spent" in raised.value.sentence
        monkeypatch.setattr(record_sink._CALLS, "write", real_write)
        sink.close()

    def test_a_non_space_artifact_failure_raises_classified(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sink = RecordSink(tmp_path / "events.jsonl")

        def write(fd: int, data: Any) -> int:
            del fd, data
            raise OSError(errno.EIO, "Input/output error")

        monkeypatch.setattr(record_sink._CALLS, "write", write)
        with pytest.raises(RecordSinkError) as raised:
            sink.seal_write(tmp_path / "outcome.json", "{}\n")
        assert raised.value.out_of_room is False
        assert "Input/output error" in raised.value.sentence
        sink.release_reserve()
        sink.close()


# --------------------------------------------------------------------------
# The run-level regressions: what the arm does when the record tears.
# --------------------------------------------------------------------------


class _RecordingSession:
    """The session surface ``run_session_episode`` drives, with events."""

    def __init__(self, events: list[Any]) -> None:
        self._tools = [SimpleNamespace(name=ACTION_TOOL)]
        self.mcp_startup = None
        self.mcp_manager = None
        self.aborted: str | None = None
        self._sinks: list[Any] = []
        self._events = events

    def subscribe(self, sink: Any) -> Any:
        self._sinks.append(sink)
        return lambda: None

    def set_tool_confinement(self, root: Any) -> None:
        del root

    async def prompt(self, text: str, *, images: Any = None) -> None:
        del text, images
        for event in self._events:
            for sink in self._sinks:
                sink(event)

    async def abort(self, reason: str | None = None) -> None:
        self.aborted = reason


class _SessionContext:
    def __init__(self, session: _RecordingSession) -> None:
        self._session = session

    async def __aenter__(self) -> Any:
        return self._session

    async def __aexit__(self, *exc: Any) -> bool:
        del exc
        return False


class _SessionFakeAdapter(FakeAdapter):
    """FakeAdapter with the one capability the session arm requires."""

    async def handshake(self, *, timeout: float = 10.0) -> Any:
        base = await super().handshake(timeout=timeout)
        capabilities = base.metadata.capabilities.model_copy(
            update={"ask_user_answer_owner": "adapter"}
        )
        return base.model_copy(
            update={"metadata": base.metadata.model_copy(update={"capabilities": capabilities})}
        )


def _spec_and_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scratch_root: Path
) -> tuple[Any, Any, SessionRoots]:
    home = scratch_root / "home"
    config = home / ".local-operator"
    agent_home = home / "local-operator-home"
    work = home / "work"
    for path in (config, agent_home, work):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    monkeypatch.setenv("LOCAL_OPERATOR_HOME", str(agent_home))
    roots = SessionRoots(config_dir=config, agent_home=agent_home, cwd=work, allow_volatile=True)
    spec = build_spec("ep-record")
    config_obj = build_config(tmp_path)
    return spec, config_obj, roots


def _runner(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scratch_root: Path,
    events: list[Any],
    launched: list[Any] | None = None,
) -> tuple[Any, _RecordingSession, Any]:
    spec, episode_config, roots = _spec_and_roots(tmp_path, monkeypatch, scratch_root)
    adapter = _SessionFakeAdapter(
        tmp_path, spec.episode_id, score=ScoreArtifact(status="scored", binary=1)
    )
    session = _RecordingSession(events)
    session_spec = SessionSpec(
        hosting="test", model="mock", approvals=ApprovalPolicy.auto(), name="arm-record"
    )

    async def rescue(descriptor: Any, **kwargs: Any) -> Any:
        del descriptor, kwargs
        return SimpleNamespace(complete=True, receipts=(), rescue_required=False)

    def launch(selected: Any) -> Any:
        if launched is not None:
            launched.append(selected)
        return adapter

    def opener(selected: Any, *, roots: Any, mode: Any = None) -> Any:
        # The signature ``open_episode_session`` calls: the spec positionally,
        # ``roots``/``mode`` as keywords. It returns the CONTEXT only -- the
        # subscription and the ``EpisodeSession`` wrapper are the real
        # function's job, and a fake that did them too would stop exercising it.
        del selected, roots, mode
        return _SessionContext(session)

    async def run() -> Any:
        return await run_session_episode(
            spec=spec,
            config=episode_config,
            selector=selector(tmp_path),
            roots=roots,
            scratch_root=scratch_root,
            session_spec=session_spec,
            secrets=(),
            launch=launch,
            rescue=rescue,
            session_opener=opener,
        )

    return run, session, spec


class TestRunRefusesBeforeLaunch:
    @pytest.mark.asyncio
    async def test_a_hopeless_volume_never_reaches_launch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, short_scratch: Path
    ) -> None:
        _free(monkeypatch, RECORD_EXPECTED_BYTES)  # one byte short of the budget
        launched: list[Any] = []
        run, _, spec = _runner(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            scratch_root=short_scratch,
            events=[],
            launched=launched,
        )
        outcome = await run()
        assert launched == [], "the sink must refuse before anything is launched"
        assert outcome.status == "failed_pre_bundle"
        # The refusal NAMES the phase it died in, for the record's reader.
        assert outcome.terminal_reason == "record-sink"
        assert outcome.record_incomplete is False
        assert outcome.diagnostic is not None
        assert outcome.diagnostic.startswith("refusing to start")


class TestTornRecordKeepsTheOutcome:
    @pytest.mark.asyncio
    async def test_the_run_keeps_its_status_and_the_record_says_what_it_lost(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, short_scratch: Path
    ) -> None:
        # Enough events to exhaust the volume mid-run; each is a realistic
        # agent-event line (~4 KB of reasoning delta, the shape that dominated
        # the measured 65-80 MB records).
        events = [
            ReasoningDeltaEvent(message_id=f"m{index}", delta="r" * 4000) for index in range(40)
        ]
        run, session, spec = _runner(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            scratch_root=short_scratch,
            events=events,
        )

        budget = {"left": 20 * 4200}  # space for about half the stream
        real_write = record_sink._CALLS.write
        real_unlink = record_sink._CALLS.unlink

        def write(fd: int, data: Any) -> int:
            if budget["left"] <= 0:
                half = max(1, len(data) // 2)
                real_write(fd, bytes(data[:half]))
                raise OSError(errno.ENOSPC, "No space left on device")
            budget["left"] -= len(data)
            return real_write(fd, data)

        def unlink(path: str) -> None:
            real_unlink(path)
            if path.endswith(SEAL_RESERVE_NAME):
                budget["left"] += SEAL_RESERVE_BYTES  # the reserve is real room

        monkeypatch.setattr(record_sink._CALLS, "write", write)
        monkeypatch.setattr(record_sink._CALLS, "unlink", unlink)

        outcome = await run()
        monkeypatch.setattr(record_sink._CALLS, "write", real_write)

        # The old defect, pinned: this is the run that used to read
        # ``failed / steps: 0 / tool_names: []`` with `score: null`.
        assert outcome.status == "agent_stop", outcome.diagnostic
        assert outcome.record_incomplete is True
        assert outcome.record_diagnostic is not None
        assert "ran out of room" in outcome.record_diagnostic
        assert len(outcome.tool_names) >= 1
        assert outcome.score is not None

        record_root = outcome.record_root
        assert record_root is not None
        # The record on disk is complete through its last finished line, and
        # the seal artifacts all landed: score, the terminal marker, outcome.
        lines = _parse_lines(record_root / "events.jsonl")
        assert lines[-1]["kind"] == "record_incomplete"
        assert lines[-1]["out_of_room"] is True
        assert (record_root / "score.json").exists()
        written_outcome = json.loads((record_root / "outcome.json").read_text(encoding="utf-8"))
        assert written_outcome["status"] == "agent_stop"
        assert written_outcome["record_incomplete"] is True
        assert written_outcome["record_diagnostic"].startswith("the volume holding")
        # The reserve is released at the end no matter how the seal went.
        assert not (record_root / SEAL_RESERVE_NAME).exists()

    @pytest.mark.asyncio
    async def test_a_clean_run_is_not_marked_incomplete(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, short_scratch: Path
    ) -> None:
        events = [ReasoningDeltaEvent(message_id="m0", delta="short")]
        run, _, _ = _runner(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            scratch_root=short_scratch,
            events=events,
        )
        outcome = await run()
        assert outcome.status == "agent_stop", outcome.diagnostic
        assert outcome.record_incomplete is False
        assert outcome.record_diagnostic is None
        record_root = outcome.record_root
        assert record_root is not None
        written_outcome = json.loads((record_root / "outcome.json").read_text(encoding="utf-8"))
        assert written_outcome["record_incomplete"] is False
        assert not (record_root / SEAL_RESERVE_NAME).exists()
        kinds = [line["kind"] for line in _parse_lines(record_root / "events.jsonl")]
        assert "record_incomplete" not in kinds
        assert "agent_event" in kinds

    @pytest.mark.asyncio
    async def test_a_failed_outcome_seal_still_reports_the_record_incomplete(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, short_scratch: Path
    ) -> None:
        # Review round 1 (R1-F3): the disk copy of the summary is part of the
        # archive ``record_incomplete`` describes, so a failed OUTCOME seal must
        # reach the returned outcome rather than stopping at stderr. Injected at
        # the arm's own call site -- the sink's ENOSPC ladder has its unit tests
        # above -- and only for the outcome target, so the score seal still
        # exercises the real path.
        run, _, _ = _runner(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            scratch_root=short_scratch,
            events=[ReasoningDeltaEvent(message_id="m0", delta="short")],
        )

        real_seal = RecordSink.seal_write

        def refusing_seal(self: RecordSink, target: Path, text: str) -> None:
            if Path(target).name == "outcome.json":
                raise RecordSinkError(
                    f"the seal artifact {target} could not be written: the volume "
                    "holding it ran out of room (ENOSPC) with the seal reserve "
                    "already spent",
                    path=Path(target),
                    out_of_room=True,
                )
            real_seal(self, target, text)

        monkeypatch.setattr(RecordSink, "seal_write", refusing_seal)
        outcome = await run()

        assert outcome.status == "agent_stop", outcome.diagnostic
        assert outcome.record_incomplete is True
        assert outcome.record_diagnostic is not None
        assert "ran out of room" in outcome.record_diagnostic
        record_root = outcome.record_root
        assert record_root is not None
        # The OTHER seal artifact still landed, the disk copy of the outcome is
        # the thing that did not, and the reserve is released either way -- all
        # three facts the outcome above now carries instead of hiding.
        assert (record_root / "score.json").exists()
        assert not (record_root / "outcome.json").exists()
        assert not (record_root / SEAL_RESERVE_NAME).exists()
