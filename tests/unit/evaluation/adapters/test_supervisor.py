from __future__ import annotations

import hashlib
import os
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.evaluation.adapters.api import (
    AckResult,
    AdapterCapabilities,
    AdapterMetadata,
    AdapterSelector,
    AskUserExchangeParams,
    AskUserExchangeResult,
    CleanupResult,
    Handshake,
    PythonRuntime,
    RescueDescriptor,
    observation_content_id,
)
from local_operator.evaluation.adapters.supervisor import (
    MAX_DIAGNOSTIC_TAIL,
    HostVerifier,
    SupervisionError,
    _Tail,
    _terminate_process_group,
    load_pending_rescue,
    minimal_environment,
    persist_rescue,
    run_rescue,
    verify_artifact,
)
from local_operator.evaluation.lifecycle import (
    CleanupAction,
    CleanupPlan,
    record_cleanup,
)
from local_operator.evaluation.protocol import ArtifactRef, Observation


def selector(tmp_path: Path) -> AdapterSelector:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return AdapterSelector(
        schema_version="1.0",
        adapter_id="tiny",
        distribution="tiny-adapter",
        version="1.0",
        entry_point="tiny_adapter:create",
        package_digest="a" * 64,
        release_digest="b" * 64,
        python_executable=str(Path(sys.executable).resolve()),
        workspace=str(workspace),
        route_capability="computer",
    )


def metadata() -> AdapterMetadata:
    return AdapterMetadata(
        adapter_id="tiny",
        distribution="tiny-adapter",
        version="1.0",
        entry_point="tiny_adapter:create",
        package_digest="a" * 64,
        release_digest="b" * 64,
        schema_version="1.0",
        capabilities=AdapterCapabilities(routes=("computer",), ask_user=False, scoring=True),
    )


def handshake(tmp_path: Path) -> Handshake:
    return Handshake(
        selector=selector(tmp_path),
        metadata=metadata(),
        python=PythonRuntime.current(),
        workspace_digest="c" * 64,
        selected_route="computer",
    )


def plan() -> CleanupPlan:
    return CleanupPlan(
        episode_id="episode",
        actions=(
            CleanupAction(
                action_id="release",
                kind="release_instance",
                resource_ref="resource",
                timeout_ms=100,
                max_attempts=2,
            ),
        ),
    )


def descriptor(tmp_path: Path) -> RescueDescriptor:
    return RescueDescriptor(
        schema_version="1.0",
        selector=selector(tmp_path),
        handshake=handshake(tmp_path),
        episode_id="episode",
        cleanup_plan=plan(),
        secret_refs=(),
        infra_values=(),
        artifact_root=str(tmp_path),
    )


def observation(
    task_id: str, episode_id: str, sequence: int, *, text: str = "state"
) -> Observation:
    provisional = Observation(
        task_id=task_id,
        episode_id=episode_id,
        sequence=sequence,
        observation_id="provisional",
        text=text,
    )
    return provisional.model_copy(update={"observation_id": observation_content_id(provisional)})


def test_host_verifier_rejects_cross_episode_without_mutating_current(tmp_path: Path) -> None:
    verifier = HostVerifier("task", "episode", tmp_path)
    initial = observation("task", "episode", 0)
    verifier.accept_initial(initial)
    with pytest.raises(SupervisionError, match="another task or episode"):
        verifier.accept_output(observation("other", "episode", 1))
    assert verifier.current_observation == initial
    with pytest.raises(SupervisionError, match="another task or episode"):
        verifier.accept_output(observation("task", "other", 1))
    assert verifier.current_observation == initial


def test_host_verifier_enforces_initial_next_and_snapshot_modes(tmp_path: Path) -> None:
    verifier = HostVerifier("task", "episode", tmp_path)
    with pytest.raises(SupervisionError, match="sequence zero"):
        verifier.accept_initial(observation("task", "episode", 99))
    assert verifier.current_observation is None
    initial = observation("task", "episode", 0)
    verifier.accept_initial(initial)
    for sequence in (0, 2, 99):
        with pytest.raises(SupervisionError, match="exact next"):
            verifier.accept_output(observation("task", "episode", sequence))
        assert verifier.current_observation == initial
    verifier.verify_current_snapshot(initial)
    with pytest.raises(SupervisionError, match="snapshot differs"):
        verifier.verify_current_snapshot(observation("task", "episode", 0, text="changed content"))
    assert verifier.current_observation == initial
    next_observation = observation("task", "episode", 1)
    verifier.accept_output(next_observation)
    assert verifier.current_observation == next_observation


def test_ask_exchange_requires_expected_episode_and_preserves_pending_on_error(
    tmp_path: Path,
) -> None:
    verifier = HostVerifier("task", "episode", tmp_path)
    wrong_begin = AskUserExchangeParams(
        operation_id="ask-wrong", episode_id="other", ask_id="ask", prompt="Question?"
    )
    with pytest.raises(SupervisionError, match="another episode"):
        verifier.begin_ask(wrong_begin)
    begin = AskUserExchangeParams(
        operation_id="ask-begin", episode_id="episode", ask_id="ask", prompt="Question?"
    )
    verifier.begin_ask(begin)
    wrong_finish = AskUserExchangeParams(
        operation_id="ask-finish-wrong",
        episode_id="other",
        ask_id="ask",
        prompt="Question?",
        answer="Answer",
    )
    result = AskUserExchangeResult(ask_id="ask", accepted=True)
    with pytest.raises(SupervisionError, match="stale, unsolicited, or mismatched"):
        verifier.finish_ask(wrong_finish, result)
    # The failed finish must leave the original exchange outstanding.
    with pytest.raises(SupervisionError, match="begin once"):
        verifier.begin_ask(begin.model_copy(update={"operation_id": "again"}))
    correct_finish = wrong_finish.model_copy(
        update={"operation_id": "ask-finish", "episode_id": "episode"}
    )
    verifier.finish_ask(correct_finish, result)
    verifier.begin_ask(begin.model_copy(update={"operation_id": "next", "ask_id": "next"}))


def test_environment_is_constructed_without_parent_secrets() -> None:
    environment = minimal_environment(
        {
            "LANG": "en_US.UTF-8",
            "TMPDIR": "/tmp",
            "PATH": "/secret/path",
            "OPENAI_API_KEY": "marker-openai",
            "OPENROUTER_API_KEY": "marker-openrouter",
            "MODEL_TOKEN": "marker-model",
        },
        protocol_fds={
            "LO_ADAPTER_OWNER_FD": 3,
            "LO_ADAPTER_REQUEST_FD": 4,
            "LO_ADAPTER_RESPONSE_FD": 5,
        },
    )
    assert environment == {
        "LANG": "en_US.UTF-8",
        "TMPDIR": "/tmp",
        "LO_ADAPTER_OWNER_FD": "3",
        "LO_ADAPTER_REQUEST_FD": "4",
        "LO_ADAPTER_RESPONSE_FD": "5",
    }
    assert not any("marker" in value for value in environment.values())


def test_diagnostic_tail_is_bounded() -> None:
    tail = _Tail()
    tail.append(b"x" * (MAX_DIAGNOSTIC_TAIL + 10))
    assert tail.bytes() == b"x" * MAX_DIAGNOSTIC_TAIL


def test_persist_and_load_rescue_is_atomic_mode_0600(tmp_path: Path) -> None:
    rescue = descriptor(tmp_path)
    root = tmp_path / "pending"
    path = persist_rescue(root, rescue)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert load_pending_rescue(root) == rescue
    assert list(root.glob(".rescue-*")) == []


def test_load_pending_rescue_does_not_scan_and_rejects_symlink(tmp_path: Path) -> None:
    assert load_pending_rescue(tmp_path / "missing") is None
    target = tmp_path / "outside"
    target.write_text("{}")
    root = tmp_path / "pending"
    root.mkdir()
    (root / "rescue.json").symlink_to(target)
    with pytest.raises(SupervisionError, match="unsafe"):
        load_pending_rescue(root)


def test_artifact_verification_rejects_symlink_and_digest_attack(tmp_path: Path) -> None:
    data = b'{"ok":true}'
    digest = hashlib.sha256(data).hexdigest()
    reference = ArtifactRef(sha256=digest, media_type="application/json", byte_count=len(data))
    (tmp_path / digest).write_bytes(data)
    assert verify_artifact(tmp_path, reference) == data
    (tmp_path / digest).unlink()
    outside = tmp_path / "outside"
    outside.write_bytes(data)
    (tmp_path / digest).symlink_to(outside)
    with pytest.raises(SupervisionError, match="unsafe"):
        verify_artifact(tmp_path, reference)


def test_pid_mismatch_refuses_to_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session=True,
    )
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "getpgid", lambda _: process.pid + 1)
    monkeypatch.setattr(os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    try:
        with pytest.raises(SupervisionError, match="refusing"):
            _terminate_process_group(process, process.pid)
        assert killed == []
    finally:
        process.kill()
        process.wait()


def test_process_group_termination_kills_stubborn_grandchild(tmp_path: Path) -> None:
    pid_file = tmp_path / "grandchild.pid"
    script = (
        "import os,signal,subprocess,sys,time;"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        "p=subprocess.Popen([sys.executable,'-c',"
        "'import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)']);"
        "open(sys.argv[1],'w').write(str(p.pid));time.sleep(30)"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(pid_file)], start_new_session=True
    )
    deadline = time.monotonic() + 5
    while not pid_file.exists() and time.monotonic() < deadline:
        time.sleep(0.001)
    assert pid_file.exists()
    _terminate_process_group(process, process.pid)
    assert process.returncode == -signal.SIGKILL
    grandchild = int(pid_file.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(grandchild, 0)


class FakeSupervisor:
    def __init__(self, expected: RescueDescriptor, *, duplicate: bool = False) -> None:
        self.expected = expected
        self.duplicate = duplicate
        self.terminated = False
        self.methods: list[str] = []

    async def handshake(self) -> Handshake:
        return self.expected.handshake

    async def call(
        self,
        method: Any,
        params: Any,
        result_type: Any,
        *,
        timeout: float,
    ) -> CleanupResult | AckResult:
        del result_type, timeout
        self.methods.append(method)
        if method in ("begin_rescue", "close"):
            return AckResult()
        assert method == "cleanup"
        action = self.expected.cleanup_plan.actions[0]
        receipt = record_cleanup(
            self.expected.cleanup_plan,
            action.action_id,
            status="succeeded",
            evidence_code="released",
            duration_ms=1,
        )
        receipts = (receipt, receipt) if self.duplicate else (receipt,)
        return CleanupResult(receipts=receipts)

    async def terminate(self) -> None:
        self.terminated = True


@pytest.mark.asyncio
async def test_rescue_reconciles_every_action_and_records_aggregate(tmp_path: Path) -> None:
    rescue = descriptor(tmp_path)
    fake = FakeSupervisor(rescue)
    aggregate = await run_rescue(rescue, launch=lambda _: fake)  # type: ignore[arg-type]
    assert aggregate.complete
    assert len(aggregate.receipts) == 1
    assert fake.methods == ["begin_rescue", "cleanup", "close"]
    assert fake.terminated


@pytest.mark.asyncio
async def test_rescue_rejects_duplicate_receipts_and_reaps(tmp_path: Path) -> None:
    rescue = descriptor(tmp_path)
    fake = FakeSupervisor(rescue, duplicate=True)
    with pytest.raises(SupervisionError, match="missing or duplicate"):
        await run_rescue(rescue, launch=lambda _: fake)  # type: ignore[arg-type]
    assert fake.terminated
