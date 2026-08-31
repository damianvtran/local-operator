"""POSIX adapter-tree supervision, host verification, and durable rescue."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import field_validator, model_validator

from local_operator.evaluation.adapters.api import (
    AckResult,
    AdapterResult,
    AdapterSelector,
    AskUserExchangeParams,
    AskUserExchangeResult,
    BeginRescueParams,
    CleanupParams,
    CleanupResult,
    CloseParams,
    ExecuteParams,
    ExecuteResult,
    Handshake,
    HelloParams,
    InspectRequirementsParams,
    ObservationResult,
    ObserveParams,
    PrepareParams,
    PrepareResult,
    RequirementsResult,
    RescueDescriptor,
    ResetStartParams,
    ScoreParams,
    ScoreResult,
    validate_execution,
    validate_observation,
)
from local_operator.evaluation.adapters.discovery import (
    ResolvedLaunch,
    resolve_launch,
    validate_resolved_launch,
    worker_argv,
)
from local_operator.evaluation.adapters.rpc import RpcClient
from local_operator.evaluation.evidence.media import (
    MediaValidationError,
    validate_media,
)
from local_operator.evaluation.evidence.models import canonical_digest
from local_operator.evaluation.lifecycle import (
    CleanupPlan,
    CleanupReceipt,
    aggregate_cleanup,
    record_cleanup,
)
from local_operator.evaluation.protocol import ArtifactRef, ProtocolModel
from local_operator.evaluation.receipts import ZERO_DIGEST, Digest

if os.name != "posix":  # pragma: no cover - import itself is the explicit diagnostic
    raise RuntimeError("evaluation adapter supervision requires POSIX process groups")

MAX_DIAGNOSTIC_TAIL = 64 * 1024
TERM_GRACE_SECONDS = 5.0
OWNER_FD_ENV = "LO_ADAPTER_OWNER_FD"
LAUNCH_IDENTITY_ENV = "LO_ADAPTER_LAUNCH_IDENTITY"
REQUEST_FD_ENV = "LO_ADAPTER_REQUEST_FD"
RESPONSE_FD_ENV = "LO_ADAPTER_RESPONSE_FD"
_RESCUE_FILE = "rescue.json"

# Locale and temporary-directory settings are behavior inputs, not credentials.
# Everything else is omitted instead of copying an ambient environment that may
# contain paid-provider, model-route, or benchmark secrets.
_ENV_ALLOW = frozenset({"LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "TEMP", "TMP"})
_DENIED_MARKERS = ("OPENAI", "OPENROUTER", "ANTHROPIC", "MODEL", "PROVIDER")


class SupervisionError(RuntimeError):
    pass


class _Tail:
    def __init__(self, limit: int = MAX_DIAGNOSTIC_TAIL) -> None:
        self._limit = limit
        self._chunks: deque[bytes] = deque()
        self._size = 0
        self._lock = threading.Lock()

    def append(self, chunk: bytes) -> None:
        with self._lock:
            self._chunks.append(chunk)
            self._size += len(chunk)
            while self._size > self._limit and self._chunks:
                excess = self._size - self._limit
                first = self._chunks[0]
                if len(first) <= excess:
                    self._chunks.popleft()
                    self._size -= len(first)
                else:
                    self._chunks[0] = first[excess:]
                    self._size -= excess

    def bytes(self) -> bytes:
        with self._lock:
            return b"".join(self._chunks)


def minimal_environment(
    source: Mapping[str, str],
    *,
    protocol_fds: Mapping[str, int],
) -> dict[str, str]:
    """Construct, never copy, the worker environment from a closed allowlist."""

    environment = {
        name: value
        for name, value in source.items()
        if name in _ENV_ALLOW and not any(marker in name.upper() for marker in _DENIED_MARKERS)
    }
    for name, fd in protocol_fds.items():
        if name not in {OWNER_FD_ENV, REQUEST_FD_ENV, RESPONSE_FD_ENV} or fd < 0:
            raise ValueError("unknown or invalid protocol descriptor")
        environment[name] = str(fd)
    return environment


def _reset_child_signals() -> None:
    """Undo supervisor masks/defaults without mutating host signal state."""

    signal.pthread_sigmask(signal.SIG_SETMASK, [])
    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGPIPE):
        signal.signal(number, signal.SIG_DFL)


def _owned_group(pgid: int) -> bool:
    try:
        return os.getpid() == pgid == os.getpgrp() == os.getpgid(os.getpid())
    except ProcessLookupError:
        return False


def _signal_owned_group(pgid: int, number: int) -> bool:
    """Refuse a stale/recycled PGID immediately before every signal."""

    if not _owned_group(pgid):
        return False
    try:
        os.killpg(pgid, number)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _owner_watch(owner_fd: int, pgid: int) -> None:
    try:
        while os.read(owner_fd, 4096):
            pass
    except OSError:
        pass
    finally:
        try:
            os.close(owner_fd)
        except OSError:
            pass
    if not _signal_owned_group(pgid, signal.SIGTERM):
        return
    time.sleep(TERM_GRACE_SECONDS)
    _signal_owned_group(pgid, signal.SIGKILL)


def supervise_worker(argv: Sequence[str]) -> int:
    """Remain group leader so descendants cannot outlive ownership loss."""

    if not argv:
        raise SupervisionError("supervisor requires a worker command")
    owner_fd = int(os.environ.pop(OWNER_FD_ENV))
    request_fd = int(os.environ[REQUEST_FD_ENV])
    response_fd = int(os.environ[RESPONSE_FD_ENV])
    pgid = os.getpid()
    if not _owned_group(pgid):
        raise SupervisionError("supervisor is not a fresh process-group leader")
    # Core owns group termination.  The leader ignores TERM so its live PID keeps
    # proving that the numeric PGID was not recycled during the grace interval.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    watcher = threading.Thread(target=_owner_watch, args=(owner_fd, pgid), daemon=True)
    watcher.start()
    child_environment = minimal_environment(
        os.environ,
        protocol_fds={REQUEST_FD_ENV: request_fd, RESPONSE_FD_ENV: response_fd},
    )
    identity_payload = os.environ.pop(LAUNCH_IDENTITY_ENV, "")
    try:
        resolved = ResolvedLaunch(**json.loads(identity_payload))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise SupervisionError("supervisor launch identity is missing or malformed") from error
    validate_resolved_launch(resolved)
    child = subprocess.Popen(
        list(argv),
        stdin=subprocess.DEVNULL,
        stdout=None,
        stderr=None,
        env=child_environment,
        close_fds=True,
        pass_fds=(request_fd, response_fd),
        preexec_fn=_reset_child_signals,
    )
    child.wait()
    # Worker EOF must not release descendants.  The leader waits until owner EOF
    # and is then removed with the entire group after the bounded grace period.
    while True:
        signal.pause()


class AdapterSupervisor:
    """Parent-owned handles for one isolated adapter process tree."""

    def __init__(self, selector: AdapterSelector, process: subprocess.Popen[bytes]) -> None:
        self.selector = selector
        self.process = process
        self.pgid = process.pid
        self.stdout_tail = _Tail()
        self.stderr_tail = _Tail()
        self._owner_fd = -1
        self._request_fd = -1
        self._response_fd = -1
        self.rpc: RpcClient | None = None
        self._drainers: list[threading.Thread] = []
        self._closed = False

    @classmethod
    def launch(
        cls,
        selector: AdapterSelector,
        *,
        environment: Mapping[str, str] | None = None,
    ) -> "AdapterSupervisor":
        resolved = resolve_launch(selector)
        owner_read, owner_write = os.pipe()
        request_read, request_write = os.pipe()
        response_read, response_write = os.pipe()
        protocol_fds = {
            OWNER_FD_ENV: owner_read,
            REQUEST_FD_ENV: request_read,
            RESPONSE_FD_ENV: response_write,
        }
        env = minimal_environment(environment or os.environ, protocol_fds=protocol_fds)
        env[LAUNCH_IDENTITY_ENV] = json.dumps(resolved.__dict__, sort_keys=True)
        argv = (
            selector.python_executable,
            "-I",
            "-s",
            "-E",
            "-m",
            "local_operator.evaluation.adapters.supervisor",
            "--supervise",
            *worker_argv(selector),
        )
        process: subprocess.Popen[bytes] | None = None
        try:
            # The supervisor cannot make subprocess.exec race-free, but comparing
            # captured dev/inode/mode immediately before spawn closes alias and
            # ordinary swap attacks at both parent and leader boundaries.
            validate_resolved_launch(resolved)
            process = subprocess.Popen(
                argv,
                cwd=selector.workspace,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                close_fds=True,
                pass_fds=(owner_read, request_read, response_write),
                start_new_session=True,
            )
            os.close(owner_read)
            os.close(request_read)
            os.close(response_write)
            owner_read = request_read = response_write = -1
            try:
                if process.pid != os.getpgid(process.pid):
                    raise SupervisionError("spawned supervisor is not its process-group leader")
            except ProcessLookupError as error:
                raise SupervisionError("adapter supervisor exited during startup") from error
            instance = cls(selector, process)
            instance._owner_fd = owner_write
            instance._request_fd = request_write
            instance._response_fd = response_read
            instance._start_drainers()
            instance.rpc = RpcClient(
                request_write,
                response_read,
                terminate=instance.terminate,
            )
            return instance
        except Exception:
            for fd in (
                owner_read,
                owner_write,
                request_read,
                request_write,
                response_read,
                response_write,
            ):
                if fd >= 0:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
            if process is not None:
                _terminate_process_group(process, process.pid)
            raise

    def _start_drainers(self) -> None:
        assert self.process.stdout is not None and self.process.stderr is not None
        for stream, tail in (
            (self.process.stdout, self.stdout_tail),
            (self.process.stderr, self.stderr_tail),
        ):
            thread = threading.Thread(target=_drain, args=(stream, tail), daemon=True)
            thread.start()
            self._drainers.append(thread)

    async def _call_raw(
        self,
        method: Any,
        params: ProtocolModel,
        result_type: type[AdapterResult],
        *,
        timeout: float,
    ) -> AdapterResult:
        if self.rpc is None:
            raise SupervisionError("adapter RPC is unavailable")
        payload = await self.rpc.call(method, params, timeout=timeout)
        return result_type.model_validate(payload, strict=True)

    async def handshake(self, *, timeout: float = 10.0) -> Handshake:
        result = await self._call_raw(
            "hello", HelloParams(selector=self.selector), Handshake, timeout=timeout
        )
        assert isinstance(result, Handshake)
        # Constructing Handshake already enforces every repeated exact pin; this
        # equality additionally prevents a caller from accepting another selector.
        if result.selector != self.selector:
            await self.terminate()
            raise SupervisionError("adapter handshake selector differs")
        return result

    async def terminate(self) -> None:
        if self._closed:
            return
        self._closed = True
        _terminate_process_group(self.process, self.pgid)
        self._close_fds()
        await asyncio.to_thread(self.process.wait)
        for thread in self._drainers:
            thread.join(timeout=1)

    async def close(self) -> None:
        if self._closed:
            return
        # Owner EOF is the normal teardown path and exercises the same rescue
        # invariant used when the parent dies unexpectedly.
        if self._owner_fd >= 0:
            os.close(self._owner_fd)
            self._owner_fd = -1
        await asyncio.to_thread(self.process.wait)
        self._closed = True
        self._close_fds()
        for thread in self._drainers:
            thread.join(timeout=1)

    def _close_fds(self) -> None:
        for name in ("_owner_fd", "_request_fd", "_response_fd"):
            fd = getattr(self, name)
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
                setattr(self, name, -1)


def _drain(stream: Any, tail: _Tail) -> None:
    try:
        while chunk := stream.read(65536):
            tail.append(chunk)
    except OSError:
        pass
    finally:
        stream.close()


def _terminate_process_group(process: subprocess.Popen[bytes], pgid: int) -> None:
    if process.poll() is not None:
        return
    try:
        identity_matches = process.pid == pgid == os.getpgid(process.pid)
    except ProcessLookupError:
        return
    if not identity_matches:
        raise SupervisionError("refusing to signal an unverified process group")
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=TERM_GRACE_SECONDS)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        if process.pid != pgid or os.getpgid(process.pid) != pgid:
            raise SupervisionError("process-group identity changed before SIGKILL")
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait(timeout=TERM_GRACE_SECONDS)


class HostVerifier:
    """Parent-owned episode truth; adapters never grant lifecycle authority."""

    def __init__(self, task_id: str, episode_id: str, artifact_root: Path) -> None:
        self.task_id = task_id
        self.episode_id = episode_id
        self.artifact_root = artifact_root
        self.current_observation: Any | None = None
        self._sequences: set[int] = set()
        self._outstanding_ask: tuple[AskUserExchangeParams, Digest] | None = None
        self._score_ids: set[str] = set()

    def accept_initial(self, observation: Any) -> None:
        if self.current_observation is not None or observation.sequence != 0:
            raise SupervisionError("initial adapter observation must be sequence zero")
        self._validate_observation_content(observation)
        self._sequences.add(observation.sequence)
        self.current_observation = observation

    def accept_output(self, observation: Any) -> None:
        self._validate_output(observation)
        self._sequences.add(observation.sequence)
        self.current_observation = observation

    def verify_current_snapshot(self, observation: Any) -> None:
        current = self.current_observation
        if current is None or observation != current:
            raise SupervisionError("adapter snapshot differs from the current observation")
        self._validate_observation_content(observation)

    def _validate_observation_content(self, observation: Any) -> None:
        if (observation.task_id, observation.episode_id) != (
            self.task_id,
            self.episode_id,
        ):
            raise SupervisionError("adapter observation belongs to another task or episode")
        validate_observation(observation)
        for frame in observation.frames:
            verify_artifact(self.artifact_root, frame.artifact)

    def validate_execution_result(self, params: ExecuteParams, result: ExecuteResult) -> None:
        if self.current_observation is None:
            raise SupervisionError("execution has no current observation")
        validate_execution(
            params,
            result,
            self.current_observation,
            seen_sequences=self._sequences,
        )
        self._validate_output(result.observation)

    def accept_execution(self, params: ExecuteParams, result: ExecuteResult) -> None:
        self.validate_execution_result(params, result)
        self._sequences.add(result.observation.sequence)
        self.current_observation = result.observation

    def _validate_output(self, observation: Any) -> None:
        current = self.current_observation
        if current is None or observation.sequence != current.sequence + 1:
            raise SupervisionError("adapter output must be the exact next sequence")
        self._validate_observation_content(observation)

    def begin_ask(self, params: AskUserExchangeParams) -> None:
        if params.episode_id != self.episode_id:
            raise SupervisionError("ask-user exchange belongs to another episode")
        if self._outstanding_ask is not None or params.answer is not None:
            raise SupervisionError("ask-user exchange must begin once without an answer")
        request_digest = self._ask_request_digest(params)
        self._outstanding_ask = (params, request_digest)

    def validate_ask_completion(
        self,
        params: AskUserExchangeParams,
        result: AskUserExchangeResult | None = None,
    ) -> None:
        outstanding = self._outstanding_ask
        request_digest = self._ask_request_digest(params)
        if (
            params.episode_id != self.episode_id
            or outstanding is None
            or outstanding[1] != request_digest
            or outstanding[0].ask_id != params.ask_id
            or (result is not None and result.ask_id != params.ask_id)
            or params.answer is None
        ):
            raise SupervisionError("ask-user response is stale, unsolicited, or mismatched")

    def finish_ask(self, params: AskUserExchangeParams, result: AskUserExchangeResult) -> None:
        self.validate_ask_completion(params, result)
        self._outstanding_ask = None

    @staticmethod
    def _ask_request_digest(params: AskUserExchangeParams) -> Digest:
        # The answer and retry operation key are completion data. Everything
        # model-visible in the initiating request remains immutable.
        return canonical_digest(
            "adapter-ask-user-request-v1",
            params.model_dump(mode="json", exclude={"operation_id", "answer"}),
        )

    def accept_score(self, params: ScoreParams, result: ScoreResult) -> None:
        if params.episode_id != self.episode_id:
            raise SupervisionError("score belongs to another episode")
        if result.score.score_id in self._score_ids:
            raise SupervisionError("score artifact identity is duplicated")
        if result.score.details is not None:
            reference = ArtifactRef.model_validate(
                result.score.details.model_dump(mode="json"), strict=True
            )
            verify_artifact(self.artifact_root, reference)
        # Consume identity only after every referenced byte passed verification,
        # so a missing artifact can be repaired and retried safely.
        self._score_ids.add(result.score.score_id)


class VerifiedAdapterSession:
    """Typed application surface that cannot bypass parent-owned verification."""

    def __init__(
        self,
        supervisor: AdapterSupervisor,
        verifier: HostVerifier,
        *,
        rescue_required: Callable[[], None] | None = None,
    ) -> None:
        self._supervisor = supervisor
        self.verifier = verifier
        self._rescue_required = rescue_required or (lambda: None)
        self._rescue_descriptor_id: Digest | None = None
        self._cleanup_plan: CleanupPlan | None = None
        self._poisoned = False

    def mark_rescue_persisted(self, descriptor_id: Digest) -> None:
        self._rescue_descriptor_id = descriptor_id

    def _ensure_usable(self) -> None:
        if self._poisoned:
            raise SupervisionError("adapter session is poisoned; rescue is required")

    async def _poison_after_ambiguous_mutation(self) -> None:
        if self._poisoned:
            return
        self._poisoned = True
        self._rescue_required()
        await asyncio.shield(self._supervisor.terminate())

    async def _mutating_call(
        self,
        method: Any,
        params: ProtocolModel,
        result_type: type[AdapterResult],
        *,
        timeout: float,
        validate: Callable[[AdapterResult], None],
    ) -> AdapterResult:
        self._ensure_usable()
        try:
            result = await self._supervisor._call_raw(method, params, result_type, timeout=timeout)
            validate(result)
            return result
        except BaseException:
            await self._poison_after_ambiguous_mutation()
            raise

    async def inspect_requirements(
        self, params: InspectRequirementsParams, *, timeout: float
    ) -> RequirementsResult:
        result = await self._supervisor._call_raw(
            "inspect_requirements", params, RequirementsResult, timeout=timeout
        )
        assert isinstance(result, RequirementsResult)
        return result

    async def prepare(self, params: PrepareParams, *, timeout: float) -> PrepareResult:
        if params.episode_id != self.verifier.episode_id:
            raise SupervisionError("prepare belongs to another episode")
        self._ensure_usable()
        if self._rescue_descriptor_id is None:
            raise SupervisionError("prepare requires a persisted rescue descriptor")

        def validate(result: AdapterResult) -> None:
            if not isinstance(result, PrepareResult):
                raise SupervisionError("prepare returned the wrong result")
            if result.cleanup_plan.episode_id != self.verifier.episode_id:
                raise SupervisionError("cleanup plan belongs to another episode")

        result = await self._mutating_call(
            "prepare", params, PrepareResult, timeout=timeout, validate=validate
        )
        assert isinstance(result, PrepareResult)
        self._cleanup_plan = result.cleanup_plan
        return result

    async def reset_start(self, params: ResetStartParams, *, timeout: float) -> ObservationResult:
        if (params.task_id, params.episode_id) != (
            self.verifier.task_id,
            self.verifier.episode_id,
        ):
            raise SupervisionError("reset belongs to another task or episode")
        self._ensure_usable()
        ack = await self._mutating_call(
            "reset_start",
            params,
            AckResult,
            timeout=timeout,
            validate=lambda result: (
                None
                if isinstance(result, AckResult)
                else (_ for _ in ()).throw(SupervisionError("reset returned the wrong result"))
            ),
        )
        assert isinstance(ack, AckResult)
        initial = await self._supervisor._call_raw(
            "observe",
            ObserveParams(episode_id=params.episode_id),
            ObservationResult,
            timeout=timeout,
        )
        assert isinstance(initial, ObservationResult)
        self.verifier.accept_initial(initial.observation)
        return initial

    async def observe(self, params: ObserveParams, *, timeout: float) -> ObservationResult:
        self._ensure_usable()
        if params.episode_id != self.verifier.episode_id:
            raise SupervisionError("observe belongs to another episode")
        try:
            result = await self._supervisor._call_raw(
                "observe", params, ObservationResult, timeout=timeout
            )
            assert isinstance(result, ObservationResult)
            self.verifier.verify_current_snapshot(result.observation)
        except BaseException:
            await self._poison_after_ambiguous_mutation()
            raise
        return result

    async def execute(self, params: ExecuteParams, *, timeout: float) -> ExecuteResult:
        current = self.verifier.current_observation
        if current is None:
            raise SupervisionError("execute has no current observation")
        params.action_batch.validate_for(current)
        self._ensure_usable()
        result = await self._mutating_call(
            "execute",
            params,
            ExecuteResult,
            timeout=timeout,
            validate=lambda value: (
                self.verifier.validate_execution_result(params, value)
                if isinstance(value, ExecuteResult)
                else (_ for _ in ()).throw(SupervisionError("execute returned the wrong result"))
            ),
        )
        assert isinstance(result, ExecuteResult)
        self.verifier.accept_execution(params, result)
        return result

    def begin_ask(self, params: AskUserExchangeParams) -> None:
        # Every application entry point passes the same poison gate, so a
        # poisoned session cannot accumulate new episode state locally either.
        self._ensure_usable()
        self.verifier.begin_ask(params)

    async def finish_ask(
        self, params: AskUserExchangeParams, *, timeout: float
    ) -> AskUserExchangeResult:
        self._ensure_usable()
        # This prospective check rejects changed prompts before an adapter call.
        self.verifier.validate_ask_completion(params)
        result = await self._mutating_call(
            "ask_user_exchange",
            params,
            AskUserExchangeResult,
            timeout=timeout,
            validate=lambda value: (
                self.verifier.validate_ask_completion(params, value)
                if isinstance(value, AskUserExchangeResult)
                else (_ for _ in ()).throw(SupervisionError("ask returned the wrong result"))
            ),
        )
        assert isinstance(result, AskUserExchangeResult)
        self.verifier.finish_ask(params, result)
        return result

    async def score(self, params: ScoreParams, *, timeout: float) -> ScoreResult:
        if params.episode_id != self.verifier.episode_id:
            raise SupervisionError("score belongs to another episode")
        self._ensure_usable()
        result = await self._mutating_call(
            "score",
            params,
            ScoreResult,
            timeout=timeout,
            validate=lambda value: (
                self.verifier.accept_score(params, value)
                if isinstance(value, ScoreResult)
                else (_ for _ in ()).throw(SupervisionError("score returned the wrong result"))
            ),
        )
        assert isinstance(result, ScoreResult)
        return result

    async def cleanup(self, params: CleanupParams, *, timeout: float) -> tuple[CleanupReceipt, ...]:
        self._ensure_usable()
        plan = self._cleanup_plan
        if (
            plan is None
            or params.cleanup_plan != plan
            or params.cleanup_plan.episode_id != self.verifier.episode_id
        ):
            raise SupervisionError("cleanup differs from the prepared cleanup plan")
        selected = {
            action.action_id: action
            for action in plan.actions
            if action.action_id in params.action_ids
        }
        if set(selected) != set(params.action_ids):
            raise SupervisionError("cleanup selects an action outside the prepared plan")
        receipts: tuple[CleanupReceipt, ...] = ()

        def validate(value: AdapterResult) -> None:
            nonlocal receipts
            if not isinstance(value, CleanupResult):
                raise SupervisionError("cleanup returned the wrong result")
            if len(value.outcomes) != len(selected) or {
                outcome.action_id for outcome in value.outcomes
            } != set(selected):
                raise SupervisionError("cleanup outcomes differ from selected actions")
            receipts = tuple(
                record_cleanup(
                    plan,
                    outcome.action_id,
                    status=outcome.status,
                    evidence_code=outcome.evidence_code,
                    duration_ms=outcome.duration_ms,
                )
                for outcome in value.outcomes
            )

        await self._mutating_call(
            "cleanup", params, CleanupResult, timeout=timeout, validate=validate
        )
        return receipts

    async def close(self, params: CloseParams, *, timeout: float) -> AckResult:
        if params.episode_id not in (None, self.verifier.episode_id):
            raise SupervisionError("close belongs to another episode")
        result = await self._supervisor._call_raw("close", params, AckResult, timeout=timeout)
        assert isinstance(result, AckResult)
        return result


def verify_artifact(root: Path, reference: ArtifactRef) -> bytes:
    """Read one content-addressed artifact without following attacker links."""

    name = reference.sha256
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    root_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        fd = os.open(name, flags, dir_fd=root_fd)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size != reference.byte_count:
                raise SupervisionError("artifact is not a matching regular file")
            data = bytearray()
            while chunk := os.read(fd, min(65536, reference.byte_count + 1 - len(data))):
                data.extend(chunk)
                if len(data) > reference.byte_count:
                    raise SupervisionError("artifact exceeds its declared size")
        finally:
            os.close(fd)
    except OSError as error:
        raise SupervisionError("artifact path is unsafe or unavailable") from error
    finally:
        os.close(root_fd)
    raw = bytes(data)
    if hashlib.sha256(raw).hexdigest() != name:
        raise SupervisionError("artifact digest differs")
    try:
        validate_media(raw, reference.media_type)
    except MediaValidationError as error:
        raise SupervisionError("artifact media differs") from error
    return raw


def persist_rescue(root: Path, descriptor: RescueDescriptor) -> Path:
    """Parent-owned atomic persistence before any prepare/reset side effect."""

    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = root / _RESCUE_FILE
    fd, temporary = tempfile.mkstemp(prefix=".rescue-", dir=root)
    try:
        os.fchmod(fd, 0o600)
        payload = descriptor.to_canonical_json()
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("rescue write made no progress")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.replace(temporary, path)
        directory_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return path


def load_pending_rescue(root: Path) -> RescueDescriptor | None:
    """Explicitly load one descriptor; importing or startup never scans."""

    path = root / _RESCUE_FILE
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags)
    except FileNotFoundError:
        return None
    except OSError as error:
        raise SupervisionError("rescue descriptor path is unsafe") from error
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
            raise SupervisionError("rescue descriptor ownership mode is unsafe")
        payload = os.read(fd, 1024 * 1024 + 1)
        if len(payload) > 1024 * 1024:
            raise SupervisionError("rescue descriptor is oversized")
    finally:
        os.close(fd)
    return RescueDescriptor.from_canonical_json(payload)


class RescueAggregate(ProtocolModel):
    descriptor_id: Digest
    receipts: tuple[CleanupReceipt, ...]
    cleanup_result_id: Digest
    complete: bool
    aggregate_id: Digest = ZERO_DIGEST

    @field_validator("receipts", mode="before")
    @classmethod
    def _freeze_receipts(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _identify(self) -> "RescueAggregate":
        expected = canonical_digest(
            "adapter-rescue-aggregate-v1",
            self.model_dump(mode="json", exclude={"aggregate_id"}),
        )
        if self.aggregate_id not in (ZERO_DIGEST, expected):
            raise ValueError("rescue aggregate identity differs")
        object.__setattr__(self, "aggregate_id", expected)
        return self


async def run_rescue(
    descriptor: RescueDescriptor,
    *,
    launch: Callable[[AdapterSelector], AdapterSupervisor] = AdapterSupervisor.launch,
) -> RescueAggregate:
    """Use a fresh exact worker and reconcile every declared cleanup action."""

    supervisor = launch(descriptor.selector)
    receipts: list[CleanupReceipt] = []
    try:
        handshake = await supervisor.handshake()
        if handshake != descriptor.handshake:
            raise SupervisionError("rescue worker handshake differs from persisted pins")
        begin = await supervisor._call_raw(
            "begin_rescue",
            BeginRescueParams(
                operation_id=f"rescue-begin-{descriptor.descriptor_id[:32]}",
                descriptor=descriptor,
                descriptor_id=descriptor.descriptor_id,
                episode_id=descriptor.episode_id,
                cleanup_plan_id=descriptor.cleanup_plan.cleanup_plan_id,
                selector_digest=canonical_digest("adapter-rescue-selector-v1", descriptor.selector),
                handshake_digest=canonical_digest(
                    "adapter-rescue-handshake-v1", descriptor.handshake
                ),
            ),
            AckResult,
            timeout=10.0,
        )
        assert isinstance(begin, AckResult)
        for action in descriptor.cleanup_plan.actions:
            # One action per call preserves each action's own timeout/attempt cap
            # and makes missing or duplicate receipts independently detectable.
            operation_id = f"rescue-{descriptor.descriptor_id[:16]}-{action.action_id}"
            result = await supervisor._call_raw(
                "cleanup",
                CleanupParams(
                    operation_id=operation_id,
                    cleanup_plan=descriptor.cleanup_plan,
                    action_ids=(action.action_id,),
                ),
                CleanupResult,
                timeout=action.timeout_ms / 1000 * action.max_attempts,
            )
            assert isinstance(result, CleanupResult)
            if len(result.outcomes) != 1 or result.outcomes[0].action_id != action.action_id:
                raise SupervisionError("rescue cleanup returned missing or duplicate outcomes")
            outcome = result.outcomes[0]
            receipts.append(
                record_cleanup(
                    descriptor.cleanup_plan,
                    outcome.action_id,
                    status=outcome.status,
                    evidence_code=outcome.evidence_code,
                    duration_ms=outcome.duration_ms,
                )
            )
        cleanup = aggregate_cleanup(descriptor.cleanup_plan, receipts)
        closed = await supervisor._call_raw(
            "close",
            CloseParams(
                operation_id=f"rescue-close-{descriptor.descriptor_id[:32]}",
                episode_id=descriptor.episode_id,
            ),
            AckResult,
            timeout=10.0,
        )
        assert isinstance(closed, AckResult)
        return RescueAggregate(
            descriptor_id=descriptor.descriptor_id,
            receipts=tuple(receipts),
            cleanup_result_id=cleanup.cleanup_result_id,
            complete=not cleanup.rescue_required,
        )
    finally:
        await supervisor.terminate()


def main() -> int:
    if len(sys.argv) >= 3 and sys.argv[1] == "--supervise":
        return supervise_worker(sys.argv[2:])
    print("adapter supervisor is an internal module", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
