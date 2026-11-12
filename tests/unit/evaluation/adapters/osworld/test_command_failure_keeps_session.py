"""One failed command must not cost the episode: the session stays usable.

WHY THIS IS ITS OWN FILE. ``test_remote_command.py`` proves the decoder answers
instead of raising; this proves what that buys, through the layer that actually
owns the poison decision. The sealed run ``t27b-t003-20261108-090532`` (episode
``ep-640575925798``) ended ``failed / bridge-wedged`` with ``score: null``: the
adapter raised on a command the target refused, ``_mutating_call`` read that as
an ambiguous mutation, poisoned the session, and every later command AND the
score died with "adapter session is poisoned; rescue is required".

So the two properties pinned here are the two the run lost: the NEXT command
succeeds, and scoring still runs. ``run_command`` goes through the real
``AwsProvider.run_command`` (stubbed HTTP) rather than a double, because the
raise used to come from there and a double would only restate the fix.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from lop_osworld_v2_adapter.providers.aws import AwsProvider, _Clients

from local_operator.evaluation.action_surface import LEGACY_ACTION_SURFACE
from local_operator.evaluation.adapters.api import ScoreParams, ScoreResult
from local_operator.evaluation.adapters.supervisor import (
    HostVerifier,
    VerifiedAdapterSession,
)
from local_operator.evaluation.evidence.models import EvidenceArtifactRef, ScoreArtifact
from tests.unit.evaluation.adapters.test_supervisor import observation

EPISODE = "episode"


class _Worker:
    """The raw-supervisor seam, with ``run_command`` running the real provider."""

    action_surface = LEGACY_ACTION_SURFACE

    def __init__(
        self,
        bodies: list[Any],
        score: ScoreResult,
        *,
        http_get: Any = None,
        execute_error: BaseException | None = None,
    ) -> None:
        self.bodies = bodies
        self.score = score
        self.calls: list[str] = []
        self.terminated = False
        # The guest's liveness endpoint (``/platform``) as the provider's probe
        # reads it: a status code, or an exception for a guest that is gone.
        self.http_get = http_get or (lambda url, timeout: 200)
        self.execute_error = execute_error

    async def terminate(self) -> None:
        self.terminated = True

    async def _call_raw(
        self,
        method: str,
        params: Any,
        result_type: type[Any],
        *,
        timeout: float,
        **kwargs: Any,
    ) -> Any:
        del result_type, timeout, kwargs
        self.calls.append(method)
        if method == "score":
            return self.score
        if method == "execute" and self.execute_error is not None:
            raise self.execute_error
        if method != "run_command":
            raise AssertionError(f"unexpected method {method}")

        def http_post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
            del url, payload, timeout
            body = self.bodies.pop(0)
            if isinstance(body, BaseException):
                raise body
            return body

        provider = object.__new__(AwsProvider)
        provider._public_ip = "127.0.0.1"
        provider._clients = _Clients(
            ec2=None,
            scheduler=None,
            http_get=self.http_get,
            http_post_json=http_post_json,
        )
        return await provider.run_command(params)


def _score(tmp_path: Path) -> ScoreResult:
    data = b'{"detail":true}'
    digest = hashlib.sha256(data).hexdigest()
    (tmp_path / digest).write_bytes(data)
    return ScoreResult(
        score=ScoreArtifact(
            status="scored",
            binary=1,
            details=EvidenceArtifactRef(
                sha256=digest, media_type="application/json", byte_count=len(data)
            ),
        )
    )


def _params(operation_id: str, command: str) -> Any:
    from local_operator.evaluation.adapters.api import RunCommandParams

    return RunCommandParams(
        operation_id=operation_id,
        episode_id=EPISODE,
        command=command,
        timeout_s=5.0,
        max_output_bytes=1024,
    )


def _answer(output: str, *, status: str = "success", returncode: int = 0) -> dict[str, Any]:
    return {"status": status, "output": output, "error": "", "returncode": returncode}


def test_a_failed_command_leaves_the_session_usable_and_scoring_works(tmp_path: Path) -> None:
    verifier = HostVerifier("task", EPISODE, tmp_path)
    verifier.accept_initial(observation("task", EPISODE, 0))
    good = {
        "exit_code": 0,
        "timed_out": False,
        "stdout": "after\n",
        "stderr": "",
        "stdout_bytes": 6,
        "stderr_bytes": 0,
        "stdout_truncated": False,
        "stderr_truncated": False,
    }
    worker = _Worker(
        [
            # 1. the measured body: the guest killed the wrapper (SIGTERM)
            _answer("", returncode=-15),
            # 2. the next command must still run
            _answer(json.dumps(good)),
        ],
        _score(tmp_path),
    )
    session = VerifiedAdapterSession(worker, verifier)  # type: ignore[arg-type]

    async def run() -> None:
        failed = await session.run_command(_params("command_001", "compose"), timeout=5)
        assert failed.error is not None and failed.exit_code is None

        # The session was never poisoned: this is the next command, not a refusal.
        after = await session.run_command(_params("command_002", "echo after"), timeout=5)
        assert (after.stdout, after.error) == ("after\n", None)

        # ...and scoring still runs, which is the "score: null" the run lost.
        scored = await session.score(
            ScoreParams(operation_id="score", episode_id=EPISODE), timeout=5
        )
        assert scored.score.binary == 1
        assert not session._poisoned

    asyncio.run(run())


def test_an_unclassified_command_failure_still_poisons(tmp_path: Path) -> None:
    """A failure the provider cannot classify as transport still ends the session.

    Only a ``requests`` transport failure earns the liveness probe (see the
    lost-reply tests below); anything else escaping ``run_command`` -- here a
    bare builtin ``ConnectionError`` no ``requests`` call raises -- keeps the
    pre-existing rule: a command that may have run must never be replayed, so
    the session ends rather than guessing.
    """

    verifier = HostVerifier("task", EPISODE, tmp_path)
    verifier.accept_initial(observation("task", EPISODE, 0))
    worker = _Worker([ConnectionError("lost command reply")], _score(tmp_path))
    session = VerifiedAdapterSession(worker, verifier)  # type: ignore[arg-type]

    async def run() -> None:
        try:
            await session.run_command(_params("command_001", "compose"), timeout=5)
        except ConnectionError:
            pass
        else:  # pragma: no cover - the assertion below says what went wrong
            raise AssertionError("a lost command reply must fail the call")
        assert session._poisoned
        assert worker.terminated

    asyncio.run(run())


# --------------------------------------------------------------------------
# A LOST REPLY from a guest that still answers (sealed runs
# ``t29a-t002-20261111-185136`` row 670 and ``t29a-t004-20261111-185213`` row
# 5996). Both episodes ran a trivial command (``grep``/``sed``, ``pdffonts``),
# ``requests`` raised a ``ConnectionError`` wrapping ``ReadTimeoutError`` from
# the BODY read, the session was poisoned, and every later command, the final
# GUI batch, ``score`` and ``cleanup`` failed: ``score: null``. The tests below
# pin the split: a live guest -> unknown-outcome answer, session alive; a dead
# guest -> poison exactly as before; the GUI batch -> unchanged.
# --------------------------------------------------------------------------

#: The measured exception, shaped as ``requests`` raised it: its text names the
#: guest's address, which must never reach the model.
_GUEST_IP = "34.239.130.39"


def _sealed_read_timeout() -> BaseException:
    import requests

    return requests.ConnectionError(
        f"HTTPConnectionPool(host='{_GUEST_IP}', port=5000): Read timed out."
    )


_GOOD = {
    "exit_code": 0,
    "timed_out": False,
    "stdout": "after\n",
    "stderr": "",
    "stdout_bytes": 6,
    "stderr_bytes": 0,
    "stdout_truncated": False,
    "stderr_truncated": False,
}


def test_a_lost_reply_from_a_live_guest_is_answered_and_the_episode_continues(
    tmp_path: Path,
) -> None:
    from lop_osworld_v2_adapter.remote_command import UNKNOWN_OUTCOME_CAUTION

    verifier = HostVerifier("task", EPISODE, tmp_path)
    verifier.accept_initial(observation("task", EPISODE, 0))
    probes: list[str] = []

    def http_get(url: str, timeout: float) -> int:
        probes.append(url)
        return 200

    worker = _Worker(
        [_sealed_read_timeout(), _answer(json.dumps(_GOOD))],
        _score(tmp_path),
        http_get=http_get,
    )
    session = VerifiedAdapterSession(worker, verifier)  # type: ignore[arg-type]

    async def run() -> None:
        lost = await session.run_command(
            _params("command_001", "grep YEAR /tmp/syl.txt"), timeout=5
        )
        # An unknown outcome, said as one: no invented exit code or timeout
        # flag, the caution that forbids a blind re-issue, and no guest address.
        assert (lost.exit_code, lost.timed_out, lost.stdout, lost.stderr) == (None, False, "", "")
        assert lost.error is not None
        assert "reply to this command was lost" in lost.error
        assert "ConnectionError" in lost.error
        assert lost.error.endswith(UNKNOWN_OUTCOME_CAUTION)
        assert _GUEST_IP not in lost.error

        # The very next command runs: the sealed run's row 682 was a refusal.
        after = await session.run_command(_params("command_002", "echo after"), timeout=5)
        assert (after.stdout, after.error) == ("after\n", None)

        # ...and scoring runs, which is the ``score: null`` both runs lost.
        scored = await session.score(
            ScoreParams(operation_id="score", episode_id=EPISODE), timeout=5
        )
        assert scored.score.binary == 1

    asyncio.run(run())
    assert not session._poisoned and not worker.terminated
    # Exactly one probe, of the cheap liveness endpoint -- never a replay.
    assert probes == ["http://127.0.0.1:5000/platform"]


def test_a_lost_reply_from_a_dead_guest_still_poisons(tmp_path: Path) -> None:
    import requests

    verifier = HostVerifier("task", EPISODE, tmp_path)
    verifier.accept_initial(observation("task", EPISODE, 0))
    original = _sealed_read_timeout()

    def http_get(url: str, timeout: float) -> int:
        raise requests.ConnectTimeout("guest gone")

    worker = _Worker([original], _score(tmp_path), http_get=http_get)
    session = VerifiedAdapterSession(worker, verifier)  # type: ignore[arg-type]

    async def run() -> None:
        try:
            await session.run_command(_params("command_001", "grep x"), timeout=5)
        except requests.ConnectionError as error:
            # The ORIGINAL loss escapes, not the probe's failure.
            assert error is original
        else:  # pragma: no cover - the assertion below says what went wrong
            raise AssertionError("a lost reply from a dead guest must fail the call")

    asyncio.run(run())
    assert session._poisoned and worker.terminated


def test_a_reply_that_outruns_the_wall_clock_deadline_is_a_lost_reply(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """``requests`` times out per socket read, so the provider bounds the wall.

    Both sealed losses took 75.3/75.4 s against a 70 s socket timeout -- the
    status line arrived, then the body stalled -- which had already consumed the
    harness's RPC slack. A stub that never returns stands in for that stall.
    """

    import threading

    from lop_osworld_v2_adapter.providers import aws

    monkeypatch.setattr(aws, "_COMMAND_SOCKET_SLACK_S", 0.0)
    verifier = HostVerifier("task", EPISODE, tmp_path)
    verifier.accept_initial(observation("task", EPISODE, 0))
    release = threading.Event()

    class _Stalled(_Worker):
        async def _call_raw(self, method: str, params: Any, result_type: Any, **kw: Any) -> Any:
            if method != "run_command":
                return await super()._call_raw(method, params, result_type, **kw)

            def http_post_json(url: str, payload: dict[str, Any], timeout: float) -> Any:
                release.wait(30)
                raise AssertionError("the stalled read must not be awaited")

            provider = object.__new__(AwsProvider)
            provider._public_ip = "127.0.0.1"
            provider._clients = _Clients(
                ec2=None,
                scheduler=None,
                http_get=lambda url, timeout: 200,
                http_post_json=http_post_json,
            )
            return await provider.run_command(params)

    worker = _Stalled([], _score(tmp_path))
    session = VerifiedAdapterSession(worker, verifier)  # type: ignore[arg-type]
    params = _params("command_001", "pdffonts out.pdf").model_copy(update={"timeout_s": 0.2})

    async def run() -> Any:
        try:
            return await session.run_command(params, timeout=5)
        finally:
            # Released INSIDE the loop: ``asyncio.run`` joins the default
            # executor on exit, so the abandoned read must end before it does.
            release.set()

    lost = asyncio.run(run())
    assert lost.error is not None and "TimeoutError" in lost.error
    assert not session._poisoned


def test_a_gui_batch_transport_failure_still_poisons(tmp_path: Path) -> None:
    """The control: ``apply_actions``' poison semantics are not touched.

    The same transport loss on ``execute`` keeps today's path -- no budget, so
    no probe, and the session ends -- because a GUI batch's lost reply is owned
    by ``guest_loss`` and the harness's ambiguous-delivery gate, not by the
    command split.
    """

    import pytest

    from local_operator.evaluation.adapters.rpc import RpcErrorDetail, RpcRemoteError
    from tests.unit.evaluation.adapters.test_supervisor import _execute_params

    verifier = HostVerifier("task", EPISODE, tmp_path)
    verifier.accept_initial(observation("task", EPISODE, 0))
    lost = RpcRemoteError(
        "adapter_error",
        "adapter operation failed",
        RpcErrorDetail(
            exception_type="ConnectionError",
            message="Read timed out.",
            method="execute",
            operation_id="exec-0",
        ),
    )
    probes: list[str] = []
    worker = _Worker(
        [],
        _score(tmp_path),
        http_get=lambda url, timeout: probes.append(url) or 200,
        execute_error=lost,
    )
    session = VerifiedAdapterSession(worker, verifier)  # type: ignore[arg-type]

    async def run() -> None:
        with pytest.raises(RpcRemoteError):
            await session.execute(_execute_params(verifier.current_observation), timeout=5)

    asyncio.run(run())
    assert session._poisoned and worker.terminated
    assert probes == []


def test_the_harness_rpc_deadline_leaves_room_for_the_lost_reply_probe() -> None:
    """Bind the harness's RPC deadline to the provider's own worst case.

    The provider answers a lost reply only after its wall-clock wait
    (``timeout_s + _COMMAND_SOCKET_SLACK_S``) plus one probe whose ``requests``
    timeout applies to connect and read separately. An RPC deadline at or under
    that sum fires first and tears the channel down -- the episode is lost
    anyway, just later. At the old ``timeout_s + 15`` the margin was ~1 s.
    """

    from lop_osworld_v2_adapter.providers import aws

    from local_operator.evaluation.session_arm import _make_run_command

    seen: list[float] = []

    class _Session:
        async def run_command(self, params: Any, *, timeout: float) -> None:
            seen.append(timeout)

    timeout_s = 60.0

    async def run() -> None:
        await _make_run_command(_Session(), EPISODE)("true", timeout_s)

    asyncio.run(run())
    worst_case = (
        timeout_s + aws._COMMAND_SOCKET_SLACK_S + 2 * aws._COMMAND_REPLY_LOST_PROBE_TIMEOUT_S
    )
    assert seen and seen[0] - worst_case >= 5.0
