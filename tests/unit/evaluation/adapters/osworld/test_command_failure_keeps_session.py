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

    def __init__(self, bodies: list[Any], score: ScoreResult) -> None:
        self.bodies = bodies
        self.score = score
        self.calls: list[str] = []
        self.terminated = False

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
            http_get=lambda url, timeout: 200,
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


def test_a_lost_command_reply_still_poisons(tmp_path: Path) -> None:
    """The residual case, pinned: no reply at all may already have mutated.

    A command that may have run must never be replayed, so the session ends
    rather than guessing -- the same rule a lost ``execute`` reply follows. This
    is the ONLY way a command can still wedge an episode.
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
