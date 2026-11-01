"""Exercise the target shell wrapper, without allocating a target or calling a model."""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace
from typing import Any, cast

import pytest
from lop_osworld_v2_adapter.providers.aws import AwsProvider, _Clients

from local_operator.evaluation.adapters.api import EvaluationAdapter, RunCommandParams


def _params(command: str, **overrides: Any) -> RunCommandParams:
    return RunCommandParams(
        operation_id="command_001",
        episode_id="episode_001",
        command=command,
        timeout_s=overrides.get("timeout_s", 2.0),
        max_output_bytes=overrides.get("max_output_bytes", 128),
    )


def _provider(post: Any = None) -> AwsProvider:
    provider = object.__new__(AwsProvider)
    provider._public_ip = "127.0.0.1"

    def execute(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        assert url.endswith("/execute")
        assert payload["shell"] is False
        assert payload["timeout"] < timeout
        # This executes the exact argv sent to the existing control endpoint.
        completed = subprocess.run(
            payload["command"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "status": "success",
            "output": completed.stdout,
            "error": completed.stderr,
            "returncode": completed.returncode,
        }

    provider._clients = _Clients(
        ec2=None,
        scheduler=None,
        http_get=lambda _url, _timeout: 200,
        http_post_json=post or execute,
    )
    return provider


@pytest.mark.asyncio
async def test_keyed_worker_replay_runs_the_shell_exactly_once() -> None:
    import asyncio
    import os

    from local_operator.evaluation.adapters.rpc import (
        IncrementalReader,
        IncrementalWriter,
        RpcRequest,
    )
    from local_operator.evaluation.adapters.worker import Worker
    from tests.unit.evaluation.adapters.test_worker import exchange

    reads, writes = os.pipe()
    response_read, response_write = os.pipe()
    worker = Worker(reads, response_write)
    worker._state = "RUNNING"
    calls: list[str] = []
    provider = _provider()

    async def run(params: RunCommandParams) -> Any:
        calls.append(params.command)
        return await provider.run_command(params)

    # Only run_command is reachable in this RUNNING-state transport fixture.
    worker._adapter = cast(EvaluationAdapter, SimpleNamespace(run_command=run))
    task = asyncio.create_task(worker.run())
    writer, reader = IncrementalWriter(writes), IncrementalReader(response_read)
    params = _params("printf once").model_dump(mode="json")
    try:
        replies = [
            await exchange(
                writer,
                reader,
                RpcRequest(
                    jsonrpc="2.0",
                    id=number,
                    method="run_command",
                    params=params,
                ),
            )
            for number in (1, 2)
        ]
        assert calls == ["printf once"]
        assert replies[0].error is None
        assert replies[0].result == replies[1].result
        assert replies[0].result is not None
        assert replies[0].result["stdout"] == "once"
    finally:
        os.close(writes)
        assert await asyncio.wait_for(task, 2) == 0
        for fd in (reads, response_read, response_write):
            os.close(fd)


@pytest.mark.asyncio
async def test_supervised_command_transport_failure_poisoned_without_retry(tmp_path: Any) -> None:
    from local_operator.evaluation.adapters.supervisor import (
        AdapterSupervisor,
        HostVerifier,
        VerifiedAdapterSession,
    )
    from tests.unit.evaluation.adapters.test_supervisor import (
        RawSupervisor,
        observation,
    )

    verifier = HostVerifier("task", "episode_001", tmp_path)
    verifier.accept_initial(observation("task", "episode_001", 0))
    raw = RawSupervisor([ConnectionError("lost command reply")])
    # The fixture supplies only the raw-call and termination transport seam.
    session = VerifiedAdapterSession(cast(AdapterSupervisor, raw), verifier)
    with pytest.raises(ConnectionError, match="lost command reply"):
        await session.run_command(_params("true"), timeout=5)
    assert raw.calls == ["run_command"]
    assert raw.terminated
    assert session._poisoned


@pytest.mark.asyncio
async def test_shell_result_keeps_streams_and_nonzero_exit_separate() -> None:
    result = await _provider().run_command(_params("printf output; printf warning >&2; exit 7"))
    assert result.exit_code == 7
    assert result.stdout == "output"
    assert result.stderr == "warning"
    assert not result.timed_out
    assert not result.stdout_truncated


@pytest.mark.asyncio
async def test_output_cap_keeps_utf8_prefix_and_original_byte_count() -> None:
    result = await _provider().run_command(
        _params("printf abcdef; printf xyz >&2", max_output_bytes=4)
    )
    assert result.stdout == "abcd"
    assert result.stdout_bytes == 6
    assert result.stdout_truncated
    assert result.stderr == "xyz"
    assert not result.stderr_truncated


@pytest.mark.asyncio
async def test_target_timeout_returns_partial_output_not_a_transport_error() -> None:
    # exec removes the intermediate shell: some shells emit their own job-
    # termination diagnostic while a killed child exits, which is valid stderr
    # but would make an assertion about our explicit output platform-dependent.
    result = await _provider().run_command(
        _params("printf before; printf warning >&2; exec sleep 30", timeout_s=2)
    )
    assert result.exit_code is None
    assert result.timed_out
    assert result.stdout == "before"
    assert result.stderr == "warning"


@pytest.mark.asyncio
async def test_transport_error_is_not_a_command_exit() -> None:
    def fail(*args: Any) -> Any:
        raise ConnectionError("control server unreachable")

    with pytest.raises(ConnectionError, match="control server unreachable"):
        await _provider(fail).run_command(_params("true"))


@pytest.mark.asyncio
async def test_multibyte_output_is_not_split_at_the_cap() -> None:
    result = await _provider().run_command(_params("printf 'ééé'", max_output_bytes=3))
    assert result.stdout == "é"
    assert result.stdout_bytes == 6
    assert result.stdout_truncated


@pytest.mark.asyncio
async def test_malformed_control_reply_is_an_error() -> None:
    for body in (
        {"output": "{}", "returncode": 0},
        {"status": "success", "output": json.dumps({}), "returncode": 0},
        {"status": "success", "output": "", "returncode": 1, "error": "wrapper failed"},
    ):
        with pytest.raises((ValueError, RuntimeError)):
            await _provider(lambda *args: body).run_command(_params("true"))
