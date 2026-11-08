"""Exercise the target shell wrapper, without allocating a target or calling a model."""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace
from typing import Any, cast

import pytest
from lop_osworld_v2_adapter.providers.aws import AwsProvider, _Clients
from lop_osworld_v2_adapter.remote_command import (
    UNKNOWN_OUTCOME_CAUTION,
    answer_from_error_body,
)

from local_operator.evaluation.adapters.api import (
    EvaluationAdapter,
    RunCommandParams,
    RunCommandResult,
)


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
async def test_malformed_control_reply_is_a_result_not_an_error() -> None:
    """A payload this layer cannot use is the TARGET's failure, not the call's.

    It used to raise, which the parent reads as an adapter failure on a
    mutating call: the session is poisoned and the episode is lost. The three
    bodies here are one per way that can happen -- no ``status``/``returncode``
    at all, a payload that is not a command result, and a wrapper that exited
    nonzero after writing its stderr (the sealed run's shape).
    """

    cases = (
        # No payload at all and no server status: the wrapper never reported.
        ({"output": "", "returncode": 0}, "finished without reporting a result"),
        (
            {"status": "success", "output": json.dumps({}), "returncode": 0},
            "is not a command result",
        ),
        (
            {"status": "success", "output": "", "returncode": 1, "error": "wrapper failed"},
            "wrapper failed",
        ),
    )
    for body, expected in cases:
        result = await _provider(lambda *args, body=body: body).run_command(_params("true"))
        assert isinstance(result, RunCommandResult)
        assert result.exit_code is None
        assert not result.timed_out
        assert result.error is not None and expected in result.error
        assert (result.stdout, result.stderr) == ("", "")


@pytest.mark.asyncio
async def test_a_wrapper_killed_by_a_signal_is_a_result_that_names_it() -> None:
    """The sealed run's exact body: the guest deadline SIGTERM'd the wrapper.

    ``t27b-t003-20261108-090532`` (episode ``ep-640575925798``) answered a
    command with ``{'status': 'success', 'output': '', 'error': '',
    'returncode': -15}``; the resulting ``RuntimeError`` poisoned the session,
    three later commands failed, and ``score`` failed with them (``score:
    null``). ``-15`` is a signal in an exit code's costume, so the message names
    it rather than leaving a reader to look up the number.
    """

    body = {"status": "success", "output": "", "error": "", "returncode": -15}
    result = await _provider(lambda *args: body).run_command(_params("compose"))

    assert result.exit_code is None
    # Not a timeout: nothing in the body says the command's own deadline
    # elapsed, and this layer does not invent one.
    assert not result.timed_out
    assert result.error is not None
    assert "SIGTERM" in result.error
    # The server's own status said "success" and the wrapper died anyway: quoting
    # it beside ``exit_code: none`` reads as a contradiction, so it is dropped.
    assert "success" not in result.error
    # The shell is in its own session (``start_new_session=True``), so a dead
    # wrapper says nothing about it -- and the sealed run's agent re-issued a
    # near-identical command one turn later because nothing said so.
    assert UNKNOWN_OUTCOME_CAUTION in result.error


@pytest.mark.asyncio
async def test_a_guest_error_answer_is_a_result_carrying_its_reason() -> None:
    """The control server answering an error body is the target talking."""

    body = {
        "status": "error",
        "output": "",
        "error": "Command timed out after 60 seconds",
        "returncode": 1,
    }
    result = await _provider(lambda *args: body).run_command(_params("sleep 99"))

    assert result.exit_code is None
    assert not result.timed_out
    assert result.error is not None and "timed out after 60 seconds" in result.error


@pytest.mark.asyncio
async def test_the_guest_deadline_answer_is_a_result_and_warns_about_the_shell() -> None:
    """The body the guest really sends when ITS OWN deadline fires.

    ``desktop_env/server/main.py:107-112`` (upstream ``36a6adf4``) catches every
    exception from ``/execute`` -- ``subprocess.TimeoutExpired`` among them -- and
    answers 500 ``{"status": "error", "message": str(e)}``. No ``output``, no
    ``returncode``: the old filter rejected that body and the call raised, so a
    command the TARGET reported on was still treated as a channel failure and
    poisoned the session. It is also the same physical event as a dead wrapper --
    the handler kills only its direct child -- so the answer carries the same
    caution.
    """

    import requests

    message = (
        "Command '['python3', '-c', '\\nimport base64, codecs, json, ...', "
        "'ZWNobyBoaQ==', '60.0', '262144']' timed out after 65.0 seconds"
    )
    body = {"status": "error", "message": message}

    class Response:
        content = json.dumps(body).encode()
        status_code = 500

    def post(*args: Any) -> Any:
        raise requests.HTTPError("500 Server Error", response=Response())

    result = await _provider(post).run_command(_params("true"))

    assert result.exit_code is None
    assert not result.timed_out
    assert result.error is not None
    assert "timed out after 65.0 seconds" in result.error
    assert UNKNOWN_OUTCOME_CAUTION in result.error


@pytest.mark.asyncio
async def test_a_huge_guest_diagnostic_keeps_its_conclusion_and_the_caution() -> None:
    """A quoted argv must not push the actionable part out of the bound.

    ``str(TimeoutExpired)`` puts the whole command list in FRONT of "timed out
    after N seconds", and this layer now quotes that message, so a head-only cut
    would drop exactly the fact the model needs.
    """

    import requests

    message = "Command '['" + "x" * 4000 + "']' timed out after 65.0 seconds"
    body = {"status": "error", "message": message}

    class Response:
        content = json.dumps(body).encode()
        status_code = 500

    def post(*args: Any) -> Any:
        raise requests.HTTPError("500 Server Error", response=Response())

    result = await _provider(post).run_command(_params("true"))

    assert result.error is not None and len(result.error) <= 4096
    assert "timed out after 65.0 seconds" in result.error
    assert UNKNOWN_OUTCOME_CAUTION in result.error


@pytest.mark.asyncio
async def test_a_command_cannot_match_its_own_wrapper_in_the_process_table() -> None:
    """The measured root cause, not its symptom.

    ``pkill -f``/``pgrep -f`` match a process's WHOLE argv. The sealed run's
    command contained ``pkill -f citysheet``, and ``payload_for`` put the command
    TEXT in the wrapper's argv, so the command SIGTERM'd the wrapper running it
    (``returncode: -15``, 6.0 s into a 60 s deadline) and the failure was blamed
    on a deadline that never fired. The command is base64'd now, so it is not in
    the process table at all.

    The pattern is assembled at runtime on purpose: a literal would sit in the
    shell's own argv and match itself, which would prove nothing about the
    wrapper.

    ``argv=`` is the load-bearing half on Darwin, where ``pgrep`` does not report
    a calling process's ancestors at all (measured: it misses the wrapper even
    with the command text in its argv). ``wrapper=`` is the half that carries on
    the Linux guest, whose ``pgrep -f`` matches ``/proc/<pid>/cmdline`` -- the
    same bytes ``ps`` prints -- and is where the sealed run's self-kill happened.
    """

    from lop_osworld_v2_adapter.remote_command import payload_for

    command = (
        'pat=city; snap=sheet; term="$pat$snap"; '
        "parent=$(ps -o ppid= -p $$ | tr -d ' '); "
        'hits=$(pgrep -f "$term" | tr -d \' \' | tr "\\n" " "); '
        'case " $hits " in *" $parent "*) w=MATCHED;; *) w=CLEAN;; esac; '
        'case "$(ps -o args= -p "$parent")" in *"$term"*) a=PRESENT;; *) a=ABSENT;; esac; '
        'echo "wrapper=$w argv=$a"'
    )

    argv = " ".join(str(part) for part in payload_for(_params(command))["command"])
    assert "citysheet" not in argv

    # And the behaviour, through the real wrapper this module sends.
    result = await _provider().run_command(_params(command, timeout_s=15.0))
    assert result.stdout.strip() == "wrapper=CLEAN argv=ABSENT"
    assert result.exit_code == 0


@pytest.mark.asyncio
async def test_a_2xx_body_this_layer_cannot_read_is_a_result_not_a_raise() -> None:
    """The target ANSWERED, unreadably; that is not a lost reply.

    ``http_post_json`` raises for these before ``decode_result`` is reached --
    ``response.json()`` on a non-JSON body, and its own ``ValueError`` for a JSON
    body that is not an object -- so the decode branch could never see them.
    """

    for error in (
        ValueError("guest control server returned a non-object body"),
        json.JSONDecodeError("Expecting value", "<html>502 Bad Gateway</html>", 0),
    ):

        def post(*args: Any, error: Exception = error) -> Any:
            raise error

        result = await _provider(post).run_command(_params("true"))

        assert result.exit_code is None
        assert not result.timed_out
        assert result.error is not None and UNKNOWN_OUTCOME_CAUTION in result.error


@pytest.mark.asyncio
async def test_a_non_success_status_does_not_discard_a_valid_payload() -> None:
    """QA N1: a server complaint is not a reason to throw the command away.

    The payload is written by the wrapper AFTER it has waited for the shell, so
    it is the only evidence about what the command did; the server's own status
    is about the request and says nothing about the command. Dropping the payload
    (the previous rule) lost the answer without gaining any certainty.
    """

    payload = {"exit_code": 0, "timed_out": False, "stdout": "hi\n", "stderr": ""}
    for body in (
        {"status": "error", "output": json.dumps(payload), "error": "", "returncode": 0},
        {"status": "error", "output": json.dumps(payload), "error": "", "returncode": 3},
    ):
        result = await _provider(lambda *args, body=body: body).run_command(_params("echo hi"))

        assert result.exit_code == 0
        assert result.stdout == "hi\n"
        assert result.error is None


@pytest.mark.asyncio
async def test_an_oversized_payload_is_capped_not_refused() -> None:
    """A wrapper that ignored the cap reports truncation, not a failure.

    The shipped wrapper enforces the cap itself, so this body is what a
    drifted or older guest would send. Raising ``ValueError`` for it (the old
    behaviour) lost the command's real exit status AND poisoned the session.
    """

    payload = {
        "exit_code": 0,
        "timed_out": False,
        "stdout": "x" * 5000,
        "stderr": "y" * 5000,
        "stdout_bytes": 5000,
        "stderr_bytes": 5000,
        "stdout_truncated": False,
        "stderr_truncated": False,
    }
    body = {"status": "success", "output": json.dumps(payload), "returncode": 0, "error": ""}
    result = await _provider(lambda *args: body).run_command(_params("emit", max_output_bytes=100))

    assert result.exit_code == 0
    assert result.error is None
    assert (len(result.stdout.encode()), len(result.stderr.encode())) == (100, 100)
    assert result.stdout_truncated and result.stderr_truncated
    assert (result.stdout_bytes, result.stderr_bytes) == (5000, 5000)


@pytest.mark.asyncio
async def test_a_non_2xx_body_carrying_the_guest_answer_is_decoded() -> None:
    """A four-key body on a non-2xx is decoded rather than raised on.

    DEFENSIVE, not observed: upstream's ``/execute`` answers 200 with this
    four-key body and 500 with its ``{status, message}`` envelope -- the
    envelope is what
    ``test_the_guest_deadline_answer_is_a_result_and_warns_about_the_shell``
    pins. This arm exists so a guest build that DOES answer a command outcome
    with an error status still reaches the model; only a body that is not the
    guest's answer may still fail the call.
    """

    import requests

    body = {"status": "error", "output": "", "error": "Command timed out", "returncode": 1}

    class Response:
        content = json.dumps(body).encode()
        status_code = 500

    def post(*args: Any) -> Any:
        raise requests.HTTPError("500 Server Error", response=Response())

    result = await _provider(post).run_command(_params("true"))
    assert result.error is not None and "Command timed out" in result.error


@pytest.mark.asyncio
async def test_a_non_2xx_body_that_is_not_the_guest_answer_still_fails_the_call() -> None:
    """A proxy page is a CHANNEL failure: the command may or may not have run."""

    import requests

    class Response:
        content = b"<html>502 Bad Gateway</html>"

    def post(*args: Any) -> Any:
        raise requests.HTTPError("502 Bad Gateway", response=Response())

    with pytest.raises(requests.HTTPError):
        await _provider(post).run_command(_params("true"))


def test_answer_from_error_body_reads_only_the_guest_answer_shapes() -> None:
    """Both shapes the AMI's handler really emits, and nothing else."""

    body = {"status": "success", "output": "", "error": "", "returncode": 0}
    assert answer_from_error_body(json.dumps(body).encode()) == body
    # The handler's exception envelope (its own deadline included).
    deadline = {"status": "error", "message": "Command '['sleep']' timed out after 65 seconds"}
    assert answer_from_error_body(json.dumps(deadline).encode()) == deadline
    for content in (
        b"",
        b"<html>502</html>",
        b'{"message": "upstream connect error"}',
        b"[]",
        # An envelope with nothing to quote is not an answer.
        b'{"status": "error"}',
        # A word the control server does not use is not the control server.
        b'{"status": "weird", "output": "", "returncode": 0}',
        b'{"output": "", "returncode": 0}',
    ):
        assert answer_from_error_body(content) is None


@pytest.mark.asyncio
async def test_the_failure_text_is_bounded() -> None:
    """A huge guest diagnostic must not become a huge RPC frame."""

    body = {"status": "error", "output": "", "error": "E" * 100_000, "returncode": 1}
    result = await _provider(lambda *args: body).run_command(_params("true"))
    assert result.error is not None and len(result.error) <= 4096
    # The caution is the last sentence, so a bound that dropped the tail would
    # delete the one instruction a re-issuing model has to read.
    assert UNKNOWN_OUTCOME_CAUTION in result.error


@pytest.mark.asyncio
async def test_a_target_failure_survives_the_real_worker_wire_as_a_result() -> None:
    """The whole point: the failure crosses the RPC as a RESULT, not as an error.

    The sealed run's poison started here -- an ``adapter_error`` frame built from
    the adapter's raise. A result carrying ``error`` must instead come back as
    ``result`` (no ``error`` frame), so the parent's mutating-call gate never
    sees a failure at all.
    """

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
    # The measured body: the guest's deadline killed the wrapper (SIGTERM).
    body = {"status": "success", "output": "", "error": "", "returncode": -15}
    provider = _provider(lambda *args: body)

    async def run(params: RunCommandParams) -> Any:
        return await provider.run_command(params)

    worker._adapter = cast(EvaluationAdapter, SimpleNamespace(run_command=run))
    task = asyncio.create_task(worker.run())
    writer, reader = IncrementalWriter(writes), IncrementalReader(response_read)
    params = _params("compose", timeout_s=5.0).model_dump(mode="json")
    try:
        reply = await exchange(
            writer,
            reader,
            RpcRequest(jsonrpc="2.0", id=1, method="run_command", params=params),
        )
        assert reply.error is None, reply.error
        assert reply.result is not None
        assert reply.result["exit_code"] is None
        assert reply.result["timed_out"] is False
        assert "SIGTERM" in reply.result["error"]
    finally:
        os.close(writes)
        assert await asyncio.wait_for(task, 2) == 0
        for fd in (reads, response_read, response_write):
            os.close(fd)
